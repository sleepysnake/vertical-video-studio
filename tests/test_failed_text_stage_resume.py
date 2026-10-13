"""A failed text stage must stay resumable without accepting ASR fallback.

This integration uses cached ASR and fixed offline API replies.  It verifies
failure/retry ownership and actual rendering, not a live model's judgment.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw

from studio.bailian import BailianClient, ModelOutputError
from studio.models import Person, VideoItem
from studio.pipeline import JobConfig, Pipeline, work_dir
from studio.render import default_font
from studio.toolchain import detect_tools
from tests.test_contextual_api_pipeline import TextOnlySession, fixture_transcription


class FailedTextStageResumeTests(unittest.TestCase):
    def test_edit_failure_preserves_cached_asr_and_retry_builds_a_new_owned_plan(self):
        encoder, inspector = detect_tools()
        font = default_font()
        if not encoder[0] or not inspector[0] or not font:
            self.skipTest("FFmpeg, FFprobe and a usable font are required")
        ffmpeg, ffprobe = encoder[0], inspector[0]
        raw, edit_reply, caption_reply, expected = fixture_transcription()
        endpoint = "http://offline-test.invalid/custom-text-api"
        session = TextOnlySession(endpoint, edit_reply, caption_reply)
        client = BailianClient("http://offline-test.invalid/asr", endpoint,
                               "offline-test-key", "offline-asr-model", "offline-compatible-model",
                               session, upload_url="http://offline-test.invalid/upload",
                               task_url="http://offline-test.invalid/task/{task_id}")

        with tempfile.TemporaryDirectory(prefix="failed-text-stage-resume-") as temporary:
            folder = Path(temporary)
            video, template, output = folder / "source.mp4", folder / "template.png", folder / "output"
            subprocess.run([ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                            "color=c=gray:s=320x180:r=25:d=8", "-f", "lavfi", "-i",
                            "sine=frequency=440:sample_rate=48000:duration=8",
                            "-shortest", "-c:v", "libx264", "-c:a", "aac", str(video)],
                           check=True, capture_output=True, timeout=30)
            image = Image.new("RGBA", (540, 960), "#2D65A4")
            ImageDraw.Draw(image).rectangle((0, 326, 539, 629), fill=(0, 0, 0, 0))
            image.save(template)
            item = VideoItem(video, Person("示例医院", "医生", "测试人"))
            config = JobConfig(template, output, "仅供测试", (0, 326, 540, 304), ffmpeg, ffprobe,
                               font=font)
            cache = work_dir(item, output)
            cache.mkdir(parents=True)
            fingerprint = {"path": str(video.resolve()), "size": video.stat().st_size,
                           "mtime_ns": video.stat().st_mtime_ns,
                           "asr_url": client.asr_url, "upload_url": client.upload_url,
                           "task_url": client.task_url, "asr_model": client.asr_model}
            (cache / "task.json").write_text(json.dumps({"fingerprint": fingerprint,
                    "task_id": "already-finished-asr"}), encoding="utf-8")
            raw_path = cache / "transcription.json"
            raw_path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
            original_asr_bytes = raw_path.read_bytes()
            old_plan = {"run_id": "older-completed-run", "review_count": 5,
                        "decisions": [{"action": "KEEP", "need_review": True,
                                       "source_text": "旧计划中的复核项"}], "cuts": []}
            plan_path = cache / "edit-plan.json"
            plan_path.write_text(json.dumps(old_plan, ensure_ascii=False), encoding="utf-8")
            old_plan_bytes = plan_path.read_bytes()
            events = []
            pipeline = Pipeline(client, config, lambda *event: events.append(event), threading.Event())

            with patch.object(client, "upload_audio", side_effect=AssertionError("ASR upload forbidden")) as upload, \
                    patch.object(client, "submit_asr", side_effect=AssertionError("ASR submit forbidden")) as submit, \
                    patch.object(client, "get_task", side_effect=AssertionError("ASR polling forbidden")) as poll, \
                    patch.object(client, "download_transcription", side_effect=AssertionError("ASR download forbidden")) as download:
                with patch.object(client, "chat_json", side_effect=ModelOutputError(
                        "offline edit request exhausted retries")) as failed_text:
                    with self.assertRaisesRegex(ModelOutputError, "剪辑模型调用未成功"):
                        pipeline.run_one(item)
                    self.assertEqual(failed_text.call_count, 1)
                    self.assertEqual(failed_text.call_args.kwargs["phase"], "edit")

                failed_run = dict(pipeline.last_run)
                self.assertEqual(failed_run["status"], "失败")
                self.assertFalse(failed_run["plan_ready"])
                self.assertTrue(failed_run["asr_cache_reused"])
                self.assertNotEqual(failed_run["run_id"], old_plan["run_id"])
                self.assertNotIn("plan_path", failed_run)
                self.assertEqual(plan_path.read_bytes(), old_plan_bytes)
                self.assertEqual(session.calls, [])
                self.assertEqual(raw_path.read_bytes(), original_asr_bytes)
                self.assertFalse((cache / "captions.json").exists())

                partial_path = cache / "captions.partial.json"
                progress = json.loads(partial_path.read_text(encoding="utf-8"))["progress"]
                self.assertEqual((progress["edited"], progress["reviewed"], progress["captioned"]), (0, 0, 0))
                self.assertEqual(progress["cuts"], [])
                self.assertEqual(progress["captions"], [])
                self.assertEqual(progress["fallback_groups"], [])

                target = pipeline.run_one(item)
                self.assertTrue(target.is_file())
                self.assertEqual(len(session.calls), 3)
                payloads = [json.loads(call["payload"]["messages"][1]["content"])
                            for call in session.calls]
                self.assertEqual([payload["phase"] for payload in payloads],
                                 ["edit", "semantic_review", "caption"])
                successful_run = pipeline.last_run
                self.assertEqual(successful_run["status"], "完成")
                self.assertTrue(successful_run["plan_ready"])
                self.assertTrue(successful_run["asr_cache_reused"])
                self.assertNotEqual(successful_run["run_id"], failed_run["run_id"])
                self.assertNotEqual(successful_run["run_id"], old_plan["run_id"])
                self.assertEqual(successful_run["plan_path"], str(plan_path))
                actual_plan = json.loads(plan_path.read_text(encoding="utf-8"))
                self.assertEqual(actual_plan["run_id"], successful_run["run_id"])
                self.assertEqual(actual_plan["version"], 6)
                self.assertEqual(actual_plan["mode"], "model_authored")
                self.assertEqual(len(actual_plan["cuts"]), 1)
                self.assertEqual(actual_plan["cuts"][0]["removed_word_indices"], expected["removed_indices"])
                source_captions = json.loads((cache / "captions.json").read_text(encoding="utf-8"))
                self.assertEqual(source_captions["caption_fallback_groups"], [])
                self.assertEqual(source_captions["uncertain"], [])
                self.assertTrue(source_captions["coverage_audit"]["retained_words_assigned_once_in_order"])
                self.assertEqual([caption["text"] for caption in source_captions["captions"]],
                                 [expected["summary"], expected["repaired"], expected["clean"], expected["numbers"]])
                self.assertFalse(partial_path.exists())
                self.assertEqual(raw_path.read_bytes(), original_asr_bytes)
                runs = json.loads((cache / "runs.json").read_text(encoding="utf-8"))["runs"]
                self.assertEqual([entry["run_id"] for entry in runs],
                                 [failed_run["run_id"], successful_run["run_id"]])
                self.assertEqual([entry["plan_ready"] for entry in runs], [False, True])
                for forbidden in (upload, submit, poll, download):
                    forbidden.assert_not_called()


if __name__ == "__main__":
    unittest.main()
