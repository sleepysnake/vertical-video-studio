"""Offline integration: real API adapter, cached ASR, planning and FFmpeg output.

The HTTP session supplies fixed model replies. This verifies the independent
tool's execution contract, not a live model's semantic judgment quality.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw

from studio.bailian import BailianClient
from studio.models import Person, VideoItem
from studio.pipeline import JobConfig, Pipeline, work_dir
from studio.render import default_font, probe
from studio.toolchain import detect_tools


class Response:
    status_code = 200

    def __init__(self, content: dict, number: int):
        self.data = {"id": f"offline-request-{number}", "model": "offline-compatible-model",
                     "choices": [{"finish_reason": "stop", "message": {
                         "content": json.dumps(content, ensure_ascii=False)}}],
                     "usage": {"prompt_tokens": 100 * number,
                               "completion_tokens": 20 * number,
                               "total_tokens": 120 * number}}
        self.text = json.dumps(self.data, ensure_ascii=False)

    def raise_for_status(self):
        return None

    def json(self):
        return self.data


class TextOnlySession:
    """Reject requests outside the configured edit/review/caption API sequence."""
    def __init__(self, endpoint: str, edit_reply: dict, caption_reply: dict,
                 review_reply: dict | None = None):
        self.endpoint = endpoint
        self.replies = [edit_reply, deepcopy(edit_reply) if review_reply is None else review_reply,
                        caption_reply]
        self.calls = []

    def get(self, *_args, **_kwargs):
        raise AssertionError("Cached ASR must not perform upload-policy/task/download GETs")

    def post(self, url, **kwargs):
        if url != self.endpoint or len(self.calls) >= len(self.replies):
            raise AssertionError("Only three configured text API requests are allowed")
        self.calls.append({"url": url, "payload": deepcopy(kwargs["json"]),
                           "timeout": kwargs["timeout"]})
        return Response(self.replies[len(self.calls) - 1], len(self.calls))


def fixture_transcription() -> tuple[dict, dict, dict, dict]:
    summary = "也就是说我今天并没有吃多少"
    prefix, abandoned, replacement = "我今天下午", "只喝了一个奶", "只喝了一杯奶茶"
    hesitant, clean = "咦我吃吃得不多", "我吃得不多"
    numbers = "每天摄入1800大卡"
    sentences = []
    cursor = 0
    bounds = []
    for text in (summary, prefix + abandoned + replacement, hesitant, numbers):
        first = cursor
        words = []
        for offset, character in enumerate(text):
            words.append({"begin_time": 400 + 120 * cursor,
                          "end_time": 400 + 120 * (cursor + 1), "text": character,
                          "punctuation": "。" if offset == len(text) - 1 else ""})
            cursor += 1
        sentences.append({"begin_time": words[0]["begin_time"], "end_time": words[-1]["end_time"],
                          "text": text + "。", "words": words})
        bounds.append((first, cursor))
    start = bounds[1][0] + len(prefix)
    end = start + len(abandoned)
    edit = {"cuts": [{"kind": "repair_restart", "start_word": start, "end_word": end,
                      "replacement_range": {"start_word": end, "end_word": bounds[1][1]},
                      "reason": "前半表达未完成且说错量词，后半完整改口替代该片段",
                      "joined_text": prefix + replacement}], "pause_compressions": [],
            "display_only_suggestions": [], "uncertain": []}
    segments = []
    token_cursor = 0
    for source, display in ((summary, summary), (prefix + replacement, prefix + replacement),
                            (hesitant, clean), (numbers, numbers)):
        last = token_cursor + len(source)
        segments.append({"start_token": token_cursor, "end_token": last, "text": display})
        token_cursor = last
    captions = {"segments": segments, "changes": [], "uncertain": []}
    raw = {"transcripts": [{"text": "".join(item["text"] for item in sentences),
                            "sentences": sentences}]}
    expected = {"summary": summary, "repaired": prefix + replacement,
                "hesitant": hesitant, "clean": clean, "numbers": numbers,
                "cut_start": (400 + 120 * start) / 1000,
                "cut_end": (400 + 120 * end) / 1000,
                "removed_indices": list(range(start, end)),
                "hesitant_indices": list(range(*bounds[2]))}
    return raw, edit, captions, expected


class ContextualApiPipelineTests(unittest.TestCase):
    def test_cached_asr_calls_three_text_stages_and_reuses_result_on_second_render(self):
        encoder, inspector = detect_tools()
        font = default_font()
        if not encoder[0] or not inspector[0] or not font:
            self.skipTest("FFmpeg, FFprobe and a usable font are required for media integration")
        ffmpeg, ffprobe = encoder[0], inspector[0]
        raw, edit_reply, caption_reply, expected = fixture_transcription()
        endpoint = "http://offline-test.invalid/custom-text-api"
        session = TextOnlySession(endpoint, edit_reply, caption_reply)
        client = BailianClient("http://offline-test.invalid/asr", endpoint,
                               "offline-test-key", "offline-asr-model", "offline-compatible-model",
                               session, upload_url="http://offline-test.invalid/upload",
                               task_url="http://offline-test.invalid/task/{task_id}")
        with tempfile.TemporaryDirectory(prefix="contextual-api-pipeline-") as temporary:
            folder = Path(temporary)
            video, template, output = folder / "source.mp4", folder / "template.png", folder / "output"
            source_duration = 8.0
            subprocess.run([ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                            f"color=c=gray:s=320x180:r=25:d={source_duration}", "-f", "lavfi", "-i",
                            f"sine=frequency=440:sample_rate=48000:duration={source_duration}",
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
                    "task_id": "offline-already-finished"}), encoding="utf-8")
            (cache / "transcription.json").write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
            events = []
            pipeline = Pipeline(client, config, lambda *event: events.append(event), threading.Event())
            with patch.object(client, "upload_audio", side_effect=AssertionError("ASR upload forbidden")) as upload, \
                    patch.object(client, "submit_asr", side_effect=AssertionError("ASR submit forbidden")) as submit, \
                    patch.object(client, "get_task", side_effect=AssertionError("ASR polling forbidden")) as poll, \
                    patch.object(client, "download_transcription", side_effect=AssertionError("ASR download forbidden")) as download:
                target = pipeline.run_one(item)
                self.assertTrue(target.is_file())
                self.assertEqual(len(session.calls), 3)
                payloads = [json.loads(call["payload"]["messages"][1]["content"]) for call in session.calls]
                self.assertEqual([payload["phase"] for payload in payloads],
                                 ["edit", "semantic_review", "caption"])
                self.assertEqual(payloads[0]["context_scope"], "whole_video")
                self.assertIn(expected["summary"], payloads[0]["full_transcript"])
                self.assertIn(expected["numbers"], payloads[0]["full_transcript"])
                self.assertEqual([call["payload"]["max_tokens"] for call in session.calls], [8192] * 3)
                self.assertEqual([call["payload"]["model"] for call in session.calls],
                                 ["offline-compatible-model"] * 3)
                self.assertEqual(payloads[1]["draft_plan"]["cuts"], edit_reply["cuts"])
                self.assertEqual(payloads[1]["full_transcript"], payloads[0]["full_transcript"])
                self.assertEqual(payloads[2]["applied_cuts"], edit_reply["cuts"])
                kept_indices = [token["word_index"] for token in payloads[2]["kept_tokens"]]
                self.assertTrue(set(expected["removed_indices"]).isdisjoint(kept_indices))
                self.assertTrue(set(expected["hesitant_indices"]).issubset(kept_indices))

                source_captions = json.loads((cache / "captions.json").read_text(encoding="utf-8"))
                self.assertEqual(source_captions["mode"], "model_authored")
                self.assertEqual(source_captions["edit_plan"]["version"], 6)
                self.assertEqual(source_captions["edit_plan"]["mode"], "model_authored")
                self.assertFalse(source_captions["caption_fallback_groups"])
                self.assertTrue(source_captions["coverage_audit"]["retained_words_assigned_once_in_order"])
                self.assertEqual([caption["text"] for caption in source_captions["captions"]],
                                 [expected["summary"], expected["repaired"], expected["clean"], expected["numbers"]])
                cuts = source_captions["edit_plan"]["cuts"]
                self.assertEqual(len(cuts), 1)
                self.assertAlmostEqual(cuts[0]["start"], expected["cut_start"], places=3)
                self.assertAlmostEqual(cuts[0]["end"], expected["cut_end"], places=3)
                self.assertEqual(cuts[0]["removed_word_indices"], expected["removed_indices"])
                cleaned = source_captions["captions"][2]
                self.assertEqual(cleaned["source_word_indices"], expected["hesitant_indices"])
                self.assertEqual(cleaned["source_text"], expected["hesitant"])
                self.assertEqual(sorted(item["source_text"] for item in cleaned["display_cleanup"]), ["吃", "咦"])
                self.assertTrue(all(item["audio_kept"] for item in cleaned["display_cleanup"]))

                timeline = json.loads((cache / "caption_timeline.json").read_text(encoding="utf-8"))
                clean_mapping = next(caption for caption in timeline["captions"] if caption["text"] == expected["clean"])
                self.assertTrue(all(not word["removed"] for word in clean_mapping["asr_words"]))
                self.assertTrue(all(item["output_start"] is not None and item["audio_kept"]
                                    for item in clean_mapping["display_cleanup"]))
                original_seconds, _, _ = probe(video, ffprobe)
                final_seconds, final_width, final_height = probe(target, ffprobe)
                self.assertEqual((final_width, final_height), (540, 960))
                removed_seconds = expected["cut_end"] - expected["cut_start"]
                self.assertAlmostEqual(final_seconds, original_seconds - removed_seconds, delta=.08)

                audit = json.loads((cache / "model-calls.json").read_text(encoding="utf-8"))
                self.assertEqual([record["phase"] for record in audit["requests"]],
                                 ["edit", "semantic_review", "caption"])
                self.assertEqual([record["request_number"] for record in audit["requests"]], [1, 2, 3])
                self.assertEqual([record["attempts"][0]["request_id"] for record in audit["requests"]],
                                 ["offline-request-1", "offline-request-2", "offline-request-3"])
                self.assertEqual(audit["statistics"]["logical_requests"], 3)
                self.assertEqual(audit["statistics"]["http_attempts"], 3)
                self.assertEqual(audit["statistics"]["usage"], {"prompt_tokens": 600,
                                 "completion_tokens": 120, "total_tokens": 720})
                self.assertEqual(pipeline.last_run["text_api_logical_requests"], 3)
                self.assertNotIn("offline-test-key", json.dumps(audit))

                events.clear()
                self.assertEqual(pipeline.run_one(item), target)
                self.assertEqual(len(session.calls), 3)
                self.assertEqual(len(client.request_records), 3)
                self.assertEqual(pipeline.last_run["text_api_logical_requests"], 0)
                self.assertEqual(pipeline.last_run["text_api_http_attempts"], 0)
                self.assertTrue(any("使用已保存的字幕" in event[2] for event in events))
                audit_after_cache = json.loads((cache / "model-calls.json").read_text(encoding="utf-8"))
                self.assertEqual(audit_after_cache["requests"], [])
                self.assertEqual(audit_after_cache["statistics"]["logical_requests"], 3)
                for forbidden in (upload, submit, poll, download):
                    forbidden.assert_not_called()


if __name__ == "__main__":
    unittest.main()
