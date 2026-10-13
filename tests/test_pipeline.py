from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import patch
import requests

from PIL import Image, ImageDraw

from studio.alignment import align_captions, build_timeline
from studio.bailian import ApiError, BailianClient, ModelOutputError, endpoints_from_base_url, saved_base_url
from studio.captions import Word, build_captions, extract_words, shorten_captions, validate_caption_document
from studio.edits import build_cut_plan, detect_long_silence_cuts, retime_captions
from studio.edit_plan import build_edit_plan
from studio.models import (DEFAULT_ASR_MODEL, DEFAULT_ASR_URL, DEFAULT_TEXT_URL, Person,
                           VideoItem, scan_folder, slot_from_display_rect)
from studio.pipeline import JobConfig, Pipeline, captions_path, output_path, work_dir
from studio.prompts import PromptSet
from studio.caption_style import CaptionStyle
from studio.render import default_font, graphics, hospital_and_department, probe, read_captions, render_video
from studio.sentence_flow import asr_sentences, build_sentence_project
from studio.toolchain import detect_tools


TRANSCRIPTION = {
    "transcripts": [{"sentences": [{"begin_time": 200, "end_time": 2200,
        "text": "武汉同济院的医生。", "words": [
            {"begin_time": 200, "end_time": 1050, "text": "武汉同济院", "punctuation": ""},
            {"begin_time": 1050, "end_time": 1450, "text": "的", "punctuation": ""},
            {"begin_time": 1450, "end_time": 2200, "text": "医生", "punctuation": "。"},
        ]}]}]
}


class FakeResponse:
    def __init__(self, data=None, status_code=200):
        self.data = data or {}
        self.status_code = status_code
        self.text = json.dumps(self.data, ensure_ascii=False)

    def json(self):
        return self.data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(self.text)


class FakeSession:
    def __init__(self):
        self.calls = []
        self.polls = 0
        self.uncertain = []

    def get(self, url, **kwargs):
        self.calls.append(("GET", url, kwargs))
        if url.endswith("/uploads"):
            return FakeResponse({"data": {
                "max_file_size_mb": 100, "upload_dir": "folder", "upload_host": "https://oss.example/upload",
                "oss_access_key_id": "id", "signature": "sig", "policy": "policy",
                "x_oss_object_acl": "private", "x_oss_forbid_overwrite": "true"}})
        if "/tasks/" in url:
            self.polls += 1
            if self.polls == 1:
                return FakeResponse({"output": {"task_status": "PENDING"}})
            return FakeResponse({"output": {"task_status": "SUCCEEDED", "results": [
                {"subtask_status": "SUCCEEDED", "transcription_url": "https://oss.example/result.json"}]}})
        if url.endswith("result.json"):
            return FakeResponse(TRANSCRIPTION)
        raise AssertionError(url)

    def post(self, url, **kwargs):
        self.calls.append(("POST", url, kwargs))
        if url == "https://oss.example/upload":
            assert list(kwargs["files"])[-1] == "file"
            return FakeResponse()
        if url.endswith("/transcription"):
            assert kwargs["headers"]["X-DashScope-Async"] == "enable"
            assert kwargs["headers"]["X-DashScope-OssResourceResolve"] == "enable"
            assert kwargs["json"]["input"]["file_urls"][0].startswith("oss://")
            return FakeResponse({"output": {"task_id": "task-123"}})
        if url.endswith("/chat/completions"):
            user = kwargs["json"]["messages"][1]["content"]
            try:
                context = json.loads(user)
            except ValueError:
                context = {}
            if context.get("phase") in {"edit", "semantic_review"}:
                message = {"cuts": [], "pause_compressions": [],
                           "display_only_suggestions": [], "uncertain": []}
            else:
                message = {"segments": [{"start_token": 0, "end_token": 3,
                                         "text": "武汉同济医院的医生"}],
                           "changes": [{"from": "武汉同济院", "to": "武汉同济医院",
                                        "reason": "ASR误识别"}],
                           "uncertain": self.uncertain}
            return FakeResponse({"choices": [{"message": {"content": json.dumps(message, ensure_ascii=False)}}]})
        raise AssertionError(url)


class PipelineTests(unittest.TestCase):
    def test_transcription_get_retries_a_broken_json_response(self):
        class BrokenResponse(FakeResponse):
            def json(self):
                raise ValueError("truncated")

        class FlakySession:
            def __init__(self):
                self.calls = 0

            def get(self, _url, **_kwargs):
                self.calls += 1
                return BrokenResponse() if self.calls == 1 else FakeResponse(TRANSCRIPTION)

        session = FlakySession()
        client = BailianClient(DEFAULT_ASR_URL, DEFAULT_TEXT_URL, "sk-test",
                               DEFAULT_ASR_MODEL, "qwen-plus", session)
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "transcription.json"
            self.assertEqual(client.download_transcription("https://example.test/result", destination),
                             TRANSCRIPTION)
            self.assertEqual(json.loads(destination.read_text(encoding="utf-8")), TRANSCRIPTION)
        self.assertEqual(session.calls, 2)

    def test_edit_plan_keeps_excessive_model_cut_and_caption_audio_in_sync(self):
        parts = ["第一句", "还要", "保留", "结尾"]
        source = {"transcripts": [{"sentences": [{"words": [
            {"begin_time": i * 500, "end_time": (i + 1) * 500, "text": value}
            for i, value in enumerate(parts)]}]}]}
        words = extract_words(source)
        plan = build_edit_plan(words, [{"start_word": 0, "end_word": 3,
                                        "reason": "模型认为冗余"}], 2.0, [])
        self.assertEqual(plan["decisions"][0]["action"], "KEEP")
        self.assertTrue(plan["decisions"][0]["need_review"])
        self.assertEqual(plan["cuts"], [])

        def reply(system, user):
            if "视频剪辑师" in system:
                return {"cuts": [{"start_word": 0, "end_word": 3,
                                  "reason": "模型认为冗余"}], "uncertain": []}
            tokens = json.loads(user)["kept_tokens"]
            self.assertEqual([item["text"] for item in tokens], parts)
            return {"segments": [{"start_token": 0, "end_token": 4,
                                  "text": "".join(parts)}]}

        result = build_sentence_project(source, Person("机构", "职务", "姓名"), reply,
                                        plan_builder=lambda proposals: build_edit_plan(
                                            words, proposals, 2.0, []))
        self.assertEqual(result["cut_proposals"], [])
        self.assertEqual("".join(item["text"] for item in result["captions"]), "".join(parts))

    def test_model_output_failure_keeps_speech_and_marks_review(self):
        def broken(_system, _user):
            raise ModelOutputError("返回内容不是有效 JSON")

        result = build_sentence_project(TRANSCRIPTION, Person("机构", "职务", "姓名"), broken)
        self.assertEqual(result["cut_proposals"], [])
        self.assertTrue(result["caption_fallback_groups"])
        self.assertTrue(result["uncertain"])
        self.assertEqual("".join(item["text"] for item in result["captions"]),
                         "武汉同济院的医生")

    def test_model_owns_clause_boundary_without_keyword_rejection(self):
        parts = ["优质的", "碳水", "那", "启动", "一天", "能量代谢"]
        source = {"transcripts": [{"sentences": [{"words": [
            {"begin_time": i * 400, "end_time": (i + 1) * 400,
             "text": value, "punctuation": "，" if i == 1 else ""}
            for i, value in enumerate(parts)]}]}]}
        attempts = 0

        def reply(system, _user):
            nonlocal attempts
            if "视频剪辑师" in system:
                return {"cuts": [], "uncertain": []}
            attempts += 1
            if attempts == 1:
                return {"segments": [{"start_token": 0, "end_token": 6,
                                      "text": "优质的碳水那启动一天能量代谢"}]}
            return {"segments": [{"start_token": 0, "end_token": 2,
                                  "text": "优质的碳水"},
                                 {"start_token": 2, "end_token": 6,
                                  "text": "那启动一天能量代谢"}]}

        result = build_sentence_project(source, Person("机构", "职务", "姓名"), reply)
        self.assertEqual(attempts, 1)
        self.assertEqual([item["text"] for item in result["captions"]],
                         ["优质的碳水那启动一天能量代谢"])

    def test_sentence_first_edit_decision_drives_audio_and_caption_timeline(self):
        texts = ["第二个", "热量", "是", "是", "第二个", "热量", "要", "均衡"]
        source = {"transcripts": [{"sentences": [
            {"begin_time": 0, "end_time": 1600, "text": "第二个热量是是",
             "words": [{"begin_time": i * 400, "end_time": (i + 1) * 400,
                        "text": value} for i, value in enumerate(texts[:4])]},
            {"begin_time": 1800, "end_time": 3400, "text": "第二个热量要均衡",
             "words": [{"begin_time": 1800 + i * 400, "end_time": 2200 + i * 400,
                        "text": value} for i, value in enumerate(texts[4:])]}]}]}
        calls = []
        def reply(system, user):
            payload = json.loads(user)
            calls.append(system)
            if "视频剪辑师" in system:
                self.assertEqual(len(payload["target_sentences"]), 2)
                self.assertEqual(payload["target_sentences"][0]["start_ms"], 0)
                self.assertEqual(payload["target_sentences"][1]["start_word"], 4)
                return {"cuts": [{"start_word": 2, "end_word": 3,
                                  "reason": "局部重复起音"}]}
            self.assertEqual([token["word_index"] for token in payload["kept_tokens"]],
                             [0, 1, 3, 4, 5, 6, 7])
            return {"segments": [{"start_token": 0, "end_token": 3,
                                  "text": "第二个热量是"},
                                 {"start_token": 3, "end_token": 7,
                                  "text": "第二个热量要均衡"}]}
        from studio.strategy import Strategy
        result = build_sentence_project(source, Person("医院", "医生", "甲"), reply,
                                        strategy=Strategy(max_subtitle_cleanup_seconds=.35))
        self.assertEqual(len(calls), 2)
        self.assertEqual(result["captions"][1]["start_word"], 4)
        self.assertEqual(result["asr_sentences"][1]["start_ms"], 1800)
        cuts = build_cut_plan(extract_words(source), result["cut_proposals"], 3.4)
        self.assertEqual((cuts[0]["start"], cuts[0]["end"]), (0.8, 1.2))

    def test_caption_stage_keeps_audio_when_model_omits_uncut_filler(self):
        source = {"transcripts": [{"sentences": [{"begin_time": 0, "end_time": 900,
            "text": "嗯今天好", "words": [
                {"begin_time": i * 300, "end_time": (i + 1) * 300, "text": value}
                for i, value in enumerate(("嗯", "今天", "好"))]}]}]}
        def reply(system, _user):
            if "视频剪辑师" in system:
                return {"cuts": []}
            return {"segments": [{"start_token": 0, "end_token": 3, "text": "今天好"}]}
        result = build_sentence_project(source, Person("医院", "医生", "甲"), reply)
        self.assertEqual(result["cut_proposals"], [])
        self.assertEqual(result["caption_fallback_groups"], [])
        self.assertEqual("".join(item["text"] for item in result["captions"]), "今天好")
        self.assertEqual(result["captions"][0]["display_cleanup"][0]["source_text"], "嗯")

    def test_sentence_phase_resumes_without_repeating_completed_edit_group(self):
        source = {"transcripts": [{"sentences": [
            {"begin_time": 0, "end_time": 500, "text": "第一句", "words": [
                {"begin_time": 0, "end_time": 500, "text": "第一句"}]},
            {"begin_time": 31000, "end_time": 31500, "text": "第二句", "words": [
                {"begin_time": 31000, "end_time": 31500, "text": "第二句"}]}]}]}
        saved = []
        edits = []
        def interrupted(system, user):
            if "视频剪辑师" in system:
                target = json.loads(user)["target_sentences"][0]["id"]
                edits.append(target)
                if target == 1:
                    raise RuntimeError("模拟文本接口中断")
                return {"cuts": []}
            raise AssertionError("剪辑阶段完成前不应生成字幕")
        with patch("studio.sentence_flow.sentence_groups", side_effect=lambda sentences: [[s] for s in sentences]), \
                self.assertRaisesRegex(RuntimeError, "模拟文本接口中断"):
            build_sentence_project(source, Person("医院", "医生", "甲"), interrupted,
                                   checkpoint=lambda state: saved.append(
                                       json.loads(json.dumps(state, ensure_ascii=False))))
        self.assertEqual(edits, [0, 1])
        self.assertEqual(saved[-1]["edited"], 1)
        def continued(system, user):
            if "视频剪辑师" in system:
                edits.append(json.loads(user)["target_sentences"][0]["id"])
                return {"cuts": []}
            tokens = json.loads(user)["kept_tokens"]
            return {"segments": [{"start_token": 0, "end_token": len(tokens),
                                  "text": tokens[0]["text"]}]}
        with patch("studio.sentence_flow.sentence_groups", side_effect=lambda sentences: [[s] for s in sentences]):
            result = build_sentence_project(source, Person("医院", "医生", "甲"), continued,
                                            resume=saved[-1])
        self.assertEqual(edits, [0, 1, 1])
        self.assertEqual(len(result["captions"]), 2)

    def test_caption_fallback_handles_one_unusually_long_asr_token(self):
        source = {"transcripts": [{"sentences": [{"begin_time": 0, "end_time": 3400,
            "text": "长" * 34, "words": [{"begin_time": 0, "end_time": 3400,
                                       "text": "长" * 34}]}]}]}
        result = build_sentence_project(source, Person("医院", "医生", "甲"),
            lambda system, _user: {"cuts": []} if "视频剪辑师" in system
            else {"segments": []})
        self.assertTrue(result["caption_fallback_groups"])
        self.assertTrue(all(0 < len(part["text"]) <= 16 for part in result["captions"]))
        self.assertEqual("".join(part["text"] for part in result["captions"]), "长" * 34)

    def test_output_name_uses_person_contract_without_overwriting_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            person = Person("武汉同济医院", "副主任医师", "李景平")
            item = VideoItem(folder / "random.mp4", person)
            self.assertEqual(output_path(item, folder).name,
                             "武汉同济医院_副主任医师_李景平.mp4")
            named_source = VideoItem(folder / "武汉同济医院_副主任医师_李景平.mp4", person)
            self.assertEqual(output_path(named_source, folder),
                             folder / "竖屏成片" / named_source.path.name)
            alternate = VideoItem(folder / "another.mp4", Person("医院", "主任/教授", "张三"))
            self.assertEqual(output_path(alternate, folder).name, "医院_主任／教授_张三.mp4")
            tagged = output_path(item, folder, "20260924-154217-123456")
            self.assertEqual(tagged.name, "武汉同济医院_副主任医师_李景平_20260924-154217-123456.mp4")
            self.assertNotEqual(tagged, output_path(item, folder, "20260924-154218-123456"))

    def test_failed_attempt_records_start_end_and_elapsed(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            item = VideoItem(base / "missing.mp4", Person("医院", "医生", "张三"))
            client = BailianClient(DEFAULT_ASR_URL, DEFAULT_TEXT_URL, "sk-test",
                                   DEFAULT_ASR_MODEL, "qwen-plus", FakeSession())
            config = JobConfig(base / "template.png", base / "out", "提示", None,
                               "ffmpeg", "ffprobe")
            pipeline = Pipeline(client, config, lambda *_: None, threading.Event())
            with self.assertRaisesRegex(ValueError, "视频不存在"):
                pipeline.run_one(item)
            runs = json.loads((work_dir(item, config.output_dir) / "runs.json")
                              .read_text(encoding="utf-8"))["runs"]
            self.assertEqual(runs[0]["status"], "失败")
            self.assertIn("started_at", runs[0])
            self.assertIn("ended_at", runs[0])
            self.assertGreaterEqual(runs[0]["elapsed_seconds"], 0)
            self.assertNotIn("sk-test", json.dumps(runs))

    def test_preview_drag_touching_image_edge_stays_inside_template(self):
        slot = slot_from_display_rect((1063, 1890), (366, 650), (0, 220), (366, 430))
        self.assertEqual(slot[0], 0)
        self.assertEqual(slot[0] + slot[2], 1063)
        self.assertLessEqual(slot[1] + slot[3], 1890)
        self.assertEqual(slot_from_display_rect((1063, 1890), (366, 650),
                                                (-20, -20), (400, 700)), (0, 0, 1063, 1890))

    def test_long_caption_splits_at_meaningful_phrase_and_keeps_timing(self):
        text = "也就是说我今天并没有吃多少只吃了一个汉堡或喝了一杯奶茶"
        words = [Word(i * 0.3, (i + 1) * 0.3, char) for i, char in enumerate(text)]
        original = {"start": 0.0, "end": len(text) * 0.3, "text": text,
                    "start_word": 0, "end_word": len(words)}
        parts = shorten_captions([original], words)
        self.assertEqual([item["text"] for item in parts],
                         ["也就是说我今天并没有吃多少", "只吃了一个汉堡或喝了一杯奶茶"])
        self.assertTrue(all(len(item["text"]) <= 16 for item in parts))
        self.assertEqual(parts[0]["end_word"], parts[1]["start_word"])
        self.assertLessEqual(parts[0]["end"], parts[1]["start"])

    def test_length_fallback_keeps_organization_and_name_together(self):
        text = "各位朋友大家好我是武汉同济医院李景平医生"
        parts = ["各位", "朋友", "大家好", "我是", "武汉", "同济", "医院", "李景平", "医生"]
        words = [Word(i * 0.3, (i + 1) * 0.3, part,
                      "，" if i == 2 else "") for i, part in enumerate(parts)]
        original = {"start": 0.0, "end": len(parts) * 0.3, "text": text,
                    "start_word": 0, "end_word": len(parts)}
        captions = shorten_captions([original], words, ("武汉同济医院", "李景平", "医生"))
        self.assertEqual([item["text"] for item in captions],
                         ["各位朋友大家好", "我是武汉同济医院李景平医生"])

    def test_qwen37_caption_request_disables_thinking(self):
        fake = FakeSession()
        client = BailianClient(DEFAULT_ASR_URL, DEFAULT_TEXT_URL, "sk-test", DEFAULT_ASR_MODEL,
                               "qwen3.7-flash-2026-07-15", fake)
        result = client.chat_json("JSON", "test")
        self.assertIn("segments", result)
        payload = fake.calls[-1][2]["json"]
        self.assertIs(payload["enable_thinking"], False)
        self.assertEqual(payload["max_tokens"], 2048)
        self.assertEqual(fake.calls[-1][2]["timeout"], (30, 90))

    def test_chat_recovers_truncated_and_malformed_json_with_larger_budget(self):
        class Responses:
            def __init__(self):
                self.budgets = []

            def post(self, _url, **kwargs):
                self.budgets.append(kwargs["json"]["max_tokens"])
                if len(self.budgets) == 1:
                    return FakeResponse({"request_id": "req-first", "choices": [{
                        "finish_reason": "length", "message": {"content": '{"segments":[]}'}}]})
                if len(self.budgets) == 2:
                    return FakeResponse({"request_id": "req-second", "choices": [{
                        "finish_reason": "stop", "message": {"content": '{"segments":'}}]})
                return FakeResponse({"request_id": "req-third", "choices": [{
                    "finish_reason": "stop", "message": {"content": [
                        {"type": "text", "text": "```json\n{\"segments\": []}\n```"}]}}]})

        responses = Responses()
        client = BailianClient(DEFAULT_ASR_URL, DEFAULT_TEXT_URL, "sk-test", DEFAULT_ASR_MODEL,
                               "qwen-plus", responses)
        retries = []
        self.assertEqual(client.chat_json("JSON", "test", lambda attempt, reason:
                                          retries.append((attempt, reason))), {"segments": []})
        self.assertEqual(responses.budgets, [2048, 4096, 8192])
        self.assertEqual([attempt for attempt, _ in retries], [1, 2])

    def test_chat_failure_reports_finish_reason_and_request_id(self):
        class EmptyResponses:
            def post(self, _url, **_kwargs):
                return FakeResponse({"request_id": "req-empty", "choices": [{
                    "finish_reason": "length", "message": {"content": None}}]})

        client = BailianClient(DEFAULT_ASR_URL, DEFAULT_TEXT_URL, "sk-test", DEFAULT_ASR_MODEL,
                               "qwen-plus", EmptyResponses())
        with self.assertRaises(ApiError) as error:
            client.chat_json("JSON", "test")
        self.assertIn("finish_reason=length", str(error.exception))
        self.assertIn("req-empty", str(error.exception))
        self.assertNotIn("sk-test", str(error.exception))

    def test_caption_checkpoint_resumes_after_a_failed_chunk(self):
        source = {"transcripts": [{"sentences": [{"begin_time": 0, "end_time": 15000,
            "text": "甲" * 150, "words": [
                {"begin_time": i * 100, "end_time": (i + 1) * 100, "text": "甲"}
                for i in range(150)]}]}]}
        checkpoint = {}
        calls = []

        def reply(_system, user):
            data = json.loads(user)
            calls.append(data["word_range"])
            if len(calls) == 2:
                raise RuntimeError("temporary model failure")
            start, end = data["word_range"]
            return {"segments": [{"start_word": i, "end_word": min(i + 10, end),
                                  "text": "甲" * (min(i + 10, end) - i)}
                                 for i in range(start, end, 10)],
                    "changes": [], "uncertain": []}

        with self.assertRaisesRegex(RuntimeError, "temporary"):
            build_captions(source, Person("示例医院", "医生", "示例人"), reply,
                           checkpoint=lambda value: checkpoint.update(value))
        self.assertEqual(checkpoint["next_section"], 1)
        result = build_captions(source, Person("示例医院", "医生", "示例人"), reply,
                                resume=checkpoint)
        self.assertTrue(result["reviewed"])
        self.assertEqual(len(calls), 3)
        self.assertEqual(result["captions"][0]["start_word"], 0)
        self.assertEqual(result["captions"][-1]["end_word"], 150)

    def test_bailian_sdk_base_url_expands_to_rest_endpoints(self):
        base = "https://llm-example.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
        client = BailianClient(base, base, "test", "qwen-audio-3.0-asr-flash-filetrans",
                               "qwen-plus", upload_url=base,
                               task_url=base + "/tasks/{task_id}")
        self.assertEqual(client.asr_url, "https://llm-example.cn-beijing.maas.aliyuncs.com/api/v1/services/audio/asr/transcription")
        self.assertEqual(client.upload_url, "https://llm-example.cn-beijing.maas.aliyuncs.com/api/v1/uploads")
        self.assertEqual(client.task_url, "https://llm-example.cn-beijing.maas.aliyuncs.com/api/v1/tasks/{task_id}")
        self.assertEqual(client.text_url, base + "/chat/completions")

    def test_one_bailian_url_supplies_all_endpoints_and_migrates_old_setting(self):
        base = "https://llm-example.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
        client = BailianClient.from_base_url(base, "test", DEFAULT_ASR_MODEL, "qwen-plus")
        self.assertEqual((client.asr_url, client.upload_url, client.task_url, client.text_url),
                         endpoints_from_base_url(base))
        self.assertEqual(saved_base_url({"text_url": base + "/chat/completions"}), base)
        self.assertEqual(saved_base_url({"bailian_url": base}), base)
        with self.assertRaises(ValueError):
            endpoints_from_base_url(base + "/chat/completions")

    def test_ffmpeg_detection_rejects_nonworking_binary(self):
        with tempfile.TemporaryDirectory() as temporary:
            fake = Path(temporary) / "ffmpeg.exe"
            fake.write_text("invalid", encoding="utf-8")
            ffmpeg, _ = detect_tools(str(fake), "")
            self.assertTrue(ffmpeg is None or ffmpeg[0] != str(fake))

    def test_known_synchronous_asr_model_has_actionable_message(self):
        with self.assertRaisesRegex(ValueError, "qwen-audio-3.0-asr-flash-filetrans"):
            BailianClient(DEFAULT_ASR_URL, DEFAULT_TEXT_URL, "test", "qwen-audio-3.0-asr-flash", "qwen-plus")

    def test_custom_urls_are_used_without_fixed_path_validation(self):
        class CustomSession:
            def __init__(self):
                self.calls = []

            def get(self, url, **kwargs):
                self.calls.append(("GET", url))
                return FakeResponse({"output": {"task_status": "PENDING"}})

            def post(self, url, **kwargs):
                self.calls.append(("POST", url))
                if url == "http://localhost:8900/custom-asr":
                    return FakeResponse({"output": {"task_id": "task/one"}})
                return FakeResponse({"choices": [{"message": {"content": "{}"}}]})

        session = CustomSession()
        client = BailianClient("http://localhost:8900/custom-asr", "http://localhost:8900/text",
                               "test", "my-asr-model", "my-text-model", session,
                               upload_url="http://localhost:8900/upload-policy",
                               task_url="http://localhost:8900/check/{task_id}")
        task_id = client.submit_asr("oss://example")
        client.get_task(task_id)
        client.chat_json("system", "user")
        self.assertEqual(session.calls, [("POST", "http://localhost:8900/custom-asr"),
                                         ("GET", "http://localhost:8900/check/task%2Fone"),
                                         ("POST", "http://localhost:8900/text")])

    def test_batch_filenames_reject_invalid_video(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / "武汉同济医院__副主任医师__耿珊.mp4").touch()
            (folder / "随意名称.mp4").touch()
            (folder / "notes.txt").touch()
            valid, invalid = scan_folder(folder)
            self.assertEqual(len(valid), 1)
            self.assertEqual(len(invalid), 1)
            self.assertEqual(valid[0].person.name, "耿珊")

    def test_batch_accepts_multiple_clips_for_same_person_without_output_collisions(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            for name in ("医院_医生_张三_01.mp4", "医院_医生_张三_02.mp4",
                         "医院_医生_李四.mp4", "医院_医生_李四.mov",
                         "医院_医生_李四_mp4.mp4"):
                (folder / name).touch()
            valid, invalid = scan_folder(folder)
            self.assertEqual(invalid, [])
            self.assertEqual(len(valid), 5)
            names = [output_path(item, folder / "out", "20260924-154217-123456").name for item in valid]
            self.assertEqual(len(names), len(set(name.casefold() for name in names)))
            self.assertIn("医院_医生_张三_01_20260924-154217-123456.mp4", names)
            self.assertIn("医院_医生_李四_mov_20260924-154217-123456.mp4", names)

    def test_single_underscore_long_organization_and_legacy_double_underscore(self):
        organization = "中山大学孙逸仙纪念医院创伤与足踝外科"
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / f"{organization}_副主任医师_张帆.mp4").touch()
            (folder / "旧医院__主任__李医生.mp4").touch()
            valid, invalid = scan_folder(folder)
            self.assertEqual(invalid, [])
            self.assertEqual(len(valid), 2)
            person = next(item.person for item in valid if item.person.name == "张帆")
            self.assertEqual((person.organization, person.title, person.name),
                             (organization, "副主任医师", "张帆"))

    @unittest.skipUnless(default_font(), "Chinese font not installed")
    def test_long_organization_fits_nameplate_at_540_pixels(self):
        self.assertEqual(hospital_and_department("中山大学孙逸仙纪念医院创伤与足踝外科"),
                         ("中山大学孙逸仙纪念医院", "创伤与足踝外科"))
        with tempfile.TemporaryDirectory() as temporary:
            overlay = Path(temporary) / "overlay.png"
            graphics((540, 960), (0, 326, 540, 304), "张帆",
                     "中山大学孙逸仙纪念医院创伤与足踝外科", "副主任医师",
                     "", default_font(), overlay)
            self.assertTrue(overlay.is_file())
            with Image.open(overlay) as image:
                self.assertIsNotNone(image.crop((0, 326, 215, 500)).getchannel("A").getbbox())
                self.assertIsNone(image.crop((215, 375, 540, 500)).getchannel("A").getbbox())

    def test_bailian_contract_and_caption_correction(self):
        with tempfile.TemporaryDirectory() as temporary:
            audio = Path(temporary) / "audio.mp3"
            audio.write_bytes(b"a" * 2000)
            fake = FakeSession()
            client = BailianClient(DEFAULT_ASR_URL, DEFAULT_TEXT_URL, "sk-test", DEFAULT_ASR_MODEL, "qwen-plus", fake)
            url = client.upload_audio(audio)
            self.assertEqual(url, "oss://folder/audio.mp3")
            task_id = client.submit_asr(url)
            self.assertEqual(task_id, "task-123")
            self.assertEqual(client.task_status(client.get_task(task_id)), "PENDING")
            task = client.get_task(task_id)
            result = client.download_transcription(client.transcription_url(task), Path(temporary) / "raw.json")
            captions = build_sentence_project(result, Person("武汉同济医院", "副主任医师", "耿珊"), client.chat_json)
            self.assertEqual(captions["captions"][0]["text"], "武汉同济医院的医生")
            self.assertTrue(captions["reviewed"])

    def test_invalid_model_segments_fail_after_automatic_retries(self):
        calls = 0
        def invalid_json(_system, _user):
            nonlocal calls
            calls += 1
            return {"segments": [{"start_word": 0, "end_word": 3, "text": "的武汉同济院医生。"}]}

        with self.assertRaisesRegex(ValueError, "连续三次返回无效字幕"):
            build_captions(TRANSCRIPTION, Person("武汉同济医院", "医师", "示例人"), invalid_json)
        self.assertEqual(calls, 3)

    def test_natural_sentence_boundaries_and_soft_length(self):
        parts = ["各位", "朋友", "大家好", "我是", "武汉同济医院", "李景平", "医生"]
        source = {"transcripts": [{"sentences": [{"begin_time": 0, "end_time": len(parts) * 500,
            "words": [{"begin_time": i * 500, "end_time": (i + 1) * 500, "text": part}
                      for i, part in enumerate(parts)]}]}]}

        def split(system, _user):
            self.assertIn("开场寒暄、自我介绍", system)
            return {"segments": [{"start_word": 0, "end_word": 3, "text": "各位朋友大家好"},
                                 {"start_word": 3, "end_word": 7, "text": "我是武汉同济医院李景平医生"}],
                    "changes": [], "uncertain": ["医生姓名未从音频复核"]}

        result = build_captions(source, Person("武汉同济医院", "医生", "李景平"), split)
        self.assertTrue(result["reviewed"])
        self.assertEqual([item["text"] for item in result["captions"]],
                         ["各位朋友大家好", "我是武汉同济医院李景平医生"])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "captions.json"
            path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
            validate_caption_document(result, 4)
            self.assertEqual(len(read_captions(path, 4)), 2)
            longer = {"reviewed": True, "uncertain": [], "captions": [
                {"start": 0, "end": 3, "text": "如果出现不适请及时到当地医院就诊"}]}
            self.assertGreater(len(longer["captions"][0]["text"]), 14)
            validate_caption_document(longer, 4)
            path.write_text(json.dumps(longer, ensure_ascii=False), encoding="utf-8")
            self.assertEqual(len(read_captions(path, 4)), 1)

    def test_long_sentence_is_returned_to_model_before_fallback_split(self):
        parts = ["各位", "朋友", "大家好", "我是", "武汉", "同济", "医院", "李景平", "医生"]
        source = {"transcripts": [{"sentences": [{"words": [
            {"begin_time": i * 350, "end_time": (i + 1) * 350, "text": part,
             "punctuation": "，" if i == 2 else ""}
            for i, part in enumerate(parts)]}]}]}
        requests = []
        def reply(_system, user):
            requests.append(user)
            if len(requests) == 1:
                return {"segments": [{"start_word": 0, "end_word": 9,
                                     "text": "各位朋友大家好我是武汉同济医院李景平医生"}]}
            return {"segments": [{"start_word": 0, "end_word": 3, "text": "各位朋友大家好"},
                                {"start_word": 3, "end_word": 9,
                                 "text": "我是武汉同济医院李景平医生"}]}
        result = build_captions(source, Person("武汉同济医院", "医生", "李景平"), reply)
        self.assertEqual(len(requests), 2)
        self.assertIn("超过 16 字", requests[1])
        self.assertEqual([part["text"] for part in result["captions"]],
                         ["各位朋友大家好", "我是武汉同济医院李景平医生"])

    def test_model_punctuation_becomes_timed_clause_boundaries(self):
        parts = ["中午", "要", "吃得", "像", "大臣", "八分饱", "能量", "均衡"]
        source = {"transcripts": [{"sentences": [{"words": [
            {"begin_time": i * 400, "end_time": (i + 1) * 400, "text": part,
             "punctuation": "，" if i in (4, 5) else ""}
            for i, part in enumerate(parts)]}]}]}
        calls = 0
        def reply(_system, _user):
            nonlocal calls
            calls += 1
            return {"segments": [{"start_word": 0, "end_word": len(parts),
                                  "text": "中午要吃得像大臣，八分饱，能量均衡"}]}
        result = build_captions(source, Person("示例医院", "医生", "示例人"), reply)
        self.assertEqual(calls, 1)
        self.assertEqual([item["text"] for item in result["captions"]],
                         ["中午要吃得像大臣", "八分饱", "能量均衡"])
        self.assertEqual([(item["start_word"], item["end_word"]) for item in result["captions"]],
                         [(0, 5), (5, 6), (6, 8)])
        validate_caption_document(result, 3.2)

    def test_disfluency_plan_retimes_subtitles_and_rejects_bad_indexes(self):
        words = [Word(0.0, 0.4, "咦"), Word(0.4, 1.0, "我吃"),
                 Word(1.0, 1.3, "吃得"), Word(1.3, 2.0, "不多")]
        with self.assertRaisesRegex(ValueError, "未能映射"):
            build_cut_plan(words, [{"word_index": -1, "start_char": 0,
                                    "end_char": 1}], 2.0)
        cuts = build_cut_plan(words, [{"start_word": 0, "end_word": 1, "reason": "语气词"},
                                      {"word_index": 1, "start_char": 1, "end_char": 2,
                                       "reason": "重复起音"}], 2.0)
        self.assertEqual([(cut["start"], cut["end"]) for cut in cuts],
                         [(0.0, 0.4), (0.7, 1.0)])
        shifted = retime_captions([{"start": 0.4, "end": 2.0,
                                    "text": "我吃得不多"}], cuts)
        self.assertEqual((shifted[0]["start"], shifted[0]["end"]), (0.0, 1.3))

    def test_text_alignment_repairs_shifted_word_indexes(self):
        parts = ["我们", "已经", "控制", "饮食", "为什么", "还", "瘦", "不下来", "这里", "有", "答案"]
        words = [Word(i * 0.3, (i + 1) * 0.3, part) for i, part in enumerate(parts)]
        captions = [{"start_word": 0, "end_word": 6, "start": 0, "end": 1.8,
                     "text": "我们已经控制饮食"},
                    {"start_word": 6, "end_word": 9, "start": 1.8, "end": 2.7,
                     "text": "为什么还瘦不下来"},
                    {"start_word": 9, "end_word": 11, "start": 2.7, "end": 3.3,
                     "text": "这里有答案"}]
        fixed = align_captions(captions, words, [])
        self.assertEqual([(part["start_word"], part["end_word"]) for part in fixed],
                         [(0, 4), (4, 8), (8, 11)])
        self.assertEqual(fixed[1]["model_start_word"], 6)
        self.assertEqual((fixed[1]["start"], fixed[1]["end"]), (1.2, 2.4))
        timeline = build_timeline(Path("source.mp4"), Path("output.mp4"), words, fixed,
                                  [{"start": 0.3, "end": 0.6, "reason": "测试"}], 3.3)
        self.assertEqual(timeline["captions"][1]["source_start"], 1.2)
        self.assertEqual(timeline["captions"][1]["output_start"], 0.9)
        self.assertEqual(timeline["keep_segments"][1]["output_start"], 0.3)

    def test_alignment_excludes_unspoken_caption_padding(self):
        parts = ["那么", "中午", "呢", "要", "吃的", "像", "大臣", "也就是说", "八", "分", "饱"]
        words = [Word(i * 0.3, (i + 1) * 0.3, part) for i, part in enumerate(parts)]
        captions = [{"start_word": 0, "end_word": 8, "start": 0, "end": 2.4,
                     "text": "中午要吃得像大臣"},
                    {"start_word": 8, "end_word": 11, "start": 2.4, "end": 3.3,
                     "text": "八分饱"}]
        fixed = align_captions(captions, words, [])
        self.assertEqual([(part["start_word"], part["end_word"]) for part in fixed],
                         [(1, 7), (8, 11)])

    def test_recognition_correction_does_not_remove_real_audio(self):
        words = [Word(0, 0.3, "中午"), Word(0.3, 0.6, "吃的"), Word(0.6, 0.9, "像")]
        from studio.sentence_flow import _check_cuts
        with self.assertRaisesRegex(ValueError, "ASR 纠错"):
            _check_cuts([{ "kind": "asr_correction", "start_word": 1, "end_word": 3,
                           "reason": "ASR误识别"}], 0, len(words), words)

    def test_caption_text_is_not_silently_cleaned_without_audio_cut(self):
        parts = ["嗯", "今天", "好"]
        source = {"transcripts": [{"sentences": [{"words": [
            {"begin_time": i * 300, "end_time": (i + 1) * 300, "text": part}
            for i, part in enumerate(parts)]}]}]}
        person = Person("示例医院", "医生", "示例人")
        retained = build_captions(source, person, lambda _s, _u: {
            "segments": [{"start_word": 0, "end_word": 3, "text": "嗯今天好"}],
            "cuts": []})
        self.assertEqual(retained["captions"][0]["text"], "嗯今天好")
        removed = build_captions(source, person, lambda _s, _u: {
            "segments": [{"start_word": 1, "end_word": 3, "text": "今天好"}],
            "cuts": [{"start_word": 0, "end_word": 1, "reason": "口癖"}]})
        self.assertEqual(removed["cut_proposals"][0]["start_word"], 0)
        self.assertEqual(build_cut_plan(extract_words(source), removed["cut_proposals"], 0.9)[0]["end"], 0.3)

        calls = 0
        def repair_missing_cut(_system, _user):
            nonlocal calls
            calls += 1
            if calls == 1:
                return {"segments": [{"start_word": 0, "end_word": 3,
                                      "text": "今天好"}], "cuts": []}
            return {"segments": [{"start_word": 0, "end_word": 3,
                                  "text": "今天好"}],
                    "cuts": [{"start_word": 0, "end_word": 1, "reason": "口癖"}]}
        repaired = build_captions(source, person, repair_missing_cut)
        self.assertEqual(calls, 2)
        self.assertEqual(repaired["cut_proposals"][0]["end_word"], 1)

    def test_repeated_syllable_requires_matching_word_internal_audio_cut(self):
        parts = ["我吃", "吃得", "不多"]
        source = {"transcripts": [{"sentences": [{"words": [
            {"begin_time": i * 400, "end_time": (i + 1) * 400, "text": part}
            for i, part in enumerate(parts)]}]}]}
        calls = 0
        def reply(_system, _user):
            nonlocal calls
            calls += 1
            return {"segments": [{"start_word": 0, "end_word": 3,
                                  "text": "我吃得不多"}],
                    "cuts": ([{"word_index": 0, "start_char": 1,
                               "end_char": 2, "reason": "重复起音"}]
                             if calls > 1 else [])}
        result = build_captions(source, Person("示例医院", "医生", "示例人"), reply)
        self.assertEqual(calls, 2)
        self.assertEqual(result["cut_proposals"][0]["word_index"], 0)

    def test_model_decides_restarted_thought_and_keeps_emphasis(self):
        texts = ["今天", "我们", "介绍", "睡眠", "规律", "就是", "早睡", "早起", "是", "是",
                 "今天", "我们", "介绍", "睡眠", "规律", "就是", "早睡", "早起"]
        words = [Word(i * 0.4, (i + 1) * 0.4, value,
                      "。" if i == len(texts) - 1 else "") for i, value in enumerate(texts)]
        # The second utterance starts after a distinct restart pause.
        words[10] = Word(4.4, 4.8, words[10].text)
        for i in range(11, len(words)):
            words[i] = Word(4.4 + (i - 10) * 0.4, 4.8 + (i - 10) * 0.4,
                            words[i].text, words[i].punctuation)
        self.assertEqual(build_cut_plan(words, [], words[-1].end), [])
        transcription = {"transcripts": [{"sentences": [{"words": [
            {"begin_time": round(word.start * 1000), "end_time": round(word.end * 1000),
             "text": word.text, "punctuation": word.punctuation} for word in words]}]}]}

        def restarted(system, _user):
            self.assertIn("刻意强调", system)
            return {"segments": [{"start_word": 10, "end_word": 18,
                                  "text": "今天我们介绍睡眠规律就是早睡早起"}],
                    "cuts": [{"start_word": 0, "end_word": 10,
                              "reason": "前一次说到一半后改口重说"}]}

        revised = build_captions(transcription, Person("示例医院", "医生", "示例人"), restarted)
        self.assertEqual(revised["captions"][0]["start_word"], 10)
        cuts = build_cut_plan(words, revised["cut_proposals"], words[-1].end)
        self.assertEqual([(cut["start"], cut["end"]) for cut in cuts], [(0.0, 4.0)])

        emphasized_words = words[:8] + words[10:]
        emphasized_source = {"transcripts": [{"sentences": [{"words": [
            {"begin_time": round(word.start * 1000), "end_time": round(word.end * 1000),
             "text": word.text, "punctuation": word.punctuation}
            for word in emphasized_words]}]}]}
        def emphasized(_system, _user):
            return {"segments": [{"start_word": 0, "end_word": 8,
                                  "text": "今天我们介绍睡眠规律就是早睡早起"},
                                 {"start_word": 8, "end_word": 16,
                                  "text": "今天我们介绍睡眠规律就是早睡早起"}], "cuts": []}

        kept = build_captions(emphasized_source, Person("示例医院", "医生", "示例人"), emphasized)
        self.assertEqual(len(kept["captions"]), 2)
        self.assertEqual(build_cut_plan(emphasized_words, kept["cut_proposals"], words[-1].end), [])

    def test_model_full_sentence_cut_is_not_silently_limited_to_six_seconds(self):
        words = [Word(i * 0.8, (i + 1) * 0.8, f"词{i}") for i in range(26)]
        cuts = build_cut_plan(words, [{"start_word": 0, "end_word": 13,
                                       "reason": "口误后完整重说"}], words[-1].end)
        self.assertEqual([(cut["start"], cut["end"]) for cut in cuts], [(0.0, 10.4)])

    def test_long_pause_requires_both_asr_gap_and_detected_silence(self):
        words = [Word(0, 0.5, "先说"), Word(2.6, 3.0, "继续"),
                 Word(3.7, 4.1, "结束")]
        cuts = detect_long_silence_cuts(words, [(0.5, 2.5), (3.1, 3.65)])
        self.assertEqual(len(cuts), 1)
        self.assertEqual((cuts[0]["start"], cuts[0]["end"]), (0.62, 2.38))
        self.assertEqual(detect_long_silence_cuts(words, [(0.51, 2.49)]), [])

    def test_semantic_cleanup_cuts_spoken_false_start_with_caption(self):
        parts = ["第二个", "要", "大量的", "吃", "输液", "呃", "绿叶", "蔬菜", "来", "蔬菜"]
        source = {"transcripts": [{"sentences": [{"begin_time": 0, "end_time": len(parts) * 300,
            "words": [{"begin_time": i * 300, "end_time": (i + 1) * 300, "text": part}
                      for i, part in enumerate(parts)]}]}]}

        def cleaned(system, _user):
            self.assertIn("ASR 误识别的词仍然保留其音视频", system)
            return {"segments": [{"start_word": 0, "end_word": len(parts),
                                  "text": "第二个要大量的吃蔬菜"}],
                    "cuts": [{"start_word": 4, "end_word": 7, "reason": "说错后重新说"},
                             {"start_word": 8, "end_word": 10, "reason": "重复起音"}],
                    "changes": [{"from": "输液呃绿叶", "to": "", "reason": "口误"}],
                    "uncertain": []}

        result = build_captions(source, Person("示例医院", "医生", "示例人"), cleaned)
        self.assertTrue(result["reviewed"])
        self.assertEqual(result["captions"][0]["text"], "第二个要大量的吃蔬菜")
        self.assertEqual(result["captions"][0]["end"], 3.0)
        cuts = build_cut_plan(extract_words(source), result["cut_proposals"], 3.0)
        self.assertEqual([(cut["start"], cut["end"]) for cut in cuts],
                         [(1.2, 2.1), (2.4, 3.0)])

    def test_missing_word_index_retries_instead_of_silently_hiding_audio(self):
        parts = ["一千", "大", "卡", "那么", "要注意"]
        source = {"transcripts": [{"sentences": [{"begin_time": 0, "end_time": 2500,
            "words": [{"begin_time": i * 500, "end_time": (i + 1) * 500,
                       "text": part, "punctuation": "。" if i == 2 else ""}
                      for i, part in enumerate(parts)]}]}]}

        calls = 0
        def off_by_one(_system, _user):
            nonlocal calls
            calls += 1
            if calls == 1:
                return {"segments": [{"start_word": 0, "end_word": 2, "text": "一千大卡"},
                                     {"start_word": 3, "end_word": 5, "text": "那么要注意"}],
                        "cuts": []}
            return {"segments": [{"start_word": 0, "end_word": 3, "text": "一千大卡"},
                                 {"start_word": 3, "end_word": 5, "text": "那么要注意"}],
                    "cuts": []}

        result = build_captions(source, Person("示例医院", "医生", "示例人"), off_by_one)
        first, second = result["captions"]
        self.assertEqual((first["start_word"], first["end_word"]), (0, 3))
        self.assertEqual(first["end"], 1.5)
        self.assertEqual(second["start_word"], 3)
        self.assertEqual(calls, 2)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg not installed")
    def test_render_physically_removes_video_and_audio_interval(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            video, template, captions, output = [base / name for name in
                                                 ("source.mp4", "template.png", "captions.json", "out.mp4")]
            subprocess.run([shutil.which("ffmpeg"), "-y", "-v", "error", "-f", "lavfi", "-i",
                            "color=c=gray:s=640x360:r=25:d=3", "-f", "lavfi", "-i",
                            "sine=frequency=440:duration=3", "-shortest", "-c:v", "libx264",
                            "-c:a", "aac", str(video)], check=True)
            image = Image.new("RGBA", (540, 960), "#2D65A4")
            ImageDraw.Draw(image).rectangle((0, 326, 539, 629), fill=(0, 0, 0, 0))
            image.save(template)
            cuts = [{"start": 0.4, "end": 0.6, "source_text": "嗯", "reason": "语气词"},
                    {"start": 1.0, "end": 1.5, "source_text": "呃", "reason": "语气词"}]
            data = {"reviewed": True, "captions": retime_captions(
                [{"start": 0.0, "end": 3.0, "text": "大家好"}], cuts)}
            captions.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            render_video(video, template, captions, output, "李景平", "武汉同济医院", "医生",
                         "医学科普 仅供参考", ffmpeg_path=shutil.which("ffmpeg"),
                         ffprobe_path=shutil.which("ffprobe"), cuts=cuts)
            duration, width, height = probe(output)
            self.assertAlmostEqual(duration, 2.3, delta=0.12)
            self.assertEqual((width, height), (540, 960))
            streams = json.loads(subprocess.run([shutil.which("ffprobe"), "-v", "error",
                                                 "-show_entries", "stream=codec_type", "-of", "json",
                                                 str(output)], capture_output=True, text=True,
                                                check=True).stdout)["streams"]
            self.assertEqual({stream["codec_type"] for stream in streams}, {"video", "audio"})

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg not installed")
    def test_complete_video_and_resume_from_saved_transcription(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            video = base / "source.mp4"
            subprocess.run([shutil.which("ffmpeg"), "-y", "-v", "error", "-f", "lavfi", "-i",
                            "color=c=gray:s=640x360:r=25:d=3", "-f", "lavfi", "-i",
                            "sine=frequency=440:duration=3", "-shortest", "-c:v", "libx264", "-c:a", "aac",
                            str(video)], check=True)
            template = base / "template.png"
            image = Image.new("RGBA", (540, 960), "#2D65A4")
            ImageDraw.Draw(image).rectangle((0, 326, 539, 629), fill=(0, 0, 0, 0))
            image.save(template)
            item = VideoItem(video, Person("武汉同济医院", "副主任医师", "耿珊"))
            fake = FakeSession()
            client = BailianClient(DEFAULT_ASR_URL, DEFAULT_TEXT_URL, "sk-test", DEFAULT_ASR_MODEL, "qwen-plus", fake)
            config = JobConfig(template, base / "out", "医学科普 仅供参考", None,
                               shutil.which("ffmpeg"), shutil.which("ffprobe"))
            progress = []
            pipeline = Pipeline(client, config, lambda stage, pct, detail: progress.append(stage), threading.Event())
            output = pipeline.run_one(item)
            self.assertTrue(output.is_file())
            text_calls = [call for call in fake.calls if call[0] == "POST"
                          and call[1].endswith("/chat/completions")]
            self.assertEqual([json.loads(call[2]["json"]["messages"][1]["content"])["phase"]
                              for call in text_calls], ["edit", "semantic_review", "caption"])
            self.assertIn("recognized", progress)
            duration, width, height = probe(output)
            self.assertAlmostEqual(duration, 3, delta=0.2)
            self.assertEqual((width, height), (540, 960))
            before = len(fake.calls)
            second = pipeline.run_one(item)
            self.assertEqual(second, output)
            self.assertEqual(len(fake.calls), before)
            self.assertTrue((work_dir(item, config.output_dir) / "sentences.json").is_file())
            plan = json.loads((work_dir(item, config.output_dir) / "edit-plan.json")
                              .read_text(encoding="utf-8"))
            self.assertEqual(plan["version"], 6)
            self.assertEqual(plan["mode"], "model_authored")
            self.assertEqual(plan["source"]["path"], str(video.resolve()))

            stale = json.loads(captions_path(item, config.output_dir).read_text(encoding="utf-8"))
            stale["caption_revision"] = 6
            captions_path(item, config.output_dir).write_text(
                json.dumps(stale, ensure_ascii=False), encoding="utf-8")
            before = len(fake.calls)
            self.assertEqual(pipeline.run_one(item), output)
            upgrade_calls = fake.calls[before:]
            self.assertTrue(upgrade_calls)
            self.assertTrue(all(call[0] == "POST" and call[1].endswith("/chat/completions")
                                for call in upgrade_calls))

            captions_path(item, config.output_dir).unlink()
            fake.uncertain = ["示例疑点，保留原词"]
            before = len(fake.calls)
            self.assertEqual(pipeline.run_one(item), output)
            later_calls = fake.calls[before:]
            self.assertTrue(later_calls)
            self.assertTrue(all(call[0] == "POST" and call[1].endswith("/chat/completions")
                                for call in later_calls))
            saved = json.loads(captions_path(item, config.output_dir).read_text(encoding="utf-8"))
            self.assertEqual(saved["uncertain"], fake.uncertain)
            runs = json.loads((work_dir(item, config.output_dir) / "runs.json")
                              .read_text(encoding="utf-8"))["runs"]
            self.assertEqual(len(runs), 4)
            self.assertTrue(all(run["status"] == "完成" for run in runs))
            self.assertTrue(all(run["output"] == str(output) for run in runs))

            restyled = replace(config, caption_style=CaptionStyle(30, "#FFD233", "#1D3050", 2),
                               nameplate_scale=.75, nameplate_box=(320, 390, 180, 100))
            before = len(fake.calls)
            Pipeline(client, restyled, lambda *_: None, threading.Event()).run_one(item)
            self.assertEqual(len(fake.calls), before)
            restyled_plan = json.loads((work_dir(item, config.output_dir) / "edit-plan.json").read_text(encoding="utf-8"))
            self.assertEqual(restyled_plan["template"]["caption_style"], restyled.caption_style.snapshot())
            self.assertEqual(restyled_plan["template"]["nameplate_scale"], .75)
            self.assertEqual(restyled_plan["template"]["nameplate_box"], [320, 390, 180, 100])

            changed_prompts = PromptSet(config.prompts.edit + "\n补充剪辑判断。", config.prompts.caption,
                                        config.prompts.review)
            updated = replace(config, prompts=changed_prompts)
            before = len(fake.calls)
            Pipeline(client, updated, lambda *_: None, threading.Event()).run_one(item)
            prompt_calls = fake.calls[before:]
            self.assertTrue(prompt_calls)
            self.assertTrue(all(call[0] == "POST" and call[1].endswith("/chat/completions")
                                for call in prompt_calls))
            plan = json.loads((work_dir(item, config.output_dir) / "edit-plan.json")
                              .read_text(encoding="utf-8"))
            self.assertEqual(plan["prompts_sha256"], changed_prompts.digest())


if __name__ == "__main__":
    unittest.main()
