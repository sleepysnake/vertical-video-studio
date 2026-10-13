from __future__ import annotations

import json
import unittest

from studio.bailian import ModelOutputError
from studio.models import Person
from studio.sentence_flow import build_sentence_project


def source():
    values = ("比例是", "6", ":", "3", ":", "1")
    return {"transcripts": [{"sentences": [{"text": "比例是6:3:1", "words": [
        {"text": text, "begin_time": i * 300, "end_time": (i + 1) * 300}
        for i, text in enumerate(values)]}]}]}


class PunctuationCoverageTests(unittest.TestCase):
    def test_punctuation_only_words_keep_original_indices_but_are_not_missing_speech(self):
        def model(_system, user):
            payload = json.JSONDecoder().raw_decode(user)[0]
            if payload["phase"] == "edit":
                return {"cuts": []}
            self.assertEqual([t["word_index"] for t in payload["kept_tokens"]], [0, 1, 3, 5])
            return {"segments": [{"start_token": 0, "end_token": 4, "text": "比例是6比3比1"}]}
        result = build_sentence_project(source(), Person("机构", "头衔", "姓名"), model)
        self.assertEqual(result["coverage_audit"]["punctuation_only_word_indices"], [2, 4])
        self.assertTrue(result["coverage_audit"]["retained_words_assigned_once_in_order"])
        self.assertEqual(result["captions"][0]["source_word_indices"], [0, 1, 3, 5])

    def test_literal_fallback_uses_the_same_captionable_word_set(self):
        result = build_sentence_project(source(), Person("机构", "头衔", "姓名"),
            lambda _system, _user: (_ for _ in ()).throw(ModelOutputError("超时")))
        self.assertTrue(result["caption_fallback_groups"])
        self.assertTrue(result["coverage_audit"]["retained_words_assigned_once_in_order"])
        self.assertEqual(result["coverage_audit"]["captionable_words"], 4)

    def test_api_failure_does_not_become_completed_phase_in_strict_pipeline(self):
        checkpoints, calls = [], []
        def model(_system, user):
            calls.append(json.loads(user)["phase"])
            raise ModelOutputError("输出截断后超时")
        with self.assertRaisesRegex(ModelOutputError, "剪辑模型调用未成功"):
            build_sentence_project(source(), Person("机构", "头衔", "姓名"), model,
                strict_model_failures=True, checkpoint=lambda state: checkpoints.append(json.loads(json.dumps(state))))
        self.assertEqual(calls, ["edit"])
        self.assertEqual(checkpoints[-1]["edited"], 0)
        self.assertEqual(checkpoints[-1]["captioned"], 0)
        retry_calls = []
        def recovered(_system, user):
            context = json.JSONDecoder().raw_decode(user)[0]
            retry_calls.append(context["phase"])
            if context["phase"] == "edit":
                return {"cuts": []}
            return {"segments": [{"start_token": 0, "end_token": 4, "text": "比例是6比3比1"}]}
        result = build_sentence_project(source(), Person("机构", "头衔", "姓名"), recovered,
            strict_model_failures=True, resume=checkpoints[-1])
        self.assertEqual(retry_calls, ["edit", "caption"])
        self.assertFalse(result["caption_fallback_groups"])
        self.assertFalse(result["uncertain"])

    def test_caption_api_failure_reuses_successful_edit_phase(self):
        checkpoints, calls = [], []
        def model(_system, user):
            phase = json.JSONDecoder().raw_decode(user)[0]["phase"]
            calls.append(phase)
            if phase == "edit":
                return {"cuts": []}
            raise ModelOutputError("字幕请求超时")
        with self.assertRaisesRegex(ModelOutputError, "字幕模型调用未成功"):
            build_sentence_project(source(), Person("机构", "头衔", "姓名"), model,
                strict_model_failures=True, checkpoint=lambda state: checkpoints.append(json.loads(json.dumps(state))))
        self.assertEqual(calls, ["edit", "caption"])
        self.assertEqual(checkpoints[-1]["edited"], 1)
        self.assertEqual(checkpoints[-1]["captioned"], 0)
        self.assertEqual(checkpoints[-1]["fallback_groups"], [])
        retry_calls = []
        def recovered(_system, user):
            retry_calls.append(json.JSONDecoder().raw_decode(user)[0]["phase"])
            return {"segments": [{"start_token": 0, "end_token": 4, "text": "比例是6比3比1"}]}
        result = build_sentence_project(source(), Person("机构", "头衔", "姓名"), recovered,
            strict_model_failures=True, resume=checkpoints[-1])
        self.assertEqual(retry_calls, ["caption"])
        self.assertFalse(result["caption_fallback_groups"])
