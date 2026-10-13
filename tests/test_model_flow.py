"""The final semantic review, rather than code policy, owns edit choices."""
from copy import deepcopy
import json
import unittest

from studio.bailian import ModelOutputError
from studio.models import Person
from studio.sentence_flow import build_sentence_project


def source(text, step=300):
    words = [{"text": character, "begin_time": 300 + i * step,
              "end_time": 300 + (i + 1) * step, "punctuation": ""}
             for i, character in enumerate(text)]
    return {"transcripts": [{"text": text, "sentences": [{"text": text,
             "begin_time": words[0]["begin_time"], "end_time": words[-1]["end_time"],
             "words": words}]}]}


def cut(first, last, reason="模型结合上下文决定删除", **extra):
    return {"start_word": first, "end_word": last, "reason": reason, **extra}


def captions(context, text=None):
    tokens = context["kept_tokens"]
    return {"segments": [{"start_token": 0, "end_token": len(tokens),
                          "text": text or "".join(t["text"] for t in tokens)}],
            "changes": [], "uncertain": []}


class ModelFlowTests(unittest.TestCase):
    def test_independent_review_withdraws_wrong_cut_and_adds_missing_cut(self):
        calls = []
        initial = {"cuts": [cut(1, 2)], "pause_compressions": [],
                   "display_only_suggestions": [], "uncertain": []}
        final = {"cuts": [cut(3, 5)], "pause_compressions": [],
                 "display_only_suggestions": [{"word_index": 6, "reason": "只修文字"}],
                 "uncertain": [], "review_changes": [
                     {"action": "withdraw", "reason": "原内容含独有信息"},
                     {"action": "add", "reason": "补漏掉的改口"}]}

        def model(system, user):
            context = json.loads(user)
            calls.append(context)
            if context["phase"] == "edit":
                return deepcopy(initial)
            if context["phase"] == "semantic_review":
                self.assertEqual(context["draft_plan"], initial)
                self.assertEqual(context["full_transcript"], "我今天吃吃得不多")
                return deepcopy(final)
            self.assertEqual(context["display_only_suggestions"], final["display_only_suggestions"])
            return captions(context)

        result = build_sentence_project(source("我今天吃吃得不多"), Person("医院", "医生", "姓名"),
                                        model, model_authored=True)
        self.assertEqual([c["phase"] for c in calls], ["edit", "semantic_review", "caption"])
        self.assertEqual(result["cut_proposals"], final["cuts"])
        self.assertEqual(result["model_edit_plan"]["draft_plans"], [initial])
        self.assertEqual(result["model_edit_plan"]["final_plans"], [final])
        self.assertEqual(result["captions"][0]["source_word_indices"], [0, 1, 2, 5, 6, 7])
        self.assertTrue(result["coverage_audit"]["retained_words_assigned_once_in_order"])

    def test_review_failure_resumes_review_and_caption_without_repeating_edit(self):
        saved, calls = [], []

        def fail_review(system, user):
            context = json.loads(user)
            calls.append(context["phase"])
            if context["phase"] == "edit":
                return {"cuts": [], "uncertain": []}
            raise ModelOutputError("离线模拟复审请求失败")

        raw = source("大家好")
        person = Person("医院", "医生", "姓名")
        with self.assertRaisesRegex(ModelOutputError, "语义复审模型调用未成功"):
            build_sentence_project(raw, person, fail_review,
                                   checkpoint=lambda s: saved.append(deepcopy(s)), model_authored=True)
        self.assertEqual((saved[-1]["edited"], saved[-1]["reviewed"], saved[-1]["captioned"]), (1, 0, 0))
        self.assertEqual(calls, ["edit", "semantic_review"])
        calls.clear()

        def resume_model(system, user):
            context = json.loads(user)
            calls.append(context["phase"])
            return {"cuts": [], "uncertain": []} if context["phase"] == "semantic_review" else captions(context)

        result = build_sentence_project(raw, person, resume_model, resume=saved[-1], model_authored=True)
        self.assertEqual(calls, ["semantic_review", "caption"])
        self.assertEqual(result["captions"][0]["text"], "大家好")

    def test_caption_model_wording_is_not_overridden_by_duration_or_similarity(self):
        raw_text, display = "咦嗯呃我我我我我我我我我吃得不多", "我吃得不多"

        def model(system, user):
            context = json.loads(user)
            if context["phase"] != "caption":
                return {"cuts": [], "uncertain": []}
            return captions(context, display)

        result = build_sentence_project(source(raw_text), Person("医院", "医生", "姓名"),
                                        model, model_authored=True)
        self.assertEqual(result["captions"][0]["text"], display)
        self.assertFalse(result["caption_fallback_groups"])
        self.assertEqual(result["edit_plan"]["cuts"], [])
        cleanup = result["captions"][0]["display_cleanup"]
        self.assertTrue(all(c["audio_kept"] for c in cleanup))
        self.assertTrue(any(c["source_end"] - c["source_start"] > 1 for c in cleanup))

    def test_bad_execution_indices_stop_and_checkpoint_without_policy_fallback(self):
        calls, saved = [], []

        def model(system, user):
            calls.append(user)
            return {"cuts": [cut(1, 99)], "uncertain": []}

        with self.assertRaisesRegex(ModelOutputError, "两次返回不可执行的计划"):
            build_sentence_project(source("大家好"), Person("医院", "医生", "姓名"), model,
                                   checkpoint=lambda s: saved.append(deepcopy(s)), model_authored=True)
        self.assertEqual(len(calls), 2)
        self.assertEqual(saved[-1]["edited"], 0)
        self.assertEqual(saved[-1]["cuts"], [])
        self.assertEqual(len(saved[-1]["edit_attempts"]), 2)

    def test_full_character_range_is_normalized_before_caption_mapping(self):
        proposal = {"word_index": 0, "start_char": 0, "end_char": 1, "reason": "模型决定"}

        def model(system, user):
            context = json.loads(user)
            return {"cuts": [proposal], "uncertain": []} if context["phase"] != "caption" else captions(context)

        result = build_sentence_project(source("嗯大家好"), Person("医院", "医生", "姓名"),
                                        model, model_authored=True)
        self.assertEqual(result["captions"][0]["source_word_indices"], [1, 2, 3])
        self.assertEqual(result["captions"][0]["text"], "大家好")


if __name__ == "__main__":
    unittest.main()
