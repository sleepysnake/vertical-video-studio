from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from studio.bailian import ModelOutputError
from studio.captions import Word, extract_words
from studio.edit_plan import build_edit_plan
from studio.models import Person
from studio.sentence_flow import _check_cuts, build_sentence_project
from studio.strategy import Strategy


PERSON = Person("示例医院", "医生", "张帆")


def transcription(prefix, resumed, separate=False):
    first = [{"begin_time": 500, "end_time": 950, "text": prefix[0]},
             {"begin_time": 950, "end_time": 1300, "text": prefix[1]}]
    after = [{"begin_time": 2700, "end_time": 3500, "text": resumed[0]},
             {"begin_time": 3500, "end_time": 4800, "text": resumed[1]}]
    parts = [first, after] if separate else [first + after]
    return {"transcripts": [{"sentences": [
        {"text": "".join(word["text"] for word in part), "words": part}
        for part in parts]}]}


def caption_response(context):
    return {"segments": [{"start_token": sentence["start_token"],
                          "end_token": sentence["end_token"], "text": sentence["text"]}
                         for sentence in context["sentences"]], "changes": [], "uncertain": []}


def context_from(user):
    # Retries include the earlier response after the input JSON.
    return json.JSONDecoder().raw_decode(user)[0]


def proposal(resume_word=2):
    return {"kind": "abandoned_start", "start_word": 0, "end_word": 2,
            "resume_word": resume_word,
            "reason": "短起句尚未完成观点，停顿后独立重起，无独有信息或刻意强调"}


class AbandonedFlowTests(unittest.TestCase):
    def project(self, raw, model, silences=((1.7, 2.7),), **kwargs):
        words = extract_words(raw)
        return build_sentence_project(raw, PERSON, model,
            plan_builder=lambda proposals: build_edit_plan(words, proposals, 5.5, list(silences)),
            **kwargs)

    def test_model_proposals_for_different_short_starts_reach_plan_and_caption_only_retained_speech(self):
        fixtures = [(('接下来', '我想'), ('实际方案', '只有两个步骤')),
                    (('这部分', '如果'), ('关键条件', '是保持运动')),
                    (('先从', '这个'), ('另一项', '需要单独检查'))]
        for prefix, resumed in fixtures:
            with self.subTest(prefix=prefix):
                contexts = []
                def model(_system, user):
                    context = context_from(user)
                    contexts.append(context)
                    return {"cuts": [proposal()], "uncertain": []} if "target_sentences" in context else caption_response(context)
                project = self.project(transcription(prefix, resumed), model)
                self.assertEqual(project["edit_plan"]["applied_proposals"], [proposal()])
                self.assertEqual("".join(c["text"] for c in project["captions"]), "".join(resumed))
                self.assertEqual([token["word_index"] for token in contexts[-1]["kept_tokens"]], [2, 3])
                self.assertEqual(project["captions"][0]["start"], 2.7)
                self.assertEqual(project["edit_attempts"][0]["response"]["cuts"], [proposal()])

    def test_pause_metadata_and_constraints_include_next_word_for_combined_asr_sentence(self):
        contexts = []
        def model(_system, user):
            context = context_from(user)
            contexts.append(context)
            return {"cuts": []} if "target_sentences" in context else caption_response(context)
        self.project(transcription(('接下来', '我想'), ('实际方案', '只有两个步骤')), model)
        context = contexts[0]
        self.assertEqual(context["abandoned_start_constraints"], {
            "min_pause_seconds": Strategy().abandoned_pause_seconds,
            "max_spoken_seconds": Strategy().max_abandoned_start_seconds})
        self.assertEqual(context["words"][1]["pause_after_ms"], 1400)
        self.assertEqual(context["full_transcript"], "接下来我想实际方案只有两个步骤")
        self.assertEqual(contexts[-1]["full_transcript"], context["full_transcript"])
        self.assertEqual(context["words"][1]["next_word_index"], 2)
        self.assertIsNone(context["words"][-1]["next_word_index"])
        self.assertEqual(context["words"][-1]["pause_after_ms"], 0)

    def test_group_boundary_allows_next_word_in_readonly_after_but_never_deletes_it(self):
        raw = transcription(('接下来', '我想'), ('实际方案', '只有两个步骤'), separate=True)
        raw["transcripts"][0]["text"] = "接下来我想。实际方案只有两个步骤。"
        contexts = []
        def model(_system, user):
            context = context_from(user)
            contexts.append(context)
            if "target_sentences" in context:
                cuts = [proposal()] if context["target_sentences"][0]["start_word"] == 0 else []
                return {"cuts": cuts}
            return caption_response(context)
        with patch("studio.sentence_flow.sentence_groups", side_effect=lambda sentences: [[s] for s in sentences]):
            project = self.project(raw, model)
        self.assertEqual(contexts[0]["words"][-1]["next_word_index"], 2)
        self.assertTrue(all(c["full_transcript"] == raw["transcripts"][0]["text"] for c in contexts))
        self.assertEqual(contexts[0]["after"][0]["start_word"], 2)
        self.assertEqual(project["cut_proposals"], [proposal()])
        self.assertEqual("".join(c["text"] for c in project["captions"]), "实际方案只有两个步骤")
        self.assertEqual(project["captions"][0]["start_word"], 2)

    def test_wrong_resume_retries_then_preserves_speech_and_records_errors(self):
        raw = transcription(('接下来', '我想'), ('实际方案', '只有两个步骤'))
        def model(_system, user):
            context = context_from(user)
            return {"cuts": [proposal(3)]} if "target_sentences" in context else caption_response(context)
        project = self.project(raw, model)
        self.assertEqual(project["model_cut_proposals"], [])
        self.assertEqual(project["edit_plan"]["applied_proposals"], [])
        self.assertTrue("".join(c["text"] for c in project["captions"]).startswith("接下来我想"))
        self.assertEqual(len(project["edit_attempts"]), 2)
        self.assertTrue(all("紧邻" in item["error"] for item in project["edit_attempts"]))
        self.assertEqual(project["edit_attempts"][1]["response"]["cuts"], [proposal(3)])

    def test_no_audio_silence_evidence_cuts_only_model_identified_words(self):
        raw = transcription(('接下来', '我想'), ('实际方案', '只有两个步骤'))
        def model(_system, user):
            context = context_from(user)
            return {"cuts": [proposal()]} if "target_sentences" in context else caption_response(context)
        project = self.project(raw, model, silences=())
        self.assertEqual(project["model_cut_proposals"], [proposal()])
        self.assertEqual(project["edit_plan"]["applied_proposals"], [proposal()])
        self.assertEqual(project["edit_plan"]["cuts"][0]["end"], 1.3)
        self.assertEqual("".join(c["text"] for c in project["captions"]), "实际方案只有两个步骤")
        self.assertEqual(project["edit_attempts"][0]["error"], "")

    def test_new_kind_requires_complete_current_words_existing_next_word_and_valid_fields(self):
        words = [Word(0, .5, "先从"), Word(.5, 1, "这个"), Word(2, 3, "实际方案")]
        self.assertEqual(_check_cuts([proposal()], 0, 2, words), [proposal()])
        invalid = [{**proposal(), "kind": "delete_sentence"},
                   {**proposal(), "resume_word": 3},
                   {**proposal(), "word_index": 0, "start_char": 0, "end_char": 1},
                   {**proposal(), "end_ms": 2000},
                   {**proposal(), "start_word": 2, "end_word": 3, "resume_word": 3}]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                _check_cuts([value], 0, 2, words)
        # Semantic validation uses the declared category, not magic reason words.
        value = {**proposal(), "reason": "ASR 合并了停顿前后内容，前段未完成观点且后段独立重起"}
        self.assertEqual(_check_cuts([value], 0, 2, words), [value])
        with self.assertRaisesRegex(ValueError, "完整保留"):
            _check_cuts([proposal(), {"start_word": 2, "end_word": 3, "reason": "另一个局部起音"}], 0, 3, words)

    def test_old_resume_without_attempt_log_and_model_failure_are_auditable(self):
        raw = transcription(('接下来', '我想'), ('实际方案', '只有两个步骤'))
        resume = {"signature": [[0, 4]], "edited": 0, "captioned": 0,
                  "cuts": [], "captions": [], "changes": [], "uncertain": [], "fallback_groups": []}
        def model(_system, user):
            context = context_from(user)
            if "target_sentences" in context:
                raise ModelOutputError("模型输出被截断")
            return caption_response(context)
        project = self.project(raw, model, resume=resume)
        self.assertEqual(project["edit_plan"]["applied_proposals"], [])
        self.assertEqual(project["edit_attempts"], [{"group": 1, "attempt": 1,
            "response": None, "error": "模型输出被截断"}])
        self.assertTrue(project["captions"])


if __name__ == "__main__":
    unittest.main()
