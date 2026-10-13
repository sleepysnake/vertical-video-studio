from __future__ import annotations

import json
import unittest

from studio.alignment import align_captions
from studio.captions import Word, extract_words
from studio.models import Person
from studio.sentence_flow import asr_sentences, budget_sentences, build_sentence_project, sentence_groups


def raw_document(parts):
    sentences = []
    for number, values in enumerate(parts):
        sentence_words = []
        for i, text in enumerate(values):
            start = number * 10000 + i * 500
            sentence_words.append({"text": text, "begin_time": start, "end_time": start + 400})
        sentences.append({"text": "".join(values), "words": sentence_words})
    return {"transcripts": [{"text": "".join(s["text"] for s in sentences), "sentences": sentences}]}


class ContextualFlowTests(unittest.TestCase):
    def test_whole_video_context_and_frozen_edit_before_caption_api(self):
        raw = raw_document([("先说明一个", "完整观点"),
                            ("我想说米", "我想说米饭", "要保留数量"),
                            ("后面还要", "保留总结")])
        calls = []

        def model(system, user):
            payload = json.loads(user)
            calls.append(payload)
            if payload["phase"] == "edit":
                self.assertEqual(payload["context_scope"], "whole_video")
                self.assertEqual(len(payload["target_sentences"]), 3)
                return {"cuts": [{"id": "C001", "kind": "repair_restart", "start_word": 2, "end_word": 3,
                                   "replacement_range": {"start_word": 3, "end_word": 5},
                                   "reason": "前半名词未完成后从头完整重说"}]}
            kept = payload["kept_tokens"]
            self.assertEqual([t["word_index"] for t in kept], [0, 1, 3, 4, 5, 6])
            return {"segments": [{"start_token": i, "end_token": i + 1, "text": t["text"]}
                                 for i, t in enumerate(kept)]}

        project = build_sentence_project(raw, Person("机构", "头衔", "姓名"), model)
        self.assertEqual(len(calls), 2)
        self.assertTrue(project["coverage_audit"]["retained_words_assigned_once_in_order"])
        self.assertEqual(project["edit_plan"]["cuts"][0]["source_text"], "我想说米")
        self.assertEqual(project["captions"][2]["start_word"], 3)

    def test_long_caption_returns_to_model_for_semantic_split(self):
        raw = raw_document([("各位朋友大家好", "我是", "这家医院", "的主任医师")])
        seen = []

        def model(_system, user):
            payload = json.loads(user.split("\n上次字幕无效", 1)[0])
            if payload["phase"] == "edit":
                return {"cuts": []}
            seen.append(user)
            if len(seen) == 1:
                return {"segments": [{"start_token": 0, "end_token": 4,
                                       "text": "各位朋友大家好我是这家医院的主任医师"}]}
            return {"segments": [{"start_token": 0, "end_token": 1, "text": "各位朋友大家好"},
                                  {"start_token": 1, "end_token": 4, "text": "我是这家医院的主任医师"}]}

        project = build_sentence_project(raw, Person("这家医院", "主任医师", "姓名"), model)
        self.assertEqual(len(seen), 2)
        self.assertIn("自然意群", seen[-1])
        self.assertEqual([c["text"] for c in project["captions"]],
                         ["各位朋友大家好", "我是这家医院的主任医师"])
        self.assertFalse(project["caption_fallback_groups"])

    def test_valid_source_indices_survive_text_cleanup_and_repeated_words(self):
        raw = raw_document([("嗯", "重要条件", "仍然要保留", "重要条件", "后续解释")])

        def model(_system, user):
            payload = json.loads(user)
            if payload["phase"] == "edit":
                return {"cuts": []}
            return {"segments": [{"start_token": 0, "end_token": 3, "text": "重要条件仍然要保留"},
                                  {"start_token": 3, "end_token": 5, "text": "重要条件后续解释"}]}

        project = build_sentence_project(raw, Person("机构", "头衔", "姓名"), model)
        aligned = align_captions(project["captions"], extract_words(raw), [])
        self.assertEqual(aligned[0]["source_word_indices"], [0, 1, 2])
        self.assertEqual(aligned[0]["start"], 0)
        self.assertEqual(aligned[1]["start_word"], 3)
        self.assertTrue(project["coverage_audit"]["retained_words_assigned_once_in_order"])

    def test_repeated_overlong_model_reply_repairs_layout_without_asr_fallback(self):
        raw = raw_document([tuple("字" for _ in range(31))])

        def model(_system, user):
            payload = json.loads(user.split("\n上次字幕无效", 1)[0])
            if payload["phase"] == "edit":
                return {"cuts": []}
            return {"segments": [{"start_token": 0, "end_token": 31, "text": "字" * 31}]}

        project = build_sentence_project(raw, Person("机构", "头衔", "姓名"), model)
        self.assertEqual(project["caption_fallback_groups"], [])
        self.assertTrue(all(len(c["text"]) <= 16 for c in project["captions"]))
        self.assertTrue(all(c["formatting_fallback"] for c in project["captions"]))
        self.assertEqual("".join(c["text"] for c in project["captions"]), "字" * 31)
        self.assertEqual(len(project["caption_formatting_repairs"]), 1)
        self.assertTrue(project["coverage_audit"]["retained_words_assigned_once_in_order"])
        aligned = align_captions(project["captions"], extract_words(raw), [])
        self.assertTrue(all(a["end"] <= b["start"] for a, b in zip(aligned, aligned[1:])))

    def test_repeated_conflicting_edits_keep_independent_valid_candidates(self):
        raw = raw_document([("前次起说", "完整替代", "另一起说", "接续", "起音", "完整", "废弃", "保留")])
        cuts = [
            {"id": "A", "kind": "repair_restart", "start_word": 0, "end_word": 1,
             "replacement_range": {"start_word": 1, "end_word": 2}, "reason": "明确重新起说"},
            {"id": "B", "kind": "repair_restart", "start_word": 2, "end_word": 3,
             "replacement_range": {"start_word": 3, "end_word": 6}, "reason": "重新起说"},
            {"id": "C", "kind": "repair_restart", "start_word": 4, "end_word": 5,
             "replacement_range": {"start_word": 5, "end_word": 6}, "reason": "独立起音"},
            {"id": "D", "kind": "repair_restart", "start_word": 6, "end_word": 7,
             "replacement_range": {"start_word": 7, "end_word": 8}, "reason": "明确废弃重说"},
        ]
        edit_users = []
        def model(_system, user):
            context = json.JSONDecoder().raw_decode(user)[0]
            if context["phase"] == "edit":
                edit_users.append(user)
                return {"cuts": cuts}
            return {"segments": [{"start_token": i, "end_token": i + 1, "text": t["text"]}
                                 for i, t in enumerate(context["kept_tokens"])]}
        project = build_sentence_project(raw, Person("机构", "头衔", "姓名"), model)
        self.assertEqual(len(edit_users), 2)
        self.assertIn('"proposal_id": "B"', edit_users[1])
        self.assertIn('"proposal_id": "C"', edit_users[1])
        self.assertEqual([p["id"] for p in project["model_cut_proposals"]], ["A", "D"])
        self.assertEqual([p["id"] for p in project["edit_plan"]["applied_proposals"]], ["A", "D"])
        self.assertEqual({p["proposal_id"] for p in project["rejected_model_proposals"]}, {"B", "C"})
        self.assertEqual([i for c in project["captions"] for i in c["source_word_indices"]], [1, 2, 3, 4, 5, 7])

    def test_one_overlong_caption_does_not_discard_another_caption_cleanup(self):
        raw = raw_document([("咦", "我", "吃", "吃", "得", "不", "多") + tuple("内容" for _ in range(17))])
        def model(_system, user):
            context = json.JSONDecoder().raw_decode(user)[0]
            if context["phase"] == "edit":
                return {"cuts": []}
            return {"segments": [{"start_token": 0, "end_token": 7, "text": "我吃得不多"},
                                  {"start_token": 7, "end_token": 24, "text": "内容" * 17}]}
        project = build_sentence_project(raw, Person("机构", "头衔", "姓名"), model)
        first = project["captions"][0]
        self.assertEqual(first["text"], "我吃得不多")
        self.assertEqual({p["source_text"] for p in first["display_cleanup"]}, {"咦", "吃"})
        self.assertFalse(project["caption_fallback_groups"])
        self.assertTrue(project["coverage_audit"]["retained_words_assigned_once_in_order"])

    def test_short_kept_word_at_exact_cut_boundary_is_not_rejected(self):
        result = align_captions([{"start": 1, "end": 1.02, "text": "我",
                                  "source_word_indices": [0]}],
                                [Word(1, 1.02, "我")], [{"start": .5, "end": 1}])
        self.assertEqual((result[0]["start"], result[0]["end"]), (1, 1.02))

    def test_long_video_chapters_preserve_all_sentence_ranges(self):
        raw = raw_document([tuple("内容" * 100 for _ in range(3)) for _ in range(10)])
        words = extract_words(raw)
        sentences = asr_sentences(raw, words)
        groups = sentence_groups(sentences)
        self.assertGreater(len(groups), 1)
        self.assertEqual([s for group in groups for s in group], sentences)

    def test_one_oversized_asr_sentence_is_bounded_without_losing_source_words(self):
        raw = raw_document([tuple("讲话" for _ in range(3000))])
        words = extract_words(raw)
        bounded = budget_sentences(asr_sentences(raw, words), words)
        groups = sentence_groups(bounded)
        self.assertGreater(len(groups), 1)
        self.assertTrue(all(g[-1]["end_word"] - g[0]["start_word"] <= 1200 for g in groups))
        self.assertEqual([i for s in bounded for i in range(s["start_word"], s["end_word"])],
                         list(range(len(words))))


if __name__ == "__main__":
    unittest.main()
