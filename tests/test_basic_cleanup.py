from __future__ import annotations

import json
from pathlib import Path
import unittest

from studio.alignment import align_captions, build_timeline
from studio.captions import Word, extract_words
from studio.edit_plan import build_edit_plan
from studio.models import Person
from studio.sentence_flow import _caption_segments, build_sentence_project
from studio.strategy import Strategy


def tokens(words):
    return [{"token": i, "word_index": i, "text": w.text, "punctuation": w.punctuation,
             "start_ms": round(w.start * 1000), "end_ms": round(w.end * 1000)} for i, w in enumerate(words)]


def caption(words, text, cleanup=None):
    return {"segments": [{"start_token": 0, "end_token": len(words), "text": text,
                          **({"display_cleanup": cleanup} if cleanup is not None else {})}]}


class BasicCleanupTests(unittest.TestCase):
    def test_audio_policy_rejects_complete_clause_and_semantic_restart(self):
        for phrase in ("我今天并没有吃多少", "今天下午只喝了一个奶", "三餐的热量分布"):
            words = [Word(0, 1, phrase), Word(1, 2, "继续解释")]
            plan = build_edit_plan(words, [{"start_word": 0, "end_word": 1,
                                            "reason": "删掉重复句子"}], 2, [])
            self.assertEqual(plan["cuts"], [])
            self.assertEqual(plan["decisions"][0]["action"], "KEEP")

    def test_short_filler_keeps_audio_but_long_filler_is_cut(self):
        for duration, expected in ((.1, "KEEP"), (.2, "KEEP"), (.3, "KEEP"), (1.2, "CUT")):
            words = [Word(.4, .4 + duration, "呃"), Word(.4 + duration, 4, "我吃得不多")]
            plan = build_edit_plan(words, [{"start_word": 0, "end_word": 1, "reason": "孤立口气词"}], 4, [])
            self.assertEqual(plan["decisions"][0]["action"], expected)
            if expected == "KEEP":
                self.assertFalse(plan["decisions"][0]["need_review"])
                self.assertFalse(plan["cuts"])

    def test_short_stutter_caption_inside_one_word_preserves_full_audio_range(self):
        words = [Word(0, .1, "我"), Word(.1, .7, "吃吃的"), Word(.7, 1.7, "怎么怎么样")]
        result = _caption_segments(caption(words, "我吃的怎么怎么样", [
            {"start_char": 1, "end_char": 2, "kind": "stutter", "reason": "短促结巴保留原声"}]), tokens(words), words)
        self.assertEqual(result[0]["text"], "我吃的怎么怎么样")
        self.assertEqual((result[0]["start"], result[0]["end"]), (0, 1.7))
        self.assertEqual(result[0]["display_cleanup"][0]["source_text"], "吃")
        self.assertTrue(result[0]["display_cleanup"][0]["audio_kept"])
        aligned = align_captions(result, words, [])
        self.assertEqual((aligned[0]["start_word"], aligned[0]["end_word"]), (0, 3))

    def test_word_internal_cut_is_kept_when_boundary_is_only_interpolated(self):
        words = [Word(0, .1, "我"), Word(.1, .7, "吃吃的"), Word(.7, 1.7, "怎么怎么样")]
        plan = build_edit_plan(words, [{"word_index": 1, "start_char": 0, "end_char": 1,
                                        "reason": "重复起音"}], 1.7, [])
        self.assertEqual(plan["cuts"], [])
        self.assertIn("均分估算", plan["decisions"][0]["policy_reason"])

    def test_stutter_across_asr_words_is_allowed_with_explicit_cleanup(self):
        words = [Word(i * .1, (i + 1) * .1, value) for i, value in enumerate(("我", "吃", "吃", "得", "不多"))]
        result = _caption_segments(caption(words, "我吃得不多", [
            {"start_char": 1, "end_char": 2, "kind": "stutter", "reason": "短促结巴"}]), tokens(words), words)
        self.assertEqual(result[0]["end_word"], 5)
        inferred = _caption_segments(caption(words, "我吃得不多"), tokens(words), words)
        self.assertEqual(inferred[0]["text"], "我吃得不多")
        self.assertEqual(inferred[0]["display_cleanup"][0]["source_text"], "吃")

    def test_subtitle_cleanup_rejects_long_audio(self):
        words = [Word(0, 1, "我"), Word(1, 2.2, "吃"), Word(2.2, 3, "吃"), Word(3, 4, "得不多")]
        result = _caption_segments(caption(words, "我吃得不多", [
            {"start_char": 1, "end_char": 2, "kind": "stutter", "reason": "结巴"}]), tokens(words), words)
        self.assertTrue(result[0]["caption_fallback"])
        self.assertEqual(result[0]["text"], "我吃吃得不多")

    def test_model_decides_stutter_despite_asr_punctuation_or_short_pause(self):
        for punctuation, gap in (("，", 0), ("", .4)):
            words = [Word(0, .1, "我"), Word(.1, .2, "吃", punctuation),
                     Word(.2 + gap, .3 + gap, "吃"), Word(.3 + gap, 1, "得不多")]
            result = _caption_segments(caption(words, "我吃得不多", [
                {"start_char": 1, "end_char": 2, "kind": "stutter", "reason": "上下文确认无意起音"}]), tokens(words), words)
            self.assertEqual(result[0]["text"], "我吃得不多")
            self.assertEqual((result[0]["start"], result[0]["end"]), (0, 1))

    def test_model_cleanup_is_not_limited_to_fixed_vocabulary(self):
        words = [Word(0, .3, "这什么"), Word(.3, 1, "我吃得不多")]
        result = _caption_segments(caption(words, "我吃得不多", [
            {"start_char": 0, "end_char": 3, "kind": "filler", "reason": "上下文确认是无意义口头起音"}]), tokens(words), words)
        self.assertEqual(result[0]["display_cleanup"][0]["source_text"], "这什么")
        self.assertEqual(result[0]["start"], 0)

    def test_combined_hesitation_and_stutter_preserves_audio_and_both_records(self):
        values = [(0, .5, "咦"), (.5, .6, "我"), (.6, .8, "吃"),
                  (.8, 1, "吃"), (1, 1.5, "得不多")]
        words = [Word(*value) for value in values]
        raw = {"transcripts": [{"sentences": [{"words": [
            {"begin_time": round(w.start * 1000), "end_time": round(w.end * 1000), "text": w.text}
            for w in words]}]}]}
        def model(_system, user):
            context = json.loads(user)
            if "target_sentences" in context:
                # Even a suggested small cut must leave the original audio.
                return {"cuts": [{"start_word": 0, "end_word": 1, "reason": "无意义起音"}]}
            return caption(words, "我吃得不多", [
                {"start_char": 0, "end_char": 1, "kind": "filler", "reason": "无意义起音原声保留"},
                {"start_char": 2, "end_char": 3, "kind": "stutter", "reason": "短促结巴原声保留"}])
        project = build_sentence_project(raw, Person("医院", "医生", "姓名"), model)
        self.assertEqual(project["edit_plan"]["cuts"], [])
        aligned = align_captions(project["captions"], words, [])
        self.assertEqual(aligned[0]["text"], "我吃得不多")
        self.assertEqual((aligned[0]["start"], aligned[0]["end"]), (0, 1.5))
        timeline = build_timeline(Path("source.mp4"), Path("output.mp4"), words, aligned, [], 1.5)
        self.assertEqual(timeline["output_duration"], 1.5)
        self.assertEqual([item["source_text"] for item in timeline["captions"][0]["display_cleanup"]], ["咦", "吃"])
        self.assertTrue(all(item["audio_kept"] for item in timeline["captions"][0]["display_cleanup"]))

    def test_small_filler_normalization_keeps_timestamp_and_auditable_timeline(self):
        words = [Word(0, .1, "嗯"), Word(.1, .7, "我今天"), Word(.7, 1.7, "并没有吃多少")]
        raw = {"transcripts": [{"sentences": [{"words": [
            {"begin_time": round(w.start * 1000), "end_time": round(w.end * 1000), "text": w.text}
            for w in words]}]}]}
        def model(_system, user):
            context = json.loads(user)
            if "target_sentences" in context:
                return {"cuts": [{"start_word": 0, "end_word": 1, "reason": "孤立口气词"}]}
            return caption(words, "我今天并没有吃多少", [
                {"start_char": 0, "end_char": 1, "kind": "filler", "reason": "原声很短只修整字幕"}])
        project = build_sentence_project(raw, Person("医院", "医生", "姓名"), model,
                      plan_builder=lambda proposals: build_edit_plan(extract_words(raw), proposals, 1.7, []))
        self.assertEqual(project["edit_plan"]["cuts"], [])
        aligned = align_captions(project["captions"], words, [])
        self.assertEqual(aligned[0]["start"], 0)
        self.assertEqual(aligned[0]["start_word"], 0)
        timeline = build_timeline(Path("source.mp4"), Path("output.mp4"), words, aligned, [], 1.7)
        self.assertEqual(timeline["output_duration"], 1.7)
        self.assertFalse(timeline["captions"][0]["asr_words"][0]["removed"])
        omission = timeline["captions"][0]["display_cleanup"][0]
        self.assertEqual((omission["source_start"], omission["output_start"], omission["output_end"]), (0, 0, .1))

    def test_asr_correction_still_preserves_sentence(self):
        words = [Word(0, 1, "武汉同济院")]
        self.assertEqual(_caption_segments(caption(words, "武汉同济医院"), tokens(words), words)[0]["text"], "武汉同济医院")
        words = [Word(0, 1, "我吃的不多")]
        response = caption(words, "我吃得不多")
        response["changes"] = [{"from": "的", "to": "得", "reason": "ASR同音错字"}]
        self.assertEqual(_caption_segments(response, tokens(words), words)[0]["text"], "我吃得不多")
        words = [Word(0, .2, "我吃"), Word(.2, .4, "的"), Word(.4, 1, "不多")]
        response = caption(words, "我吃得不多")
        response["changes"] = [{"from": "的", "to": "得", "reason": "同音错字"}]
        self.assertEqual(_caption_segments(response, tokens(words), words)[0]["text"], "我吃得不多")
        words = [Word(0, 1, "我今天并没有吃多少")]
        with self.assertRaisesRegex(ValueError, "差异过大"):
            _caption_segments(caption(words, "天气晴朗适合出门"), tokens(words), words)

    def test_new_timing_limits_are_configurable_and_validated(self):
        configured = Strategy.from_dict({**Strategy().snapshot(), "min_model_cut_seconds": .25})
        self.assertEqual(configured.min_model_cut_seconds, .25)
        with self.assertRaises(ValueError):
            Strategy.from_dict({**Strategy().snapshot(), "max_subtitle_cleanup_seconds": 10})


if __name__ == "__main__":
    unittest.main()
