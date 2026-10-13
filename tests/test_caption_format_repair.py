from __future__ import annotations

from copy import deepcopy
import unittest
from unittest.mock import patch

from studio.caption_format_repair import repair_caption_lengths
from studio.captions import MAX_CAPTION_CHARS, Word


def caption(words, indices, text=None, **extra):
    source = "".join(words[index].text for index in indices)
    return {"start_word": indices[0], "end_word": indices[-1] + 1,
            "source_word_indices": indices, "source_text": source,
            "start": words[indices[0]].start, "end": words[indices[-1]].end,
            "text": source if text is None else text, "model_wording": True, **extra}


class CaptionFormatRepairTests(unittest.TestCase):
    def assert_mapping(self, original, repaired, words):
        self.assertEqual("".join(part["text"] for part in repaired), original["text"])
        self.assertEqual("".join(part["source_text"] for part in repaired), original["source_text"])
        self.assertEqual([index for part in repaired for index in part["source_word_indices"]],
                         original["source_word_indices"])
        for part in repaired:
            self.assertLessEqual(len(part["text"]), MAX_CAPTION_CHARS)
            self.assertEqual(part["start"], round(words[part["source_word_indices"][0]].start, 3))
            self.assertEqual(part["end"], round(words[part["source_word_indices"][-1]].end, 3))
            self.assertTrue(part["length_split"])
            self.assertTrue(part["formatting_fallback"])
            self.assertNotIn("caption_fallback", part)
            self.assertNotIn("timing_estimated", part)

    def test_only_long_caption_changes_and_model_wording_is_preserved(self):
        text = "各位朋友大家好我是武汉同济院李景平医生"
        words = [Word(index * .2, (index + 1) * .2, char) for index, char in enumerate(text)]
        greeting = caption(words, list(range(7)))
        long = caption(words, list(range(len(words))),
                       "各位朋友大家好我是武汉同济医院李景平医生", id="original-model-3",
                       model_text_changes=[{"from": "同济院", "to": "同济医院", "reason": "机构名称纠错"}])
        ending = caption(words, [len(words) - 2, len(words) - 1])
        snapshot = deepcopy([greeting, long, ending])
        repaired = repair_caption_lengths([greeting, long, ending], words, ("武汉同济医院", "李景平"))
        self.assertIs(repaired[0], greeting)
        self.assertIs(repaired[-1], ending)
        parts = repaired[1:-1]
        self.assert_mapping(long, parts, words)
        self.assertEqual(sum("model_text_changes" in part for part in parts), 1)
        self.assertEqual(parts[0]["model_text_changes"], long["model_text_changes"])
        self.assertEqual(parts[0]["model_text_changes_scope"]["source_word_range"], [0, len(words)])
        self.assertTrue(all(part["formatting_source_segment_id"] == "original-model-3" for part in parts))
        self.assertEqual([greeting, long, ending], snapshot)

    def test_non_contiguous_tokens_exclude_deleted_and_punctuation_words(self):
        text = "我今天下午只喝了一杯奶茶所以我觉得吃得不多"
        words = []
        indices = []
        for char in text:
            index = len(words)
            words.append(Word(index * .2, (index + 1) * .2, char))
            indices.append(index)
            if index % 5 == 0:
                words.extend([Word((index + 1) * .2, (index + 2) * .2, "删除原声"),
                              Word((index + 2) * .2, (index + 3) * .2, ":")])
        original = caption(words, indices)
        repaired = repair_caption_lengths([original], words)
        self.assert_mapping(original, repaired, words)
        self.assertTrue(all("删除原声" not in part["source_text"] and ":" not in part["source_text"]
                            for part in repaired))

    def test_short_filler_cleanup_is_distributed_without_duplication(self):
        source = "咦我吃吃得不多为什么还在长胖因为热量摄入超过消耗"
        words = [Word(index * .1, (index + 1) * .1, char) for index, char in enumerate(source)]
        original = caption(words, list(range(len(words))),
                           "我吃得不多为什么还在长胖因为热量摄入超过消耗",
                           display_cleanup=[{"start_char": 0, "end_char": 1, "kind": "filler", "reason": "起音"},
                                            {"start_char": 2, "end_char": 3, "kind": "stutter", "reason": "结巴"}])
        repaired = repair_caption_lengths([original], words)
        self.assert_mapping(original, repaired, words)
        omissions = [item for part in repaired for item in part.get("display_cleanup", [])]
        self.assertEqual([item["source_text"] for item in omissions], ["咦", "吃"])
        for part in repaired:
            for omission in part.get("display_cleanup", []):
                self.assertEqual(part["source_text"][omission["start_char"]:omission["end_char"]],
                                 omission["source_text"])
                self.assertTrue(part["start"] <= omission["source_start"] < omission["source_end"] <= part["end"])

    def test_cleanup_crossing_split_is_clipped_to_each_source_range(self):
        words = [Word(index * .1, (index + 1) * .1, "甲") for index in range(30)]
        original = caption(words, list(range(30)), "甲" * 28, id="s7", display_cleanup=[
            {"start_char": 15, "end_char": 17, "source_text": "甲甲",
             "kind": "model_cleanup", "reason": "短重复", "audio_kept": True}])
        with patch("studio.caption_format_repair.shorten_captions", return_value=[
                {"start_word": 0, "end_word": 16, "text": "甲" * 15},
                {"start_word": 16, "end_word": 30, "text": "甲" * 13}]):
            repaired = repair_caption_lengths([original], words)
        self.assert_mapping(original, repaired, words)
        first = repaired[0]["display_cleanup"][0]
        second = repaired[1]["display_cleanup"][0]
        self.assertEqual((first["start_char"], first["end_char"], first["source_start"], first["source_end"]),
                         (15, 16, 1.5, 1.6))
        self.assertEqual((second["start_char"], second["end_char"], second["source_start"], second["source_end"]),
                         (0, 1, 1.6, 1.7))
        self.assertEqual(first["original_source_char_range"], [15, 17])
        self.assertEqual(second["original_segment_id"], "s7")

    def test_word_internal_cut_retained_source_text_can_be_allocated(self):
        values = ["咦我", "吃吃得", "不多", "为什么", "还在长胖", "因为", "摄入热量", "超过消耗"]
        words = [Word(index * .5, (index + 1) * .5, text) for index, text in enumerate(values)]
        source = "咦我吃得不多为什么还在长胖因为摄入热量超过消耗"
        original = caption(words, list(range(len(words))), "我吃得不多为什么还在长胖因为摄入热量超过消耗",
                           display_cleanup=[{"start_char": 0, "end_char": 1, "kind": "filler", "reason": "起音"}])
        original["source_text"] = source
        repaired = repair_caption_lengths([original], words)
        self.assert_mapping(original, repaired, words)

    def test_unreliable_timing_raises_instead_of_splitting_time_evenly(self):
        words = [Word(0, 10, "字幕必须保留原文不能通过均分时间伪造字幕与音频关系")]
        with self.assertRaisesRegex(ValueError, "真实源词时间"):
            repair_caption_lengths([caption(words, [0])], words)

    def test_invalid_source_mapping_does_not_silently_drop_tokens(self):
        source = "字幕必须保留原文而且应该按照原有词语的时间来分配"
        words = [Word(index * .1, (index + 1) * .1, char) for index, char in enumerate(source)]
        original = caption(words, list(range(len(words))))
        original["source_word_indices"][3] = original["source_word_indices"][2]
        with self.assertRaisesRegex(ValueError, "按原顺序唯一"):
            repair_caption_lengths([original], words)


if __name__ == "__main__":
    unittest.main()
