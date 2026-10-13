"""Model-authored cuts are limited by executable timing, not editing policy."""
from __future__ import annotations

from copy import deepcopy
import math
import unittest

from studio.captions import Word
from studio.model_plan import compile_model_plan, validate_executable_cuts
from studio.strategy import Strategy


def cut(first: int, last: int, **metadata) -> dict:
    return {"start_word": first, "end_word": last, "reason": "模型决定删除此处", **metadata}


class ModelPlanTests(unittest.TestCase):
    def test_long_cut_high_removal_ratio_and_short_retained_fragment_are_executed(self):
        words = [Word(0, .03, "保留"), Word(.03, 16, "模型删除的长段"), Word(16, 20, "结尾")]
        proposals = [cut(1, 2, kind="任意模型分类")]
        strategy = Strategy(max_single_model_cut=.1, max_removal_ratio=.01, min_retained_fragment=3)
        plan = compile_model_plan(words, proposals, 20, [], [], strategy=strategy)
        self.assertEqual(plan["mode"], "model_authored")
        self.assertEqual(plan["applied_proposals"], proposals)
        self.assertEqual([(c["start"], c["end"]) for c in plan["cuts"]], [(.03, 16)])
        self.assertEqual(plan["retained_word_audit"]["word_indices"], [0, 2])
        self.assertEqual(plan["review_count"], 0)

    def test_sub_point_two_second_and_short_isolated_filler_model_choices_execute(self):
        words = [Word(0, 1, "前文"), Word(1, 1.1, "嗯"), Word(1.1, 2, "后文")]
        for kind in ("isolated_filler", "stutter", None):
            with self.subTest(kind=kind):
                proposal = cut(1, 2, kind=kind)
                plan = compile_model_plan(words, [proposal], 2, [], [])
                self.assertEqual(plan["applied_proposals"], [proposal])
                self.assertAlmostEqual(plan["cuts"][0]["end"] - plan["cuts"][0]["start"], .1)

    def test_kind_and_replacement_metadata_do_not_veto_other_model_choices(self):
        words = [Word(i, i + 1, "词") for i in range(6)]
        proposals = [cut(1, 2, kind={"model_label": "自定义"},
                         replacement_range={"start_word": 2, "end_word": 4}),
                     cut(2, 3, kind="another_label")]
        original = deepcopy(proposals)
        checked = validate_executable_cuts(proposals, 0, len(words), words)
        self.assertEqual(checked, proposals)
        self.assertIsNot(checked[0], proposals[0])
        self.assertIsNot(checked[0]["kind"], proposals[0]["kind"])
        plan = compile_model_plan(words, proposals, 6, [], [])
        self.assertEqual(plan["applied_proposals"], original)
        self.assertEqual([(c["start"], c["end"]) for c in plan["cuts"]], [(1, 3)])
        self.assertEqual(proposals, original)

    def test_overlapping_deletions_are_merged_without_an_order_winner(self):
        words = [Word(i, i + 1, "词") for i in range(8)]
        proposals = [cut(1, 4, id="a"), cut(3, 6, id="b")]
        plan = compile_model_plan(words, proposals, 8, [], [])
        reverse = compile_model_plan(words, list(reversed(proposals)), 8, [], [])
        self.assertEqual(len(plan["applied_proposals"]), 2)
        self.assertEqual([(c["start"], c["end"]) for c in plan["cuts"]], [(1, 6)])
        self.assertEqual(plan["cuts"][0]["removed_word_indices"], [1, 2, 3, 4, 5])
        self.assertEqual([(c["start"], c["end"]) for c in reverse["cuts"]], [(1, 6)])

    def test_unselected_silence_and_filler_never_add_automatic_cuts(self):
        words = [Word(1, 2, "嗯"), Word(7, 8, "后文")]
        plan = compile_model_plan(words, [], 10, [(2, 7)], [])
        self.assertEqual(plan["cuts"], [])
        self.assertEqual(plan["decisions"], [])
        self.assertEqual(plan["retained_word_audit"]["word_indices"], [0, 1])

    def test_abandoned_start_does_not_extend_into_unselected_pause(self):
        words = [Word(0, 1, "前文"), Word(1, 2, "废弃起句"), Word(5, 6, "接续")]
        proposal = cut(1, 2, kind="abandoned_start", resume_word=2)
        plan = compile_model_plan(words, [proposal], 6, [(2, 5)], [])
        self.assertEqual([(c["start"], c["end"]) for c in plan["cuts"]], [(1, 2)])
        selected = compile_model_plan(words, [proposal], 6, [(2, 5)],
                                      [{"gap_id": "S1", "keep_before_ms": 0, "keep_after_ms": 120}])
        self.assertEqual([(c["start"], c["end"]) for c in selected["cuts"]], [(1, 4.88)])

    def test_model_explicitly_extends_a_cut_to_before_the_next_source_word(self):
        words = [Word(0, 1, "前文"), Word(1, 2, "废弃起句"), Word(5, 6, "接续")]
        proposal = cut(1, 2, end_before_word=2)
        plan = compile_model_plan(words, [proposal], 6, [], [])
        self.assertEqual([(c["start"], c["end"]) for c in plan["cuts"]], [(1, 4.88)])
        self.assertEqual(plan["cuts"][0]["removed_word_indices"], [1])
        self.assertIn(2, plan["retained_word_audit"]["word_indices"])
        custom = compile_model_plan(words, [{**proposal, "keep_before_ms": 300}], 6, [], [])
        self.assertEqual(custom["cuts"][0]["end"], 4.7)

    def test_explicit_extension_must_use_the_immediate_next_word_and_valid_padding(self):
        words = [Word(0, 1, "前文"), Word(1, 2, "删除"), Word(2.1, 3, "接续"), Word(3, 4, "后文")]
        invalid = [cut(1, 2, end_before_word=3), cut(1, 2, end_before_word=True),
                   cut(1, 2, end_before_word=2, keep_before_ms=-1),
                   cut(1, 2, end_before_word=2, keep_before_ms=math.inf),
                   cut(1, 2, end_before_word=2, keep_before_ms=120),
                   cut(3, 4, end_before_word=4)]
        for proposal in invalid:
            with self.subTest(proposal=proposal), self.assertRaises(ValueError):
                compile_model_plan(words, [proposal], 4, [], [])
        plan = compile_model_plan(words, [cut(1, 2, end_before_word=2, keep_before_ms=50)], 4, [], [])
        self.assertEqual(plan["cuts"][0]["end"], 2.05)

    def test_selected_pause_is_clipped_to_confirmed_quiet_and_retained_word_gaps(self):
        words = [Word(0, 1, "前文"), Word(2, 3, "保留的轻声词"), Word(4, 5, "后文")]
        plan = compile_model_plan(words, [], 5, [(.8, 4.2)], [{"gap_id": "S1"}])
        self.assertEqual([(c["start"], c["end"]) for c in plan["cuts"]], [(1.12, 1.88), (3.12, 3.88)])
        self.assertTrue(all(c["retained_word_check"]["verified"] for c in plan["cuts"]))
        self.assertEqual(plan["retained_word_audit"]["word_indices"], [0, 1, 2])
        self.assertEqual(plan["applied_pause_compressions"], [{"gap_id": "S1"}])

    def test_pause_choices_cannot_supply_unconfirmed_timestamps(self):
        words = [Word(0, 1, "前文"), Word(4, 5, "后文")]
        for proposal in ({"gap_id": "S2"}, {"gap_id": "S0"}, {"gap_id": "other"},
                         {"gap_id": "S1", "keep_before_ms": -1},
                         {"gap_id": "S1", "keep_after_ms": math.inf},
                         {"gap_id": "S1", "keep_before_ms": True}):
            with self.subTest(proposal=proposal), self.assertRaises(ValueError):
                compile_model_plan(words, [], 5, [(1, 4)], [proposal])
        plan = compile_model_plan(words, [], 5, [(2, 3)],
                                  [{"gap_id": "S1", "start": 0, "end": 5,
                                    "keep_before_ms": 0, "keep_after_ms": 0}])
        self.assertEqual([(c["start"], c["end"]) for c in plan["cuts"]], [(2, 3)])

    def test_unknown_or_unusable_quiet_intervals_are_execution_errors(self):
        words = [Word(0, 1, "前文"), Word(2, 3, "后文")]
        for silence in ((math.nan, 2), (2, math.inf), (2, 1), (0, .5), (1, 1.1)):
            with self.subTest(silence=silence), self.assertRaises(ValueError):
                compile_model_plan(words, [], 3, [silence], [{"gap_id": "S1"}])

    def test_partial_character_cut_requires_exact_character_times(self):
        words = [Word(0, 1, "前文"), Word(1, 2, "吃吃"), Word(2, 3, "后文")]
        partial = {"word_index": 1, "start_char": 0, "end_char": 1, "reason": "模型删除重复起音"}
        with self.assertRaisesRegex(ValueError, "准确字符时间.*整词.*字幕"):
            validate_executable_cuts([partial], 0, 3, words)
        complete = {**partial, "end_char": 2}
        plan = compile_model_plan(words, [complete], 3, [], [])
        self.assertEqual([(c["start"], c["end"]) for c in plan["cuts"]], [(1, 2)])
        self.assertEqual(plan["retained_word_audit"]["word_indices"], [0, 2])
        self.assertEqual(plan["applied_proposals"], [cut(1, 2, reason=complete["reason"])])
        self.assertEqual(plan["raw_model_decisions"]["cuts"], [complete])

    def test_only_valid_integer_source_indices_and_finite_times_execute(self):
        words = [Word(0, 1, "前文"), Word(1, 2, "删除"), Word(2, 3, "后文")]
        bad = [None, {**cut(1, 2), "start_word": True}, {**cut(1, 2), "end_word": 2.0},
               {**cut(1, 2), "start_word": "1"}, cut(-1, 1), cut(2, 4),
               {**cut(1, 2), "reason": ""}, {**cut(1, 2), "start_char": 0}]
        for proposal in bad:
            with self.subTest(proposal=proposal), self.assertRaises(ValueError):
                validate_executable_cuts([proposal], 0, 3, words)
        for value in (None, {}, "[]"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_executable_cuts(value, 0, 3, words)
        for time in (math.nan, math.inf, -1, True):
            broken = [words[0], Word(time, 2, "删除"), words[2]]
            with self.subTest(time=time), self.assertRaises(ValueError):
                validate_executable_cuts([cut(1, 2)], 0, 3, broken)

    def test_out_of_source_time_and_empty_output_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "范围"):
            compile_model_plan([Word(1, 3, "删除")], [cut(0, 1)], 2, [], [])
        with self.assertRaisesRegex(ValueError, "没有可保留"):
            compile_model_plan([Word(0, 2, "删除")], [cut(0, 1)], 2, [], [])
        for duration in (0, -1, math.nan, math.inf, True):
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                compile_model_plan([], [], duration, [], [])

    def test_word_timing_overlap_cannot_delete_a_model_retained_word(self):
        words = [Word(0, 1.04, "保留前文"), Word(1, 2, "删除"), Word(2, 3, "保留后文")]
        with self.assertRaisesRegex(ValueError, "未选择删除的源词.*0"):
            compile_model_plan(words, [cut(1, 2)], 3, [], [])

    def test_adjacent_cuts_do_not_swallow_a_short_retained_fragment(self):
        words = [Word(0, 1, "前文"), Word(1, 2, "删除"), Word(2, 2.000008, "短保留词"),
                 Word(2.000008, 3, "删除"), Word(3, 4, "后文")]
        plan = compile_model_plan(words, [cut(1, 2), cut(3, 4)], 4, [], [])
        self.assertEqual([(c["start"], c["end"]) for c in plan["cuts"]], [(1, 2), (2.000008, 3)])
        self.assertIn(2, plan["retained_word_audit"]["word_indices"])

    def test_estimated_word_times_are_not_used_for_audio_cuts(self):
        words = [Word(0, 1, "前文"), Word(1, 2, "删除"), Word(2, 3, "后文")]
        with self.assertRaisesRegex(ValueError, "估算"):
            compile_model_plan(words, [cut(1, 2)], 3, [], [], estimated_timing=True)
        plan = compile_model_plan(words, [], 3, [], [], estimated_timing=True)
        self.assertEqual(plan["cuts"], [])

    def test_reference_ten_core_word_spans_and_two_explicit_pauses_execute(self):
        # Exact approved source-word spans. Two extensions are explicit model
        # choices via end_before_word, rather than effects of a kind label.
        source_spans = [(31.28, 32.24), (69.28, 70.28), (92.6, 93.6), (107.37, 109.01),
                        (122.76, 129.76), (141.52, 145.4), (164.78, 166.14),
                        (178.7, 179.18), (262.87, 263.87), (317.74, 318.82)]
        words = [Word(a, b, "待删除原声") for a, b in source_spans]
        words += [Word(47, 48.26, "简单地"), Word(49.5, 49.98, "说"),
                  Word(71.44, 71.92, "所以说"), Word(179.82, 180.54, "高蛋白"),
                  Word(236.67, 237.75, "那么这种顺序"), Word(240.19, 240.75, "就是这样的")]
        words.sort(key=lambda word: word.start)
        selected = [cut(index, index + 1, id=f"C{number + 1:02}")
                    for number, span in enumerate(source_spans)
                    for index, word in enumerate(words) if (word.start, word.end) == span]
        for item in selected:
            if item["id"] == "C02":
                item["end_before_word"] = item["end_word"]
            if item["id"] == "C08":
                item.update({"end_before_word": item["end_word"], "keep_before_ms": 154})
        plan = compile_model_plan(words, selected, 329.495542,
                                  [(48.212229, 49.365875), (237.636062, 240.219812)],
                                  [{"gap_id": "S1"},
                                   {"gap_id": "S2", "keep_before_ms": 200, "keep_after_ms": 200}])
        self.assertEqual(len(plan["applied_proposals"]), 10)
        self.assertEqual(len(plan["applied_pause_compressions"]), 2)
        self.assertEqual(len(plan["cuts"]), 12)
        self.assertEqual(len(plan["decisions"]), 12)
        self.assertTrue(all(d["action"] == "CUT" for d in plan["decisions"]))
        self.assertTrue(all(c["retained_word_check"]["verified"] for c in plan["cuts"]))
        self.assertIn((237.95, 239.99), [(c["start"], c["end"]) for c in plan["cuts"]])
        self.assertIn((69.28, 71.32), [(c["start"], c["end"]) for c in plan["cuts"]])
        self.assertIn((178.7, 179.666), [(c["start"], c["end"]) for c in plan["cuts"]])
        self.assertAlmostEqual(sum(c["end"] - c["start"] for c in plan["cuts"]), 23.832, delta=.001)


if __name__ == "__main__":
    unittest.main()
