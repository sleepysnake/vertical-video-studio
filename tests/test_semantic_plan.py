from __future__ import annotations

from dataclasses import replace
import unittest

from studio.captions import Word
from studio.edit_plan import PLAN_VERSION, build_edit_plan
from studio.edits import build_cut_plan, keep_intervals
from studio.model_edits import validate_model_cuts
from studio.strategy import Strategy


def restart(first=1, last=3, replacement_first=3, replacement_last=5,
            kind="repair_restart"):
    return {"kind": kind, "start_word": first, "end_word": last,
            "replacement_range": {"start_word": replacement_first,
                                  "end_word": replacement_last},
            "reason": "前次表达废弃，后续完整重说覆盖全部信息",
            "joined_text": "前文后续完整表达"}


class SemanticPlanTests(unittest.TestCase):
    def setUp(self):
        self.words = [Word(0, 2, "前文"), Word(2, 5, "废弃首次表达"),
                      Word(5, 9, "以及未完成后半句"), Word(9, 11, "完整重说前半句"),
                      Word(11, 13, "完整重说后半句"), Word(13, 30, "继续解释")]

    def test_seven_second_contextual_restart_and_nonidentical_take_are_applied(self):
        for kind in ("repair_restart", "repeated_take"):
            with self.subTest(kind=kind):
                proposal = restart(kind=kind)
                plan = build_edit_plan(self.words, [proposal], 30, [])
                self.assertEqual(plan["applied_proposals"], [proposal])
                self.assertEqual(plan["decisions"][0]["action"], "CUT")
                self.assertEqual((plan["cuts"][0]["start"], plan["cuts"][0]["end"]), (2, 9))
                self.assertEqual(plan["cuts"][0]["removed_word_indices"], [1, 2])
                self.assertEqual(plan["retained_word_audit"]["word_indices"], [0, 3, 4, 5])
                self.assertEqual(plan["version"], PLAN_VERSION)
                self.assertEqual(plan["mode"], "contextual_cleanup")

    def test_reason_is_explanation_not_a_keyword_classifier(self):
        proposal = {**restart(), "reason": "ASR识别出的词时间用于定位，前次说到一半被后续完整重说替代"}
        self.assertEqual(validate_model_cuts([proposal], 0, 6, self.words), [proposal])
        plan = build_edit_plan(self.words, [proposal], 30, [])
        self.assertEqual(plan["decisions"][0]["action"], "CUT")
        with self.assertRaisesRegex(ValueError, "ASR"):
            validate_model_cuts([{**proposal, "kind": "asr_correction"}], 0, 6, self.words)

    def test_abandoned_words_can_be_cut_without_audio_silence_evidence(self):
        words = [Word(0, 1, "前文"), Word(1, 2, "起句说到一半"), Word(2, 10, "独立完整下句")]
        for legacy in (False, True):
            proposal = {"kind": "abandoned_start", "start_word": 1, "end_word": 2,
                        "reason": "上下文确认起句废弃且无独有信息"}
            if legacy:
                proposal["resume_word"] = 2
            plan = build_edit_plan(words, [proposal], 10, [])
            self.assertEqual(plan["applied_proposals"], [proposal])
            self.assertEqual(plan["cuts"][0]["end"], 2)
            self.assertIsNone(plan["decisions"][0]["pause_evidence"])

    def test_abandoned_pause_extension_needs_actual_quiet_and_keeps_resume_word(self):
        words = [Word(0, 1, "前文"), Word(1, 2, "废弃起句"), Word(3.2, 10, "完整下句")]
        proposal = {"kind": "abandoned_start", "start_word": 1, "end_word": 2,
                    "reason": "废弃起句后停顿再开始完整下句"}
        for quiet, end in (([], 2), ([(2.1, 2.7)], 2), ([(2.5, 3.2)], 3.08)):
            with self.subTest(quiet=quiet):
                plan = build_edit_plan(words, [proposal], 10, quiet)
                self.assertEqual(plan["cuts"][0]["end"], end)
                self.assertIn(2, plan["retained_word_audit"]["word_indices"])

    def test_declared_isolated_filler_does_not_require_fixed_chinese_word_list(self):
        words = [Word(0, 1, "前文"), Word(1, 2.2, "这什么来着"), Word(2.2, 10, "完整下句")]
        proposal = {"kind": "isolated_filler", "start_word": 1, "end_word": 2,
                    "reason": "上下文确认无信息的口头干扰"}
        self.assertEqual(build_edit_plan(words, [proposal], 10, [])["applied_proposals"], [proposal])
        words[1] = Word(1, 1.3, "这什么来着")
        plan = build_edit_plan(words, [proposal], 10, [])
        self.assertEqual(plan["cuts"], [])
        self.assertIn("只修整字幕", plan["decisions"][0]["policy_reason"])

    def test_short_word_internal_edits_still_preserve_audio(self):
        words = [Word(0, 1, "前文"), Word(1, 1.6, "吃吃"), Word(1.6, 10, "后文")]
        proposal = {"word_index": 1, "start_char": 0, "end_char": 1, "reason": "短促结巴"}
        self.assertEqual(validate_model_cuts([proposal], 0, 3, words), [proposal])
        plan = build_edit_plan(words, [proposal], 10, [])
        self.assertEqual(plan["cuts"], [])
        self.assertIn("词内", plan["decisions"][0]["policy_reason"])

    def test_timing_limits_are_strategy_controls_not_schema_hardcoded_caps(self):
        proposal = restart()
        self.assertEqual(validate_model_cuts([proposal], 0, 6, self.words), [proposal])
        for strategy, message in ((replace(Strategy(), max_single_model_cut=6), "上限"),
                                  (replace(Strategy(), max_removal_ratio=.2), "比例")):
            plan = build_edit_plan(self.words, [proposal], 30, [], strategy=strategy)
            self.assertEqual(plan["cuts"], [])
            self.assertIn(message, plan["decisions"][0]["policy_reason"])
        plan = build_edit_plan(self.words, [proposal], 30, [], estimated_timing=True)
        self.assertEqual(plan["cuts"], [])
        self.assertIn("估算", plan["decisions"][0]["policy_reason"])

    def test_replacement_must_exist_follow_cut_and_remain_kept(self):
        bad = [{**restart(), "replacement_range": None},
               restart(replacement_first=2), restart(replacement_last=7)]
        for proposal in bad:
            with self.subTest(proposal=proposal), self.assertRaises(ValueError):
                validate_model_cuts([proposal], 0, 6, self.words)
        second = {"kind": "isolated_filler", "start_word": 4, "end_word": 5,
                  "reason": "另一独立模型结果"}
        with self.assertRaisesRegex(ValueError, "完整保留"):
            validate_model_cuts([restart(), second], 0, 6, self.words)
        for proposals in ([restart(), second], [second, restart()]):
            plan = build_edit_plan(self.words, proposals, 30, [])
            self.assertEqual(plan["applied_proposals"], [proposals[0]])
            self.assertTrue(any("最终剪辑" in item["policy_reason"] for item in plan["decisions"]))

    def test_replacement_can_reference_following_context_outside_target_group(self):
        self.assertEqual(validate_model_cuts([restart()], 1, 3, self.words), [restart()])

    def test_strict_indices_overlaps_and_mixed_schemas_are_rejected(self):
        cases = [{**restart(), "start_word": True}, {**restart(), "start_word": "1"},
                 {**restart(), "end_word": 3.0}, {**restart(), "start_word": -1},
                 {**restart(), "word_index": 1, "start_char": 0, "end_char": 1},
                 {**restart(), "kind": {"type": "repair_restart"}},
                 {**restart(), "end_ms": 9000}]
        for item in cases:
            with self.subTest(item=item), self.assertRaises(ValueError):
                validate_model_cuts([item], 0, 6, self.words)
        with self.assertRaisesRegex(ValueError, "重叠"):
            validate_model_cuts([restart(), restart(first=2)], 0, 6, self.words)

    def test_asr_timing_overlap_cannot_remove_any_retained_word(self):
        words = [Word(0, 1.04, "保留前文"), Word(1, 2, "废弃起句"), Word(2, 10, "完整下句")]
        proposal = {"kind": "abandoned_start", "start_word": 1, "end_word": 2,
                    "reason": "模型确认废弃"}
        with self.assertRaisesRegex(ValueError, "侵入"):
            build_cut_plan(words, [proposal], 10)
        plan = build_edit_plan(words, [proposal], 10, [])
        self.assertEqual(plan["cuts"], [])
        self.assertIn("侵入", plan["decisions"][0]["policy_reason"])

    def test_inward_serialization_and_short_retained_fragment_never_drop_word(self):
        words = [Word(0, 1.0004, "前文"), Word(1.0004, 2.0004, "废弃"),
                 Word(2.0004, 2.0084, "短词"), Word(2.0084, 3.0084, "另一口癖"),
                 Word(3.0084, 10, "后文")]
        proposals = [{"start_word": 1, "end_word": 2, "reason": "第一处"},
                     {"start_word": 3, "end_word": 4, "reason": "第二处"}]
        cuts = build_cut_plan(words, proposals, 10)
        self.assertEqual(len(cuts), 2)
        self.assertTrue(all(item["retained_word_check"]["verified"] for item in cuts))
        kept = keep_intervals(cuts, 10)
        self.assertTrue(any(start <= words[2].start and end >= words[2].end for start, end in kept))


if __name__ == "__main__":
    unittest.main()
