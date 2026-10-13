"""Partition model errors without discarding independent proposed edits."""
from __future__ import annotations

import unittest

from studio.captions import Word
from studio.model_edits import partition_model_cuts, validate_model_cuts


def whole(first: int, last: int, identifier: str | None = None) -> dict:
    return {"start_word": first, "end_word": last, "reason": "上下文判断的裁剪",
            **({"id": identifier} if identifier is not None else {})}


def restart(first: int, last: int, replacement_first: int, replacement_last: int,
            identifier: str) -> dict:
    return {**whole(first, last, identifier), "kind": "repair_restart",
            "replacement_range": {"start_word": replacement_first, "end_word": replacement_last}}


class CutPartitionTests(unittest.TestCase):
    def setUp(self):
        self.words = [Word(i, i + 1, "吃吃喝喝") for i in range(12)]

    def test_outer_list_and_group_contract_remain_strict(self):
        for value in (None, {}, "[]"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "数组"):
                partition_model_cuts(value, 0, len(self.words), self.words)
        for first, last in ((-1, 12), (0, 13), (True, 12), (3, 2)):
            with self.subTest(first=first, last=last), self.assertRaisesRegex(ValueError, "句组"):
                partition_model_cuts([], first, last, self.words)

    def test_bad_index_is_isolated_and_unrelated_valid_cuts_survive(self):
        proposals = [whole(1, 2, "left"), whole(12, 13), whole(8, 9, "right")]
        safe, issues = partition_model_cuts(proposals, 0, 12, self.words)
        self.assertEqual(safe, [proposals[0], proposals[2]])
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0]["proposal_position"], 2)
        self.assertIsNone(issues[0]["proposal_id"])
        self.assertEqual(issues[0]["proposal"], proposals[1])
        self.assertIn("词索引", issues[0]["reason"])
        self.assertEqual(issues[0]["conflicts"], [])

    def test_single_candidate_schema_is_not_relaxed(self):
        bad_values = [None, {**whole(2, 3), "start_word": True},
                      {**whole(2, 3), "end_ms": 2000},
                      {**whole(2, 3), "reason": ""},
                      {**whole(2, 3), "kind": "repair_restart"}]
        for bad in bad_values:
            with self.subTest(bad=bad):
                safe, issues = partition_model_cuts([bad, whole(9, 10)], 0, 12, self.words)
                self.assertEqual(safe, [whole(9, 10)])
                self.assertEqual([item["proposal_position"] for item in issues], [1])

    def test_adjacent_character_cuts_in_same_word_do_not_conflict(self):
        proposals = [{"word_index": 2, "start_char": 0, "end_char": 1, "reason": "第一次起音"},
                     {"word_index": 2, "start_char": 1, "end_char": 2, "reason": "第二次起音"}]
        safe, issues = partition_model_cuts(proposals, 0, 12, self.words)
        self.assertEqual(safe, proposals)
        self.assertEqual(issues, [])

    def test_overlapping_character_cuts_isolate_both_parties_with_exact_chars(self):
        proposals = [{"word_index": 2, "start_char": 0, "end_char": 2, "reason": "第一处"},
                     {"word_index": 2, "start_char": 1, "end_char": 3, "reason": "第二处"},
                     whole(9, 10)]
        safe, issues = partition_model_cuts(proposals, 0, 12, self.words)
        self.assertEqual(safe, proposals[2:])
        self.assertEqual([item["proposal_position"] for item in issues], [1, 2])
        for issue in issues:
            conflict = issue["conflicts"][0]
            self.assertEqual(conflict["kind"], "deletion_overlap")
            self.assertEqual(conflict["word_indices"], [2])
            self.assertEqual(conflict["char_ranges"], [{"word_index": 2, "start_char": 1, "end_char": 2}])

    def test_whole_word_and_partial_word_overlap_isolate_both(self):
        proposals = [whole(2, 3),
                     {"word_index": 2, "start_char": 1, "end_char": 2, "reason": "局部起音"},
                     whole(9, 10)]
        safe, issues = partition_model_cuts(proposals, 0, 12, self.words)
        self.assertEqual(safe, proposals[2:])
        self.assertEqual([item["proposal_position"] for item in issues], [1, 2])
        self.assertEqual(issues[0]["conflicts"][0]["char_ranges"],
                         [{"word_index": 2, "start_char": 1, "end_char": 2}])

    def test_replacement_protection_also_conflicts_with_partial_word_cut(self):
        proposals = [restart(1, 2, 2, 4, "restart"),
                     {"id": "partial", "word_index": 3, "start_char": 2, "end_char": 3,
                      "reason": "局部起音"}, whole(9, 10, "unrelated")]
        safe, issues = partition_model_cuts(proposals, 0, 12, self.words)
        self.assertEqual(safe, proposals[2:])
        self.assertEqual({item["proposal_id"] for item in issues}, {"restart", "partial"})
        for issue in issues:
            conflict = issue["conflicts"][0]
            self.assertEqual(conflict["protection"], "replacement")
            self.assertEqual(conflict["word_indices"], [3])
            self.assertEqual(conflict["protected_by_proposal_position"], 1)
            self.assertEqual(conflict["deleted_by_proposal_position"], 2)

    def test_resume_conflict_isolates_both_and_preserves_independent_proposal(self):
        abandoned = {**whole(1, 2, "abandoned"), "kind": "abandoned_start"}
        proposals = [abandoned, whole(2, 3, "resume-cut"), whole(9, 10, "independent")]
        safe, issues = partition_model_cuts(proposals, 0, 12, self.words)
        self.assertEqual(safe, proposals[2:])
        self.assertEqual([item["proposal_position"] for item in issues], [1, 2])
        self.assertEqual(issues[0]["conflicts"][0]["protection"], "resume")
        self.assertEqual(issues[0]["word_indices"], [2])

    def test_chained_overlap_isolates_every_party_without_an_order_winner(self):
        proposals = [whole(1, 3, "a"), whole(2, 4, "b"), whole(3, 5, "c"),
                     whole(9, 10, "independent")]
        safe, issues = partition_model_cuts(proposals, 0, 12, self.words)
        self.assertEqual(safe, proposals[3:])
        self.assertEqual({item["proposal_id"] for item in issues}, {"a", "b", "c"})
        self.assertEqual(len(issues[1]["conflicts"]), 2)
        reverse_safe, reverse_issues = partition_model_cuts(list(reversed(proposals)), 0, 12, self.words)
        self.assertEqual({item["id"] for item in safe}, {item["id"] for item in reverse_safe})
        self.assertEqual({item["proposal_id"] for item in issues},
                         {item["proposal_id"] for item in reverse_issues})

    def test_repeated_ids_do_not_merge_distinct_proposal_positions(self):
        proposals = [whole(1, 3, "same-id"), whole(2, 4, "same-id"), whole(9, 10, "same-id")]
        safe, issues = partition_model_cuts(proposals, 0, 12, self.words)
        self.assertEqual(safe, proposals[2:])
        self.assertEqual([item["proposal_position"] for item in issues], [1, 2])

    def test_cached_fourteen_proposal_ranges_keep_ten_and_isolate_four(self):
        # Exact source/replacement ranges from the 2026-10-04 API cache.
        ranges = [(122, 124, 124, 154), (197, 198, 198, 227), (269, 273, 273, 291),
                  (317, 322, 322, 336), (326, 328, 328, 336), (357, 359, 359, 363),
                  (432, 433, 433, 439), (437, 439, 439, 448), (553, 555, 555, 558),
                  (563, 564, 564, 576), (642, 646, 646, 650), (686, 687, 687, 691),
                  (738, 740, 740, 745), (762, 763, 763, 771)]
        words = [Word(i, i + 1, "词") for i in range(792)]
        proposals = [restart(*bounds, f"C{position:03}") for position, bounds in enumerate(ranges, 1)]
        safe, issues = partition_model_cuts(proposals, 0, 792, words)
        self.assertEqual(len(safe), 10)
        self.assertEqual([item["proposal_position"] for item in issues], [4, 5, 7, 8])
        self.assertEqual({item["proposal_id"] for item in issues}, {"C004", "C005", "C007", "C008"})
        self.assertEqual(issues[0]["word_indices"], [326, 327])
        self.assertEqual(issues[2]["word_indices"], [437, 438])
        self.assertEqual(validate_model_cuts(safe, 0, 792, words), safe)
        reverse_safe, reverse_issues = partition_model_cuts(list(reversed(proposals)), 0, 792, words)
        self.assertEqual({item["id"] for item in safe}, {item["id"] for item in reverse_safe})
        self.assertEqual({item["proposal_id"] for item in issues},
                         {item["proposal_id"] for item in reverse_issues})


if __name__ == "__main__":
    unittest.main()
