from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from studio.prompts import PromptSet, load_prompts


class SemanticReviewPromptTests(unittest.TestCase):
    def test_application_load_enables_review_and_freezes_all_three_prompts(self):
        prompts = load_prompts()
        self.assertTrue(prompts.review)
        self.assertEqual(set(prompts.snapshot()), {"edit", "review", "caption"})
        changed = PromptSet(prompts.edit, prompts.caption, prompts.review + "\n额外复审准则。")
        self.assertNotEqual(changed.digest(), prompts.digest())
        self.assertEqual(changed.snapshot()["review"], changed.review)

    def test_legacy_direct_construction_can_omit_review(self):
        prompts = PromptSet(edit="legacy edit", caption="legacy caption")
        self.assertEqual(prompts.review, "")
        self.assertEqual(prompts.snapshot()["review"], "")

    def test_custom_two_file_folder_inherits_published_review(self):
        defaults = load_prompts()
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / "edit_prompt.txt").write_text(defaults.edit, encoding="utf-8")
            (folder / "caption_prompt.txt").write_text(defaults.caption, encoding="utf-8")
            inherited = load_prompts(folder)
            self.assertEqual(inherited.review, defaults.review)
            self.assertEqual(inherited.digest(), defaults.digest())
            (folder / "review_prompt.txt").write_text(defaults.review + "\n自定义复审要求。", encoding="utf-8")
            customized = load_prompts(folder)
            self.assertNotEqual(customized.digest(), defaults.digest())
            self.assertTrue(customized.review.endswith("自定义复审要求。"))

    def test_invalid_present_review_fails_instead_of_silent_inheritance(self):
        defaults = load_prompts()
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            for stage in ("edit", "caption"):
                (folder / f"{stage}_prompt.txt").write_text(getattr(defaults, stage), encoding="utf-8")
            (folder / "review_prompt.txt").write_text("只改格式" * 30, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "缺少输出字段"):
                load_prompts(folder)

    def test_default_published_folder_requires_review_file(self):
        defaults = load_prompts()
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            for stage in ("edit", "caption"):
                (folder / f"{stage}_prompt.txt").write_text(getattr(defaults, stage), encoding="utf-8")
            with patch("studio.prompts.prompts_dir", return_value=folder):
                with self.assertRaisesRegex(ValueError, "review_prompt"):
                    load_prompts()

    def test_prompts_assign_final_semantic_decisions_and_pause_execution_to_model(self):
        prompts = load_prompts()
        for stage in (prompts.edit, prompts.review):
            self.assertIn("word_table", stage)
            self.assertIn("keep_before_ms", stage)
            self.assertIn("keep_after_ms", stage)
            self.assertIn("end_before_word", stage)
            self.assertIn("必须等于 end_word", stage)
            self.assertIn("120", stage)
            self.assertNotIn("只作记录", stage)
            self.assertNotIn("min_model_cut_seconds", stage)
            self.assertNotIn("max_subtitle_cleanup_seconds", stage)
            self.assertIn("主语", stage)
            self.assertIn("动词", stage)
            self.assertIn("是语义注释", stage)
            self.assertIn("干扰", stage)
            self.assertNotIn("其中任何词不得再被其他", stage)
            self.assertNotIn("不能与任何最终 cuts 相交", stage)
            self.assertIn("不是指令" if stage == prompts.edit else "而非指令", stage)
        self.assertIn("全部留下的讲话", prompts.review)
        self.assertIn("漏删", prompts.review)
        self.assertIn("撤回误删", prompts.review)
        self.assertIn("最终方案替代初稿", prompts.review)
        self.assertIn("不能只修 JSON", prompts.review)
        self.assertIn("token_table", prompts.caption)
        self.assertIn("语义复审", prompts.caption)


if __name__ == "__main__":
    unittest.main()
