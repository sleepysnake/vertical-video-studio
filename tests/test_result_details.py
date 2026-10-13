"""Result details must keep historical plans separate from the latest failure."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication, QLineEdit, QPlainTextEdit

from qt_app import App
from studio.models import Person, VideoItem
from studio.pipeline import work_dir


class ResultDetailsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.item = VideoItem(self.folder / "source.mp4", Person("医院", "医生", "姓名"))
        self.plan_path = work_dir(self.item, self.folder) / "edit-plan.json"
        self.plan_path.parent.mkdir(parents=True)
        self.window = SimpleNamespace(
            selected_item=Mock(return_value=self.item), sync_video_preview=Mock(),
            show_preview=Mock(), results={}, output_dir=QLineEdit(str(self.folder)),
            detail=QPlainTextEdit())

    def write_plan(self, run_id=None):
        plan = {"decisions": [{"start": 49.5, "end": 49.7, "source_text": "说",
                              "need_review": True, "policy_reason": "保留原声"}]}
        if run_id is not None:
            plan["run_id"] = run_id
        self.plan_path.write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")

    def show_result(self, run):
        self.window.results[str(self.item.path)] = (
            "失败", "字幕未按原顺序完整覆盖实际保留的源词", run)
        App.show_selected(self.window)
        return self.window.detail.toPlainText()

    def test_failed_new_attempt_does_not_append_old_plan_reviews(self):
        self.write_plan("previous-attempt")
        details = self.show_result({"run_id": "latest-attempt", "plan_ready": False,
                                    "plan_path": str(self.plan_path)})
        self.assertIn("字幕未按原顺序完整覆盖", details)
        self.assertIn("本次任务尚未生成", details)
        self.assertNotIn("建议复核", details)
        self.assertNotIn("49.5", details)
        self.assertNotIn("本次剪辑计划：", details)

    def test_matching_current_attempt_displays_retained_reviews_and_plan(self):
        self.write_plan("latest-attempt")
        details = self.show_result({"run_id": "latest-attempt", "plan_ready": True,
                                    "plan_path": str(self.plan_path)})
        self.assertIn("建议复核 1 处", details)
        self.assertIn("不是任务失败原因", details)
        self.assertIn("49.5–49.7 秒  说", details)
        self.assertIn(f"本次剪辑计划：{self.plan_path}", details)

    def test_ready_flag_does_not_override_mismatched_attempt_identity(self):
        self.write_plan("previous-attempt")
        details = self.show_result({"run_id": "latest-attempt", "plan_ready": True})
        self.assertIn("本次任务尚未生成", details)
        self.assertNotIn("建议复核", details)

    def test_legacy_earlier_plan_is_labelled_historical_without_reviews(self):
        self.write_plan()
        stamp = datetime(2026, 10, 3, tzinfo=timezone.utc).timestamp()
        os.utime(self.plan_path, (stamp, stamp))
        details = self.show_result({"started_at": "2026-10-04T00:00:00+00:00",
                                    "ended_at": "2026-10-04T00:01:00+00:00"})
        self.assertIn("历史剪辑计划（非本次任务）", details)
        self.assertNotIn("建议复核", details)
        self.assertNotIn("49.5", details)

    def test_legacy_plan_written_during_attempt_still_displays_reviews(self):
        self.write_plan()
        stamp = datetime(2026, 10, 4, 0, 0, 30, tzinfo=timezone.utc).timestamp()
        os.utime(self.plan_path, (stamp, stamp))
        details = self.show_result({"started_at": "2026-10-04T00:00:00+00:00",
                                    "ended_at": "2026-10-04T00:01:00+00:00"})
        self.assertIn("建议复核 1 处", details)
        self.assertIn("本次剪辑计划：", details)

    def test_current_plan_exposes_adopted_counts_and_local_rejections(self):
        self.write_plan("latest-attempt")
        plan = json.loads(self.plan_path.read_text(encoding="utf-8"))
        plan["processing_summary"] = {"model_proposal_count": 14, "adopted_model_proposal_count": 10,
            "rejected_model_proposal_count": 4, "removed_seconds": 12.5, "caption_count": 151,
            "caption_literal_fallback_count": 1, "caption_formatting_repair_count": 2}
        plan["rejected_model_proposals"] = [{"proposal_id": "C004", "reason": "保留范围与删除范围冲突"}]
        self.plan_path.write_text(json.dumps(plan), encoding="utf-8")
        details = self.show_result({"run_id": "latest-attempt", "plan_ready": True})
        self.assertIn("模型建议 14 处，采用 10 处", details)
        self.assertIn("原文回退 1 条", details)
        self.assertIn("C004", details)
        self.assertIn("相关原声已保留", details)


if __name__ == "__main__":
    unittest.main()
