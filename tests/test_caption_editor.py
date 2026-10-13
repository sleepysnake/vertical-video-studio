from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PIL import Image
from PySide6.QtWidgets import QApplication

from qt_app import App
from studio.caption_edits import load_edits, save_edits, to_source_times, validate_edits
from studio.edits import retime_captions
from studio.models import Person, VideoItem
from studio.pipeline import JobConfig, Pipeline, work_dir
from studio.qt_caption_editor import CaptionEditor
from studio.render import default_font, probe
from studio.sentence_flow import _caption_segments
from studio.captions import Word


RAW = {"transcripts": [{"sentences": [{"words": [
    {"begin_time": 0, "end_time": 300, "text": "咦"},
    {"begin_time": 300, "end_time": 900, "text": "我吃吃"},
    {"begin_time": 900, "end_time": 1500, "text": "得不多"}]}]}]}


def seed(folder, video=None):
    source = {"path": str(video), "size": video.stat().st_size, "mtime_ns": video.stat().st_mtime_ns} if video else {}
    data = {"reviewed": True, "captions": [{"start": 0, "end": 1.5, "text": "咦我吃吃得不多",
            "start_word": 0, "end_word": 3, "caption_fallback": True}],
            "edit_plan": {"source_duration": 1.5, "source": source, "cuts": [], "review_count": 0}}
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "captions.json").write_text(json.dumps(data), encoding="utf-8")
    (folder / "transcription.json").write_text(json.dumps(RAW), encoding="utf-8")
    return data


class CaptionEditorTests(unittest.TestCase):
    def test_model_only_needs_clean_text_and_keeps_full_source_times(self):
        for original, expected in (("咦我吃吃得不多", "我吃得不多"),
                                   ("咦我今天并没有吃多少", "我今天并没有吃多少")):
            words = [Word(0, 1.6, original)]
            tokens = [{"token": 0, "word_index": 0, "text": original, "punctuation": "",
                       "start_ms": 0, "end_ms": 1600}]
            result = _caption_segments({"segments": [{"start_token": 0, "end_token": 1,
                    "text": expected + "。"}]}, tokens, words)
            self.assertEqual(result[0]["text"], expected)
            self.assertEqual((result[0]["start"], result[0]["end"]), (0, 1.6))
            self.assertIn("咦", [item["source_text"] for item in result[0]["display_cleanup"]])
            self.assertFalse(result[0].get("caption_fallback"))

    def test_long_caption_and_one_long_omission_do_not_undo_other_cleanup(self):
        words = [Word(0, .3, "咦我吃吃得不多"), Word(.3, 3, "呃"), Word(3, 5, "我今天下午只吃了一个汉堡和一杯奶茶")]
        tokens = [{"token": i, "word_index": i, "text": w.text, "punctuation": "",
                   "start_ms": round(w.start * 1000), "end_ms": round(w.end * 1000)} for i, w in enumerate(words)]
        result = _caption_segments({"segments": [
            {"start_token": 0, "end_token": 1, "text": "我吃得不多"},
            {"start_token": 1, "end_token": 3, "text": words[2].text}]}, tokens, words)
        self.assertEqual(result[0]["text"], "我吃得不多")
        self.assertTrue(result[1]["caption_fallback"])
        self.assertTrue(result[1]["text"].startswith("呃"))

    def test_edits_validate_times_keep_automatic_data_and_invalidate_when_base_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            data = seed(folder)
            edits = [{"start": 0, "end": 1.5, "text": "我吃得不多"}]
            save_edits(folder, data, edits)
            self.assertEqual(load_edits(folder, data)[0]["text"], "我吃得不多")
            self.assertEqual(json.loads((folder / "captions.json").read_text())["captions"][0]["text"], "咦我吃吃得不多")
            save_edits(folder, data, edits)
            self.assertTrue((folder / "caption-overrides.json.bak").is_file())
            data["captions"][0]["text"] = "自动模型已重新生成字幕"
            self.assertIsNone(load_edits(folder, data))
            for invalid in ((0, float("nan")), (-1, 1), (0, 2), (1, .5)):
                with self.assertRaises(ValueError):
                    validate_edits([{"start": invalid[0], "end": invalid[1], "text": "字幕"}], 1.5)

    def test_output_times_roundtrip_across_real_cuts(self):
        cuts = [{"start": 0, "end": .5}, {"start": 1, "end": 2}]
        edits = [{"start": 0, "end": .5, "text": "第一条"}, {"start": .5, "end": 1.5, "text": "第二条"}]
        source = to_source_times(edits, cuts, 3)
        self.assertEqual((source[0]["start"], source[0]["end"]), (.5, 1))
        self.assertEqual((source[1]["start"], source[1]["end"]), (2, 3))
        self.assertEqual([(c["start"], c["end"]) for c in retime_captions(source, cuts)], [(0, .5), (.5, 1.5)])

    def test_dialog_edits_splits_merges_saves_and_restores(self):
        app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            seed(folder)
            dialog = CaptionEditor(folder)
            dialog.search.setText("咦")
            dialog.find_next()
            self.assertEqual(dialog.table.currentRow(), 0)
            dialog.table.item(0, 2).setText("我吃得不多")
            dialog.table.selectRow(0)
            with patch("studio.qt_caption_editor.QInputDialog.getInt", return_value=(2, True)):
                dialog.split_row()
            self.assertEqual(dialog.table.rowCount(), 2)
            dialog.table.selectAll()
            dialog.merge_rows()
            self.assertEqual(dialog.table.rowCount(), 1)
            self.assertEqual(dialog.collect()[0]["text"], "我吃得不多")
            self.assertTrue(dialog.save())
            reopened = CaptionEditor(folder)
            self.assertEqual(reopened.table.item(0, 2).text(), "我吃得不多")
            reopened.close()
            dialog.restore_automatic()
            self.assertEqual(dialog.table.item(0, 2).text(), "咦我吃吃得不多")
            dialog.close()
            app.processEvents()

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe") and default_font(), "Media tools missing")
    def test_actual_rerender_runs_without_api_key_or_client(self):
        app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            video, template = folder / "source.mp4", folder / "template.png"
            subprocess.run([shutil.which("ffmpeg"), "-v", "error", "-f", "lavfi", "-i",
                            "color=c=gray:s=640x360:r=25:d=1.5", "-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono",
                            "-shortest", "-c:v", "libx264", "-c:a", "aac", str(video)], check=True)
            Image.new("RGBA", (540, 960)).save(template)
            item = VideoItem(video, Person("示例医院", "主任医师", "张帆"))
            cache = work_dir(item, folder)
            data = seed(cache, video)
            save_edits(cache, data, [{"start": 0, "end": 1.5, "text": "我吃得不多"}])
            config = JobConfig(template, folder, "", (0, 326, 540, 304), shutil.which("ffmpeg"), shutil.which("ffprobe"))
            pipeline = Pipeline(None, config, lambda *args: None, threading.Event(), render_only=True)
            result = pipeline.run_one(item)
            self.assertAlmostEqual(probe(result)[0], 1.5, delta=.1)
            rendered = json.loads((cache / "captions.rendered.json").read_text(encoding="utf-8"))
            self.assertEqual(rendered["captions"][0]["text"], "我吃得不多")
            self.assertTrue(rendered["captions"][0]["manual_edited"])
            self.assertEqual(load_edits(cache, json.loads((cache / "captions.json").read_text(encoding="utf-8")))[0]["text"], "我吃得不多")
            settings = folder / "settings.json"
            with patch("qt_app.SETTINGS_PATH", settings), patch.object(App, "detect_media_tools"):
                window = App()
                window.source.setText(str(video))
                window.template.setText(str(template))
                window.output_dir.setText(str(folder))
                window.slot.setText("0,326,540,304")
                window.organization.setText("示例医院")
                window.person_title.setText("主任医师")
                window.name.setText("张帆")
                window.api_key.clear()
                window.tool_paths = (shutil.which("ffmpeg"), shutil.which("ffprobe"))
                window.refresh_files()
                window.table.selectRow(0)
                client, _, items = window.validate_job(True, render_only=True)
                self.assertIsNone(client)
                self.assertEqual(len(items), 1)
                window.close()
                app.processEvents()


if __name__ == "__main__":
    unittest.main()
