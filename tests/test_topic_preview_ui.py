from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PIL import Image
from PySide6.QtWidgets import QApplication

from qt_app import App


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "Media tools missing")
class TopicPreviewUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def wait_frame(self, window, channel):
        deadline = time.monotonic() + 6
        while time.monotonic() < deadline:
            self.application.processEvents()
            frame = window.preview_frame
            if frame is not None and frame.getpixel((100, 100))[channel] > 200:
                return
            time.sleep(.01)
        self.fail(f"Video frame did not reach preview: {window.preview_video_status.text()}")

    def fixture(self, folder, filename, color):
        video = folder / filename
        subprocess.run([shutil.which("ffmpeg"), "-v", "error", "-f", "lavfi", "-i",
                        f"color=c={color}:s=640x360:r=25:d=1", "-c:v", "libx264", str(video)], check=True)
        return video

    def test_import_frame_composites_with_overlays_and_style_settings_persist(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            video = self.fixture(folder, "source.mp4", "red")
            template = folder / "template.png"
            Image.new("RGB", (540, 960), "#305F90").save(template)
            with patch("qt_app.SETTINGS_PATH", folder / "settings.json"), patch.object(App, "detect_media_tools"):
                window = App()
                try:
                    window.tool_paths = shutil.which("ffmpeg"), shutil.which("ffprobe")
                    window.source.setText(str(video))
                    self.wait_frame(window, 0)
                    self.assertIn("选择竖屏模板", window.preview_feedback.text())
                    window.template.setText(str(template))
                    window.slot.setText("0,326,540,304")
                    window.topic_box.setText("20,30,500,180")
                    window.topic.setText("健康饮食注意事项")
                    window.topic_auto_size.setChecked(False)
                    window.topic_size.setValue(40)
                    window.topic_background.setCurrentIndex(window.topic_background.findData("rounded"))
                    window.topic_effect.setCurrentIndex(window.topic_effect.findData("gradient"))
                    window.topic_background_color = "#A03030"
                    window.topic_background_opacity.setValue(76)
                    window.show_preview()
                    self.assertTrue(window.preview_seek.isEnabled())
                    image = window.preview.pixmap().toImage()
                    sx, sy = image.width() / 540, image.height() / 960
                    self.assertGreater(image.pixelColor(round(270 * sx), round(510 * sy)).red(), 200)
                    window.preview_scope.setCurrentIndex(window.preview_scope.findData("video"))
                    zoomed = window.preview.pixmap().toImage()
                    self.assertGreater(zoomed.width(), image.width())
                    self.assertGreater(zoomed.pixelColor(zoomed.width() // 2, zoomed.height() // 2).red(), 200)
                    window.preview_scope.setCurrentIndex(window.preview_scope.findData("full"))
                    before = image.copy()
                    window.nameplate_box.setText("300,390,180,100")
                    window.style.setCurrentIndex(window.style.findData("teal_gradient"))
                    self.assertNotEqual(before, window.preview.pixmap().toImage())
                    window.save_current_settings()
                    window.close()
                    restored = App()
                    try:
                        self.assertFalse(restored.topic_auto_size.isChecked())
                        self.assertEqual(restored.current_topic_style().snapshot(), window.current_topic_style().snapshot())
                        self.assertTrue(restored.topic_accent_button.isEnabled())
                        restored.topic_background.setCurrentIndex(restored.topic_background.findData("none"))
                        self.assertFalse(restored.topic_background_opacity.isEnabled())
                    finally:
                        restored.close()
                finally:
                    window.close()

    def test_batch_selection_replaces_video_frame_without_old_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            red = self.fixture(folder, "医院__医生__甲.mp4", "red")
            blue = self.fixture(folder, "医院__医生__乙.mp4", "blue")
            with patch("qt_app.SETTINGS_PATH", folder / "settings.json"), patch.object(App, "detect_media_tools"):
                window = App()
                try:
                    window.tool_paths = shutil.which("ffmpeg"), shutil.which("ffprobe")
                    window.mode.setCurrentIndex(window.mode.findData("folder"))
                    window.source.setText(str(folder))
                    window.refresh_files()
                    window.table.selectRow(next(i for i, item in enumerate(window.items) if item.path == red))
                    self.wait_frame(window, 0)
                    window.table.selectRow(next(i for i, item in enumerate(window.items) if item.path == blue))
                    self.assertIsNone(window.preview_frame)
                    self.wait_frame(window, 2)
                    self.assertEqual(Path(window._preview_source_key[0]), blue.resolve())
                    window.source.clear()
                    self.assertIsNone(window.preview_frame)
                    self.assertFalse(window.preview_seek.isEnabled())
                finally:
                    window.close()


if __name__ == "__main__":
    unittest.main()
