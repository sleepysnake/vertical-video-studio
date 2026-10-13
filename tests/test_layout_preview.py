from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PIL import Image
from PySide6.QtWidgets import QApplication

from qt_app import App
from studio.render import NAMEPLATE_STYLES, default_font, graphics, render_video


@unittest.skipUnless(default_font(), "Chinese font missing")
class LayoutPreviewTests(unittest.TestCase):
    def test_topic_region_updates_preview_before_video_region_and_settings_roundtrip(self):
        app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            template = folder / "template.png"
            Image.new("RGB", (540, 960), "#305F90").save(template)
            settings = folder / "settings.json"
            with patch("qt_app.SETTINGS_PATH", settings), patch.object(App, "detect_media_tools"):
                window = App()
                window.template.setText(str(template))
                original = window.preview.pixmap().toImage()
                window.topic_box.setText("20,30,500,150")
                selected = window.preview.pixmap().toImage()
                self.assertNotEqual(original, selected)
                window.topic.setText("健康饮食注意事项")
                self.assertNotEqual(selected, window.preview.pixmap().toImage())
                # An opaque template with no video slot still shows topic text.
                self.assertNotIn("文字预览", window.preview_feedback.text())
                window.slot.setText("0,326,540,304")
                window.style.setCurrentIndex(window.style.findData("teal_gradient"))
                normal = window.preview.pixmap().toImage()
                window.nameplate_size.setValue(65)
                self.assertNotEqual(normal, window.preview.pixmap().toImage())
                window.nameplate_box.setText("300,390,180,100")
                self.assertFalse(window.nameplate_size.isEnabled())
                window.save_current_settings()
                window.close()
                restored = App()
                self.assertEqual(restored.nameplate_box.text(), "300,390,180,100")
                self.assertEqual(restored.nameplate_size.value(), 65)
                self.assertFalse(restored.nameplate_size.isEnabled())
                restored.nameplate_box.clear()
                self.assertTrue(restored.nameplate_size.isEnabled())
                restored.topic_box.setText("500,30,500,150")
                self.assertIn("主题区域无效", restored.preview_feedback.text())
                restored.close()
                app.processEvents()

    def test_whole_nameplate_scales_and_fits_custom_region_for_every_style(self):
        for _, style in NAMEPLATE_STYLES:
            with self.subTest(style=style):
                def overlay(**kwargs):
                    return graphics((540, 960), (0, 326, 540, 304), "张帆", "示例医院", "主任医师",
                                    "", default_font(), None, style, **kwargs)
                original = overlay().getchannel("A").getbbox()
                smaller = overlay(nameplate_scale=.5).getchannel("A").getbbox()
                self.assertAlmostEqual(smaller[2] - smaller[0], (original[2] - original[0]) / 2, delta=2)
                self.assertAlmostEqual(smaller[3] - smaller[1], (original[3] - original[1]) / 2, delta=2)
                placed = overlay(nameplate_box=(300, 400, 180, 100)).getchannel("A").getbbox()
                self.assertGreaterEqual(placed[0], 300)
                self.assertGreaterEqual(placed[1], 400)
                self.assertLessEqual(placed[2], 480)
                self.assertLessEqual(placed[3], 500)
        with self.assertRaises(ValueError):
            overlay(nameplate_box=(530, 0, 200, 100))

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "Media tools missing")
    def test_export_uses_selected_nameplate_region_and_topic_text(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            source, template, captions = folder / "source.mp4", folder / "template.png", folder / "captions.json"
            subprocess.run([shutil.which("ffmpeg"), "-v", "error", "-f", "lavfi", "-i",
                            "color=c=gray:s=640x360:r=25:d=0.5", "-c:v", "libx264", str(source)], check=True)
            Image.new("RGBA", (540, 960)).save(template)
            captions.write_text(json.dumps({"reviewed": True, "captions": [
                {"start": 0, "end": .5, "text": "各位朋友大家好"}]}), encoding="utf-8")
            output, frame = folder / "out.mp4", folder / "frame.png"
            render_video(source, template, captions, output, "张帆", "示例医院", "主任医师", "",
                         slot=(0, 326, 540, 304), nameplate_style="teal_gradient",
                         nameplate_box=(300, 390, 180, 100), topic="健康饮食", topic_box=(20, 30, 500, 150),
                         ffmpeg_path=shutil.which("ffmpeg"), ffprobe_path=shutil.which("ffprobe"))
            subprocess.run([shutil.which("ffmpeg"), "-v", "error", "-i", str(output), "-frames:v", "1", str(frame)], check=True)
            with Image.open(frame) as image:
                red, green, blue = image.getpixel((310, 420))
                self.assertGreater(green - red, 20)
                red, green, blue = image.getpixel((24, 395))
                self.assertLess(max(red, green, blue) - min(red, green, blue), 10)
                crop = image.crop((20, 30, 520, 180)).convert("L")
                self.assertGreater(max(getattr(crop, "get_flattened_data", crop.getdata)()), 150)


if __name__ == "__main__":
    unittest.main()
