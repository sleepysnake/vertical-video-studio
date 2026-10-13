from pathlib import Path
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PIL import Image
from PySide6.QtWidgets import QApplication

from studio.caption_style import CaptionStyle
from studio.render import caption_layout, default_font, font_family_name, make_ass, render_video
from qt_app import App


class CaptionStyleTests(unittest.TestCase):
    def test_defaults_and_bad_input(self):
        self.assertEqual(CaptionStyle.from_dict({}), CaptionStyle())
        for changes in ({"font_size": 8}, {"text_color": "red"}, {"outline_width": -1}, {"font_path": 2}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                CaptionStyle.from_dict(changes)

    @unittest.skipUnless(default_font(), "Chinese font missing")
    def test_manual_size_is_preserved_with_wrapping_and_ass_colors(self):
        style = CaptionStyle(48, "#FFD233", "#123456", 2)
        size, lines = caption_layout("只吃了一个汉堡或喝了一杯奶茶", 540, default_font(), style)
        self.assertEqual(size, 48)
        self.assertEqual(len(lines), 2)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "caption.ass"
            make_ass([{"start": 0, "end": 1, "text": "只吃了一个汉堡或喝了一杯奶茶"}],
                     (540, 960), (0, 326, 540, 304), path, font_family_name(default_font()), default_font(), caption_style=style)
            source = path.read_text(encoding="utf-8-sig")
            self.assertIn("&H0033D2FF", source)
            self.assertIn("&H00563412", source)
            self.assertIn(r"\fs48", source)
            self.assertIn(r"\N", source)

    @unittest.skipUnless(default_font(), "Chinese font missing")
    def test_font_family_comes_from_font_metadata_not_filename(self):
        with tempfile.TemporaryDirectory() as temporary:
            renamed = Path(temporary) / "custom-name.ttf"
            shutil.copyfile(default_font(), renamed)
            self.assertEqual(font_family_name(renamed), font_family_name(default_font()))

    def test_ui_restores_saves_and_previews_caption_settings(self):
        app = QApplication.instance() or QApplication([])
        style = CaptionStyle(42, "#FFD233", "#1D3050", 2)
        with tempfile.TemporaryDirectory() as temporary:
            settings = Path(temporary) / "settings.json"
            settings.write_text(json.dumps({"caption_style": style.snapshot()}), encoding="utf-8")
            with patch("qt_app.SETTINGS_PATH", settings), patch.object(App, "detect_media_tools"):
                window = App()
                self.assertEqual(window.current_caption_style(), style)
                window.caption_preview.resize(460, 110)
                first = window.caption_preview.grab().toImage()
                window.caption_size.setValue(32)
                second = window.caption_preview.grab().toImage()
                self.assertNotEqual(first, second)
                window.save_current_settings()
                saved = json.loads(settings.read_text(encoding="utf-8"))
                self.assertEqual(saved["caption_style"]["font_size"], 32)
                self.assertEqual(saved["caption_style"]["text_color"], "#FFD233")
                self.assertNotIn("api_key", saved)
                window.close()
                app.processEvents()

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe") and default_font(), "Media tools missing")
    def test_actual_export_obeys_subtitle_color_and_size(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            source = folder / "source.mp4"
            subprocess.run([shutil.which("ffmpeg"), "-v", "error", "-f", "lavfi", "-i",
                            "color=c=gray:s=640x360:r=25:d=0.5", "-c:v", "libx264", str(source)], check=True)
            template, captions = folder / "template.png", folder / "captions.json"
            Image.new("RGBA", (540, 960), (0, 0, 0, 0)).save(template)
            captions.write_text(json.dumps({"reviewed": True, "captions": [
                {"start": 0, "end": .5, "text": "各位朋友大家好"}]}, ensure_ascii=False), encoding="utf-8")
            counts = []
            for size in (18, 42):
                output, frame = folder / f"out-{size}.mp4", folder / f"frame-{size}.png"
                render_video(source, template, captions, output, "李景平", "示例医院", "医生", "",
                             slot=(0, 326, 540, 304), caption_style=CaptionStyle(size, "#FFD233", "#1D3050", 2),
                             ffmpeg_path=shutil.which("ffmpeg"), ffprobe_path=shutil.which("ffprobe"))
                subprocess.run([shutil.which("ffmpeg"), "-v", "error", "-i", str(output), "-frames:v", "1", str(frame)], check=True)
                with Image.open(frame) as image:
                    crop = image.crop((0, 525, 540, 630))
                    pixels = getattr(crop, "get_flattened_data", crop.getdata)()
                    counts.append(sum(r > 170 and g > 140 and b < 100 for r, g, b in pixels))
            self.assertGreater(counts[0], 100)
            self.assertGreater(counts[1], counts[0] * 2)


if __name__ == "__main__":
    unittest.main()
