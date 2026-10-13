from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PIL import Image
from PySide6.QtWidgets import QApplication, QComboBox, QLineEdit

from qt_app import NameplatePreview
from studio.render import NAMEPLATE_STYLES, default_font, graphics, probe, render_video


@unittest.skipUnless(default_font(), "Chinese font not installed")
class NameplateTests(unittest.TestCase):
    def test_every_style_fits_long_person_details_and_notice_in_video_area(self):
        hashes = set()
        for _, style in NAMEPLATE_STYLES:
            with self.subTest(style=style):
                image = graphics((540, 960), (0, 326, 540, 304), "张帆",
                                 "中山大学孙逸仙纪念医院创伤与足踝外科", "副主任医师",
                                 "医学科普 仅供参考\n如有不适请及时线下就医", default_font(), None, style)
                hashes.add(hashlib.sha256(image.tobytes()).hexdigest())
                self.assertIsNotNone(image.crop((0, 384, 215, 514)).getchannel("A").getbbox())
                self.assertIsNone(image.crop((215, 384, 540, 960)).getchannel("A").getbbox())
                self.assertIsNone(image.crop((0, 515, 540, 630)).getchannel("A").getbbox())
        self.assertEqual(len(hashes), len(NAMEPLATE_STYLES))

    def test_qt_preview_accepts_every_style_and_keeps_legacy_setting_ids(self):
        app = QApplication.instance() or QApplication([])
        fields = [QLineEdit(text) for text in ("示例医院骨科", "主任医师", "李景平")]
        choice = QComboBox()
        for label, style in NAMEPLATE_STYLES:
            choice.addItem(label, style)
        self.assertEqual(choice.findData("ass_clean"), 1)
        self.assertEqual(choice.findData("ass_dark"), 2)
        preview = NameplatePreview(*fields, choice)
        preview.resize(540, 145)
        hashes = set()
        for index in range(choice.count()):
            choice.setCurrentIndex(index)
            app.processEvents()
            image = preview.grab().toImage()
            hashes.add(hashlib.sha256(image.constBits()).hexdigest())
        self.assertEqual(len(hashes), choice.count())
        preview.close()

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg not installed")
    def test_gradient_and_slanted_nameplates_survive_actual_video_export(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            source = folder / "source.mp4"
            subprocess.run([shutil.which("ffmpeg"), "-v", "error", "-f", "lavfi", "-i",
                            "color=c=gray:s=640x360:r=25:d=0.5", "-c:v", "libx264", str(source)], check=True)
            template = folder / "template.png"
            Image.new("RGBA", (540, 960), (0, 0, 0, 0)).save(template)
            captions = folder / "captions.json"
            captions.write_text(json.dumps({"reviewed": True, "captions": [
                {"start": 0, "end": .5, "text": "各位朋友大家好"}]}, ensure_ascii=False), encoding="utf-8")
            for style in ("broadcast_slant", "teal_gradient"):
                output = folder / f"{style}.mp4"
                render_video(source, template, captions, output, "李景平", "武汉同济医院", "医生",
                             "医学科普", slot=(0, 326, 540, 304), nameplate_style=style,
                             ffmpeg_path=shutil.which("ffmpeg"), ffprobe_path=shutil.which("ffprobe"))
                self.assertEqual(probe(output)[1:], (540, 960))
                frame = folder / f"{style}.png"
                subprocess.run([shutil.which("ffmpeg"), "-v", "error", "-i", str(output),
                                "-frames:v", "1", str(frame)], check=True)
                with Image.open(frame) as picture:
                    point = picture.getpixel((24, 390))
                    self.assertGreater(max(point) - min(point), 20)


if __name__ == "__main__":
    unittest.main()
