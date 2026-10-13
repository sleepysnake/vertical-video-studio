from pathlib import Path
import json
import shutil
import subprocess
import tempfile
import unittest

from PIL import Image, ImageDraw

from studio.render import default_font, draw_topic, fit_font, render_video
from studio.topic_style import TopicStyle, TOPIC_BACKGROUNDS, TOPIC_EFFECTS, topic_layout


class TopicStyleTests(unittest.TestCase):
    def test_roundtrip_and_validation(self):
        style = TopicStyle(48, "gradient", "rounded", "#185B87", 65, "#FFD54A", "#091827", 4)
        self.assertEqual(TopicStyle.from_dict(style.snapshot()), style)
        self.assertEqual(TopicStyle.from_dict({}), TopicStyle())
        invalid = ({"font_size": 11}, {"font_size": 301}, {"font_size": True},
                   {"effect": "unknown"}, {"background": []}, {"background_color": "red"},
                   {"background_opacity": 101}, {"accent_color": "yellow"},
                   {"outline_color": 0}, {"outline_width": -1}, {"extra": 1})
        for changes in invalid:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                TopicStyle.from_dict(changes)

    @unittest.skipUnless(default_font(), "Chinese font missing")
    def test_manual_size_changes_visible_text_and_wraps_without_shrinking(self):
        bounds = []
        for size in (24, 60):
            layer = Image.new("RGBA", (540, 960))
            result = draw_topic(layer, "健康生活", (40, 40, 460, 220), "#FFFFFF", default_font(),
                                topic_style=TopicStyle(size, "plain"))
            bounds.append(layer.getchannel("A").getbbox())
            self.assertEqual(result["font_size"], size)
        self.assertGreater(bounds[1][2] - bounds[1][0], (bounds[0][2] - bounds[0][0]) * 2)
        layout = topic_layout("今天一起学习如何改善健康生活", (400, 240), default_font(), TopicStyle(52))
        self.assertEqual(layout["font_size"], 52)
        self.assertGreater(len(layout["lines"]), 1)
        self.assertEqual("".join(layout["lines"]), "今天一起学习如何改善健康生活")
        with self.assertRaisesRegex(ValueError, "减小字号或扩大主题区域"):
            topic_layout("健康生活", (80, 40), default_font(), TopicStyle(100))

    @unittest.skipUnless(default_font(), "Chinese font missing")
    def test_every_effect_and_background_stays_inside_region(self):
        images = set()
        box = (40, 50, 440, 240)
        for _, effect in TOPIC_EFFECTS:
            for _, background in TOPIC_BACKGROUNDS:
                layer = Image.new("RGBA", (540, 960))
                style = TopicStyle(54, effect, background)
                draw_topic(layer, "健康生活\n从今天开始", box, "#F5F9FF", default_font(), topic_style=style)
                bounds = layer.getchannel("A").getbbox()
                self.assertIsNotNone(bounds)
                self.assertGreaterEqual(bounds[0], box[0])
                self.assertGreaterEqual(bounds[1], box[1])
                self.assertLessEqual(bounds[2], box[0] + box[2])
                self.assertLessEqual(bounds[3], box[1] + box[3])
                images.add(layer.tobytes())
        self.assertEqual(len(images), 25)

    @unittest.skipUnless(default_font(), "Chinese font missing")
    def test_background_opacity_and_legacy_default_pixels(self):
        layer = Image.new("RGBA", (540, 960))
        style = TopicStyle(54, "plain", "rectangle", "#13579B", 50)
        draw_topic(layer, "健康生活", (40, 50, 440, 240), "#FFFFFF", default_font(), topic_style=style)
        bounds = layer.getchannel("A").getbbox()
        self.assertEqual(layer.getpixel(bounds[:2]), (19, 87, 155, 127))
        legacy = Image.new("RGBA", (540, 960))
        expected = legacy.copy()
        box, text = (40, 50, 440, 240), "健康生活"
        draw_topic(legacy, text, box, "#FFFFFF", default_font())
        draw = ImageDraw.Draw(expected)
        chosen = fit_font(draw, text, default_font(), min(box[3] * .55, box[2] * .09), box[2] * .92)
        bbox = draw.textbbox((0, 0), text, font=chosen)
        draw.text((box[0] + (box[2] - (bbox[2] - bbox[0])) / 2 - bbox[0],
                   box[1] + (box[3] - (bbox[3] - bbox[1])) / 2 - bbox[1]),
                  text, font=chosen, fill="#FFFFFF", stroke_width=2, stroke_fill="#173863")
        self.assertEqual(legacy.tobytes(), expected.tobytes())

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe") and default_font(), "Media tools missing")
    def test_actual_ffmpeg_export_uses_topic_background_in_selected_area(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            source, template = folder / "source.mp4", folder / "template.png"
            captions, output, frame = folder / "captions.json", folder / "output.mp4", folder / "frame.png"
            subprocess.run([shutil.which("ffmpeg"), "-v", "error", "-f", "lavfi", "-i",
                            "color=c=gray:s=640x360:r=25:d=0.4", "-c:v", "libx264", str(source)], check=True)
            Image.new("RGBA", (540, 960), "#263448").save(template)
            captions.write_text(json.dumps({"reviewed": True, "captions": [
                {"start": 0, "end": .4, "text": "各位朋友大家好"}]}, ensure_ascii=False), encoding="utf-8")
            box = (40, 50, 440, 220)
            style = TopicStyle(54, "plain", "rectangle", "#EF2636", 100)
            render_video(source, template, captions, output, "李景平", "示例医院", "医生", "",
                         slot=(0, 326, 540, 304), topic="健康生活", topic_box=box,
                         topic_style=style, ffmpeg_path=shutil.which("ffmpeg"), ffprobe_path=shutil.which("ffprobe"))
            subprocess.run([shutil.which("ffmpeg"), "-v", "error", "-i", str(output),
                            "-frames:v", "1", str(frame)], check=True)
            with Image.open(frame) as actual:
                inside = actual.crop((40, 50, 480, 270))
                pixels = getattr(inside, "get_flattened_data", inside.getdata)()
                red_count = sum(r > 170 and g < 90 and b < 110 for r, g, b in pixels)
                self.assertGreater(red_count, 10000)
                self.assertEqual(actual.size, (540, 960))
                r, g, b = actual.getpixel((20, 20))
                self.assertLess(r, 100)
                self.assertLess(g, 100)


if __name__ == "__main__":
    unittest.main()
