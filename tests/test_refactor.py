from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw

from studio.captions import Word
from studio.bailian import BailianClient
from studio.brightness import analyze_brightness, choose_brightness
from studio.edits import detect_long_silence_cuts
from studio.models import scan_folder
from studio.prompts import load_prompts
from studio.render import probe, render_video
from studio.strategy import Strategy, load_strategy
from studio.sentence_flow import _caption_segments


class RefactorTests(unittest.TestCase):
    def test_editable_prompts_validate_and_change_digest(self):
        original = load_prompts()
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / "edit_prompt.txt").write_text(original.edit, encoding="utf-8")
            (folder / "caption_prompt.txt").write_text(original.caption, encoding="utf-8")
            self.assertEqual(load_prompts(folder).digest(), original.digest())
            (folder / "edit_prompt.txt").write_text(original.edit + "\n补充一条剪辑约束。", encoding="utf-8")
            self.assertNotEqual(load_prompts(folder).digest(), original.digest())
            (folder / "caption_prompt.txt").write_text("只生成字幕" * 10, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "缺少输出字段"):
                load_prompts(folder)

    def test_auto_brightness_keeps_normal_and_brightens_dark_source(self):
        self.assertEqual(choose_brightness(95, 100)["action"], "keep")
        self.assertEqual(choose_brightness(70, 100)["action"], "brighten")
        self.assertEqual(choose_brightness(70, 100)["offset"], .118)
        self.assertEqual(choose_brightness(10, 100)["offset"], .18)

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg not installed")
    def test_auto_brightness_samples_actual_video_frames(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            for name, color, expected in (("normal", "gray", "keep"),
                                          ("dark", "#303030", "brighten")):
                video = folder / f"{name}.mp4"
                subprocess.run([shutil.which("ffmpeg"), "-y", "-v", "error", "-f", "lavfi",
                                "-i", f"color=c={color}:s=640x360:r=25:d=1", "-c:v", "libx264",
                                str(video)], check=True)
                analysis = analyze_brightness(video, 1, shutil.which("ffmpeg"))
                self.assertEqual(analysis["action"], expected)
                self.assertEqual(len(analysis["samples"]), 1)

    def test_silence_boundary_and_policy_snapshot(self):
        words = [Word(2.0, 2.4, "你好")]
        cuts = detect_long_silence_cuts(words, [(0, 2), (2.4, 5)], duration=5)
        self.assertEqual([(c["start"], c["end"]) for c in cuts], [(0, 1.88), (2.52, 5)])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "strategy.json"
            path.write_text(json.dumps({**Strategy().snapshot(), "silence_seconds": 2.5}), encoding="utf-8")
            self.assertEqual(load_strategy(path).silence_seconds, 2.5)
            self.assertNotEqual(load_strategy(path).digest(), Strategy().digest())

    def test_manifest_allows_per_video_topic_without_filename_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / "clip.mp4").touch()
            (folder / "batch.csv").write_text(
                "file,organization,title,name,clip_id,topic,nameplate_style,brightness\n"
                "clip.mp4,示例医院,医师,张帆,02,足踝健康,ass_dark,bright\n", encoding="utf-8")
            items, invalid = scan_folder(folder)
            self.assertFalse(invalid)
            self.assertEqual((items[0].person.name, items[0].clip_id, items[0].topic,
                              items[0].nameplate_style, items[0].brightness),
                             ("张帆", "02", "足踝健康", "ass_dark", "bright"))

    def test_asr_download_retries_incomplete_document_before_caching(self):
        valid = {"transcripts": [{"sentences": [{"begin_time": 0, "end_time": 300,
                 "words": [{"begin_time": 0, "end_time": 300, "text": "你好"}]}]}]}
        client = BailianClient("https://example.com/asr", "https://example.com/chat",
                               "test-key", "model-a", "model-b")
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "transcription.json"
            with patch.object(client, "_get_json", side_effect=[{}, valid]) as getter, \
                    patch("studio.bailian.time.sleep"):
                self.assertEqual(client.download_transcription("https://example.com/result", target), valid)
            self.assertEqual(getter.call_count, 2)
            self.assertEqual(json.loads(target.read_text(encoding="utf-8")), valid)

    def test_model_clause_boundary_is_not_overridden_by_asr_punctuation(self):
        words = [Word(0, .2, "先说", "。"), Word(.4, .6, "后说")]
        tokens = [{"token": 0, "word_index": 0, "text": "先说", "punctuation": "。",
                   "start_ms": 0, "end_ms": 200},
                  {"token": 1, "word_index": 1, "text": "后说", "punctuation": "",
                   "start_ms": 400, "end_ms": 600}]
        result = _caption_segments({"segments": [{"start_token": 0, "end_token": 2,
                                                 "text": "先说后说"}]}, tokens, words)
        self.assertEqual(result[0]["text"], "先说后说")
        self.assertEqual((result[0]["start"], result[0]["end"]), (0, .6))

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg not installed")
    def test_ass_nameplate_topic_brightness_and_balanced_render(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            video, template, captions, output = (folder / name for name in
                                                  ("source.mp4", "template.png", "captions.json", "out.mp4"))
            subprocess.run([shutil.which("ffmpeg"), "-y", "-v", "error", "-f", "lavfi", "-i",
                            "color=c=gray:s=640x360:r=25:d=1", "-f", "lavfi", "-i",
                            "sine=frequency=440:duration=1", "-shortest", "-c:v", "libx264",
                            "-c:a", "aac", str(video)], check=True)
            image = Image.new("RGBA", (540, 960), "#2D65A4")
            ImageDraw.Draw(image).rectangle((0, 326, 539, 629), fill=(0, 0, 0, 0))
            image.save(template)
            captions.write_text(json.dumps({"reviewed": True, "captions": [
                {"start": 0.1, "end": 0.9, "text": "大家好"}]}, ensure_ascii=False), encoding="utf-8")
            render_video(video, template, captions, output, "张帆", "中山大学孙逸仙纪念医院创伤与足踝外科",
                         "副主任医师", "医学科普", ffmpeg_path=shutil.which("ffmpeg"),
                         ffprobe_path=shutil.which("ffprobe"), nameplate_style="ass_dark",
                         topic="足踝健康", topic_box=(0, 20, 540, 100), brightness="bright")
            self.assertTrue(output.is_file())
            self.assertEqual(probe(output)[1:], (540, 960))
            automatic = folder / "automatic.mp4"
            render_video(video, template, captions, automatic, "张帆", "示例医院", "医生",
                         "医学科普", ffmpeg_path=shutil.which("ffmpeg"),
                         ffprobe_path=shutil.which("ffprobe"), topic="足踝健康",
                         topic_box=(0, 20, 540, 100), topic_color="#FFD233",
                         topic_font_path=Path("C:/Windows/Fonts/simhei.ttf")
                         if Path("C:/Windows/Fonts/simhei.ttf").is_file() else None,
                         brightness="auto")
            self.assertTrue(automatic.is_file())


if __name__ == "__main__":
    unittest.main()
