"""Real-media regression tests for the source-to-output cut timeline."""
from __future__ import annotations

import array
import json
import math
from pathlib import Path
import subprocess
import tempfile
import unittest
import wave

from PIL import Image, ImageDraw

from studio.edits import no_window, output_duration
from studio.render import default_font, probe, render_video
from studio.toolchain import detect_tools


@unittest.skipUnless(default_font(), "Chinese font not installed")
class RenderSyncTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        encoder, prober = detect_tools()
        if not encoder[0] or not prober[0]:
            raise unittest.SkipTest("FFmpeg and FFprobe are required")
        cls.encoder, cls.prober = encoder[0], prober[0]

    def command(self, args):
        return subprocess.run(args, check=True, capture_output=True,
                              creationflags=no_window()).stdout

    def make_source(self, folder, *, duration=8, audio=True):
        source = folder / "source.mkv"
        inputs = [self.encoder, "-y", "-v", "error", "-f", "lavfi", "-i",
                  f"color=c=black:s=320x180:r=25:d={duration}"]
        if audio:
            # A non-repeating chirp is identifiable at both early and late
            # times, unlike a single tone whose phase repeats periodically.
            samples = array.array("h", (round(12000 * (
                .65 * math.sin(2 * math.pi * (211 * t + 37 * t * t)) +
                .35 * math.sin(2 * math.pi * (379 * t + 23 * t * t))))
                for t in (i / 16000 for i in range(round(duration * 16000)))))
            audio_path = folder / "source.wav"
            with wave.open(str(audio_path), "wb") as sound:
                sound.setnchannels(1)
                sound.setsampwidth(2)
                sound.setframerate(16000)
                sound.writeframes(samples.tobytes())
            inputs += ["-i", str(audio_path)]
        # Each frame's luminance encodes its source frame index. A center
        # sample survives overlays and lets us verify actual frame placement.
        self.command(inputs + ["-vf", "geq=lum=40+N:cb=128:cr=128", "-c:v", "ffv1",
                               "-c:a", "pcm_s16le", str(source)])
        return source

    def render(self, folder, source, cuts):
        template, captions = folder / "template.png", folder / "captions.json"
        image = Image.new("RGBA", (640, 1136), "#244974")
        ImageDraw.Draw(image).rectangle((0, 360, 639, 719), fill=(0, 0, 0, 0))
        image.save(template)
        duration = output_duration(cuts, probe(source, self.prober)[0])
        captions.write_text(json.dumps({"reviewed": True, "captions": [
            {"start": .04, "end": min(.7, duration - .04), "text": "测试字幕"}]}),
            encoding="utf-8")
        output = folder / "output.mp4"
        render_video(source, template, captions, output, "李景平", "示例医院", "医生", "",
                     ffmpeg_path=self.encoder, ffprobe_path=self.prober, cuts=cuts)
        return output, duration

    def pcm(self, path):
        return array.array("h", self.command([
            self.encoder, "-v", "error", "-i", str(path), "-map", "0:a:0", "-ac", "1",
            "-ar", "4000", "-f", "s16le", "-"]))

    def audio_offset(self, source_pcm, output_pcm, source_time, output_time):
        sample_rate, length = 4000, 480
        reference = source_pcm[round(source_time * sample_rate):
                               round(source_time * sample_rate) + length]
        energy = sum(x * x for x in reference)
        expected = round(output_time * sample_rate)
        best_score, best_start = -1, None
        for offset in range(-1200, 1201, 4):
            start = expected + offset
            actual = output_pcm[start:start + length]
            if start < 0 or len(actual) != length:
                continue
            score = sum(a * b for a, b in zip(reference, actual)) / math.sqrt(
                energy * max(1, sum(x * x for x in actual)))
            if score > best_score:
                best_score, best_start = score, start
        self.assertGreater(best_score, .9, "Could not identify the retained source audio")
        return (best_start - expected) / sample_rate

    def gray_frames(self, path, crop):
        return self.command([self.encoder, "-v", "error", "-i", str(path), "-an",
            "-vf", f"crop=8:8:{crop[0]}:{crop[1]},scale=1:1,format=gray", "-f", "rawvideo", "-"])

    def test_many_non_frame_aligned_cuts_do_not_accumulate_late_audio_drift(self):
        cuts = [{"start": round(.333 + i * .31, 3), "end": round(.396 + i * .31, 3)}
                for i in range(18)]
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            source = self.make_source(folder)
            output, expected_duration = self.render(folder, source, cuts)
            self.assertAlmostEqual(probe(output, self.prober)[0], expected_duration, delta=.065)
            source_pcm, output_pcm = self.pcm(source), self.pcm(output)
            for source_time in (1.1, 3.3, 6.7):
                with self.subTest(source_time=source_time):
                    output_time = source_time - sum(cut["end"] - cut["start"] for cut in cuts
                                                     if cut["end"] <= source_time)
                    self.assertLessEqual(abs(self.audio_offset(source_pcm, output_pcm,
                                                               source_time, output_time)), .012)
            original = self.gray_frames(source, (156, 86))
            actual = self.gray_frames(output, (312, 532))
            removed = sum(cut["end"] - cut["start"] for cut in cuts)
            # Inspect real video frames near the tail, not FFmpeg arguments.
            for frame in range(145, min(165, len(actual))):
                source_time = frame / 25 + removed
                candidates = original[max(0, round(source_time * 25) - 1):
                                      min(len(original), round(source_time * 25) + 2)]
                self.assertLessEqual(min(abs(actual[frame] - value) for value in candidates), 3)

    def test_leading_and_trailing_cuts_keep_the_first_retained_video_frame(self):
        cuts = [{"start": 0, "end": .453}, {"start": 1.407, "end": 2}]
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            source = self.make_source(folder, duration=2)
            output, expected_duration = self.render(folder, source, cuts)
            self.assertAlmostEqual(probe(output, self.prober)[0], expected_duration, delta=.065)
            offset = self.audio_offset(self.pcm(source), self.pcm(output), .8, .8 - .453)
            self.assertLessEqual(abs(offset), .012)
            original = self.gray_frames(source, (156, 86))
            actual = self.gray_frames(output, (312, 532))
            first_retained = math.ceil(.453 * 25)
            self.assertLessEqual(abs(actual[0] - original[first_retained]), 3)

    def test_video_without_audio_supports_both_cut_and_uncut_export(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            source = self.make_source(folder, duration=1.6, audio=False)
            for cuts in ([], [{"start": .203, "end": .487}]):
                with self.subTest(cuts=cuts):
                    output, expected_duration = self.render(folder, source, cuts)
                    self.assertAlmostEqual(probe(output, self.prober)[0], expected_duration, delta=.065)
                    streams = json.loads(self.command([self.prober, "-v", "error", "-show_entries",
                        "stream=codec_type", "-of", "json", str(output)]))
                    self.assertEqual([stream["codec_type"] for stream in streams["streams"]], ["video"])


if __name__ == "__main__":
    unittest.main()
