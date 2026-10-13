from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from PIL import Image, ImageDraw

from studio.alignment import build_timeline
from studio.captions import Word
from studio.edit_plan import build_edit_plan
from studio.edits import detect_silences, no_window, retime_captions
from studio.render import probe, render_video
from studio.strategy import Strategy


def abandoned(first=1, last=2):
    return {"kind": "abandoned_start", "start_word": first, "end_word": last,
            "resume_word": last, "reason": "短起句未完成观点，停顿后独立重起，无独有信息"}


class AbandonedPlanTests(unittest.TestCase):
    def test_word_timing_quiet_endpoint_and_caption_timeline_remain_consistent(self):
        words = [Word(68, 68.7, "前一结论"), Word(69.28, 70.28, "那么我们"),
                 Word(71.44, 71.92, "所以说"), Word(71.92, 76.28, "完整下一观点")]
        plan = build_edit_plan(words, [abandoned()], 80, [(70.752687, 71.448437)])
        decision = plan["decisions"][0]
        self.assertEqual(decision["action"], "CUT")
        self.assertEqual((decision["spoken_start"], decision["spoken_end"]), (69.28, 70.28))
        self.assertEqual((plan["cuts"][0]["start"], plan["cuts"][0]["end"]), (69.28, 71.32))
        self.assertEqual(decision["pause_evidence"]["duration"], .687)
        captions = [{"start": 71.44, "end": 76.28, "text": "所以说完整下一观点",
                     "start_word": 2, "end_word": 4}]
        rendered = retime_captions(captions, plan["cuts"])
        self.assertEqual((rendered[0]["start"], rendered[0]["end"]), (69.4, 74.24))
        timeline = build_timeline(Path("source.mp4"), Path("result.mp4"), words,
                                  captions, plan["cuts"], 80)
        self.assertEqual(timeline["output_duration"], 77.96)
        self.assertEqual(timeline["captions"][0]["output_start"], 69.4)
        self.assertTrue(all(not word["removed"] for word in timeline["captions"][0]["asr_words"]))

    def test_model_owns_word_deletion_and_quiet_is_required_only_for_extension(self):
        words = [Word(0, 1, "前文"), Word(1, 2, "接下来我想"), Word(3.2, 5, "独立下一观点")]
        # A short pause alone never gives code authority to classify speech.
        self.assertEqual(build_edit_plan(words, [], 6, [(2.5, 3.2)])["cuts"], [])
        for quiet in ([], [(2.1, 2.7)], [(2.8, 3.2)]):
            with self.subTest(quiet=quiet):
                plan = build_edit_plan(words, [abandoned()], 6, quiet)
                self.assertEqual(plan["applied_proposals"], [abandoned()])
                self.assertEqual(plan["decisions"][0]["action"], "CUT")
                self.assertEqual((plan["cuts"][0]["start"], plan["cuts"][0]["end"]), (1, 2))
                self.assertIsNone(plan["decisions"][0].get("pause_evidence"))

    def test_short_start_limits_and_estimated_word_times_preserve_speech(self):
        cases = [([Word(0, 1, "前文"), Word(1, 4.1, "过长起句"), Word(5, 8, "下句")], {}, "上限"),
                 ([Word(0, 1, "前文"), Word(1, 2, "短起句"), Word(3.2, 5, "下句")],
                  {"estimated_timing": True}, "估算"),
                 ([Word(0, 1, "前文"), Word(1, 2, "短起句"), Word(3.2, 5, "下句")],
                  {"strategy": replace(Strategy(), max_removal_ratio=.2)}, "比例")]
        for words, options, reason in cases:
            with self.subTest(reason=reason):
                plan = build_edit_plan(words, [abandoned()], 9, [(words[1].end, words[2].start)], **options)
                self.assertEqual(plan["applied_proposals"], [])
                self.assertIn(reason, plan["decisions"][0]["policy_reason"])
        words = [Word(0, 1, "前文"), Word(1, 2, "短起句"), Word(3.2, 4, "下句"), Word(4, 5, "结尾")]
        plan = build_edit_plan(words, [{**abandoned(), "resume_word": 3}], 6, [(2, 3.2)])
        self.assertEqual(plan["decisions"][0]["action"], "KEEP")
        self.assertIn("紧接", plan["decisions"][0]["policy_reason"])

    def test_reason_describing_asr_sentence_merge_does_not_fail_cut_mapping(self):
        words = [Word(0, 1, "前文"), Word(1, 2, "短起句"), Word(3.2, 5, "下一观点")]
        proposal = {**abandoned(), "reason": "ASR合并了停顿前后表达，识别到前段未完成且后段独立重起"}
        plan = build_edit_plan(words, [proposal], 6, [(2.5, 3.2)])
        self.assertEqual(plan["applied_proposals"], [proposal])
        self.assertEqual(plan["decisions"][0]["action"], "CUT")

    def test_final_plan_preserves_resume_word_across_independently_analyzed_groups(self):
        words = [Word(.5, 1, "起句"), Word(2, 3.5, "再起句"), Word(4.5, 5.5, "下一观点")]
        first, second = abandoned(0, 1), abandoned(1, 2)
        for proposals in ([first, second], [second, first]):
            with self.subTest(proposals=proposals):
                plan = build_edit_plan(words, proposals, 10, [(1.2, 1.85), (3.6, 4.5)])
                self.assertEqual(plan["applied_proposals"], [proposals[0]])
                rejected = next(item for item in plan["decisions"] if item["action"] == "KEEP")
                self.assertIn("最终剪辑", rejected["policy_reason"])

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg not installed")
    def test_confirmed_pause_cut_physically_removes_audio_video_and_keeps_next_voice(self):
        encoder, prober = shutil.which("ffmpeg"), shutil.which("ffprobe")
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            source, output = folder / "source.mp4", folder / "out.mp4"
            subprocess.run([encoder, "-y", "-v", "error", "-f", "lavfi", "-i",
                "color=c=gray:s=320x180:r=25:d=6", "-f", "lavfi", "-i",
                "aevalsrc=if(lt(t\\,2)\\,0.2*sin(2*PI*440*t)\\,if(lt(t\\,3.2)\\,0\\,0.2*sin(2*PI*880*t))):s=16000:d=6",
                "-shortest", "-c:v", "libx264", "-c:a", "aac", str(source)],
                check=True, creationflags=no_window(), capture_output=True)
            words = [Word(0, 1, "前文"), Word(1, 2, "未完成起句"), Word(3.2, 6, "独立下一观点")]
            quiet = detect_silences(str(source), encoder, 6)
            plan = build_edit_plan(words, [abandoned()], 6, quiet)
            self.assertEqual(plan["decisions"][0]["action"], "CUT")
            cut = plan["cuts"][0]
            self.assertAlmostEqual(cut["end"], 3.08, delta=.025)
            image = Image.new("RGBA", (540, 960), "#2D65A4")
            ImageDraw.Draw(image).rectangle((0, 326, 539, 629), fill=(0, 0, 0, 0))
            template = folder / "template.png"
            image.save(template)
            captions = folder / "captions.json"
            original = [{"start": 0, "end": 1, "text": "前文"},
                        {"start": 3.2, "end": 6, "text": "独立下一观点"}]
            captions.write_text(json.dumps({"reviewed": True,
                "captions": retime_captions(original, plan["cuts"])}, ensure_ascii=False), encoding="utf-8")
            render_video(source, template, captions, output, "张帆", "示例医院", "医生", "仅供参考",
                         cuts=plan["cuts"], ffmpeg_path=encoder, ffprobe_path=prober)
            self.assertAlmostEqual(probe(output, prober)[0], 6 - (cut["end"] - cut["start"]), delta=.12)
            next_output = retime_captions(original[1:], plan["cuts"])[0]["start"]
            pcm = subprocess.run([encoder, "-v", "error", "-ss", str(next_output + .15),
                "-i", str(output), "-t", "0.5", "-vn", "-ac", "1", "-ar", "16000",
                "-f", "s16le", "-"], check=True, capture_output=True, creationflags=no_window()).stdout
            import array
            samples = array.array("h", pcm)
            self.assertTrue(samples)
            crossings = sum(a < 0 <= b for a, b in zip(samples, samples[1:]))
            self.assertAlmostEqual(crossings / (len(samples) / 16000), 880, delta=15)


if __name__ == "__main__":
    unittest.main()
