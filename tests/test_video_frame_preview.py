from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PIL import Image, ImageDraw
from PySide6.QtWidgets import QApplication

from studio.qt_video_preview import (FramePreviewController, PreviewCancelled,
                                     _run_media, extract_frame, fit_frame, probe_video)


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "Media tools missing")
class VideoFramePreviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.temporary = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.temporary.name)
        cls.ffmpeg, cls.ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
        cls.changing = cls.folder / "两个时刻.mp4"
        cls.green = cls.folder / "绿色.mp4"
        cls.anamorphic = cls.folder / "sar.mp4"
        subprocess.run([cls.ffmpeg, "-v", "error", "-f", "lavfi", "-i", "color=red:s=320x180:r=25:d=1",
                        "-f", "lavfi", "-i", "color=blue:s=320x180:r=25:d=1",
                        "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]", "-map", "[v]",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(cls.changing)], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        for target, video_filter in ((cls.green, None), (cls.anamorphic, "setsar=2")):
            command = [cls.ffmpeg, "-v", "error", "-f", "lavfi", "-i", "color=lime:s=320x180:r=25:d=1"]
            if video_filter:
                command += ["-vf", video_filter]
            subprocess.run(command + ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(target)],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def wait_for(self, condition, timeout=5):
        deadline = time.monotonic() + timeout
        while not condition() and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(.01)
        self.app.processEvents()
        self.assertTrue(condition(), "Preview signal did not arrive")

    @staticmethod
    def color(frame):
        return frame.getpixel((frame.width // 2, frame.height // 2))

    def test_real_frame_decode_and_sample_aspect_ratio_and_crop(self):
        info = probe_video(self.changing, self.ffprobe)
        self.assertAlmostEqual(info.duration, 2, places=2)
        self.assertAlmostEqual(info.fps, 25, places=2)
        red = extract_frame(self.changing, .2, self.ffmpeg)
        blue = extract_frame(self.changing, 1.2, self.ffmpeg)
        self.assertEqual(red.size, (320, 180))
        self.assertGreater(self.color(red)[0], 230)
        self.assertGreater(self.color(blue)[2], 230)
        square_pixels = extract_frame(self.anamorphic, .2, self.ffmpeg)
        self.assertEqual(square_pixels.size, (640, 180))
        bounded = extract_frame(self.anamorphic, .2, self.ffmpeg, max_edge=120)
        self.assertEqual(bounded.width, 120)
        self.assertLessEqual(bounded.height, 120)
        frame = Image.new("RGB", (300, 100), "red")
        ImageDraw.Draw(frame).rectangle((100, 0, 199, 99), fill="blue")
        cropped = fit_frame(frame, (100, 100))
        self.assertEqual(cropped.size, (100, 100))
        self.assertEqual(cropped.getpixel((0, 50)), (0, 0, 255))
        self.assertEqual(cropped.getpixel((99, 50)), (0, 0, 255))
        with self.assertRaises(ValueError):
            fit_frame(frame, (0, 100))

    def test_controller_seek_and_same_source_cache(self):
        controller = FramePreviewController(debounce_ms=40)
        frames, durations = [], []
        controller.frame_ready.connect(lambda frame, seconds: frames.append((frame, seconds)))
        controller.duration_ready.connect(durations.append)
        try:
            controller.set_source(self.changing, self.ffmpeg, self.ffprobe)
            self.wait_for(lambda: len(frames) == 1)
            self.assertGreater(self.color(frames[-1][0])[0], 230)
            self.assertAlmostEqual(controller.duration, 2)
            controller.set_source(self.changing, self.ffmpeg, self.ffprobe)
            self.app.processEvents()
            self.assertEqual(len(frames), 1)
            with patch("studio.qt_video_preview.probe_video", side_effect=AssertionError("Unexpected repeat probe")):
                # Only the last slider request is decoded after the debounce.
                for seconds in (.3, .5, 1.25):
                    controller.seek(seconds)
                self.wait_for(lambda: len(frames) == 2)
                self.assertGreater(self.color(frames[-1][0])[2], 230)
                self.assertAlmostEqual(frames[-1][1], 1.25)
                self.assertIs(controller.frame, frames[-1][0])
                controller.seek(100)
                self.wait_for(lambda: len(frames) == 3)
                self.assertLess(frames[-1][1], 2)
                self.assertGreater(self.color(frames[-1][0])[2], 230)
        finally:
            controller.close()
        self.assertFalse(controller._worker.isRunning())

    def test_switch_source_rejects_stale_frames_and_recovers_errors(self):
        controller = FramePreviewController(debounce_ms=0)
        frames, statuses = [], []
        controller.frame_ready.connect(lambda frame, seconds: frames.append(frame))
        controller.status_changed.connect(statuses.append)
        try:
            # Signals queued from the first source must not populate the second.
            controller.set_source(self.changing, self.ffmpeg, self.ffprobe)
            time.sleep(.1)
            controller.set_source(self.green, self.ffmpeg, self.ffprobe)
            self.wait_for(lambda: len(frames) == 1)
            self.assertGreater(self.color(frames[0])[1], 230)
            self.assertLess(self.color(frames[0])[0], 20)
            controller.set_source(self.folder / "missing.mp4", self.ffmpeg, self.ffprobe)
            self.assertIsNone(controller.frame)
            self.assertIn("文件不存在", statuses[-1])
            broken = self.folder / "broken.mp4"
            broken.write_text("not a video", encoding="utf-8")
            controller.set_source(broken, self.ffmpeg, self.ffprobe)
            self.wait_for(lambda: any("视频预览不可用" in status and "文件不存在" not in status for status in statuses))
            controller.set_source(self.changing, self.ffmpeg, self.ffprobe)
            self.wait_for(lambda: len(frames) == 2)
            self.assertGreater(self.color(frames[-1])[0], 230)
            controller.set_source(None, self.ffmpeg, self.ffprobe)
            self.assertIsNone(controller.frame)
            self.assertEqual(controller.duration, 0)
        finally:
            controller.close()

    def test_replacing_file_at_same_path_invalidates_cached_frame(self):
        source = self.folder / "replaced.mp4"
        shutil.copyfile(self.changing, source)
        controller = FramePreviewController(debounce_ms=0)
        frames = []
        controller.frame_ready.connect(lambda frame, seconds: frames.append(frame))
        try:
            controller.set_source(source, self.ffmpeg, self.ffprobe)
            self.wait_for(lambda: len(frames) == 1)
            self.assertGreater(self.color(frames[-1])[0], 230)
            self.assertAlmostEqual(controller.duration, 2)
            shutil.copyfile(self.green, source)
            controller.set_source(source, self.ffmpeg, self.ffprobe)
            self.assertIsNone(controller.frame)
            self.wait_for(lambda: len(frames) == 2)
            self.assertGreater(self.color(frames[-1])[1], 230)
            self.assertLess(self.color(frames[-1])[0], 20)
            self.assertAlmostEqual(controller.duration, 1)
        finally:
            controller.close()
            source.unlink(missing_ok=True)

    def test_close_cancels_and_reaps_active_subprocess_and_timeout_is_clear(self):
        controller = FramePreviewController(debounce_ms=0)
        real_popen = subprocess.Popen
        processes = []

        def record_process(*args, **kwargs):
            process = real_popen(*args, **kwargs)
            processes.append(process)
            return process

        def slow_decode(path, seconds, ffmpeg, *, cancelled):
            return _run_media([sys.executable, "-c", "import time; time.sleep(30)"],
                              timeout=35, cancelled=cancelled)

        with patch("studio.qt_video_preview.subprocess.Popen", side_effect=record_process), \
                patch("studio.qt_video_preview.extract_frame", side_effect=slow_decode):
            controller.set_source(self.changing, self.ffmpeg, self.ffprobe)
            self.wait_for(lambda: len(processes) >= 2)
            started = time.monotonic()
            controller.close()
            self.assertLess(time.monotonic() - started, 1.5)
        self.assertFalse(controller._worker.isRunning())
        self.assertTrue(all(process.poll() is not None for process in processes))
        with self.assertRaisesRegex(ValueError, "超时"):
            _run_media([sys.executable, "-c", "import time; time.sleep(30)"], timeout=.1)
        with self.assertRaises(PreviewCancelled):
            _run_media([sys.executable, "-c", "print('ignored')"], timeout=1, cancelled=lambda: True)

    @unittest.skipUnless(sys.platform == "win32", "Windows process-tree cancellation")
    def test_cancel_kills_tool_wrapper_child_process(self):
        # Scoop and other tool launchers can spawn FFmpeg as a child. Exercise
        # the same process-tree cancellation with a real parent and child.
        import ctypes
        marker = self.folder / "child-pid.txt"
        marker.unlink(missing_ok=True)
        code = ("import subprocess,sys,time,pathlib; "
                "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
                f"target=pathlib.Path({json.dumps(str(marker))}); "
                "pending=target.with_suffix('.tmp'); "
                "pending.write_text(str(child.pid)); pending.replace(target); "
                "time.sleep(30)")
        controller = FramePreviewController(debounce_ms=0)

        def wrapper_decode(path, seconds, ffmpeg, *, cancelled):
            return _run_media([sys.executable, "-c", code], timeout=35, cancelled=cancelled)

        try:
            with patch("studio.qt_video_preview.extract_frame", side_effect=wrapper_decode):
                controller.set_source(self.changing, self.ffmpeg, self.ffprobe)
                self.wait_for(marker.is_file)
                child_pid = int(marker.read_text())
                controller.close()
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
            kernel.OpenProcess.restype = ctypes.c_void_p
            kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
            kernel.CloseHandle.argtypes = [ctypes.c_void_p]
            handle = kernel.OpenProcess(0x1000, False, child_pid)
            if handle:
                try:
                    exit_code = ctypes.c_uint32()
                    self.assertTrue(kernel.GetExitCodeProcess(handle, ctypes.byref(exit_code)))
                    self.assertNotEqual(exit_code.value, 259, "Child process remains active")
                finally:
                    kernel.CloseHandle(handle)
        finally:
            controller.close()
            marker.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
