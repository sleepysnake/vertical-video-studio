"""Decode source-video preview frames without platform multimedia codecs.

The controller owns one cancellable FFmpeg worker.  The GUI can compose its
cached PIL frame again when typography or placement changes, without decoding.
"""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import json
import math
from pathlib import Path
import stat
import subprocess
import sys
import threading
import time
from typing import Callable

from PIL import Image
from PySide6.QtCore import QCoreApplication, QObject, QThread, QTimer, Signal, Slot

from .edits import no_window


class PreviewCancelled(Exception):
    """A newer request or shutdown superseded this media operation."""


@dataclass(frozen=True)
class VideoInfo:
    duration: float
    fps: float = 25.0


class _WindowsProcessJob:
    """A Windows job reaps executable wrappers and their media-tool children."""
    def __init__(self, process: subprocess.Popen):
        import ctypes
        from ctypes import wintypes

        class BasicLimits(ctypes.Structure):
            _fields_ = [("process_time", ctypes.c_int64), ("job_time", ctypes.c_int64),
                        ("flags", wintypes.DWORD), ("min_working_set", ctypes.c_size_t),
                        ("max_working_set", ctypes.c_size_t), ("active_limit", wintypes.DWORD),
                        ("affinity", ctypes.c_size_t), ("priority", wintypes.DWORD),
                        ("scheduling", wintypes.DWORD)]

        class IoCounters(ctypes.Structure):
            _fields_ = [(name, ctypes.c_uint64) for name in
                        ("read_ops", "write_ops", "other_ops", "read_bytes", "write_bytes", "other_bytes")]

        class ExtendedLimits(ctypes.Structure):
            _fields_ = [("basic", BasicLimits), ("io", IoCounters),
                        ("process_memory", ctypes.c_size_t), ("job_memory", ctypes.c_size_t),
                        ("peak_process_memory", ctypes.c_size_t), ("peak_job_memory", ctypes.c_size_t)]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel.CreateJobObjectW.restype = wintypes.HANDLE
        kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self._kernel, self._handle = kernel, kernel.CreateJobObjectW(None, None)
        limits = ExtendedLimits()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self._handle:
            raise OSError(ctypes.get_last_error(), "CreateJobObject failed")
        if (not kernel.SetInformationJobObject(self._handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)) or
                not kernel.AssignProcessToJobObject(self._handle, int(process._handle))):
            error = ctypes.get_last_error()
            self.close()
            raise OSError(error, "AssignProcessToJobObject failed")

    def close(self) -> None:
        if self._handle:
            self._kernel.CloseHandle(self._handle)
            self._handle = None


def _run_media(command: list[str], *, timeout: float,
               cancelled: Callable[[], bool] = lambda: False) -> bytes:
    """Read binary pipes, periodically checking cancellation and a deadline."""
    if cancelled():
        raise PreviewCancelled()
    try:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   creationflags=no_window())
    except OSError as exc:
        raise ValueError(f"无法启动媒体工具：{exc}") from exc
    job = None
    deadline = time.monotonic() + timeout
    try:
        if sys.platform == "win32":
            job = _WindowsProcessJob(process)
        while True:
            if cancelled():
                raise PreviewCancelled()
            if time.monotonic() >= deadline:
                raise ValueError("读取视频预览超时，请检查视频文件或媒体工具")
            try:
                stdout, stderr = process.communicate(timeout=.08)
                break
            except subprocess.TimeoutExpired:
                continue
        if cancelled():
            raise PreviewCancelled()
        if process.returncode:
            detail = stderr.decode("utf-8", errors="replace").strip()
            # The last line normally contains the usable FFmpeg diagnostic.
            detail = detail.splitlines()[-1][:300] if detail else f"退出码 {process.returncode}"
            raise ValueError(f"无法读取视频：{detail}")
        return stdout
    finally:
        if job is not None:
            # Also closes any child left behind by a configured tool launcher.
            job.close()
        if process.poll() is None:
            process.kill()
        # communicate also drains pipes after cancellation and reaps the child.
        process.communicate()


def probe_video(path: Path, ffprobe: str = "ffprobe", *,
                cancelled: Callable[[], bool] = lambda: False) -> VideoInfo:
    raw = _run_media([str(ffprobe), "-v", "error", "-select_streams", "v:0",
                      "-show_entries", "format=duration:stream=duration,avg_frame_rate",
                      "-of", "json", str(path)], timeout=10, cancelled=cancelled)
    try:
        data = json.loads(raw)
        stream = data.get("streams", [])[0]
        duration = float(data.get("format", {}).get("duration") or stream.get("duration") or 0)
        numerator, denominator = (float(part) for part in str(stream.get("avg_frame_rate", "25/1")).split("/"))
        fps = numerator / denominator if denominator else 25.0
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise ValueError("视频没有可读取的画面或时长信息") from exc
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("视频时长无效，无法显示预览")
    return VideoInfo(duration, fps if math.isfinite(fps) and fps > 0 else 25.0)


def extract_frame(path: Path, seconds: float, ffmpeg: str = "ffmpeg", *,
                  max_edge: int = 960,
                  cancelled: Callable[[], bool] = lambda: False) -> Image.Image:
    """Return an RGB frame with square pixels and a bounded display size."""
    edge = max(1, int(max_edge))
    # Account for sample aspect ratio while keeping both dimensions bounded.
    # Explicit setsar works with FFmpeg versions lacking scale's reset_sar.
    ratio = f"min(1,min({edge}/(iw*sar),{edge}/ih))"
    scale = (f"scale=w='max(1,round(iw*sar*{ratio}))':"
             f"h='max(1,round(ih*{ratio}))',setsar=1")
    raw = _run_media([str(ffmpeg), "-nostdin", "-v", "error", "-ss", f"{max(0, seconds):.6f}",
                      "-i", str(path), "-map", "0:v:0", "-frames:v", "1", "-vf", scale,
                      "-f", "image2pipe", "-vcodec", "png", "pipe:1"],
                     timeout=15, cancelled=cancelled)
    if not raw:
        raise ValueError("该时间点没有可读取的视频画面")
    try:
        with Image.open(BytesIO(raw)) as frame:
            frame.load()
            return frame.convert("RGB")
    except (ValueError, OSError) as exc:
        raise ValueError("媒体工具返回的预览画面无法读取") from exc


def fit_frame(frame: Image.Image, slot_size: tuple[int, int]) -> Image.Image:
    """Scale to fill and center-crop, matching the export's video placement."""
    width, height = (int(value) for value in slot_size)
    if width <= 0 or height <= 0 or frame.width <= 0 or frame.height <= 0:
        raise ValueError("视频预览区域的宽高必须大于 0")
    ratio = max(width / frame.width, height / frame.height)
    resized = frame.resize((max(width, round(frame.width * ratio)),
                            max(height, round(frame.height * ratio))), Image.Resampling.LANCZOS)
    left, top = (resized.width - width) // 2, (resized.height - height) // 2
    return resized.crop((left, top, left + width, top + height))


@dataclass(frozen=True)
class _Request:
    generation: int
    number: int
    path: Path
    ffmpeg: str
    ffprobe: str
    seconds: float


class _FrameWorker(QThread):
    frame = Signal(int, int, object, float)
    duration = Signal(int, float)
    status = Signal(int, int, str)

    def __init__(self, parent: QObject):
        super().__init__(parent)
        self._condition = threading.Condition()
        self._pending: _Request | None = None
        self._generation = 0
        self._number = 0
        self._stopping = False

    def invalidate(self, generation: int, number: int) -> None:
        with self._condition:
            self._generation, self._number = generation, number
            self._pending = None
            self._condition.notify_all()

    def submit(self, request: _Request) -> None:
        with self._condition:
            if not self._stopping:
                self._generation, self._number = request.generation, request.number
                self._pending = request
                self._condition.notify_all()

    def stop(self) -> None:
        with self._condition:
            self._stopping = True
            self._pending = None
            self._condition.notify_all()

    def _cancelled(self, generation: int, number: int | None = None) -> bool:
        with self._condition:
            return (self._stopping or generation != self._generation or
                    (number is not None and number != self._number))

    def run(self) -> None:
        cached_generation = -1
        info: VideoInfo | None = None
        while True:
            with self._condition:
                while self._pending is None and not self._stopping:
                    self._condition.wait()
                if self._stopping:
                    return
                request, self._pending = self._pending, None
            assert request is not None
            try:
                if cached_generation != request.generation:
                    self.status.emit(request.generation, request.number, "正在读取视频信息…")
                    info = probe_video(request.path, request.ffprobe,
                                       cancelled=lambda: self._cancelled(request.generation))
                    cached_generation = request.generation
                    self.duration.emit(request.generation, info.duration)
                if self._cancelled(request.generation, request.number):
                    continue
                assert info is not None
                # The exact duration is past the final frame, so seek one frame back.
                seconds = min(max(0, request.seconds), max(0, info.duration - 1 / info.fps))
                self.status.emit(request.generation, request.number, "正在读取视频画面…")
                frame = extract_frame(request.path, seconds, request.ffmpeg,
                                      cancelled=lambda: self._cancelled(request.generation, request.number))
                if not self._cancelled(request.generation, request.number):
                    self.frame.emit(request.generation, request.number, frame, seconds)
                    self.status.emit(request.generation, request.number, "视频画面已更新")
            except PreviewCancelled:
                continue
            except Exception as exc:
                if not self._cancelled(request.generation, request.number):
                    self.status.emit(request.generation, request.number, f"视频预览不可用：{exc}")


class FramePreviewController(QObject):
    frame_ready = Signal(object, float)
    duration_ready = Signal(float)
    status_changed = Signal(str)

    def __init__(self, parent: QObject | None = None, *, debounce_ms: int = 120):
        super().__init__(parent)
        self.frame: Image.Image | None = None
        self.frame_seconds = 0.0
        self.duration = 0.0
        self._key: tuple[str, str, str, int | None, int | None] | None = None
        self._path: Path | None = None
        self._ffmpeg, self._ffprobe = "ffmpeg", "ffprobe"
        self._generation = self._number = 0
        self._seconds = 0.0
        self._closed = False
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(max(0, debounce_ms))
        self._timer.timeout.connect(self._submit)
        self._worker = _FrameWorker(self)
        self._worker.frame.connect(self._on_frame)
        self._worker.duration.connect(self._on_duration)
        self._worker.status.connect(self._on_status)
        if parent is not None:
            parent.destroyed.connect(self.close)
        application = QCoreApplication.instance()
        if application is not None:
            application.aboutToQuit.connect(self.close)

    def set_source(self, path: Path | None, ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe") -> None:
        if self._closed:
            return
        canonical = Path(path).expanduser().resolve() if path else None
        file_info, file_error = None, ""
        if canonical is not None:
            try:
                file_info = canonical.stat()
                if not stat.S_ISREG(file_info.st_mode):
                    file_error = "所选路径不是视频文件"
            except (FileNotFoundError, NotADirectoryError):
                file_error = "视频文件不存在"
            except OSError as exc:
                file_error = f"无法读取视频文件：{exc}"
        key = (str(canonical), str(ffmpeg), str(ffprobe),
               file_info.st_mtime_ns if file_info else None,
               file_info.st_size if file_info else None) if canonical else None
        if key == self._key:
            return
        self._key, self._path = key, canonical
        self._ffmpeg, self._ffprobe = str(ffmpeg), str(ffprobe)
        self._generation += 1
        self._number += 1
        self._timer.stop()
        self._worker.invalidate(self._generation, self._number)
        self.frame, self.frame_seconds, self.duration, self._seconds = None, 0.0, 0.0, 0.0
        self.duration_ready.emit(0.0)
        if canonical is None:
            self.status_changed.emit("请选择视频以预览画面")
            return
        if file_error:
            self.status_changed.emit(f"视频预览不可用：{file_error}")
            return
        self.status_changed.emit("正在读取视频信息…")
        self._submit()

    def seek(self, seconds: float) -> None:
        if self._closed or self._path is None or not self._path.is_file():
            return
        if not math.isfinite(seconds):
            return
        self._seconds = max(0.0, float(seconds))
        if self.duration > 0:
            self._seconds = min(self._seconds, self.duration)
        self._number += 1
        # Invalidate immediately, so a pending old frame cannot win during debounce.
        self._worker.invalidate(self._generation, self._number)
        self._timer.start()

    @Slot()
    def _submit(self) -> None:
        if self._closed or self._path is None:
            return
        if not self._worker.isRunning():
            self._worker.start()
        self._worker.submit(_Request(self._generation, self._number, self._path,
                                    self._ffmpeg, self._ffprobe, self._seconds))

    @Slot(int, int, object, float)
    def _on_frame(self, generation: int, number: int, frame: Image.Image, seconds: float) -> None:
        if self._closed or generation != self._generation or number != self._number:
            return
        self.frame, self.frame_seconds = frame, seconds
        self.frame_ready.emit(frame, seconds)

    @Slot(int, float)
    def _on_duration(self, generation: int, duration: float) -> None:
        if not self._closed and generation == self._generation:
            self.duration = duration
            self.duration_ready.emit(duration)

    @Slot(int, int, str)
    def _on_status(self, generation: int, number: int, status: str) -> None:
        if not self._closed and generation == self._generation and number == self._number:
            self.status_changed.emit(status)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._timer.stop()
        self._worker.stop()
        # The worker checks cancellation every 80 ms and reaps its child first.
        self._worker.wait()
        self.frame = None

    def __del__(self):
        # Explicit close in the owning window is preferred. This also covers
        # controllers used without a QObject parent in small scripts/tests.
        try:
            self.close()
        except (RuntimeError, AttributeError):
            pass
