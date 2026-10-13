"""Sample the source picture and brighten only clips below a chosen baseline."""
from __future__ import annotations

from pathlib import Path
import statistics
import subprocess

from .edits import no_window


def choose_brightness(measured: int, target: int = 100, tolerance: int = 12) -> dict:
    """The score is the upper-center median on the 0..255 luma scale."""
    if not 40 <= target <= 200:
        raise ValueError("明暗标准值应在 40 到 200 之间")
    if not 0 <= tolerance <= 30:
        raise ValueError("明暗容差应在 0 到 30 之间")
    if not 0 <= measured <= 255:
        raise ValueError("测得的亮度无效")
    dark = measured < target - tolerance
    raw_offset = (target - measured) / 255 if dark else 0.0
    offset = round(min(.18, raw_offset), 3)
    return {"measured": measured, "target": target, "tolerance": tolerance,
            "action": "brighten" if offset else "keep", "offset": offset,
            "limited": raw_offset > .18}


def sample_brightness(video: Path, duration: float, ffmpeg: str) -> dict:
    """Decode three distributed frames; ignore borders to focus on the subject area."""
    positions = [duration / 2] if duration < 3 else [duration * part for part in (.15, .5, .85)]
    scores = []
    for position in positions:
        command = [str(ffmpeg), "-hide_banner", "-loglevel", "error", "-ss", f"{position:.3f}",
                   "-i", str(video), "-map", "0:v:0", "-frames:v", "1", "-an",
                   "-vf", "scale=160:90:flags=bilinear,format=gray", "-f", "rawvideo", "-"]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                timeout=30, creationflags=no_window())
        if result.returncode or len(result.stdout) != 160 * 90:
            raise RuntimeError(f"视频亮度取样失败：{result.stderr.decode('utf-8', errors='replace')[-300:]}")
        center = sorted(result.stdout[y * 160 + x] for y in range(12, 60)
                        for x in range(32, 128))
        scores.append(center[len(center) // 2])
    return {"samples": scores, "measured": round(statistics.median(scores)),
            "metric": "画面上方中心区域亮度中位数；0–255"}


def analyze_brightness(video: Path, duration: float, ffmpeg: str,
                       target: int = 100, tolerance: int = 12) -> dict:
    sample = sample_brightness(video, duration, ffmpeg)
    return {**sample, **choose_brightness(sample["measured"], target, tolerance)}
