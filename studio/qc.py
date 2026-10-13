"""Post-render technical checks and review signals."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

from .edits import keep_intervals, no_window, output_duration


def inspect_output(path: Path, source_duration: float, cuts: list[dict],
                   ffprobe: str, *, max_removal_ratio: float = .40,
                   min_retained_fragment: float = .18,
                   policy_signals: bool = True) -> dict:
    result = subprocess.run([str(ffprobe), "-v", "error", "-show_entries",
                             "format=duration,size:stream=codec_type,width,height",
                             "-of", "json", str(path)], capture_output=True, text=True,
                            encoding="utf-8", timeout=30, creationflags=no_window())
    if result.returncode:
        raise RuntimeError(f"成片质量检测失败：{result.stderr[-400:]}")
    try:
        data = json.loads(result.stdout)
        streams = data["streams"]
        actual = float(data["format"]["duration"])
        expected = output_duration(cuts, source_duration)
    except (KeyError, ValueError, TypeError) as exc:
        raise RuntimeError("成片元数据不完整，无法确认导出成功") from exc
    if not any(stream.get("codec_type") == "video" for stream in streams):
        raise RuntimeError("成片缺少视频流")
    if not any(stream.get("codec_type") == "audio" for stream in streams):
        raise RuntimeError("成片缺少音频流")
    if abs(actual - expected) > max(0.3, expected * .01):
        raise RuntimeError(f"成片时长异常：预期 {expected:.2f} 秒，实际 {actual:.2f} 秒")
    findings = []
    ratio = (source_duration - expected) / source_duration if source_duration else 0
    if policy_signals and ratio > max_removal_ratio:
        findings.append(f"总体删除比例超过 {max_removal_ratio:.0%}，建议复核")
    if policy_signals and any(end - start < min_retained_fragment for start, end in keep_intervals(cuts, source_duration)):
        findings.append(f"存在不足 {min_retained_fragment:.2f} 秒的保留片段")
    return {"status": "建议复核" if findings else "通过", "findings": findings,
            "duration": round(actual, 3), "expected_duration": round(expected, 3),
            "removal_ratio": round(ratio, 3), "size_bytes": path.stat().st_size}
