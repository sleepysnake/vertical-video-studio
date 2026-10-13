"""User edits use final-video times and never alter the automatic source plan."""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

from .edits import keep_intervals, output_duration, retime_captions


def _base_key(data: dict, normalize_path: bool) -> str:
    source = dict(data.get("edit_plan", {}).get("source", {}))
    if normalize_path and source.get("path"):
        # Windows TEMP can use an 8.3 alias while the renderer stores the
        # resolved long path. Both refer to the same unchanged source file.
        source["path"] = str(Path(source["path"]).resolve())
    base = {"source": source, "cuts": data["edit_plan"]["cuts"],
            "captions": [{"start": c["start"], "end": c["end"], "text": c["text"]}
                         for c in retime_captions(data["captions"], data["edit_plan"]["cuts"])]}
    return hashlib.sha256(json.dumps(base, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def base_key(data: dict) -> str:
    return _base_key(data, normalize_path=True)


def validate_edits(captions: list[dict], duration: float) -> list[dict]:
    if not captions:
        raise ValueError("至少需要一条字幕")
    result, previous = [], 0.0
    for number, item in enumerate(captions, 1):
        try:
            start, end = float(item["start"]), float(item["end"])
            text = str(item["text"]).strip()
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError(f"第 {number} 条字幕时间需填写秒数") from exc
        if not math.isfinite(start) or not math.isfinite(end) or not text:
            raise ValueError(f"第 {number} 条字幕文字为空或时间无效")
        if start < previous - .001 or start < 0 or end - start <= .04 or end > duration + .001:
            raise ValueError(f"第 {number} 条字幕时间重叠或超出成片范围")
        result.append({"start": round(start, 3), "end": round(end, 3), "text": text, "manual_edited": True})
        previous = end
    return result


def save_edits(folder: Path, data: dict, captions: list[dict]) -> None:
    duration = output_duration(data["edit_plan"]["cuts"], data["edit_plan"]["source_duration"])
    checked = validate_edits(captions, duration)
    target = folder / "caption-overrides.json"
    if target.is_file():
        target.with_suffix(".json.bak").write_bytes(target.read_bytes())
    pending = target.with_suffix(".tmp")
    pending.write_text(json.dumps({"version": 1, "base_sha256": base_key(data),
                        "saved_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                        "captions": checked}, ensure_ascii=False, indent=2), encoding="utf-8")
    pending.replace(target)


def load_edits(folder: Path, data: dict) -> list[dict] | None:
    target = folder / "caption-overrides.json"
    if not target.is_file():
        return None
    value = json.loads(target.read_text(encoding="utf-8"))
    if value.get("version") != 1 or value.get("base_sha256") not in {
            base_key(data), _base_key(data, normalize_path=False)}:
        return None
    return validate_edits(value["captions"], output_duration(data["edit_plan"]["cuts"],
                                                          data["edit_plan"]["source_duration"]))


def to_source_times(captions: list[dict], cuts: list[dict], duration: float) -> list[dict]:
    intervals = keep_intervals(cuts, duration)
    def source_time(t: float, is_end: bool) -> float:
        cursor = 0.0
        for start, end in intervals:
            length = end - start
            if t < cursor + length - .0001 or (is_end and t <= cursor + length + .0001):
                return round(start + max(0, min(length, t - cursor)), 3)
            cursor += length
        return intervals[-1][1]
    return [{**c, "start": source_time(c["start"], False), "end": source_time(c["end"], True)}
            for c in captions]
