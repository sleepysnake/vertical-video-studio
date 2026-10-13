from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path
import re


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".m4v", ".avi", ".webm"}
DEFAULT_ASR_URL = "https://dashscope.aliyuncs.com/api/v1/services/audio/asr/transcription"
DEFAULT_UPLOAD_URL = "https://dashscope.aliyuncs.com/api/v1/uploads"
DEFAULT_TASK_URL = "https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}"
DEFAULT_TEXT_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
DEFAULT_ASR_MODEL = "qwen-audio-3.1-asr-flash-filetrans"
DEFAULT_TEXT_MODEL = "qwen-plus"
ASR_GUIDE = "https://docs.bailian.console.aliyun.com/zh/model-studio/non-realtime-speech-recognition-user-guide"


@dataclass(frozen=True)
class Person:
    organization: str
    title: str
    name: str

    def validate(self) -> None:
        for label, value in (("工作单位", self.organization), ("头衔", self.title), ("姓名", self.name)):
            if not value.strip():
                raise ValueError(f"{label}不能为空")
            if len(value.strip()) > 80:
                raise ValueError(f"{label}过长：{value[:30]}")


@dataclass(frozen=True)
class VideoItem:
    path: Path
    person: Person
    clip_id: str = ""
    topic: str = ""
    nameplate_style: str = ""
    brightness: str = ""


def parse_batch_filename(path: Path) -> Person:
    """Accept single underscores, plus legacy double underscores, as field separators."""
    parts = batch_filename_parts(path)
    if len(parts) not in (3, 4) or any(not part for part in parts):
        raise ValueError("文件名必须为 工作单位_头衔_姓名[_视频编号].mp4")
    if len(parts) == 4 and len(parts[3]) > 80:
        raise ValueError("视频编号过长")
    person = Person(*parts[:3])
    person.validate()
    return person


def batch_filename_parts(path: Path) -> list[str]:
    separator = "__" if "__" in path.stem else "_"
    return [part.strip() for part in path.stem.split(separator)]


def scan_folder(folder: Path) -> tuple[list[VideoItem], list[tuple[Path, str]]]:
    if not folder.is_dir():
        raise ValueError("请选择存在的视频文件夹")
    valid: list[VideoItem] = []
    invalid: list[tuple[Path, str]] = []
    manifest: dict[str, dict] = {}
    manifest_path = folder / "batch.csv"
    if manifest_path.is_file():
        with manifest_path.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                name = str(row.get("file", "")).strip()
                if not name or Path(name).name != name or name.casefold() in manifest:
                    raise ValueError(f"batch.csv 的 file 为空、重复或包含路径：{name}")
                manifest[name.casefold()] = row
    paths = [path for path in sorted(folder.iterdir(), key=lambda item: item.name.casefold())
             if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS]
    stem_counts: dict[str, int] = {}
    for path in paths:
        key = path.stem.casefold()
        stem_counts[key] = stem_counts.get(key, 0) + 1
    for path in paths:
        try:
            row = manifest.get(path.name.casefold())
            if row and all(str(row.get(key, "")).strip() for key in ("organization", "title", "name")):
                person = Person(*(str(row[key]).strip() for key in ("organization", "title", "name")))
                person.validate()
                clip_id = str(row.get("clip_id", "")).strip()
            else:
                person = parse_batch_filename(path)
                parts = batch_filename_parts(path)
                clip_id = parts[3].strip() if len(parts) == 4 else ""
            if stem_counts[path.stem.casefold()] > 1:
                clip_id = f"{clip_id}-{path.suffix[1:].lower()}" if clip_id else path.suffix[1:].lower()
            valid.append(VideoItem(path, person, clip_id,
                                   str((row or {}).get("topic", "")).strip(),
                                   str((row or {}).get("nameplate_style", "")).strip(),
                                   str((row or {}).get("brightness", "")).strip()))
        except ValueError as exc:
            invalid.append((path, str(exc)))
    unique: list[VideoItem] = []
    used: set[tuple[str, str, str, str]] = set()
    for item in valid:
        base = item.clip_id or item.path.suffix[1:].lower()
        clip_id = item.clip_id
        number = 2
        identity = (item.person.organization.casefold(), item.person.title.casefold(),
                    item.person.name.casefold())
        key = (*identity, clip_id.casefold())
        while key in used:
            clip_id = f"{base}-{number}"
            number += 1
            key = (*identity, clip_id.casefold())
        used.add(key)
        unique.append(VideoItem(item.path, item.person, clip_id, item.topic,
                                item.nameplate_style, item.brightness))
    return unique, invalid


def parse_slot(value: str) -> tuple[int, int, int, int] | None:
    if not value.strip():
        return None
    parts = re.split(r"[,，\s]+", value.strip())
    if len(parts) != 4:
        raise ValueError("视频区域请填写 x,y,宽,高，例如 0,300,1080,608")
    try:
        slot = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise ValueError("视频区域只能填写整数") from exc
    return slot  # type: ignore[return-value]


def slot_from_display_rect(image_size: tuple[int, int], display_size: tuple[int, int],
                           start: tuple[int, int], end: tuple[int, int]) -> tuple[int, int, int, int]:
    """Convert a preview drag to source pixels without rounding past the edge."""
    image_width, image_height = image_size
    display_width, display_height = display_size
    if min(image_width, image_height, display_width, display_height) <= 0:
        raise ValueError("模板或预览尺寸无效")

    def edge(value: int, shown: int, original: int) -> int:
        return max(0, min(original, round(max(0, min(shown, value)) * original / shown)))

    x0, x1 = sorted((edge(start[0], display_width, image_width),
                     edge(end[0], display_width, image_width)))
    y0, y1 = sorted((edge(start[1], display_height, image_height),
                     edge(end[1], display_height, image_height)))
    return x0, y0, x1 - x0, y1 - y0
