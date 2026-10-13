"""Editable semantic instructions, frozen for the duration of each batch."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from .strategy import strategy_path


@dataclass(frozen=True)
class PromptSet:
    edit: str
    caption: str
    # Legacy direct construction can omit the additional model phase. Normal
    # application loading always supplies the editable review prompt below.
    review: str = ""

    def snapshot(self) -> dict[str, str]:
        return {"edit": self.edit, "review": self.review, "caption": self.caption}

    def digest(self) -> str:
        payload = json.dumps(self.snapshot(), ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def prompts_dir() -> Path:
    return strategy_path().parent


def load_prompts(directory: Path | None = None) -> PromptSet:
    root = directory if directory is not None else prompts_dir()
    values = {}
    required = {"edit": ("cuts", "uncertain", "start_word", "end_word"),
                "caption": ("segments", "changes", "uncertain", "start_token", "end_token"),
                "review": ("cuts", "pause_compressions", "display_only_suggestions",
                           "uncertain", "start_word", "end_word")}
    for stage, fields in required.items():
        path = root / f"{stage}_prompt.txt"
        if stage == "review" and directory is not None and not path.exists():
            # Existing custom prompt folders contained only edit/caption. Use
            # the current published review instructions for that legacy case.
            path = prompts_dir() / "review_prompt.txt"
        try:
            value = path.read_text(encoding="utf-8-sig").strip()
        except OSError as exc:
            raise ValueError(f"提示词文件无法读取：{path}：{exc}") from exc
        if len(value) < 40 or len(value) > 20000:
            raise ValueError(f"提示词长度应在 40–20000 字符：{path}")
        missing = [field for field in fields if field not in value]
        if missing:
            raise ValueError(f"提示词缺少输出字段 {', '.join(missing)}：{path}")
        values[stage] = value
    return PromptSet(**values)
