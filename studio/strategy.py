"""Validated, versioned policy loaded once for each batch."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys


@dataclass(frozen=True)
class Strategy:
    version: int = 1
    silence_seconds: float = 1.0
    silence_margin: float = 0.12
    max_single_model_cut: float = 12.0
    max_removal_ratio: float = 0.40
    min_retained_fragment: float = 0.18
    min_model_cut_seconds: float = 0.20
    max_subtitle_cleanup_seconds: float = 1.0
    abandoned_pause_seconds: float = 0.55
    max_abandoned_start_seconds: float = 3.0

    @classmethod
    def from_dict(cls, value: dict) -> "Strategy":
        if not isinstance(value, dict) or value.get("version") != 1:
            raise ValueError("策略文件需要 version: 1")
        allowed = set(cls.__dataclass_fields__)
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"策略包含未知字段：{', '.join(sorted(unknown))}")
        result = cls(**value)
        if not (0.5 <= result.silence_seconds <= 30):
            raise ValueError("silence_seconds 应在 0.5 到 30 秒之间")
        if not (0 <= result.silence_margin < result.silence_seconds / 2):
            raise ValueError("silence_margin 超出静音区间")
        if not (0.1 <= result.max_single_model_cut <= 120):
            raise ValueError("max_single_model_cut 无效")
        if not (0 < result.max_removal_ratio < 1):
            raise ValueError("max_removal_ratio 应在 0 到 1 之间")
        if not (0.02 <= result.min_retained_fragment <= 3):
            raise ValueError("min_retained_fragment 无效")
        if not (0.02 <= result.min_model_cut_seconds <= 1):
            raise ValueError("min_model_cut_seconds 应在 0.02 到 1 秒之间")
        if not (0.02 <= result.max_subtitle_cleanup_seconds <= 1):
            raise ValueError("max_subtitle_cleanup_seconds 应在 0.02 到 1 秒之间")
        if not (0.55 <= result.abandoned_pause_seconds <= 5):
            raise ValueError("abandoned_pause_seconds 应在 0.55 到 5 秒之间")
        if not (0.2 <= result.max_abandoned_start_seconds <= 5):
            raise ValueError("max_abandoned_start_seconds 应在 0.2 到 5 秒之间")
        return result

    def snapshot(self) -> dict:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.snapshot(), sort_keys=True).encode()).hexdigest()[:12]


def strategy_path() -> Path:
    root = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent.parent
    return root / "config" / "strategy.json"


def load_strategy(path: Path | None = None) -> Strategy:
    selected = path or strategy_path()
    if not selected.exists():
        return Strategy()
    try:
        return Strategy.from_dict(json.loads(selected.read_text(encoding="utf-8-sig")))
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(f"策略文件无效：{selected}：{exc}") from exc
