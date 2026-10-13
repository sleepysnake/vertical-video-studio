"""User-facing subtitle appearance, independent of ASR and editing caches."""
from dataclasses import asdict, dataclass
from pathlib import Path
import re


@dataclass(frozen=True)
class CaptionStyle:
    font_size: int = 0  # 0 follows the video region width; other values are pixels.
    text_color: str = "#FFFFFF"
    outline_color: str = "#000000"
    outline_width: int = 3
    font_path: str = ""

    def validate(self):
        if not isinstance(self.font_size, int) or self.font_size != 0 and not 12 <= self.font_size <= 120:
            raise ValueError("字幕字号请选择自动或 12–120 像素")
        for label, color in (("字幕颜色", self.text_color), ("字幕描边颜色", self.outline_color)):
            if not isinstance(color, str) or not re.fullmatch(r"#[0-9A-Fa-f]{6}", color):
                raise ValueError(f"{label}必须为 #RRGGBB")
        if not isinstance(self.outline_width, int) or not 0 <= self.outline_width <= 12:
            raise ValueError("字幕描边粗细应在 0–12 像素之间")
        if not isinstance(self.font_path, str):
            raise ValueError("字幕字体配置应为文件路径")
        if self.font_path and not Path(self.font_path).is_file():
            raise ValueError("字幕字体文件不存在，请重新选择")

    def snapshot(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise ValueError("字幕样式配置应为对象")
        try:
            result = cls(**value)
        except TypeError as exc:
            raise ValueError("字幕样式配置包含未知字段") from exc
        result.validate()
        return result


def ass_color(value):
    return "&H00" + value[5:7] + value[3:5] + value[1:3]
