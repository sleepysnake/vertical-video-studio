"""Topic title appearance shared by the desktop preview and exported video."""
from dataclasses import asdict, dataclass
import re

from PIL import Image, ImageColor, ImageDraw, ImageFilter, ImageFont


TOPIC_EFFECTS = (
    ("普通文字", "plain"),
    ("描边文字", "outline"),
    ("柔和阴影", "shadow"),
    ("轻微立体", "raised"),
    ("双色渐变", "gradient"),
)
TOPIC_BACKGROUNDS = (
    ("无背景", "none"),
    ("矩形背景", "rectangle"),
    ("圆角背景", "rounded"),
    ("胶囊背景", "pill"),
    ("斜边背景", "slant"),
)


@dataclass(frozen=True)
class TopicStyle:
    font_size: int = 0
    effect: str = "outline"
    background: str = "none"
    background_color: str = "#164D80"
    background_opacity: int = 85
    accent_color: str = "#FFD54A"
    outline_color: str = ""
    outline_width: int = 2

    def validate(self):
        if type(self.font_size) is not int or self.font_size != 0 and not 12 <= self.font_size <= 300:
            raise ValueError("主题字号请选择自动或 12–300 像素")
        if not isinstance(self.effect, str) or self.effect not in {value for _, value in TOPIC_EFFECTS}:
            raise ValueError("未知的主题文字效果")
        if not isinstance(self.background, str) or self.background not in {value for _, value in TOPIC_BACKGROUNDS}:
            raise ValueError("未知的主题背景样式")
        for label, color in (("主题背景颜色", self.background_color),
                             ("主题渐变颜色", self.accent_color)):
            if not isinstance(color, str) or not re.fullmatch(r"#[0-9A-Fa-f]{6}", color):
                raise ValueError(f"{label}必须为 #RRGGBB")
        if self.outline_color and (not isinstance(self.outline_color, str) or
                                   not re.fullmatch(r"#[0-9A-Fa-f]{6}", self.outline_color)):
            raise ValueError("主题描边颜色必须为空或 #RRGGBB")
        if not isinstance(self.outline_color, str):
            raise ValueError("主题描边颜色必须为空或 #RRGGBB")
        if type(self.background_opacity) is not int or not 0 <= self.background_opacity <= 100:
            raise ValueError("主题背景不透明度应在 0–100% 之间")
        if type(self.outline_width) is not int or not 0 <= self.outline_width <= 12:
            raise ValueError("主题描边粗细应在 0–12 像素之间")

    def snapshot(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise ValueError("主题样式配置应为对象")
        try:
            result = cls(**value)
        except TypeError as exc:
            raise ValueError("主题样式配置包含未知字段") from exc
        result.validate()
        return result


def _wrap_topic(value, draw, chosen, max_width):
    """Honor explicit newlines and prefer word / punctuation breaks when present."""
    lines = []
    for paragraph in value.splitlines():
        remainder = paragraph.strip()
        if not remainder:
            continue
        while remainder:
            if draw.textlength(remainder, font=chosen) <= max_width:
                lines.append(remainder)
                break
            count = 0
            for end in range(1, len(remainder) + 1):
                if draw.textlength(remainder[:end], font=chosen) > max_width:
                    break
                count = end
            if not count:
                return None
            # A nearby intentional separator reads better than cutting an English word.
            choices = [index + 1 for index, character in enumerate(remainder[:count])
                       if character.isspace() or character in "，。！？；：、,.!?;:"]
            if choices and choices[-1] >= count * .65:
                count = choices[-1]
            lines.append(remainder[:count].strip())
            remainder = remainder[count:].strip()
    return lines


def topic_layout(value, box_size, font_path, style):
    """Return a layout; a requested size is never silently reduced."""
    style.validate()
    width, height = box_size
    if width <= 0 or height <= 0:
        raise ValueError("主题区域宽高必须大于零")
    draw = ImageDraw.Draw(Image.new("L", (1, 1)))
    initial = style.font_size or max(12, round(min(height * .55, width * .09)))
    sizes = (initial,) if style.font_size else range(initial, 11, -1)
    for size in sizes:
        chosen = ImageFont.truetype(str(font_path), size)
        stroke = style.outline_width if style.effect in ("outline", "raised", "gradient") else 0
        offset = max(1, round(size * .045))
        blur = max(1, round(size * .025))
        effects_pad = (offset + blur * 3 if style.effect == "shadow" else
                       offset + stroke if style.effect == "raised" else stroke)
        padding_x = max(6, round(size * .28)) if style.background != "none" else 0
        padding_y = max(4, round(size * .18)) if style.background != "none" else 0
        if style.background == "slant":
            padding_x += max(4, round(size * .18))
        available = width - (padding_x + effects_pad) * 2
        lines = _wrap_topic(value, draw, chosen, available)
        if not lines:
            continue
        bounds = [draw.textbbox((0, 0), line, font=chosen, anchor="lt") for line in lines]
        line_gap = max(2, round(size * .18))
        text_width = max(bound[2] - bound[0] for bound in bounds)
        text_height = sum(bound[3] - bound[1] for bound in bounds) + line_gap * (len(lines) - 1)
        card_width = text_width + (padding_x + effects_pad) * 2
        card_height = text_height + (padding_y + effects_pad) * 2
        if card_width > width or card_height > height:
            continue
        return {"font": chosen, "font_size": size, "lines": lines, "bounds": bounds,
                "line_gap": line_gap, "text_width": text_width, "text_height": text_height,
                "size": (card_width, card_height), "padding_x": padding_x,
                "padding_y": padding_y, "effects_pad": effects_pad, "stroke": stroke,
                "offset": offset, "blur": blur}
    if style.font_size:
        raise ValueError("主题文字按当前字号换行后仍超出主题区域，请减小字号或扩大主题区域")
    raise ValueError("主题文字无法放进主题区域，请缩短文字或扩大主题区域")


def draw_styled_topic(layer, value, box, text_color, font_path, style):
    """Draw a fitted title card, clipping neither letters nor decoration."""
    layout = topic_layout(value, box[2:], font_path, style)
    card = Image.new("RGBA", layout["size"])
    draw = ImageDraw.Draw(card)
    width, height = card.size
    color = ImageColor.getrgb(text_color)
    contrast = style.outline_color or ("#173863" if sum(color) > 380 else "#FFFFFF")
    if style.background != "none" and style.background_opacity:
        fill = (*ImageColor.getrgb(style.background_color), round(style.background_opacity * 2.55))
        if style.background == "rectangle":
            draw.rectangle((0, 0, width - 1, height - 1), fill=fill)
        elif style.background in ("rounded", "pill"):
            radius = height // 2 if style.background == "pill" else max(4, round(layout["font_size"] * .25))
            draw.rounded_rectangle((0, 0, width - 1, height - 1), radius=radius, fill=fill)
        else:
            slant = min(width // 4, max(4, round(layout["font_size"] * .18)))
            draw.polygon(((slant, 0), (width - 1, 0), (width - slant - 1, height - 1), (0, height - 1)), fill=fill)
    positions = []
    top = layout["padding_y"] + layout["effects_pad"]
    for line, bounds in zip(layout["lines"], layout["bounds"]):
        left = (width - (bounds[2] - bounds[0])) / 2 - bounds[0]
        positions.append((line, (left, top - bounds[1])))
        top += bounds[3] - bounds[1] + layout["line_gap"]
    if style.effect == "shadow":
        shadow = Image.new("RGBA", card.size)
        shade = ImageDraw.Draw(shadow)
        for line, (left, top) in positions:
            shade.text((left + layout["offset"], top + layout["offset"]), line,
                       font=layout["font"], anchor="lt", fill=(0, 0, 0, 190))
        card.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(layout["blur"])))
    elif style.effect == "raised":
        # An offset darker face gives a small extrusion, without changing the chosen front color.
        shade = tuple(round(component * .4) for component in color)
        for line, (left, top) in positions:
            draw.text((left + layout["offset"], top + layout["offset"]), line,
                      font=layout["font"], anchor="lt", fill=shade,
                      stroke_width=layout["stroke"], stroke_fill=contrast)
    for line, position in positions:
        draw.text(position, line, font=layout["font"], anchor="lt", fill=text_color,
                  stroke_width=layout["stroke"], stroke_fill=contrast)
    if style.effect == "gradient":
        mask = Image.new("L", card.size)
        brush = ImageDraw.Draw(mask)
        for line, position in positions:
            brush.text(position, line, font=layout["font"], anchor="lt", fill=255)
        gradient = Image.new("RGBA", card.size)
        brush = ImageDraw.Draw(gradient)
        accent = ImageColor.getrgb(style.accent_color)
        start = layout["padding_y"] + layout["effects_pad"]
        for row in range(height):
            mix = min(1, max(0, (row - start) / max(1, layout["text_height"] - 1)))
            fill = tuple(round(a + (b - a) * mix) for a, b in zip(color, accent))
            brush.line((0, row, width, row), fill=(*fill, 255))
        gradient.putalpha(mask)
        card.alpha_composite(gradient)
    x, y, area_width, area_height = box
    layer.alpha_composite(card, (x + (area_width - width) // 2, y + (area_height - height) // 2))
    return layout
