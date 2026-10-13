"""Compose a landscape talking-head clip into a static portrait template."""

import argparse
from fractions import Fraction
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
import sys

from .edits import keep_intervals, output_duration
from .edits import no_window
from .caption_style import CaptionStyle, ass_color
from .topic_style import TopicStyle, draw_styled_topic
from .captions import STRONG_STARTS

try:
    from PIL import Image, ImageColor, ImageDraw, ImageFont
except ImportError as exc:
    raise SystemExit("Pillow is required: python -m pip install pillow") from exc

PUNCT = re.compile(r"[，。！？、；：,.!?;:…“”‘’\"'（）()《》\[\]]")
BAD_START = set("的了着过呢啊吧吗呀")

# Stable IDs retain compatibility with existing settings and batch.csv files.
NAMEPLATE_STYLES = (
    ("经典蓝色", "classic"),
    ("简洁白字", "ass_clean"),
    ("深色底条", "ass_dark"),
    ("蓝色斜边", "broadcast_slant"),
    ("青绿渐变", "teal_gradient"),
    ("深蓝金线", "navy_gold"),
    ("浅色简约", "light_card"),
)
NAMEPLATE_STYLE_IDS = frozenset(value for _, value in NAMEPLATE_STYLES)


def probe(path, ffprobe_path="ffprobe"):
    result = subprocess.run(
        [str(ffprobe_path), "-v", "error", "-show_entries", "format=duration:stream=codec_type,width,height", "-of", "json", str(path)],
        capture_output=True, text=True, encoding="utf-8", check=True,
        creationflags=no_window(),
    )
    data = json.loads(result.stdout)
    video = next((stream for stream in data.get("streams", []) if stream.get("codec_type") == "video"), None)
    if not video:
        raise ValueError("输入文件没有视频画面")
    return float(data["format"]["duration"]), int(video["width"]), int(video["height"])


def _cut_streams(path, ffprobe_path):
    """Only cut renders need to inspect audio availability and source cadence."""
    result = subprocess.run(
        [str(ffprobe_path), "-v", "error", "-show_entries",
         "stream=codec_type,avg_frame_rate,r_frame_rate", "-of", "json", str(path)],
        capture_output=True, text=True, encoding="utf-8", check=True,
        creationflags=no_window(),
    )
    streams = json.loads(result.stdout).get("streams", [])
    video = next(stream for stream in streams if stream.get("codec_type") == "video")
    rate = None
    for key in ("avg_frame_rate", "r_frame_rate"):
        try:
            candidate = Fraction(video.get(key, "0/0"))
            if candidate > 0:
                rate = str(candidate)
                break
        except (ValueError, ZeroDivisionError):
            continue
    return any(stream.get("codec_type") == "audio" for stream in streams), rate


def _cut_filters(cuts, duration, has_audio, frame_rate=None):
    """Remove exact source time without padding audio to rounded frame lengths.

    A combined A/V concat extends each audio segment to its rounded video
    duration. Across many cuts that padding accumulates and moves speech away
    from the caption timeline. Audio is therefore concatenated independently;
    retained video frames keep their source times minus completed deletions.
    """
    keep_intervals(cuts, duration)  # Validate ordered, disjoint source intervals.
    excluded = "+".join(
        f"gte(t,{float(cut['start']):.6f})*lt(t,{float(cut['end']):.6f})" for cut in cuts)
    removed = "+".join(
        f"if(gte(T,{float(cut['end']):.6f}),"
        f"{float(cut['end']) - float(cut['start']):.6f},0)" for cut in cuts)
    # Normalize before selection: a leading cut must not reset STARTPTS to the
    # first retained frame and then subtract that same deletion a second time.
    cadence = f",fps={frame_rate}:start_time=0" if frame_rate else ""
    filters = (f"[0:v:0]setpts=PTS-STARTPTS,select='not({excluded})',"
               f"setpts='PTS-({removed})/TB'{cadence}[cutv];")
    audio_map = None
    if has_audio:
        intervals = []
        cursor = 0.0
        for cut in cuts:
            start, end = float(cut["start"]), float(cut["end"])
            if start > cursor:
                intervals.append((cursor, start))
            cursor = end
        if duration > cursor:
            intervals.append((cursor, duration))
        # Even sub-frame retained spans contain audio samples and belong to
        # the same mathematical timeline as captions; do not discard them.
        for index, (start, end) in enumerate(intervals):
            filters += (f"[0:a:0]atrim=start={start:.6f}:end={end:.6f},"
                        f"asetpts=PTS-STARTPTS[clipa{index}];")
        filters += ("".join(f"[clipa{i}]" for i in range(len(intervals)))
                    + f"concat=n={len(intervals)}:v=0:a=1[cuta];")
        audio_map = "[cuta]"
    return filters, "[cutv]", audio_map


def find_slot(image):
    if image.mode not in ("RGBA", "LA") and "transparency" not in image.info:
        raise ValueError("图片不透明，请手动框选中间的视频区域")
    alpha = image.convert("RGBA").getchannel("A")
    box = alpha.point(lambda value: 255 if value < 24 else 0).getbbox()
    if not box:
        raise ValueError("没有找到透明的视频区域，请手动框选")
    x0, y0, x1, y1 = box
    slot = (x0, y0, x1 - x0, y1 - y0)
    if abs(slot[2] / slot[3] - 16 / 9) > 0.10:
        raise ValueError("透明区域不是清晰的 16:9 比例，请手动框选")
    return slot


def validate_box(box, size, label="区域"):
    if len(box) != 4:
        raise ValueError(f"{label}需要 x,y,宽,高")
    x, y, width, height = box
    if min(x, y) < 0 or width <= 0 or height <= 0 or x + width > size[0] or y + height > size[1]:
        raise ValueError(f"{label}超出模板边界")


def validate_slot(slot, size):
    x, y, width, height = slot
    canvas_width, canvas_height = size
    if min(slot) < 0 or width <= 0 or height <= 0 or x + width > canvas_width or y + height > canvas_height:
        raise ValueError(f"视频区域超出模板边界：模板 {canvas_width}×{canvas_height}，"
                         f"选区 {x},{y},{width},{height}。请重新框选")
    if abs(width / height - 16 / 9) > 0.10:
        raise ValueError("视频区域应接近 16:9，请重新框选")
    if width < canvas_width * 0.45:
        raise ValueError("视频区域过窄，请检查框选位置")


def read_captions(path, duration):
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict) or data.get("reviewed") is not True:
        raise ValueError("字幕尚未生成或结构校验未通过")
    captions = data.get("captions")
    if not isinstance(captions, list) or not captions:
        raise ValueError("字幕文件中没有字幕")
    previous_end = 0
    for index, item in enumerate(captions, 1):
        start, end, value = float(item["start"]), float(item["end"]), str(item["text"]).strip()
        if not value or (not item.get("manual_edited") and
                         (PUNCT.search(value) or len(value.replace(" ", "")) > 16)):
            raise ValueError(f"第 {index} 条字幕含标点、以虚词开头或为空：{value}")
        if start < previous_end - 0.03 or end <= start or start < 0 or end > duration + 0.25:
            raise ValueError(f"第 {index} 条字幕时间重叠或无效：{start} 到 {end}")
        previous_end = end
    return captions


def default_font():
    candidates = [
        Path("C:/Windows/Fonts/msyhbd.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("/System/Library/Fonts/PingFang.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"),
        Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc"),
        Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
    ]
    return next((path for path in candidates if path.exists()), None)


def font(path, size):
    return ImageFont.truetype(str(path), max(12, round(size)))


def fit_font(draw, value, path, initial_size, max_width):
    for size in range(round(initial_size), 11, -1):
        current = font(path, size)
        if draw.textbbox((0, 0), value, font=current, stroke_width=1)[2] <= max_width:
            return current
    raise ValueError(f"人物信息过长，无法放进姓名条：{value}")


def wrap_notice(draw, value, current_font, max_width):
    lines = []
    for paragraph in value.splitlines():
        line = ""
        for character in paragraph:
            trial = line + character
            if line and draw.textbbox((0, 0), trial, font=current_font, stroke_width=1)[2] > max_width:
                lines.append(line)
                line = character
            else:
                line = trial
        if line:
            lines.append(line)
    return lines


def hospital_and_department(organization):
    """Split the common 'hospital + department' batch field for the nameplate."""
    marker = "医院"
    end = organization.rfind(marker)
    if end >= 0 and end + len(marker) < len(organization):
        boundary = end + len(marker)
        return organization[:boundary], organization[boundary:]
    return organization, ""


def _draw_nameplate_default(layer, slot, name, organization, title, font_path, style, card_top):
    """One renderer supplies both the desktop preview and the exported overlay."""
    if style not in NAMEPLATE_STYLE_IDS:
        raise ValueError("未知的姓名条样式")
    x, y, width, height = slot
    left = x + round(width * .025)
    hospital, department = hospital_and_department(organization)
    second = f"{department} · {title}" if department else title
    values = (hospital, second, name)
    draw = ImageDraw.Draw(layer)
    if style == "classic":
        bar_gap, bar_height = round(width * .014), round(width * .047)
        max_bar_width = round(width * .37)
        for index, value in enumerate(values[:2]):
            chosen = fit_font(draw, value, font_path, width * .028, max_bar_width - bar_gap * 2)
            bar_y = card_top + index * (bar_height + round(width * .006))
            bar_width = min(max_bar_width, round(draw.textbbox((0, 0), value, font=chosen)[2] + bar_gap * 2))
            draw.rounded_rectangle((left - bar_gap // 2, bar_y, left + bar_width, bar_y + bar_height),
                                   radius=round(width * .006), fill=(20, 103, 190, 220))
            draw.text((left, bar_y + round(width * .004)), value, font=chosen,
                      fill="white", stroke_width=1, stroke_fill="#173863")
        chosen = fit_font(draw, name, font_path, width * .043, width * .28)
        name_y = card_top + 2 * (bar_height + round(width * .006)) + round(width * .007)
        name_width = max(round(width * .3), round(draw.textbbox((0, 0), name, font=chosen)[2] + width * .08))
        name_height = round(width * .068)
        draw.polygon([(left, name_y), (left + name_width, name_y),
                      (left + name_width - round(width * .027), name_y + name_height),
                      (left + round(width * .027), name_y + name_height)], fill=(255, 255, 255, 245))
        text_width = draw.textbbox((0, 0), name, font=chosen)[2]
        draw.text((left + (name_width - text_width) / 2, name_y + round(width * .005)),
                  name, font=chosen, fill="#185a9b")
        bottom = name_y + name_height
    else:
        # Draw at 3x for smooth slanted edges and rounded shapes at small sizes.
        scale = 3
        card_width, card_height = round(width * .37), round(width * .205)
        padding, slant = round(width * .016), round(width * .03)
        canvas = Image.new("RGBA", (card_width * scale, card_height * scale))
        brush = ImageDraw.Draw(canvas)
        mask = Image.new("L", canvas.size)
        shape = ImageDraw.Draw(mask)
        cw, ch = canvas.size
        if style == "broadcast_slant":
            shape.polygon([(0, 0), (cw - slant * scale, 0), (cw, ch), (0, ch)], fill=255)
        elif style not in ("ass_clean", "ass_dark"):
            shape.rounded_rectangle((0, 0, cw - 1, ch - 1), radius=round(width * .008 * scale), fill=255)
        if style in ("broadcast_slant", "teal_gradient"):
            colors = ((17, 48, 100), (24, 90, 160)) if style == "broadcast_slant" else ((9, 75, 79), (16, 117, 110))
            for column in range(cw):
                mix = column / max(1, cw - 1)
                color = tuple(round(a + (b - a) * mix) for a, b in zip(*colors))
                brush.line((column, 0, column, ch), fill=color)
            canvas.putalpha(mask)
        elif style == "navy_gold":
            brush.rectangle((0, 0, cw, ch), fill="#172C40")
            canvas.putalpha(mask)
        elif style == "light_card":
            brush.rectangle((0, 0, cw, ch), fill=(248, 250, 249, 248))
            canvas.putalpha(mask.point(lambda a: round(a * .97)))
        if style == "broadcast_slant":
            brush.polygon([(cw - slant * scale, 0), (cw - (slant + round(width * .01)) * scale, 0),
                           (cw - round(width * .01) * scale, ch), (cw, ch)], fill="#52D2EB")
        elif style == "teal_gradient":
            brush.rectangle((padding * scale, round(ch * .60), cw - padding * scale, round(ch * .60) + 2 * scale),
                            fill="#67DECF")
        elif style == "navy_gold":
            brush.rectangle((padding * scale, round(ch * .60), cw - padding * scale, round(ch * .60) + 2 * scale),
                            fill="#DDB56C")
        elif style == "light_card":
            brush.rectangle((padding * scale, 0, cw - padding * scale, 3 * scale), fill="#167C91")
        text_limit = card_width - padding * 2 - (slant if style == "broadcast_slant" else 0)
        for index, value in enumerate(values):
            fitted = fit_font(draw, value, font_path, width * (.043 if index == 2 else .028), text_limit)
            chosen = font(font_path, fitted.size * scale)
            row_y = round(width * (.018, .067, .133)[index]) * scale
            if style == "ass_dark":
                bbox = brush.textbbox((0, 0), value, font=chosen, anchor="lt")
                brush.rounded_rectangle((0, row_y - 3 * scale, bbox[2] + padding * 2 * scale,
                                         row_y + bbox[3] + 5 * scale), radius=3 * scale, fill=(12, 21, 30, 220))
            color = ("#DDB56C" if style == "navy_gold" and index == 2 else
                     "#174254" if style == "light_card" else "#FFFFFF")
            stroke = scale if style == "ass_clean" else 0
            brush.text((padding * scale, row_y), value, font=chosen, anchor="lt", fill=color,
                       stroke_width=stroke, stroke_fill="#172C40")
        canvas = canvas.resize((card_width, card_height), Image.Resampling.LANCZOS)
        layer.alpha_composite(canvas, (left, card_top))
        bottom = card_top + card_height
    if bottom > y + height * .62:
        raise ValueError("姓名条挡住字幕区域，请缩短提示水印或调整模板")


def draw_nameplate(layer, slot, name, organization, title, font_path, style, card_top,
                   *, nameplate_scale=1.0, nameplate_box=None):
    """Scale the whole card, including text, without stretching its proportions."""
    if not .25 <= nameplate_scale <= 2:
        raise ValueError("姓名条大小应在 25% 到 200% 之间")
    if nameplate_box is not None:
        validate_box(nameplate_box, layer.size, "姓名条区域")
    if nameplate_scale == 1 and nameplate_box is None:
        return _draw_nameplate_default(layer, slot, name, organization, title, font_path, style, card_top)
    source = Image.new("RGBA", layer.size)
    _draw_nameplate_default(source, slot, name, organization, title, font_path, style, card_top)
    bounds = source.getchannel("A").getbbox()
    if not bounds:
        return
    card = source.crop(bounds)
    if nameplate_box is not None:
        x, y, width, height = nameplate_box
        factor = min(width / card.width, height / card.height)
        target_size = (max(1, round(card.width * factor)), max(1, round(card.height * factor)))
        left, top = x + (width - target_size[0]) // 2, y + (height - target_size[1]) // 2
    else:
        target_size = (max(1, round(card.width * nameplate_scale)), max(1, round(card.height * nameplate_scale)))
        left, top = bounds[:2]
    if left + target_size[0] > layer.width or top + target_size[1] > layer.height:
        raise ValueError("姓名条超出模板边界，请缩小或重新框选区域")
    layer.alpha_composite(card.resize(target_size, Image.Resampling.LANCZOS), (left, top))


def draw_topic(layer, topic, topic_box, topic_color, font_path, *, topic_style=None):
    """Render identical topic text in the template preview and export."""
    if not topic:
        return
    if not topic_box or len(topic_box) != 4:
        raise ValueError("已填写主题，请设置主题区域 x,y,宽,高")
    try:
        validate_box(topic_box, layer.size, "主题区域")
    except ValueError as exc:
        raise ValueError("主题区域超出模板边界") from exc
    if not re.fullmatch(r"#[0-9A-Fa-f]{6}", topic_color):
        raise ValueError("主题颜色必须为 #RRGGBB")
    if topic_style is not None:
        return draw_styled_topic(layer, topic, tuple(map(int, topic_box)), topic_color,
                                 font_path, topic_style)
    tx, ty, tw, th = map(int, topic_box)
    draw = ImageDraw.Draw(layer)
    color = ImageColor.getrgb(topic_color)
    stroke = "#173863" if sum(color) > 380 else "#FFFFFF"
    chosen = fit_font(draw, topic, font_path, min(th * .55, tw * .09), tw * .92)
    bounds = draw.textbbox((0, 0), topic, font=chosen)
    draw.text((tx + (tw - (bounds[2] - bounds[0])) / 2 - bounds[0],
               ty + (th - (bounds[3] - bounds[1])) / 2 - bounds[1]),
              topic, font=chosen, fill=topic_color, stroke_width=2, stroke_fill=stroke)


def graphics(size, slot, name, organization, title, notice, font_path, destination,
             nameplate_style="classic", topic="", topic_box=None,
             topic_color="#FFFFFF", topic_font_path=None,
             *, nameplate_scale=1.0, nameplate_box=None, topic_style=None):
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    x, y, width, height = slot
    left = x + round(width * 0.025)
    notice_font = font(font_path, width * 0.027)
    notice_lines = wrap_notice(draw, notice, notice_font, width * 0.63)
    if len(notice_lines) > 3:
        raise ValueError("提示水印超过三行，请缩短文案")
    line_height = round(width * 0.038)
    top = y + round(height * 0.025)
    for index, line in enumerate(notice_lines):
        draw.text((left, top + index * line_height), line, font=notice_font,
                  fill="white", stroke_width=max(2, round(width * 0.0027)), stroke_fill="#202020")

    draw_topic(layer, topic, topic_box, topic_color, topic_font_path or font_path,
               topic_style=topic_style)

    card_top = max(y + round(height * 0.19), top + len(notice_lines) * line_height + round(height * 0.015))
    draw_nameplate(layer, slot, name, organization, title, font_path, nameplate_style, card_top,
                   nameplate_scale=nameplate_scale, nameplate_box=nameplate_box)
    if destination is not None:
        layer.save(destination)
    return layer


def ass_time(seconds):
    centiseconds = round(seconds * 100)
    hours, remainder = divmod(centiseconds, 360000)
    minutes, remainder = divmod(remainder, 6000)
    sec, cs = divmod(remainder, 100)
    return f"{hours}:{minutes:02}:{sec:02}.{cs:02}"


def caption_layout(raw, width, font_path, style, protected=()):
    style.validate()
    size = style.font_size or max(22, round(width * .041))
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    limit = round(width * .94) - style.outline_width * 2
    def text_width(text):
        return measure.textlength(text, font=font(font_path, size))
    if not style.font_size:
        while size > 14 and text_width(raw) > limit:
            size -= 1
    if text_width(raw) <= limit:
        return size, [raw]
    forbidden = set()
    for phrase in protected:
        if not phrase:
            continue
        for match in re.finditer(re.escape(phrase), raw):
            forbidden.update(range(match.start() + 1, match.end()))
    choices = [i for i in range(1, len(raw)) if raw[i] not in BAD_START and i not in forbidden
               and max(text_width(raw[:i]), text_width(raw[i:])) <= limit]
    if not choices:
        raise ValueError("当前字幕字号过大，最多两行仍放不下，请减小字幕字号")
    split = min(choices, key=lambda i: (not raw[i:].startswith(STRONG_STARTS),
                                        max(text_width(raw[:i]), text_width(raw[i:]))))
    return size, [raw[:split], raw[split:]]


def font_family_name(path):
    family = font(path, 16).getname()[0]
    if "," in family or "\n" in family or "\r" in family:
        raise ValueError("字体名称不能包含逗号或换行")
    return family


def make_ass(captions, size, slot, destination, font_family, font_path,
             nameplate_style="classic", name="", organization="", title="",
             duration=0, *, caption_style=None):
    style = caption_style or CaptionStyle()
    style.validate()
    x, y, width, height = slot
    font_size = style.font_size or max(22, round(width * 0.041))
    caption_y = y + height - max(8, round(height * 0.035))
    lines = [
        "[Script Info]", "ScriptType: v4.00+", f"PlayResX: {size[0]}", f"PlayResY: {size[1]}",
        "ScaledBorderAndShadow: yes", "WrapStyle: 2", "", "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Default,{font_family},{font_size},{ass_color(style.text_color)},&H00FFFFFF,{ass_color(style.outline_color)},&H80000000,-1,0,0,0,100,100,0,0,1,{style.outline_width},1,2,10,10,10,1",
        "", "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    # Nameplates are part of graphics(), so previews and exports share pixels.
    for item in captions:
        raw = str(item["text"])
        fitted_size, text_lines = caption_layout(raw, width, font_path, style, (organization, name, title))
        if fitted_size * 1.3 * len(text_lines) > height * .45:
            raise ValueError("字幕字号过大，会占用过多视频画面，请减小字号")
        value = r"\N".join(text_lines)
        value = value.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")
        value = value.replace(r"\\N", r"\N")
        lines.append(f"Dialogue: 0,{ass_time(float(item['start']))},{ass_time(float(item['end']))},Default,,0,0,0,,{{\\an2\\pos({x + width // 2},{caption_y})\\fs{fitted_size}}}{value}")
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def render_video(video, template, captions_path, output, name, organization, title, notice,
                 slot=None, font_path=None, font_family=None, ffmpeg_path="ffmpeg",
                 ffprobe_path="ffprobe", progress=None, cancel_event=None, cuts=None,
                 *, nameplate_style="classic", topic="", topic_box=None,
                 brightness="normal", quality="balanced", brightness_target=100,
                 brightness_offset=None, topic_color="#FFFFFF", topic_font_path=None,
                 caption_style=None, nameplate_scale=1.0, nameplate_box=None, topic_style=None):
    caption_style = caption_style or CaptionStyle()
    caption_style.validate()
    if topic_style is not None:
        topic_style.validate()
    if nameplate_style not in NAMEPLATE_STYLE_IDS:
        raise ValueError("未知的人名条样式")
    if brightness not in ("normal", "bright", "dark", "auto"):
        raise ValueError("未知的明暗档位")
    quality_settings = {"balanced": (23, "128k"), "high": (18, "192k"),
                        "small": (27, "96k")}
    if quality not in quality_settings:
        raise ValueError("未知的画质档位")
    video, template, captions_path, output = map(Path, (video, template, captions_path, output))
    for path in (video, template, captions_path):
        if not path.is_file():
            raise ValueError(f"文件不存在：{path}")
    duration, video_width, video_height = probe(video, ffprobe_path)
    if video_width <= video_height:
        raise ValueError("输入视频必须是横屏")
    with Image.open(template) as source:
        image = source.copy()
    if image.width >= image.height:
        raise ValueError("模板图片必须是竖屏")
    slot = slot or find_slot(image)
    if len(slot) != 4:
        raise ValueError("视频区域必须为 x,y,宽,高")
    validate_slot(slot, image.size)
    cuts = cuts or []
    final_duration = output_duration(cuts, duration)
    captions = read_captions(captions_path, final_duration)
    font_path = Path(font_path) if font_path else default_font()
    if not font_path or not font_path.is_file():
        raise ValueError("未找到中文粗体字体，请在界面中选择字体文件")
    topic_font_path = Path(topic_font_path) if topic_font_path else None
    if topic_font_path and not topic_font_path.is_file():
        raise ValueError("主题字体文件不存在")
    default_family = font_family_name(font_path)
    font_family = font_family or default_family
    if "," in font_family or "\n" in font_family:
        raise ValueError("字体名称不能包含逗号或换行")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="vertical-video-", dir=output.parent) as folder:
        temporary = Path(folder)
        overlay = temporary / "graphics.png"
        ass = temporary / "captions.ass"
        graphics(image.size, slot, name, organization, title, notice, font_path, overlay,
                 nameplate_style, topic, topic_box, topic_color, topic_font_path,
                 nameplate_scale=nameplate_scale, nameplate_box=nameplate_box,
                 topic_style=topic_style)
        shutil.copy2(font_path, temporary / font_path.name)
        caption_font = Path(caption_style.font_path) if caption_style.font_path else font_path
        if caption_font != font_path:
            shutil.copy2(caption_font, temporary / caption_font.name)
        make_ass(captions, image.size, slot, ass,
                 font_family_name(caption_font) if caption_style.font_path else font_family, caption_font,
                 nameplate_style, name, organization, title, final_duration, caption_style=caption_style)
        x, y, width, height = slot
        edit_filters = ""
        video_input = "[0:v]"
        audio_map = "0:a?"
        frame_options = []
        if cuts:
            has_audio, frame_rate = _cut_streams(video, ffprobe_path)
            edit_filters, video_input, audio_map = _cut_filters(cuts, duration, has_audio, frame_rate)
            frame_options = ["-fps_mode", "cfr"]
            if frame_rate:
                frame_options += ["-r", frame_rate]
        if brightness == "auto":
            if brightness_offset is None:
                from .brightness import analyze_brightness
                brightness_offset = analyze_brightness(video, duration, ffmpeg_path,
                                                       brightness_target)["offset"]
            if not 0 <= brightness_offset <= .18:
                raise ValueError("自动明暗调整值无效")
            brightness_filter = f"eq=brightness={brightness_offset:.3f}," if brightness_offset else ""
        else:
            brightness_filter = {"normal": "", "bright": "eq=gamma=1.10:brightness=0.025,",
                                 "dark": "eq=gamma=0.90:brightness=-0.025,"}[brightness]
        filters = edit_filters + (
            f"{video_input}scale=trunc(iw*sar/2)*2:ih,setsar=1,"
            f"scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},setsar=1,{brightness_filter}format=rgba[main];"
            f"[1:v]format=rgba[template];[template][main]overlay={x}:{y}:shortest=1[base];"
            f"[2:v]format=rgba[graphics];[base][graphics]overlay=0:0:shortest=1,"
            f"ass=captions.ass:fontsdir=.,format=yuv420p,pad=ceil(iw/2)*2:ceil(ih/2)*2[out]"
        )
        pending_output = temporary / "rendered.mp4"
        audio_options = ["-map", audio_map] if audio_map else []
        command = [str(ffmpeg_path), "-y", "-hide_banner", "-loglevel", "error", "-progress", "pipe:1",
                   "-i", str(video.resolve()),
                   "-loop", "1", "-i", str(template.resolve()), "-loop", "1", "-i", str(overlay),
                   "-filter_complex", filters, "-map", "[out]", *audio_options, "-t", str(final_duration),
                   "-c:v", "libx264", "-crf", str(quality_settings[quality][0]), "-preset", "medium",
                   "-c:a", "aac", "-b:a", quality_settings[quality][1],
                   "-movflags", "+faststart", *frame_options, str(pending_output)]
        process = subprocess.Popen(command, cwd=temporary, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
                                   creationflags=no_window())
        try:
            for line in process.stdout:
                if cancel_event and cancel_event.is_set():
                    process.terminate()
                    raise RuntimeError("用户已取消")
                if progress and line.startswith("out_time_ms="):
                    try:
                        seconds = int(line.split("=", 1)[1]) / 1_000_000
                        progress(min(1.0, seconds / final_duration))
                    except ValueError:
                        pass
            error = process.stderr.read()
            if process.wait() != 0:
                raise RuntimeError(f"FFmpeg 合成失败：{error[-1200:]}")
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)
            if process.stdout:
                process.stdout.close()
            if process.stderr:
                process.stderr.close()
        pending_output.replace(output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("video", "template", "captions", "output"):
        parser.add_argument(f"--{option}", type=Path, required=True)
    for option in ("name", "organization", "title", "notice"):
        parser.add_argument(f"--{option}", required=True)
    parser.add_argument("--slot", help="视频区域：x,y,宽,高；透明 PNG 可留空")
    parser.add_argument("--font", type=Path)
    parser.add_argument("--font-family")
    parser.add_argument("--nameplate-scale", type=float, default=1.0, help="姓名条整体缩放倍数，0.25–2")
    parser.add_argument("--nameplate-box", help="姓名条区域：x,y,宽,高；保持比例居中放入区域")
    parser.add_argument("--caption-color", default="#FFFFFF")
    parser.add_argument("--caption-size", type=int, default=0, help="成片像素字号，0 为自动")
    parser.add_argument("--caption-outline-color", default="#000000")
    parser.add_argument("--caption-outline-width", type=int, default=3)
    parser.add_argument("--caption-font", default="")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    args = parser.parse_args()
    slot = tuple(int(part.strip()) for part in args.slot.split(",")) if args.slot else None
    render_video(args.video, args.template, args.captions, args.output, args.name,
                 args.organization, args.title, args.notice, slot, args.font,
                 args.font_family, args.ffmpeg, args.ffprobe,
                 nameplate_scale=args.nameplate_scale,
                 nameplate_box=tuple(int(part.strip()) for part in args.nameplate_box.split(",")) if args.nameplate_box else None,
                 caption_style=CaptionStyle(args.caption_size, args.caption_color,
                                            args.caption_outline_color, args.caption_outline_width, args.caption_font))
    print(f"已生成：{args.output.resolve()}")


if __name__ == "__main__":
    main()
