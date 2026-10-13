"""Export the shipped nameplates for visual inspection, optionally on a video frame."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import tempfile

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from studio.render import NAMEPLATE_STYLES, default_font, font, graphics
from studio.edits import no_window


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--output", type=Path, default=Path("docs/nameplate-styles.png"))
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="nameplate-preview-") as temporary:
        if args.video:
            frame = Path(temporary) / "frame.png"
            subprocess.run([args.ffmpeg, "-v", "error", "-ss", "15", "-i", str(args.video),
                            "-frames:v", "1", "-vf", "scale=960:540", str(frame)],
                           check=True, creationflags=no_window())
            with Image.open(frame) as image:
                background = image.convert("RGBA")
        else:
            background = Image.new("RGBA", (960, 540), "#536474")
        sheet = Image.new("RGB", (1200, 1550), "#F1F4F7")
        draw = ImageDraw.Draw(sheet)
        path = default_font()
        draw.text((24, 16), "姓名条样式", font=font(path, 26), fill="#20384D")
        for index, (label, value) in enumerate(NAMEPLATE_STYLES):
            left, top = (index % 2) * 600, 62 + (index // 2) * 370
            draw.text((left + 18, top), label, font=font(path, 22), fill="#20384D")
            layer = graphics(background.size, (0, 0, 960, 540), "姓名",
                             "单位名称", "部门 · 职称", "", path, None, value)
            image = Image.alpha_composite(background, layer).convert("RGB")
            sheet.paste(image.resize((576, 324), Image.Resampling.LANCZOS), (left + 12, top + 32))
        draw.text((636, 1360), "人物信息为排版示例", font=font(path, 22), fill="#52677A")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(args.output)
        print(args.output.resolve())


if __name__ == "__main__":
    main()
