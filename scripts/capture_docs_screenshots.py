"""Capture the real Qt interface using synthetic, isolated demo data only."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw
from PySide6.QtWidgets import QApplication, QScrollArea
from PySide6.QtGui import QFont, QFontDatabase

from qt_app import App
from studio.qt_caption_editor import CaptionEditor
from studio.render import default_font, font


def capture() -> None:
    output = ROOT / "docs" / "screenshots"
    output.mkdir(parents=True, exist_ok=True)
    application = QApplication.instance() or QApplication([])
    application.setStyle("Fusion")
    regular_font = Path("C:/Windows/Fonts/msyh.ttc")
    if regular_font.is_file():
        QFontDatabase.addApplicationFont(str(regular_font))
        application.setFont(QFont("Microsoft YaHei", 9))
    original_directory = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="vvs-docs-") as temporary:
        workspace = Path(temporary)
        os.chdir(workspace)
        try:
            source = Path("演示素材")
            source.mkdir()
            for number in range(1, 4):
                (source / f"示例单位_讲师_示例嘉宾_{number:02}.mp4").touch()
            template = Image.new("RGBA", (540, 960), "#EDF3F8")
            draw = ImageDraw.Draw(template)
            draw.rectangle((0, 326, 539, 629), fill=(0, 0, 0, 0))
            draw.rounded_rectangle((22, 730, 518, 860), radius=18, fill="#173853")
            draw.text((270, 765), "访谈 · 分享 · 知识", anchor="mt",
                      font=font(default_font(), 30), fill="white")
            draw.text((270, 815), "通用演示模板", anchor="mt",
                      font=font(default_font(), 22), fill="#B3D1E4")
            template.save("演示模板.png")
            frame = Image.new("RGB", (960, 540), "#B3C9D5")
            draw = ImageDraw.Draw(frame)
            draw.rounded_rectangle((60, 65, 280, 400), radius=18, fill="#CDDCE4")
            draw.ellipse((555, 55, 725, 230), fill="#EBD4BD")
            draw.rounded_rectangle((490, 235, 790, 560), radius=80, fill="#335D73")
            draw.text((45, 475), "合成示意画面 · 无真人素材", font=font(default_font(), 23), fill="#264759")

            with patch("qt_app.SETTINGS_PATH", workspace / "settings.json"), \
                    patch("qt_app.load_settings", return_value={}), \
                    patch.object(App, "detect_media_tools"), \
                    patch.object(App, "sync_video_preview", lambda self: None):
                window = App()
                window.resize(1440, 1050)
                window.mode.setCurrentIndex(window.mode.findData("folder"))
                window.source.setText(str(source))
                window.template.setText("演示模板.png")
                window.output_dir.setText("成片输出")
                window.slot.setText("0,326,540,304")
                window.topic.setText("让每一次分享更清晰")
                window.topic_box.setText("20,65,500,190")
                window.notice.setPlainText("内容分享 · 示例画面")
                window.refresh_files()
                window.table.setColumnWidth(0, 345)
                window.table.setColumnWidth(1, 110)
                window.table.selectRow(0)
                window.preview_frame = frame
                window.preview_duration = 180
                window.preview_time.setText("00:00 / 03:00")
                window.preview_video_status.setText("演示配置：合成示意画面；未调用 ASR 或文本模型。")
                window.tool_status.setText("FFmpeg / FFprobe：演示界面未执行检测")
                window.current_status.setText("就绪 · 3 条示例素材等待开始处理")
                window.show_preview()
                window.show()
                application.processEvents()
                window.grab().save(str(output / "main-window.png"))
                scroll = window.findChild(QScrollArea)
                scroll.ensureWidgetVisible(window.caption_preview, 0, 40)
                window.preview_scope.setCurrentIndex(window.preview_scope.findData("video"))
                application.processEvents()
                window.grab().save(str(output / "style-settings.png"))
                window.close()

            editor_folder = workspace / "caption-demo"
            editor_folder.mkdir()
            texts = ["欢迎来到今天的分享", "我们从一个简单的问题开始", "让表达更清楚让制作更轻松",
                     "这里可以修改文字和时间", "保存后重新导出成片"]
            captions, words = [], []
            for index, text in enumerate(texts):
                start = index * 3
                captions.append({"start": start, "end": start + 2.8,
                                 "start_word": index, "end_word": index + 1, "text": text})
                words.append({"begin_time": start * 1000, "end_time": start * 1000 + 2800, "text": text})
            (editor_folder / "transcription.json").write_text(json.dumps({"transcripts": [
                {"sentences": [{"words": words}]}]}, ensure_ascii=False), encoding="utf-8")
            (editor_folder / "captions.json").write_text(json.dumps({"reviewed": True, "captions": captions,
                "edit_plan": {"cuts": [], "source_duration": 15}}, ensure_ascii=False), encoding="utf-8")
            editor = CaptionEditor(editor_folder)
            editor.resize(1200, 650)
            editor.status.setText("成片时长 15.000 秒 · 通用示例字幕（演示数据）")
            editor.table.selectRow(1)
            editor.show()
            application.processEvents()
            editor.grab().save(str(output / "caption-editor.png"))
            editor.close()
            application.processEvents()
        finally:
            os.chdir(original_directory)
    print(output)


if __name__ == "__main__":
    capture()
