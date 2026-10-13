"""PySide6 task center for the existing resumable video pipeline."""
from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import webbrowser

from PIL import Image, ImageDraw
from PIL.ImageQt import ImageQt
from PySide6.QtCore import Qt, QThread, Signal, QRect, QPoint, QSize
from PySide6.QtGui import QColor, QFont, QFontDatabase, QFontMetrics, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog,
    QFileDialog, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar,
    QPushButton, QScrollArea, QSplitter, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget, QColorDialog, QSpinBox, QSlider)

from studio.bailian import ApiError, BailianClient, saved_base_url
from studio.models import (ASR_GUIDE, DEFAULT_ASR_MODEL, DEFAULT_TEXT_MODEL,
                           Person, VideoItem, VIDEO_EXTENSIONS, parse_slot, scan_folder)
from studio.pipeline import Cancelled, JobConfig, Pipeline, work_dir
from studio.prompts import load_prompts, prompts_dir, PromptSet
from studio.render import NAMEPLATE_STYLES, NAMEPLATE_STYLE_IDS, find_slot, validate_slot, validate_box
from studio.strategy import load_strategy, strategy_path
from studio.toolchain import detect_tools
from studio.caption_style import CaptionStyle
from studio.topic_style import TopicStyle, TOPIC_EFFECTS, TOPIC_BACKGROUNDS
from studio.qt_video_preview import FramePreviewController, fit_frame
from studio.version import __version__


APP_TITLE = f"竖屏访谈工坊 {__version__}"
DEFAULT_NOTICE = "医学科普 仅供参考\n如有不适请及时线下就医"
SETTINGS_PATH = Path(os.environ.get("APPDATA", str(Path.home()))) / "VerticalVideoStudio" / "settings.json"


def load_settings() -> dict:
    try:
        value = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def save_settings(value: dict) -> None:
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    pending = SETTINGS_PATH.with_suffix(".tmp")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    pending.replace(SETTINGS_PATH)


def open_path(path: Path) -> None:
    if sys.platform == "win32":
        os.startfile(str(path))  # type: ignore[attr-defined]
    else:
        subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(path)])


def result_plan_details(default_path: Path, run: dict) -> list[str]:
    """Only attach decisions produced or explicitly adopted by this attempt."""
    plan_path = Path(run.get("plan_path") or default_path)
    unavailable = "本次任务尚未生成可用的剪辑计划"
    if run.get("plan_ready") is False:
        return [unavailable]
    if run.get("run_id") and run.get("plan_ready") is not True:
        return [unavailable]
    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if not isinstance(plan, dict):
            return [unavailable] if run else []
        if run.get("run_id"):
            if plan.get("run_id") != run["run_id"]:
                return [unavailable]
        else:
            # Older versions have no attempt identity. Never present a previous
            # attempt's decisions as the reason a new attempt failed.
            started_at = run.get("started_at")
            if not started_at:
                return [f"历史剪辑计划（非本次任务）：{plan_path}"]
            started = datetime.fromisoformat(started_at).timestamp()
            modified = plan_path.stat().st_mtime
            ended_at = run.get("ended_at")
            ended = datetime.fromisoformat(ended_at).timestamp() if ended_at else None
            if modified < started or (ended is not None and modified > ended + 2):
                return [unavailable, f"历史剪辑计划（非本次任务）：{plan_path}"]
        decisions = plan.get("decisions", [])
        reviews = [d for d in decisions if isinstance(d, dict) and d.get("need_review")]
        lines = []
        summary = plan.get("processing_summary")
        if isinstance(summary, dict):
            if plan.get("mode") == "model_authored":
                lines.append(f"模型复审确认：裁剪 {summary.get('adopted_model_proposal_count', 0)} 处，"
                             f"气口压缩 {summary.get('model_pause_compression_count', 0)} 处；"
                             f"共删除 {float(summary.get('removed_seconds', 0)):.2f} 秒")
                lines.append(f"模型初稿与最终方案：{plan_path.parent / 'model-edit-plan.json'}")
            else:
                lines.append(f"剪辑：模型建议 {summary.get('model_proposal_count', 0)} 处，"
                             f"采用 {summary.get('adopted_model_proposal_count', 0)} 处，"
                             f"局部校验未通过 {summary.get('rejected_model_proposal_count', 0)} 处；"
                             f"含静音共删除 {float(summary.get('removed_seconds', 0)):.2f} 秒")
            lines.append(f"字幕：{summary.get('caption_count', 0)} 条，"
                         f"原文回退 {summary.get('caption_literal_fallback_count', 0)} 条，"
                         f"局部排版修整 {summary.get('caption_formatting_repair_count', 0)} 处")
        rejected = plan.get("rejected_model_proposals") or []
        if rejected:
            lines.append(f"以下 {len(rejected)} 条模型建议仍未通过校验，相关原声已保留：")
            for issue in rejected:
                lines.append(f"{issue.get('proposal_id') or '候选 ' + str(issue.get('proposal_position', '?'))}  "
                             f"{issue.get('reason', '剪辑建议无效')}")
        if reviews:
            lines.append(f"建议复核 {len(reviews)} 处（已保留原声，不是任务失败原因）：")
            for decision in reviews:
                lines.append(f"{float(decision['start']):.1f}–{float(decision['end']):.1f} 秒  "
                             f"{decision.get('source_text', '')}  "
                             f"{decision.get('policy_reason') or decision.get('reason', '')}")
        lines.append(f"本次剪辑计划：{plan_path}")
        return lines
    except (OSError, ValueError, KeyError, TypeError):
        return [unavailable] if run else []


class ToolWorker(QThread):
    found = Signal(str, str, str, str)

    def __init__(self, ffmpeg: str, ffprobe: str):
        super().__init__()
        self.ffmpeg, self.ffprobe = ffmpeg, ffprobe

    def run(self) -> None:
        encoder, probe = detect_tools(self.ffmpeg, self.ffprobe)
        self.found.emit(encoder[0] or "", probe[0] or "", encoder[1], probe[1])


class BatchWorker(QThread):
    progress = Signal(str, int, str)
    result = Signal(str, str, str, object)
    finished_batch = Signal(int, int, int, int)

    def __init__(self, client: BailianClient, config: JobConfig, items: list[VideoItem],
                 cancel: threading.Event, *, render_only=False):
        super().__init__()
        self.client, self.config, self.items, self.cancel = client, config, items, cancel
        self.render_only = render_only

    def run(self) -> None:
        completed = review = failed = cancelled = 0
        records = []
        started = time.monotonic()
        for item in self.items:
            if self.cancel.is_set():
                cancelled += 1
                break
            self.progress.emit(str(item.path), 0, "开始处理")
            pipeline = None
            try:
                for attempt in range(3):
                    pipeline = Pipeline(self.client, self.config,
                                        lambda phase, percent, detail, path=item.path:
                                        self.progress.emit(str(path), percent, detail), self.cancel,
                                        render_only=self.render_only)
                    try:
                        target = pipeline.run_one(item)
                        break
                    except ApiError as exc:
                        if attempt == 2 or "HTTP 401" in str(exc) or "HTTP 400" in str(exc):
                            raise
                        self.progress.emit(str(item.path), 0,
                                           f"接口暂时失败，自动重试 {attempt + 1}/2：{str(exc)[:100]}")
                        if self.cancel.wait(attempt + 1):
                            raise Cancelled("用户已取消")
                completed += 1
                run = pipeline.last_run or {}
                if run.get("quality_status") == "待复核":
                    review += 1
                records.append({"file": str(item.path), "status": "完成", "output": str(target),
                                "review_count": run.get("review_count", 0)})
                self.result.emit(str(item.path), "已完成", str(target), run)
            except Cancelled as exc:
                cancelled += 1
                self.result.emit(str(item.path), "已取消", str(exc), pipeline.last_run or {})
                break
            except Exception as exc:
                failed += 1
                detail = str(exc).replace(self.client.api_key, "***") if self.client and self.client.api_key else str(exc)
                records.append({"file": str(item.path), "status": "失败", "error": detail})
                self.result.emit(str(item.path), "失败", detail, pipeline.last_run or {})
        try:
            log_path = self.config.output_dir / ".vertical-video-studio" / "batch-runs.json"
            data = json.loads(log_path.read_text(encoding="utf-8")) if log_path.is_file() else {"runs": []}
            if not isinstance(data, dict) or not isinstance(data.get("runs"), list):
                data = {"runs": []}
            data["runs"].append({"ended_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                                 "elapsed_seconds": round(time.monotonic() - started, 2),
                                 "total_videos": len(self.items), "completed": completed,
                                 "review": review, "failed": failed, "cancelled": cancelled,
                                 "strategy_sha256": self.config.strategy.digest(),
                                 "prompts_sha256": self.config.prompts.digest(), "items": records})
            log_path.parent.mkdir(parents=True, exist_ok=True)
            pending = log_path.with_suffix(".tmp")
            pending.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            pending.replace(log_path)
        except (OSError, ValueError) as exc:
            self.progress.emit("", 100, f"批次日志保存失败：{exc}")
        self.finished_batch.emit(completed, review, failed, cancelled)


class RegionCanvas(QWidget):
    changed = Signal(str)
    HANDLE = 9
    MIN_SIZE = 6

    def __init__(self, image: QPixmap, scale: float, initial: str):
        super().__init__()
        self.scale = scale
        self.picture = image.scaled(round(image.width() * scale), round(image.height() * scale),
                                    Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        self.setFixedSize(self.picture.size())
        self.setMouseTracking(True)
        self.selection = QRect()
        self.start = None
        self.origin = QRect()
        self.drag_mode = ""
        self.drag_edges = ""
        self.set_region(initial)

    def set_region(self, value: str):
        try:
            x, y, width, height = parse_slot(value)
            self.selection = QRect(round(x * self.scale), round(y * self.scale),
                                   round(width * self.scale), round(height * self.scale))
        except (TypeError, ValueError):
            self.selection = QRect()
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self.picture)
        if not self.selection.isEmpty():
            painter.fillRect(self.selection, QColor(255, 40, 40, 55))
            painter.setPen(QPen(QColor("#FF2424"), 4))
            painter.drawRect(self.selection.adjusted(2, 2, -2, -2))
            x, y, width, height = (self.selection.x(), self.selection.y(),
                                    self.selection.width(), self.selection.height())
            for point in (QPoint(x, y), QPoint(x + width // 2, y), QPoint(x + width, y),
                          QPoint(x, y + height // 2), QPoint(x + width, y + height // 2),
                          QPoint(x, y + height), QPoint(x + width // 2, y + height),
                          QPoint(x + width, y + height)):
                painter.fillRect(QRect(point - QPoint(5, 5), point + QPoint(5, 5)),
                                 QColor("#FF2424"))

    def _hit(self, point: QPoint) -> tuple[str, str]:
        rect = self.selection
        if rect.isEmpty() or not rect.adjusted(-self.HANDLE, -self.HANDLE,
                                                self.HANDLE, self.HANDLE).contains(point):
            return "new", ""
        x, y, width, height = rect.x(), rect.y(), rect.width(), rect.height()
        horizontal = "l" if abs(point.x() - x) <= self.HANDLE else (
            "r" if abs(point.x() - x - width) <= self.HANDLE else "")
        vertical = "t" if abs(point.y() - y) <= self.HANDLE else (
            "b" if abs(point.y() - y - height) <= self.HANDLE else "")
        if horizontal or vertical:
            return "resize", horizontal + vertical
        return ("move", "") if rect.contains(point) else ("new", "")

    def _cursor(self, mode: str, edges: str):
        if mode == "move":
            self.setCursor(Qt.SizeAllCursor)
        elif mode == "resize":
            cursor = (Qt.SizeFDiagCursor if edges in ("lt", "rb") else
                      Qt.SizeBDiagCursor if edges in ("rt", "lb") else
                      Qt.SizeHorCursor if any(edge in edges for edge in "lr") else
                      Qt.SizeVerCursor)
            self.setCursor(cursor)
        else:
            self.setCursor(Qt.CrossCursor)

    def _drag(self, point: QPoint):
        x = max(0, min(self.width(), point.x()))
        y = max(0, min(self.height(), point.y()))
        if self.drag_mode == "move":
            new_x = max(0, min(self.width() - self.origin.width(),
                               self.origin.x() + x - self.start.x()))
            new_y = max(0, min(self.height() - self.origin.height(),
                               self.origin.y() + y - self.start.y()))
            self.selection = QRect(new_x, new_y, self.origin.width(), self.origin.height())
        elif self.drag_mode == "resize":
            left, top = self.origin.x(), self.origin.y()
            right, bottom = left + self.origin.width(), top + self.origin.height()
            if "l" in self.drag_edges:
                left = max(0, min(right - self.MIN_SIZE, x))
            if "r" in self.drag_edges:
                right = min(self.width(), max(left + self.MIN_SIZE, x))
            if "t" in self.drag_edges:
                top = max(0, min(bottom - self.MIN_SIZE, y))
            if "b" in self.drag_edges:
                bottom = min(self.height(), max(top + self.MIN_SIZE, y))
            self.selection = QRect(left, top, right - left, bottom - top)
        else:
            left, right = sorted((self.start.x(), x))
            top, bottom = sorted((self.start.y(), y))
            self.selection = QRect(left, top, right - left, bottom - top)
        self.update()

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return
        self.start = event.position().toPoint()
        self.origin = QRect(self.selection)
        self.drag_mode, self.drag_edges = self._hit(self.start)
        if self.drag_mode == "new":
            self.selection = QRect(self.start, self.start)
        self._cursor(self.drag_mode, self.drag_edges)
        self.update()

    def mouseMoveEvent(self, event):
        if self.start is not None:
            self._drag(event.position().toPoint())
        else:
            self._cursor(*self._hit(event.position().toPoint()))

    def mouseReleaseEvent(self, event):
        if self.start is None or event.button() != Qt.LeftButton:
            return
        self._drag(event.position().toPoint())
        self.start = None
        self.drag_mode = ""
        if self.selection.width() > 2 and self.selection.height() > 2:
            values = (self.selection.x(), self.selection.y(), self.selection.width(), self.selection.height())
            self.changed.emit(",".join(str(round(value / self.scale)) for value in values))
        self._cursor(*self._hit(event.position().toPoint()))
        self.update()


class RegionDialog(QDialog):
    """Select a visible red rectangle on the original template."""
    def __init__(self, template: Path, initial: str, allow_auto=False, parent=None):
        super().__init__(parent)
        self.setWindowTitle("在模板上框选区域")
        self.template = template
        self.image = QPixmap(str(template))
        self.scale = min(1.0, 640 / self.image.width(), 760 / self.image.height())
        self.picture = RegionCanvas(self.image, self.scale, initial)
        self.selection = QLineEdit(initial)
        self.picture.changed.connect(self.selection.setText)
        self.selection.textChanged.connect(self.picture.set_region)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("框外拖动新建；框内拖动位置；拖动红色边点调整大小。也可填写 x,y,宽,高"))
        layout.addWidget(self.picture)
        layout.addWidget(self.selection)
        if allow_auto:
            auto = QPushButton("自动识别透明区域")
            auto.clicked.connect(self.auto_detect)
            layout.addWidget(auto)
        okay = QPushButton("应用选区")
        okay.clicked.connect(self.accept)
        layout.addWidget(okay)

    def auto_detect(self):
        try:
            with Image.open(self.template) as image:
                slot = find_slot(image)
            self.selection.setText(",".join(map(str, slot)))
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "无法识别透明区域", str(exc))


class NameplatePreview(QWidget):
    def __init__(self, organization: QLineEdit, title: QLineEdit, name: QLineEdit, style: QComboBox,
                 scale_getter=lambda: 1.0):
        super().__init__()
        from studio.render import default_font
        preview_font = default_font()
        if preview_font:
            QFontDatabase.addApplicationFont(str(preview_font))
        self.organization, self.title, self.name, self.style = organization, title, name, style
        self.scale_getter = scale_getter
        self.setMinimumHeight(145)
        for field in (organization, title, name):
            field.textChanged.connect(self.update)
        style.currentIndexChanged.connect(self.update)

    def paintEvent(self, event):
        from studio.render import default_font, graphics
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.fillRect(self.rect(), QColor("#252F3A"))
        try:
            layer = graphics((1080, 608), (0, 0, 1080, 608),
                             "姓名", "单位名称", "部门 · 职称", "", default_font(), None,
                             self.style.currentData() or "classic", nameplate_scale=self.scale_getter())
            # Crop empty space but preserve the actual 37% nameplate layout.
            crop_height = max(288, round(235 * self.scale_getter()))
            image = QPixmap.fromImage(ImageQt(layer.crop((0, 96, 1080, 96 + crop_height))))
            factor = min(self.width() / 1080, self.height() / crop_height)
            target = QRect(0, round((self.height() - crop_height * factor) / 2),
                           round(1080 * factor), round(crop_height * factor))
            painter.drawPixmap(target, image)
        except (OSError, ValueError, TypeError) as exc:
            painter.setPen(QColor("#FFFFFF"))
            painter.drawText(self.rect().adjusted(12, 8, -12, -8), Qt.TextWordWrap,
                             f"无法预览：{exc}")
            return


class CaptionPreview(QWidget):
    def __init__(self, style_getter, font_getter, slot_getter, template_getter):
        super().__init__()
        self.style_getter, self.font_getter, self.slot_getter = style_getter, font_getter, slot_getter
        self.template_getter = template_getter
        self.setMinimumHeight(110)

    def paintEvent(self, event):
        from studio.render import caption_layout, font
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#536474"))
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        try:
            style, path = self.style_getter(), self.font_getter()
            slot = parse_slot(self.slot_getter())
            width = slot[2] if slot else 1080
            if not slot and Path(self.template_getter()).is_file():
                with Image.open(self.template_getter()) as template:
                    try:
                        slot = find_slot(template)
                        width = slot[2]
                    except ValueError:
                        width = template.width
            size, lines = caption_layout("在这里输入字幕内容", width, path, style)
            if slot and size * 1.3 * len(lines) > slot[3] * .45:
                raise ValueError("字幕字号过大，请减小字号")
            height = max(round(width * .20), round(size * 1.3 * len(lines) + 24))
            image = Image.new("RGBA", (width, height))
            draw = ImageDraw.Draw(image)
            for index, line in enumerate(lines):
                draw.text((width / 2, (height - size * 1.3 * len(lines)) / 2 + index * size * 1.3),
                          line, anchor="mt", font=font(path, size), fill=style.text_color,
                          stroke_fill=style.outline_color, stroke_width=style.outline_width)
            pixmap = QPixmap.fromImage(ImageQt(image))
            pixmap = pixmap.scaled(self.size() - QSize(16, 16), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            painter.drawPixmap((self.width() - pixmap.width()) // 2, (self.height() - pixmap.height()) // 2, pixmap)
        except (OSError, ValueError, TypeError) as exc:
            painter.setPen(QColor("#FFFFFF"))
            painter.drawText(self.rect().adjusted(8, 8, -8, -8), Qt.TextWordWrap, str(exc))


class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.saved = load_settings()
        try:
            self.saved_caption_style = CaptionStyle.from_dict(self.saved.get("caption_style", {}))
        except ValueError:
            self.saved_caption_style = CaptionStyle()
        try:
            self.saved_topic_style = TopicStyle.from_dict(self.saved.get("topic_style", {}))
        except ValueError:
            self.saved_topic_style = TopicStyle()
        self.items: list[VideoItem] = []
        self.invalid = []
        self.cancel = threading.Event()
        self.batch_worker: BatchWorker | None = None
        self.tool_worker: ToolWorker | None = None
        self.rows: dict[str, int] = {}
        self.results: dict[str, tuple[str, str, dict]] = {}
        self.finished_items = 0
        self.active_total = 0
        self.tool_paths = ("", "")
        self.preview_frame = None
        self.preview_seconds = 0.0
        self.preview_duration = 0.0
        self._preview_source_key = None
        self.video_preview = FramePreviewController(self)
        self.video_preview.frame_ready.connect(self.preview_frame_ready)
        self.video_preview.duration_ready.connect(self.preview_duration_ready)
        self.video_preview.status_changed.connect(self.preview_status_changed)
        self.setWindowTitle(APP_TITLE)
        self.setFont(QFont("Microsoft YaHei UI", 9))
        self.resize(1240, 870)
        self.build()
        self.refresh_files()
        self.detect_media_tools()

    def field(self, key: str, default: str = "") -> QLineEdit:
        return QLineEdit(str(self.saved.get(key, default)))

    def line(self, form: QFormLayout, title: str, key: str, default="", choose=None) -> QLineEdit:
        widget = self.field(key, default)
        if choose:
            row = QWidget()
            layout = QHBoxLayout(row)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(widget)
            button = QPushButton("选择")
            button.clicked.connect(lambda: choose(widget))
            layout.addWidget(button)
            form.addRow(title, row)
        else:
            form.addRow(title, widget)
        return widget

    def choice(self, form: QFormLayout, title: str, key: str, values: list[tuple[str, str]], default: str) -> QComboBox:
        combo = QComboBox()
        for label, value in values:
            combo.addItem(label, value)
        index = combo.findData(self.saved.get(key, default))
        combo.setCurrentIndex(max(0, index))
        form.addRow(title, combo)
        return combo

    def build(self):
        root = QWidget()
        self.setCentralWidget(root)
        whole = QVBoxLayout(root)
        heading = QLabel("竖屏访谈工坊  ·  批量生产")
        heading.setStyleSheet("font-size:20px;font-weight:600;padding:5px")
        whole.addWidget(heading)
        splitter = QSplitter(Qt.Horizontal)
        whole.addWidget(splitter, 1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        controls = QWidget()
        left = QVBoxLayout(controls)
        scroll.setWidget(controls)
        splitter.addWidget(scroll)
        splitter.setStretchFactor(0, 1)

        source_group = QGroupBox("素材与模板")
        source_form = QFormLayout(source_group)
        self.mode = self.choice(source_form, "输入方式", "mode", [("单条视频", "single"), ("整个文件夹", "folder")], "single")
        self.source = self.line(source_form, "视频 / 文件夹", "source", choose=self.choose_source)
        self.template = self.line(source_form, "竖屏模板", "template", choose=lambda w: self.choose_file(w, "图片 (*.png *.jpg *.jpeg)"))
        self.output_dir = self.line(source_form, "输出目录", "output_dir", str(Path.home() / "Videos" / "竖屏成片"), self.choose_dir)
        self.slot = self.line(source_form, "视频区域", "slot")
        slot_actions = QHBoxLayout()
        for title, action in (("自动识别透明区域", self.auto_detect_slot),
                              ("手动红框选择", lambda: self.pick_region(self.slot))):
            button = QPushButton(title)
            button.clicked.connect(action)
            slot_actions.addWidget(button)
        source_form.addRow("", slot_actions)
        self.topic = self.line(source_form, "上方主题", "topic")
        self.topic_box = self.line(source_form, "主题区域", "topic_box")
        topic_button = QPushButton("在模板上框选主题区域")
        topic_button.clicked.connect(lambda: self.pick_region(self.topic_box))
        source_form.addRow("", topic_button)
        topic_options = QHBoxLayout()
        saved_color = QColor(str(self.saved.get("topic_color", "#FFFFFF")))
        self.topic_color = saved_color.name().upper() if saved_color.isValid() else "#FFFFFF"
        self.topic_color_button = QPushButton()
        self.topic_color_button.clicked.connect(self.pick_topic_color)
        self.update_topic_color_button()
        topic_options.addWidget(self.topic_color_button)
        self.topic_font = QComboBox()
        self.topic_font.addItem("跟随主字体", "")
        for label, path in (("微软雅黑粗体", "C:/Windows/Fonts/msyhbd.ttc"),
                            ("黑体", "C:/Windows/Fonts/simhei.ttf"),
                            ("楷体", "C:/Windows/Fonts/simkai.ttf"),
                            ("宋体", "C:/Windows/Fonts/simsun.ttc")):
            if Path(path).is_file():
                self.topic_font.addItem(label, path)
        saved_topic_font = str(self.saved.get("topic_font", ""))
        if saved_topic_font and self.topic_font.findData(saved_topic_font) < 0:
            self.topic_font.addItem(Path(saved_topic_font).name, saved_topic_font)
        self.topic_font.setCurrentIndex(max(0, self.topic_font.findData(saved_topic_font)))
        topic_options.addWidget(self.topic_font, 1)
        custom_font = QPushButton("字体文件…")
        custom_font.clicked.connect(self.pick_topic_font)
        topic_options.addWidget(custom_font)
        source_form.addRow("主题颜色 / 字体", topic_options)
        topic_size_row = QHBoxLayout()
        self.topic_auto_size = QCheckBox("自动字号")
        self.topic_auto_size.setChecked(self.saved_topic_style.font_size == 0)
        self.topic_size = QSpinBox()
        self.topic_size.setRange(12, 300)
        self.topic_size.setSuffix(" px")
        self.topic_size.setValue(self.saved_topic_style.font_size or 80)
        self.topic_size.setEnabled(not self.topic_auto_size.isChecked())
        self.topic_auto_size.toggled.connect(lambda checked: self.topic_size.setEnabled(not checked))
        topic_size_row.addWidget(self.topic_auto_size)
        topic_size_row.addWidget(self.topic_size)
        self.topic_effect = QComboBox()
        for label, value in TOPIC_EFFECTS:
            self.topic_effect.addItem(label, value)
        self.topic_effect.setCurrentIndex(max(0, self.topic_effect.findData(self.saved_topic_style.effect)))
        topic_size_row.addWidget(self.topic_effect, 1)
        source_form.addRow("主题字号 / 花字", topic_size_row)
        topic_background_row = QHBoxLayout()
        self.topic_background = QComboBox()
        for label, value in TOPIC_BACKGROUNDS:
            self.topic_background.addItem(label, value)
        self.topic_background.setCurrentIndex(max(0, self.topic_background.findData(self.saved_topic_style.background)))
        topic_background_row.addWidget(self.topic_background, 1)
        self.topic_background_color = self.saved_topic_style.background_color
        self.topic_background_button = QPushButton()
        self.topic_background_button.clicked.connect(lambda: self.pick_topic_style_color("background"))
        topic_background_row.addWidget(self.topic_background_button)
        self.topic_background_opacity = QSpinBox()
        self.topic_background_opacity.setRange(0, 100)
        self.topic_background_opacity.setValue(self.saved_topic_style.background_opacity)
        self.topic_background_opacity.setSuffix("%")
        self.topic_background_opacity.setToolTip("背景不透明度；0% 全透明，100% 不透明")
        topic_background_row.addWidget(self.topic_background_opacity)
        source_form.addRow("主题背景 / 不透明度", topic_background_row)
        topic_effect_row = QHBoxLayout()
        self.topic_accent_color = self.saved_topic_style.accent_color
        self.topic_accent_button = QPushButton()
        self.topic_accent_button.clicked.connect(lambda: self.pick_topic_style_color("accent"))
        topic_effect_row.addWidget(self.topic_accent_button)
        self.topic_outline_color = self.saved_topic_style.outline_color
        self.topic_outline_button = QPushButton()
        self.topic_outline_button.clicked.connect(lambda: self.pick_topic_style_color("outline"))
        topic_effect_row.addWidget(self.topic_outline_button)
        reset_outline = QPushButton("自动描边色")
        reset_outline.clicked.connect(self.reset_topic_outline)
        topic_effect_row.addWidget(reset_outline)
        self.topic_outline_width = QSpinBox()
        self.topic_outline_width.setRange(0, 12)
        self.topic_outline_width.setValue(self.saved_topic_style.outline_width)
        self.topic_outline_width.setSuffix(" px")
        self.topic_outline_width.setToolTip("主题描边粗细")
        topic_effect_row.addWidget(self.topic_outline_width)
        source_form.addRow("花字配色 / 描边", topic_effect_row)
        self.update_topic_style_controls()
        topic_hint = QLabel("字号按成片像素设置；手动字号保持大小，放不下时请缩小字号或扩大主题区域。")
        topic_hint.setWordWrap(True)
        source_form.addRow("", topic_hint)
        self.notice = QPlainTextEdit(str(self.saved.get("notice", DEFAULT_NOTICE)))
        self.notice.setFixedHeight(58)
        source_form.addRow("提示水印", self.notice)
        left.addWidget(source_group)

        person_group = QGroupBox("人物与输出")
        person_form = QFormLayout(person_group)
        self.organization = self.line(person_form, "医院 / 单位", "organization")
        self.person_title = self.line(person_form, "头衔", "person_title")
        self.name = self.line(person_form, "姓名", "name")
        self.style = self.choice(person_form, "姓名条样式", "nameplate_style", NAMEPLATE_STYLES, "classic")
        self.nameplate_size = QSpinBox()
        self.nameplate_size.setRange(25, 200)
        self.nameplate_size.setSuffix(" %")
        self.nameplate_size.setValue(int(self.saved.get("nameplate_size", 100)))
        person_form.addRow("姓名条整体大小", self.nameplate_size)
        self.nameplate_box = self.line(person_form, "姓名条区域（可选）", "nameplate_box")
        nameplate_actions = QHBoxLayout()
        select_nameplate = QPushButton("框选位置与大小")
        select_nameplate.clicked.connect(lambda: self.pick_region(self.nameplate_box))
        reset_nameplate = QPushButton("恢复默认区域")
        reset_nameplate.clicked.connect(self.nameplate_box.clear)
        nameplate_actions.addWidget(select_nameplate)
        nameplate_actions.addWidget(reset_nameplate)
        person_form.addRow("", nameplate_actions)
        person_form.addRow(QLabel("框选后按区域等比缩放；清空区域后使用百分比大小。"))
        self.nameplate_box.textChanged.connect(lambda: self.nameplate_size.setEnabled(not self.nameplate_box.text().strip()))
        self.nameplate_size.setEnabled(not self.nameplate_box.text().strip())
        self.nameplate_preview = NameplatePreview(self.organization, self.person_title, self.name, self.style,
                                                 lambda: self.nameplate_size.value() / 100 if not self.nameplate_box.text().strip() else 1.0)
        self.nameplate_size.valueChanged.connect(self.nameplate_preview.update)
        self.nameplate_box.textChanged.connect(self.nameplate_preview.update)
        person_form.addRow("选择时预览", self.nameplate_preview)
        self.brightness = self.choice(person_form, "画面明暗", "brightness", [
            ("自动 · 仅偏暗时提亮", "auto"), ("保持原样", "normal"),
            ("适度提亮", "bright"), ("适度压暗", "dark")], "auto")
        self.brightness_target = QSpinBox()
        self.brightness_target.setRange(40, 200)
        self.brightness_target.setValue(int(self.saved.get("brightness_target", 100)))
        self.brightness_target.setToolTip("中心区域暗部亮度为 0–255；低于标准值 12 以上才提亮")
        person_form.addRow("明暗标准值 (0–255)", self.brightness_target)
        self.brightness.currentIndexChanged.connect(
            lambda: self.brightness_target.setEnabled(self.brightness.currentData() == "auto"))
        self.brightness_target.setEnabled(self.brightness.currentData() == "auto")
        self.quality = self.choice(person_form, "输出画质", "quality", [
            ("均衡 · 建议", "balanced"), ("高画质", "high"), ("小体积", "small")], "balanced")
        self.timestamp = QCheckBox("文件名带日期时间")
        self.timestamp.setChecked(bool(self.saved.get("output_timestamp", True)))
        person_form.addRow("输出命名", self.timestamp)
        left.addWidget(person_group)

        caption_group = QGroupBox("字幕样式")
        caption_form = QFormLayout(caption_group)
        saved = self.saved_caption_style
        self.caption_color, self.caption_outline_color = saved.text_color, saved.outline_color
        self.caption_color_button, self.caption_outline_color_button = QPushButton(), QPushButton()
        self.caption_color_button.clicked.connect(lambda: self.pick_caption_color(False))
        self.caption_outline_color_button.clicked.connect(lambda: self.pick_caption_color(True))
        color_row = QHBoxLayout()
        color_row.addWidget(self.caption_color_button)
        color_row.addWidget(QLabel("描边"))
        color_row.addWidget(self.caption_outline_color_button)
        caption_form.addRow("文字颜色", color_row)
        self.update_caption_color_buttons()
        size_row = QHBoxLayout()
        self.caption_auto_size = QCheckBox("自动字号")
        self.caption_auto_size.setChecked(saved.font_size == 0)
        self.caption_size = QSpinBox()
        self.caption_size.setRange(12, 120)
        self.caption_size.setSuffix(" px")
        self.caption_size.setValue(saved.font_size or 44)
        self.caption_size.setEnabled(not self.caption_auto_size.isChecked())
        self.caption_auto_size.toggled.connect(lambda auto: self.caption_size.setEnabled(not auto))
        size_row.addWidget(self.caption_auto_size)
        size_row.addWidget(self.caption_size)
        caption_form.addRow("字幕字号", size_row)
        self.caption_outline = QSpinBox()
        self.caption_outline.setRange(0, 12)
        self.caption_outline.setSuffix(" px")
        self.caption_outline.setValue(saved.outline_width)
        caption_form.addRow("描边粗细", self.caption_outline)
        font_row = QHBoxLayout()
        self.caption_font = QComboBox()
        self.caption_font.addItem("跟随中文字体", "")
        for label, path in (("微软雅黑粗体", "C:/Windows/Fonts/msyhbd.ttc"),
                            ("黑体", "C:/Windows/Fonts/simhei.ttf"),
                            ("楷体", "C:/Windows/Fonts/simkai.ttf"),
                            ("宋体", "C:/Windows/Fonts/simsun.ttc")):
            if Path(path).is_file():
                self.caption_font.addItem(label, path)
        if saved.font_path and self.caption_font.findData(saved.font_path) < 0:
            self.caption_font.addItem(Path(saved.font_path).name, saved.font_path)
        self.caption_font.setCurrentIndex(max(0, self.caption_font.findData(saved.font_path)))
        font_row.addWidget(self.caption_font, 1)
        custom_caption_font = QPushButton("字体文件…")
        custom_caption_font.clicked.connect(self.pick_caption_font)
        font_row.addWidget(custom_caption_font)
        caption_form.addRow("字幕字体", font_row)
        help_label = QLabel("字号按成片像素计算；自动字号跟随视频区域宽度。\n手动字号过长时换行，最多两行。描边设为 0 可关闭。")
        help_label.setWordWrap(True)
        caption_form.addRow(help_label)
        self.caption_preview = CaptionPreview(self.current_caption_style, self.current_caption_font,
                                               self.slot.text, self.template.text)
        caption_form.addRow("样式预览", self.caption_preview)
        for signal in (self.caption_auto_size.toggled, self.caption_size.valueChanged,
                       self.caption_outline.valueChanged, self.caption_font.currentIndexChanged,
                       self.slot.textChanged, self.template.textChanged):
            signal.connect(self.caption_preview.update)
        left.addWidget(caption_group)

        api_group = QGroupBox("百炼与本机工具")
        api_form = QFormLayout(api_group)
        self.bailian_url = self.line(api_form, "百炼 Base URL", "bailian_url", saved_base_url(self.saved))
        self.api_key = QLineEdit(os.environ.get("DASHSCOPE_API_KEY", ""))
        self.api_key.setEchoMode(QLineEdit.Password)
        api_form.addRow("API Key", self.api_key)
        self.asr_model = self.line(api_form, "语音模型", "asr_model", DEFAULT_ASR_MODEL)
        self.text_model = self.line(api_form, "文本模型", "text_model", DEFAULT_TEXT_MODEL)
        self.ffmpeg = self.line(api_form, "FFmpeg", "ffmpeg", choose=lambda w: self.choose_file(w, "程序 (*.exe);;所有文件 (*)"))
        self.ffprobe = self.line(api_form, "FFprobe", "ffprobe", choose=lambda w: self.choose_file(w, "程序 (*.exe);;所有文件 (*)"))
        self.font = self.line(api_form, "中文字体", "font", choose=lambda w: self.choose_file(w, "字体 (*.ttf *.ttc *.otf)"))
        self.font.textChanged.connect(self.caption_preview.update)
        self.tool_status = QLabel("正在检测 FFmpeg / FFprobe…")
        api_form.addRow("状态", self.tool_status)
        actions = QHBoxLayout()
        for title, fn in (("重新检测", self.detect_media_tools),
                          ("安装说明", lambda: webbrowser.open("https://ffmpeg.org/download.html")),
                          ("编辑策略文件", self.open_strategy)):
            button = QPushButton(title)
            button.clicked.connect(fn)
            actions.addWidget(button)
        api_form.addRow("", actions)
        left.addWidget(api_group)
        left.addStretch()

        right = QWidget()
        right_layout = QVBoxLayout(right)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)
        self.preview = QLabel("选择模板后显示预览")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumHeight(230)
        self.preview.setMaximumHeight(430)
        self.preview.setStyleSheet("background:#1e242d;color:#aeb8c5;border-radius:7px")
        right_layout.addWidget(self.preview)
        self.preview_feedback = QLabel("红框：视频区域 · 橙框：主题区域 · 绿框：姓名条区域")
        self.preview_feedback.setWordWrap(True)
        right_layout.addWidget(self.preview_feedback)
        preview_seek_row = QHBoxLayout()
        self.preview_scope = QComboBox()
        self.preview_scope.addItem("完整模板", "full")
        self.preview_scope.addItem("放大视频区域", "video")
        self.preview_scope.setToolTip("放大视频区域，更容易检查姓名条是否挡住人物")
        self.preview_scope.currentIndexChanged.connect(self.show_preview)
        preview_seek_row.addWidget(self.preview_scope)
        self.preview_time = QLabel("00:00 / 00:00")
        self.preview_seek = QSlider(Qt.Horizontal)
        self.preview_seek.setRange(0, 0)
        self.preview_seek.setEnabled(False)
        self.preview_seek.setToolTip("拖动查看原视频的不同画面")
        self.preview_seek.valueChanged.connect(self.seek_preview)
        preview_seek_row.addWidget(self.preview_seek, 1)
        preview_seek_row.addWidget(self.preview_time)
        right_layout.addLayout(preview_seek_row)
        self.preview_video_status = QLabel("导入视频后显示画面，可拖动时间查看姓名条与人物的位置关系。")
        self.preview_video_status.setWordWrap(True)
        right_layout.addWidget(self.preview_video_status)
        for signal in (self.template.textChanged, self.slot.textChanged, self.topic_box.textChanged,
                       self.topic.textChanged, self.topic_font.currentIndexChanged,
                       self.nameplate_box.textChanged, self.nameplate_size.valueChanged,
                       self.style.currentIndexChanged, self.organization.textChanged,
                       self.person_title.textChanged, self.name.textChanged,
                       self.notice.textChanged, self.font.textChanged):
            signal.connect(self.show_preview)
        for signal in (self.topic_auto_size.toggled, self.topic_size.valueChanged,
                       self.topic_effect.currentIndexChanged, self.topic_background.currentIndexChanged,
                       self.topic_background_opacity.valueChanged, self.topic_outline_width.valueChanged):
            signal.connect(self.show_preview)
        self.topic_effect.currentIndexChanged.connect(self.update_topic_style_controls)
        self.topic_background.currentIndexChanged.connect(self.update_topic_style_controls)
        self.source.textChanged.connect(self.sync_video_preview)
        self.source.editingFinished.connect(self.refresh_files)
        self.show_preview()
        bar = QHBoxLayout()
        for title, fn in (("刷新素材", self.refresh_files), ("开始批处理", self.start_batch),
                          ("重试选中", self.retry_selected), ("取消", self.cancel_batch),
                          ("打开输出目录", self.open_output)):
            button = QPushButton(title)
            button.clicked.connect(fn)
            bar.addWidget(button)
            if title == "开始批处理":
                self.start_button = button
            if title == "取消":
                self.cancel_button = button
        self.cancel_button.setEnabled(False)
        right_layout.addLayout(bar)
        editor_bar = QHBoxLayout()
        self.edit_captions_button = QPushButton("编辑所选字幕")
        self.edit_captions_button.clicked.connect(self.edit_selected_captions)
        self.export_captions_button = QPushButton("重新导出所选")
        self.export_captions_button.clicked.connect(lambda: self.start(True, render_only=True))
        for button in (self.edit_captions_button, self.export_captions_button):
            editor_bar.addWidget(button)
        right_layout.addLayout(editor_bar)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["素材", "人物", "状态"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemSelectionChanged.connect(self.show_selected)
        right_layout.addWidget(self.table, 2)
        self.current_status = QLabel("就绪")
        right_layout.addWidget(self.current_status)
        self.current_bar = QProgressBar()
        right_layout.addWidget(self.current_bar)
        self.batch_status = QLabel("已处理 0 / 0")
        right_layout.addWidget(self.batch_status)
        self.batch_bar = QProgressBar()
        right_layout.addWidget(self.batch_bar)
        self.detail = QPlainTextEdit()
        self.detail.setReadOnly(True)
        self.detail.setPlaceholderText("选择视频查看结果和需要复核的剪辑点")
        right_layout.addWidget(self.detail, 1)
        splitter.setSizes([470, 720])
        self.mode.currentIndexChanged.connect(self.mode_changed)
        self.mode.currentIndexChanged.connect(self.refresh_files)
        self.mode_changed()

    def mode_changed(self):
        single = self.mode.currentData() == "single"
        for widget in (self.organization, self.person_title, self.name):
            widget.setEnabled(single)

    def choose_file(self, widget: QLineEdit, file_filter: str):
        path, _ = QFileDialog.getOpenFileName(self, "选择文件", widget.text(), file_filter)
        if path:
            widget.setText(path)

    def choose_dir(self, widget: QLineEdit):
        path = QFileDialog.getExistingDirectory(self, "选择文件夹", widget.text())
        if path:
            widget.setText(path)

    def choose_source(self, widget: QLineEdit):
        if self.mode.currentData() == "folder":
            self.choose_dir(widget)
        else:
            self.choose_file(widget, "视频 (*.mp4 *.mov *.mkv *.m4v *.avi *.webm)")
        self.refresh_files()

    def pick_region(self, target: QLineEdit):
        path = Path(self.template.text().strip())
        if not path.is_file():
            QMessageBox.warning(self, "缺少模板", "请先选择模板图片")
            return
        dialog = RegionDialog(path, target.text(), target is self.slot, self)
        if dialog.exec():
            target.setText(dialog.selection.text())

    def auto_detect_slot(self):
        path = Path(self.template.text().strip())
        if not path.is_file():
            QMessageBox.warning(self, "缺少模板", "请先选择模板图片")
            return
        try:
            with Image.open(path) as image:
                slot = find_slot(image)
            self.slot.setText(",".join(map(str, slot)))
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "无法识别透明区域", str(exc))

    def update_topic_color_button(self):
        self.topic_color_button.setText(self.topic_color.upper())
        self.topic_color_button.setStyleSheet(
            f"background:{self.topic_color};color:{'#111111' if QColor(self.topic_color).lightness() > 125 else '#FFFFFF'};")

    def current_caption_style(self):
        return CaptionStyle(font_size=0 if self.caption_auto_size.isChecked() else self.caption_size.value(),
                            text_color=self.caption_color, outline_color=self.caption_outline_color,
                            outline_width=self.caption_outline.value(), font_path=self.caption_font.currentData() or "")

    def current_caption_font(self):
        from studio.render import default_font
        return self.caption_font.currentData() or self.font.text().strip() or default_font()

    def update_caption_color_buttons(self):
        for button, color in ((self.caption_color_button, self.caption_color),
                              (self.caption_outline_color_button, self.caption_outline_color)):
            button.setText(color.upper())
            button.setStyleSheet(f"background:{color};color:{'#111111' if QColor(color).lightness() > 125 else '#FFFFFF'};")

    def pick_caption_color(self, outline):
        old = self.caption_outline_color if outline else self.caption_color
        color = QColorDialog.getColor(QColor(old), self, "选择描边颜色" if outline else "选择字幕文字颜色")
        if color.isValid():
            if outline:
                self.caption_outline_color = color.name().upper()
            else:
                self.caption_color = color.name().upper()
            self.update_caption_color_buttons()
            self.caption_preview.update()

    def pick_caption_font(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择字幕字体文件", "", "字体 (*.ttf *.ttc *.otf)")
        if path:
            if self.caption_font.findData(path) < 0:
                self.caption_font.addItem(Path(path).name, path)
            self.caption_font.setCurrentIndex(self.caption_font.findData(path))

    def pick_topic_color(self):
        color = QColorDialog.getColor(QColor(self.topic_color), self, "选择主题文字颜色")
        if color.isValid():
            self.topic_color = color.name().upper()
            self.update_topic_color_button()
            self.show_preview()

    def pick_topic_font(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择主题字体", "", "字体 (*.ttf *.ttc *.otf)")
        if path:
            index = self.topic_font.findData(path)
            if index < 0:
                self.topic_font.addItem(Path(path).name, path)
                index = self.topic_font.findData(path)
            self.topic_font.setCurrentIndex(index)

    def current_topic_style(self):
        return TopicStyle(font_size=0 if self.topic_auto_size.isChecked() else self.topic_size.value(),
                          effect=self.topic_effect.currentData(), background=self.topic_background.currentData(),
                          background_color=self.topic_background_color,
                          background_opacity=self.topic_background_opacity.value(),
                          accent_color=self.topic_accent_color, outline_color=self.topic_outline_color,
                          outline_width=self.topic_outline_width.value())

    def update_topic_style_controls(self):
        for button, color, label in ((self.topic_background_button, self.topic_background_color, "背景色"),
                                     (self.topic_accent_button, self.topic_accent_color, "副色"),
                                     (self.topic_outline_button, self.topic_outline_color, "描边色")):
            button.setText(label if color else "自动描边")
            button.setToolTip(color or "自动选择与主题文字对比的颜色")
            if color:
                ink = "#111111" if QColor(color).lightness() > 125 else "#FFFFFF"
                button.setStyleSheet(f"background:{color};color:{ink};")
            else:
                button.setStyleSheet("")
        background = self.topic_background.currentData() != "none"
        self.topic_background_button.setEnabled(background)
        self.topic_background_opacity.setEnabled(background)
        self.topic_accent_button.setEnabled(self.topic_effect.currentData() == "gradient")
        self.topic_outline_button.setEnabled(self.topic_effect.currentData() != "plain")
        self.topic_outline_width.setEnabled(self.topic_effect.currentData() != "plain")

    def pick_topic_style_color(self, kind):
        attribute = f"topic_{kind}_color"
        color = QColorDialog.getColor(QColor(getattr(self, attribute) or "#173863"), self,
                                      {"background": "选择主题背景色", "accent": "选择双色花字副色",
                                       "outline": "选择主题描边色"}[kind])
        if color.isValid():
            setattr(self, attribute, color.name().upper())
            self.update_topic_style_controls()
            self.show_preview()

    def reset_topic_outline(self):
        self.topic_outline_color = ""
        self.update_topic_style_controls()
        self.show_preview()

    @staticmethod
    def preview_clock(seconds):
        value = max(0, round(seconds))
        return f"{value // 3600:02d}:{value // 60 % 60:02d}:{value % 60:02d}" if value >= 3600 else f"{value // 60:02d}:{value % 60:02d}"

    def sync_video_preview(self):
        if not hasattr(self, "preview_seek"):
            return
        path = None
        source = Path(self.source.text().strip())
        if self.mode.currentData() == "single":
            if source.is_file() and source.suffix.lower() in VIDEO_EXTENSIONS:
                path = source
        elif source.is_dir():
            item = self.selected_item() if hasattr(self, "table") else None
            item = item or (self.items[0] if self.items else None)
            if item and item.path.parent.resolve() == source.resolve():
                path = item.path
        ffmpeg, ffprobe = self.tool_paths
        if path and (not ffmpeg or not ffprobe):
            self.preview_video_status.setText("正在等待 FFmpeg / FFprobe；可在工具设置中选择程序路径。")
        ready = path if ffmpeg and ffprobe else None
        try:
            key = (str(ready.resolve()), ready.stat().st_mtime_ns, ready.stat().st_size, ffmpeg, ffprobe) if ready else None
        except OSError:
            ready, key = None, None
        if key != self._preview_source_key:
            self._preview_source_key = key
            self.preview_frame = None
            self.preview_seconds = self.preview_duration = 0.0
            self.preview_seek.blockSignals(True)
            self.preview_seek.setRange(0, 0)
            self.preview_seek.setValue(0)
            self.preview_seek.blockSignals(False)
            self.preview_seek.setEnabled(False)
            self.preview_time.setText("00:00 / 00:00")
            self.show_preview()
        self.video_preview.set_source(ready, ffmpeg, ffprobe)

    def preview_duration_ready(self, duration):
        self.preview_duration = duration
        self.preview_seek.setRange(0, max(0, round(duration * 1000)))
        self.preview_seek.setEnabled(duration > 0)
        self.preview_time.setText(f"{self.preview_clock(self.preview_seconds)} / {self.preview_clock(duration)}")

    def preview_frame_ready(self, frame, seconds):
        self.preview_frame, self.preview_seconds = frame, seconds
        if not self.preview_seek.isSliderDown():
            self.preview_seek.blockSignals(True)
            self.preview_seek.setValue(round(seconds * 1000))
            self.preview_seek.blockSignals(False)
        self.preview_time.setText(f"{self.preview_clock(seconds)} / {self.preview_clock(self.preview_duration)}")
        self.show_preview()

    def preview_status_changed(self, message):
        if hasattr(self, "preview_video_status"):
            self.preview_video_status.setText(message or "拖动时间查看画面；主题、区域和姓名条修改后立即更新。")

    def seek_preview(self, milliseconds):
        seconds = milliseconds / 1000
        self.preview_time.setText(f"{self.preview_clock(seconds)} / {self.preview_clock(self.preview_duration)}")
        self.video_preview.seek(seconds)

    def show_preview(self):
        from studio.render import default_font, graphics, draw_topic
        path = Path(self.template.text().strip())
        if not path.is_file():
            self.preview_scope.setEnabled(False)
            if self.preview_frame is not None:
                self.preview.setPixmap(QPixmap.fromImage(ImageQt(self.preview_frame)).scaled(
                    540, 420, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                self.preview_feedback.setText("已显示原视频；选择竖屏模板后可查看主题和姓名条的合成位置。")
            else:
                self.preview.setPixmap(QPixmap())
                self.preview.setText("导入视频和模板后显示预览")
                self.preview_feedback.setText("红框：视频区域 · 橙框：主题区域 · 绿框：姓名条区域")
            return
        try:
            with Image.open(path) as source:
                image = source.convert("RGBA")
        except (OSError, ValueError):
            self.preview.setPixmap(QPixmap())
            self.preview.setText("无法读取模板图片，请重新选择")
            return
        regions, problems = [], []
        parsed = {}
        for key, field, label, color in (("slot", self.slot, "视频区域", "#FF2424"),
                ("topic", self.topic_box, "主题区域", "#FFB347"),
                ("nameplate", self.nameplate_box, "姓名条区域", "#54D6A0")):
            try:
                box = parse_slot(field.text())
                if key == "slot" and box is None:
                    try:
                        box = find_slot(image)
                    except ValueError:
                        pass
                if box:
                    if key == "slot":
                        validate_slot(box, image.size)
                    else:
                        validate_box(box, image.size, label)
                    regions.append((box, label, color))
                parsed[key] = box
            except ValueError as exc:
                parsed[key] = None
                problems.append(f"{label}无效：{exc}")
        if self.preview_frame is not None and parsed.get("slot"):
            x, y, width, height = parsed["slot"]
            image.alpha_composite(fit_frame(self.preview_frame, (width, height)).convert("RGBA"), (x, y))
        try:
            font_path = self.font.text().strip() or default_font()
            topic = self.topic.text().strip() if parsed.get("topic") else ""
            item = self.selected_item() if hasattr(self, "table") and self.mode.currentData() == "folder" else None
            person = item.person if item else None
            if item and item.topic and parsed.get("topic"):
                topic = item.topic
            if parsed.get("slot"):
                overlay = graphics(image.size, parsed["slot"], person.name if person else self.name.text().strip() or "姓名",
                                   person.organization if person else self.organization.text().strip() or "工作单位",
                                   person.title if person else self.person_title.text().strip() or "头衔", self.notice.toPlainText().strip(),
                                   font_path, None, item.nameplate_style or self.style.currentData() if item else self.style.currentData(), topic, parsed.get("topic"),
                                   self.topic_color, self.topic_font.currentData() or None,
                                   nameplate_scale=self.nameplate_size.value() / 100,
                                   nameplate_box=parsed.get("nameplate"), topic_style=self.current_topic_style())
            else:
                overlay = Image.new("RGBA", image.size)
                draw_topic(overlay, topic, parsed.get("topic"), self.topic_color,
                           self.topic_font.currentData() or font_path, topic_style=self.current_topic_style())
            image = Image.alpha_composite(image, overlay)
        except (OSError, ValueError, TypeError) as exc:
            problems.append(f"文字预览：{exc}")
        slot = parsed.get("slot")
        self.preview_scope.setEnabled(bool(slot))
        origin_x, origin_y = (slot[0], slot[1]) if slot and self.preview_scope.currentData() == "video" else (0, 0)
        display = image.crop((slot[0], slot[1], slot[0] + slot[2], slot[1] + slot[3])) if slot and self.preview_scope.currentData() == "video" else image
        picture = QPixmap.fromImage(ImageQt(display)).scaled(540, 420, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        painter = QPainter(picture)
        sx, sy = picture.width() / display.width, picture.height() / display.height
        for (x, y, width, height), label, color in regions:
            rect = QRect(round((x - origin_x) * sx), round((y - origin_y) * sy), max(1, round(width * sx)), max(1, round(height * sy)))
            painter.setPen(QPen(QColor(color), 2))
            painter.drawRect(rect.adjusted(1, 1, -1, -1))
        painter.end()
        self.preview.setPixmap(picture)
        self.preview_feedback.setText("；".join(problems) if problems else
            "红框：视频区域 · 橙框：主题区域 · 绿框：姓名条区域\n区域边框仅用于预览，文字与姓名条会导出。")

    def detect_media_tools(self):
        if self.tool_worker and self.tool_worker.isRunning():
            return
        self.tool_status.setText("正在检测 FFmpeg / FFprobe…")
        self.tool_worker = ToolWorker(self.ffmpeg.text(), self.ffprobe.text())
        self.tool_worker.found.connect(self.tools_found)
        self.tool_worker.start()

    def tools_found(self, ffmpeg: str, ffprobe: str, ffmpeg_info: str, ffprobe_info: str):
        self.tool_paths = ffmpeg, ffprobe
        self.tool_status.setText(f"FFmpeg：{'可用' if ffmpeg else ffmpeg_info}；FFprobe：{'可用' if ffprobe else ffprobe_info}")
        if ffmpeg:
            self.ffmpeg.setText(ffmpeg)
        if ffprobe:
            self.ffprobe.setText(ffprobe)
        self.sync_video_preview()

    def open_strategy(self):
        path = strategy_path()
        if not path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(load_strategy().snapshot(), ensure_ascii=False, indent=2), encoding="utf-8")
        open_path(path)

    def open_prompt(self, stage: str):
        path = prompts_dir() / f"{stage}_prompt.txt"
        if path.is_file():
            open_path(path)
        else:
            QMessageBox.warning(self, "缺少提示词", f"找不到提示词文件：{path}")

    def refresh_files(self):
        self.items, self.invalid = [], []
        try:
            source = Path(self.source.text().strip())
            if self.mode.currentData() == "folder":
                if source.is_dir():
                    self.items, self.invalid = scan_folder(source)
            elif source.is_file() and source.suffix.lower() in VIDEO_EXTENSIONS:
                person = Person(self.organization.text().strip(), self.person_title.text().strip(),
                                self.name.text().strip())
                self.items = [VideoItem(source, person)]
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "素材清单", str(exc))
        self.table.setRowCount(0)
        self.rows.clear()
        for item in self.items:
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.rows[str(item.path)] = row
            self.table.setItem(row, 0, QTableWidgetItem(item.path.name))
            self.table.setItem(row, 1, QTableWidgetItem(item.person.name))
            self.table.setItem(row, 2, QTableWidgetItem("待处理"))
        self.batch_status.setText(f"已识别 {len(self.items)} 条" + (f" · {len(self.invalid)} 条命名不合规" if self.invalid else ""))
        if self.items:
            self.table.selectRow(0)
        self.sync_video_preview()
        self.show_preview()

    def selected_item(self) -> VideoItem | None:
        row = self.table.currentRow()
        return self.items[row] if 0 <= row < len(self.items) else None

    def show_selected(self):
        item = self.selected_item()
        if item is None:
            return
        self.sync_video_preview()
        status, detail, run = self.results.get(str(item.path), ("待处理", "", {}))
        lines = [f"{item.path.name} · {status}", detail]
        plan_path = work_dir(item, Path(self.output_dir.text().strip())) / "edit-plan.json"
        lines.extend(result_plan_details(plan_path, run))
        self.detail.setPlainText("\n".join(lines))
        self.show_preview()

    def edit_selected_captions(self):
        if self.batch_worker and self.batch_worker.isRunning():
            return
        item = self.selected_item()
        if item is None:
            QMessageBox.information(self, "编辑字幕", "请先在文件列表选择视频")
            return
        from studio.qt_caption_editor import CaptionEditor
        try:
            dialog = CaptionEditor(work_dir(item, Path(self.output_dir.text().strip())), self)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            QMessageBox.warning(self, "无法打开字幕", f"请先完成字幕生成，并保留原视频及输出目录。\n{exc}")
            return
        dialog.exec()
        if dialog.export_requested:
            self.start(True, render_only=True)

    def validate_job(self, selected_only=False, *, render_only=False):
        selected_path = str(self.selected_item().path) if selected_only and self.selected_item() else ""
        self.refresh_files()
        if self.invalid and not render_only:
            raise ValueError(f"有 {len(self.invalid)} 个视频命名不合规；可用 batch.csv 指定人物信息")
        if not self.items:
            raise ValueError("没有可处理的视频")
        if self.mode.currentData() == "single":
            self.items[0].person.validate()
        selected = next((item for item in self.items if str(item.path) == selected_path), None) if selected_only else None
        items = [selected] if selected_only and selected else self.items
        if selected_only and not selected:
            raise ValueError("请先选择要重试的视频")
        client = None if render_only else BailianClient.from_base_url(self.bailian_url.text(), self.api_key.text().strip(),
                                             self.asr_model.text(), self.text_model.text())
        ffmpeg, ffprobe = self.tool_paths
        if not ffmpeg or not ffprobe:
            raise ValueError("未检测到 FFmpeg / FFprobe。安装后点击“重新检测”，或手动选择 exe")
        template = Path(self.template.text().strip())
        if not template.is_file():
            raise ValueError("请选择竖屏模板")
        with Image.open(template) as image:
            slot = parse_slot(self.slot.text()) or find_slot(image)
            validate_slot(slot, image.size)
            topic_box = parse_slot(self.topic_box.text())
            nameplate_box = parse_slot(self.nameplate_box.text())
            if nameplate_box:
                try:
                    validate_box(nameplate_box, image.size, "姓名条区域")
                except ValueError as exc:
                    raise ValueError("姓名条区域超出模板边界") from exc
            if topic_box:
                x, y, w, h = topic_box
                if min(x, y) < 0 or min(w, h) <= 0 or x + w > image.width or y + h > image.height:
                    raise ValueError("主题区域超出模板边界")
        if (self.topic.text().strip() or any(item.topic for item in items)) and not topic_box:
            raise ValueError("已填写主题，请先框选主题区域")
        for item in items:
            if item.nameplate_style and item.nameplate_style not in NAMEPLATE_STYLE_IDS:
                raise ValueError(f"{item.path.name} 的姓名条样式无效，请检查 batch.csv 的 nameplate_style")
            if item.brightness and item.brightness not in ("normal", "bright", "dark", "auto"):
                raise ValueError(f"{item.path.name} 的 brightness 无效")
        output = Path(self.output_dir.text().strip())
        if not self.output_dir.text().strip():
            raise ValueError("请选择输出文件夹")
        output.mkdir(parents=True, exist_ok=True)
        font = Path(self.font.text()) if self.font.text().strip() else None
        if font and not font.is_file():
            raise ValueError("字体文件不存在")
        topic_font = Path(self.topic_font.currentData()) if self.topic_font.currentData() else None
        if topic_font and not topic_font.is_file():
            raise ValueError("主题字体文件不存在")
        config = JobConfig(template, output, self.notice.toPlainText().strip(), slot,
                           ffmpeg, ffprobe, font,
                           datetime.now().strftime("%Y%m%d-%H%M%S-%f") if self.timestamp.isChecked() else "",
                           load_strategy(), self.topic.text().strip(), topic_box,
                           self.style.currentData(), self.brightness.currentData(), self.quality.currentData(),
                           self.brightness_target.value(), self.topic_color, topic_font,
                           PromptSet("", "") if render_only else load_prompts(), caption_style=self.current_caption_style(),
                           nameplate_scale=self.nameplate_size.value() / 100, nameplate_box=nameplate_box,
                           topic_style=self.current_topic_style())
        config.caption_style.validate()
        config.topic_style.validate()
        from studio.render import default_font
        from studio.topic_style import topic_layout
        for item in items:
            value = item.topic or config.topic
            if value:
                try:
                    topic_layout(value, topic_box[2:], topic_font or font or default_font(), config.topic_style)
                except ValueError as exc:
                    raise ValueError(f"{item.path.name}：{exc}") from exc
        return client, config, items

    def start_batch(self):
        self.start(False)

    def retry_selected(self):
        self.start(True)

    def start(self, selected_only: bool, *, render_only=False):
        if self.batch_worker and self.batch_worker.isRunning():
            return
        try:
            client, config, items = self.validate_job(selected_only, render_only=render_only)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "无法开始", str(exc))
            return
        self.save_current_settings()
        self.cancel.clear()
        self.finished_items = 0
        self.active_total = len(items)
        self.current_bar.setValue(0)
        self.batch_bar.setValue(0)
        self.batch_status.setText(f"已处理 0 / {len(items)}")
        self.start_button.setEnabled(False)
        self.edit_captions_button.setEnabled(False)
        self.export_captions_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.batch_worker = BatchWorker(client, config, items, self.cancel, render_only=render_only)
        self.batch_worker.progress.connect(self.on_progress)
        self.batch_worker.result.connect(self.on_result)
        self.batch_worker.finished_batch.connect(self.on_finished)
        self.batch_worker.start()

    def on_progress(self, path: str, percent: int, detail: str):
        if path in self.rows:
            self.table.item(self.rows[path], 2).setText(detail)
        self.current_status.setText(detail)
        self.current_bar.setValue(max(0, min(100, percent)))

    def on_result(self, path: str, status: str, detail: str, run):
        self.results[path] = (status, detail, run or {})
        if path in self.rows:
            self.table.item(self.rows[path], 2).setText(status)
        self.finished_items += 1
        self.batch_bar.setValue(round(100 * self.finished_items / max(1, self.active_total)))
        self.batch_status.setText(f"已处理 {self.finished_items} / {self.active_total}")
        if status == "已完成":
            self.current_bar.setValue(100)
        self.current_status.setText(f"{Path(path).name} · {status}")
        self.show_selected()

    def on_finished(self, completed: int, review: int, failed: int, cancelled: int):
        self.start_button.setEnabled(True)
        self.edit_captions_button.setEnabled(True)
        self.export_captions_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.batch_bar.setValue(100 if not cancelled else round(100 * (completed + failed) / max(1, self.active_total)))
        self.batch_status.setText(f"已完成 {completed} · 建议复核 {review} · 失败 {failed} · 取消 {cancelled}")
        self.current_status.setText("批处理结束")

    def cancel_batch(self):
        self.cancel.set()
        self.current_status.setText("正在取消当前任务…")

    def open_output(self):
        path = Path(self.output_dir.text().strip())
        if path.is_dir():
            open_path(path)

    def save_current_settings(self):
        text_fields = ("source", "template", "output_dir", "slot", "topic", "topic_box", "nameplate_box",
                       "organization", "person_title", "name", "bailian_url", "asr_model",
                       "text_model", "ffmpeg", "ffprobe", "font")
        value = {key: getattr(self, key).text() for key in text_fields}
        value.update({"mode": self.mode.currentData(), "notice": self.notice.toPlainText(),
                      "nameplate_style": self.style.currentData(),
                      "nameplate_size": self.nameplate_size.value(),
                      "brightness": self.brightness.currentData(), "quality": self.quality.currentData(),
                      "brightness_target": self.brightness_target.value(),
                      "topic_color": self.topic_color, "topic_font": self.topic_font.currentData(),
                      "topic_style": self.current_topic_style().snapshot(),
                      "caption_style": self.current_caption_style().snapshot(),
                      "output_timestamp": self.timestamp.isChecked()})
        save_settings(value)

    def closeEvent(self, event):
        self.save_current_settings()
        if self.batch_worker and self.batch_worker.isRunning():
            self.cancel.set()
            event.ignore()
            self.current_status.setText("正在结束任务，请稍后关闭…")
            return
        if self.tool_worker and self.tool_worker.isRunning():
            self.tool_worker.wait(6000)
        self.video_preview.close()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = App()
    window.show()
    marker = os.environ.get("VVS_STARTUP_MARKER")
    if marker:
        if os.environ.get("VVS_PREVIEW_SMOKE"):
            deadline = time.monotonic() + 8
            while window.preview_frame is None and time.monotonic() < deadline:
                app.processEvents()
                time.sleep(.01)
            if window.preview_frame is None or "文字预览" in window.preview_feedback.text():
                raise RuntimeError(f"发布包视频合成预览验证失败：{window.preview_video_status.text()}")
            if window.current_topic_style().background != "rounded":
                raise RuntimeError("发布包主题样式读取失败")
            original_width = window.preview.pixmap().width()
            window.preview_scope.setCurrentIndex(window.preview_scope.findData("video"))
            if window.preview.pixmap().width() <= original_width:
                raise RuntimeError("发布包视频区域放大失败")
        editor_folder = os.environ.get("VVS_EDITOR_SMOKE")
        if editor_folder:
            from studio.qt_caption_editor import CaptionEditor
            editor = CaptionEditor(Path(editor_folder), window)
            editor.show()
            app.processEvents()
            editor.table.item(0, 2).setText("我吃得不多")
            if not editor.save():
                raise RuntimeError("字幕编辑窗口保存验证失败")
            editor.close()
        Path(marker).write_text("ready", encoding="utf-8")
    sys.exit(app.exec())
