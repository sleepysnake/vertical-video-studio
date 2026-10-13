"""Optional caption editing after fully automatic video production."""
from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QAbstractItemView, QDialog, QHBoxLayout, QHeaderView,
    QInputDialog, QLabel, QLineEdit, QMessageBox, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout)

from .caption_edits import load_edits, save_edits, validate_edits
from .captions import extract_words
from .edits import output_duration, retime_captions


class CaptionEditor(QDialog):
    def __init__(self, folder: Path, parent=None):
        super().__init__(parent)
        self.folder = folder
        self.data = json.loads((folder / "captions.json").read_text(encoding="utf-8"))
        self.raw = json.loads((folder / "transcription.json").read_text(encoding="utf-8"))
        self.words = extract_words(self.raw)
        self.automatic = retime_captions(self.data["captions"], self.data["edit_plan"]["cuts"])
        self.duration = output_duration(self.data["edit_plan"]["cuts"], self.data["edit_plan"]["source_duration"])
        self.export_requested = False
        self.setWindowTitle("编辑字幕")
        self.resize(1000, 650)
        layout = QVBoxLayout(self)
        tip = QLabel("双击修改文字或时间。时间为成片秒数；字幕修改不改变原声和画面剪辑。\n保存只更新字幕数据，重新导出后才会写入视频。")
        tip.setWordWrap(True)
        layout.addWidget(tip)
        search_row = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("搜索字幕或原始识别文字，如：咦")
        next_button = QPushButton("查找下一处")
        next_button.clicked.connect(self.find_next)
        self.search.returnPressed.connect(self.find_next)
        search_row.addWidget(self.search, 1)
        search_row.addWidget(next_button)
        layout.addLayout(search_row)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["开始时间（秒）", "结束时间（秒）", "字幕文字", "原始识别文字（参考）"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setColumnWidth(0, 125)
        self.table.setColumnWidth(1, 125)
        for column in (2, 3):
            self.table.horizontalHeader().setSectionResizeMode(column, QHeaderView.Stretch)
        layout.addWidget(self.table, 1)
        actions = QHBoxLayout()
        for text, callback in (("拆成两条", self.split_row), ("合并选中", self.merge_rows),
                               ("删除选中字幕", self.delete_rows), ("恢复自动字幕", self.restore_automatic)):
            button = QPushButton(text)
            button.clicked.connect(callback)
            actions.addWidget(button)
        layout.addLayout(actions)
        self.status = QLabel(f"成片时长 {self.duration:.3f} 秒")
        layout.addWidget(self.status)
        buttons = QHBoxLayout()
        buttons.addStretch()
        for text, callback in (("保存字幕", self.save), ("保存并重新导出", self.save_and_export), ("关闭", self.reject)):
            button = QPushButton(text)
            button.clicked.connect(callback)
            buttons.addWidget(button)
        layout.addLayout(buttons)
        self.populate(load_edits(folder, self.data) or self.automatic)

    def populate(self, captions):
        self.table.setRowCount(0)
        for item in captions:
            self.insert_row(self.table.rowCount(), item)

    def insert_row(self, row, item, reference=None):
        self.table.insertRow(row)
        if reference is None:
            first, last = item.get("start_word"), item.get("end_word")
            if isinstance(first, int) and isinstance(last, int):
                reference = "".join(w.text for w in self.words[first:last])
            else:
                indices = {index for c in self.automatic if c["end"] > item["start"] and c["start"] < item["end"]
                           for index in range(c.get("start_word", 0), c.get("end_word", 0))}
                reference = "".join(self.words[index].text for index in sorted(indices))
        for column, value in enumerate((f"{item['start']:.3f}", f"{item['end']:.3f}", item["text"], reference)):
            cell = QTableWidgetItem(value)
            if column == 3:
                cell.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            self.table.setItem(row, column, cell)

    def collect(self):
        return validate_edits([{"start": self.table.item(row, 0).text(),
                                "end": self.table.item(row, 1).text(),
                                "text": self.table.item(row, 2).text()}
                               for row in range(self.table.rowCount())], self.duration)

    def selected_rows(self):
        return sorted({cell.row() for cell in self.table.selectedIndexes()})

    def find_next(self):
        value = self.search.text().strip()
        count = self.table.rowCount()
        if not value or not count:
            return
        for offset in range(1, count + 1):
            row = (max(-1, self.table.currentRow()) + offset) % count
            if any(value in self.table.item(row, column).text() for column in (2, 3)):
                self.table.selectRow(row)
                self.table.scrollToItem(self.table.item(row, 2))
                return
        self.status.setText(f"未找到：{value}")

    def split_row(self):
        row = self.table.currentRow()
        if row < 0:
            return
        try:
            item = self.collect()[row]
        except ValueError as exc:
            QMessageBox.warning(self, "无法拆分", str(exc))
            return
        text = item["text"]
        if len(text) < 2:
            return
        position, accepted = QInputDialog.getInt(self, "拆分字幕", "在第几个字后拆分", len(text) // 2, 1, len(text) - 1)
        if not accepted:
            return
        middle = round(item["start"] + (item["end"] - item["start"]) * position / len(text), 3)
        reference = self.table.item(row, 3).text()
        self.table.removeRow(row)
        self.insert_row(row, {"start": item["start"], "end": middle, "text": text[:position]}, reference)
        self.insert_row(row + 1, {"start": middle, "end": item["end"], "text": text[position:]}, reference)

    def merge_rows(self):
        rows = self.selected_rows()
        if len(rows) < 2 or rows != list(range(rows[0], rows[-1] + 1)):
            QMessageBox.information(self, "合并字幕", "请选择相邻的两条或多条字幕")
            return
        try:
            captions = self.collect()
        except ValueError as exc:
            QMessageBox.warning(self, "无法合并", str(exc))
            return
        merged = {"start": captions[rows[0]]["start"], "end": captions[rows[-1]]["end"],
                  "text": "".join(captions[row]["text"] for row in rows)}
        reference = "".join(self.table.item(row, 3).text() for row in rows)
        for row in reversed(rows):
            self.table.removeRow(row)
        self.insert_row(rows[0], merged, reference)

    def delete_rows(self):
        for row in reversed(self.selected_rows()):
            self.table.removeRow(row)

    def restore_automatic(self):
        self.populate(self.automatic)
        self.status.setText("已恢复自动字幕；点击保存后生效")

    def save(self):
        try:
            save_edits(self.folder, self.data, self.collect())
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "无法保存字幕", str(exc))
            return False
        self.status.setText("字幕已保存；重新导出后写入视频")
        return True

    def save_and_export(self):
        if self.save():
            self.export_requested = True
            self.accept()
