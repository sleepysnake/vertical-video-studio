"""Exercise the region editor's interaction states without a visible desktop."""
from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from qt_app import RegionCanvas


class RegionCanvasTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_drag_inside_moves_and_border_resizes_with_canvas_bounds(self):
        picture = QPixmap(200, 120)
        picture.fill(Qt.white)
        canvas = RegionCanvas(picture, 1, "20,20,50,40")
        values = []
        canvas.changed.connect(values.append)

        QTest.mousePress(canvas, Qt.LeftButton, pos=QPoint(45, 40))
        QTest.mouseMove(canvas, QPoint(90, 60))
        QTest.mouseRelease(canvas, Qt.LeftButton, pos=QPoint(90, 60))
        self.assertEqual(values[-1], "65,40,50,40")

        QTest.mousePress(canvas, Qt.LeftButton, pos=QPoint(115, 60))
        QTest.mouseMove(canvas, QPoint(140, 60))
        QTest.mouseRelease(canvas, Qt.LeftButton, pos=QPoint(140, 60))
        self.assertEqual(values[-1], "65,40,75,40")

        QTest.mousePress(canvas, Qt.LeftButton, pos=QPoint(90, 60))
        QTest.mouseMove(canvas, QPoint(195, 110))
        QTest.mouseRelease(canvas, Qt.LeftButton, pos=QPoint(195, 110))
        self.assertEqual(values[-1], "125,80,75,40")


if __name__ == "__main__":
    unittest.main()
