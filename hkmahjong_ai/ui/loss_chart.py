"""Small Qt loss chart widget used by the training panel."""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget


class LossChart(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.policy_points: list[tuple[int, float]] = []
        self.value_points: list[tuple[int, float]] = []
        self.setMinimumHeight(190)

    def clear(self) -> None:
        self.policy_points.clear()
        self.value_points.clear()
        self.update()

    def add_point(self, episode: int, policy_loss: float, value_loss: float) -> None:
        self.policy_points.append((episode, policy_loss))
        self.value_points.append((episode, value_loss))
        self.policy_points = self.policy_points[-300:]
        self.value_points = self.value_points[-300:]
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#082d27"))
        area = self.rect().adjusted(46, 18, -18, -32)

        painter.setPen(QPen(QColor("#b89446"), 1))
        painter.drawRect(area)
        painter.setFont(QFont("Microsoft JhengHei", 9))
        painter.setPen(QColor("#f8edc7"))
        painter.drawText(QRectF(10, 4, self.width() - 20, 18), Qt.AlignmentFlag.AlignLeft, "Loss 曲線")

        all_values = [v for _, v in self.policy_points + self.value_points]
        if not all_values:
            painter.setPen(QColor("#d8eadf"))
            painter.drawText(area, Qt.AlignmentFlag.AlignCenter, "訓練開始後顯示 policy loss / value loss")
            return

        min_episode = min(ep for ep, _ in self.policy_points + self.value_points)
        max_episode = max(ep for ep, _ in self.policy_points + self.value_points)
        min_value = min(all_values)
        max_value = max(all_values)
        if abs(max_value - min_value) < 1e-6:
            max_value += 1.0
            min_value -= 1.0
        if max_episode == min_episode:
            max_episode += 1

        self._draw_line(painter, area, self.policy_points, min_episode, max_episode, min_value, max_value, "#f2d98b")
        self._draw_line(painter, area, self.value_points, min_episode, max_episode, min_value, max_value, "#78d6c6")

        painter.setPen(QColor("#f2d98b"))
        painter.drawText(54, self.height() - 10, "policy")
        painter.setPen(QColor("#78d6c6"))
        painter.drawText(118, self.height() - 10, "value")

    def _draw_line(
        self,
        painter: QPainter,
        area,
        points: list[tuple[int, float]],
        min_episode: int,
        max_episode: int,
        min_value: float,
        max_value: float,
        color: str,
    ) -> None:
        if len(points) < 2:
            return
        mapped = []
        for episode, value in points:
            x = area.left() + (episode - min_episode) / (max_episode - min_episode) * area.width()
            y = area.bottom() - (value - min_value) / (max_value - min_value) * area.height()
            mapped.append(QPointF(x, y))
        painter.setPen(QPen(QColor(color), 2))
        for left, right in zip(mapped, mapped[1:]):
            painter.drawLine(left, right)
