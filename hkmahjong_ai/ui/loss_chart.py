"""pyqtgraph loss chart widget used by the training panel."""

from __future__ import annotations

import pyqtgraph as pg
from PySide6.QtWidgets import QVBoxLayout, QWidget


class LossChart(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.episodes: list[int] = []
        self.policy_losses: list[float] = []
        self.value_losses: list[float] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.plot = pg.PlotWidget(background="#082d27")
        self.plot.addLegend(offset=(12, 12))
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.setLabel("left", "Loss")
        self.plot.setLabel("bottom", "Episode")
        self.plot.getAxis("left").setTextPen("#d8eadf")
        self.plot.getAxis("bottom").setTextPen("#d8eadf")
        self.policy_curve = self.plot.plot(pen=pg.mkPen("#f2d98b", width=2), name="policy loss")
        self.value_curve = self.plot.plot(pen=pg.mkPen("#78d6c6", width=2), name="value loss")
        layout.addWidget(self.plot)

    def clear(self) -> None:
        self.episodes.clear()
        self.policy_losses.clear()
        self.value_losses.clear()
        self.policy_curve.setData([], [])
        self.value_curve.setData([], [])

    def add_point(self, episode: int, policy_loss: float, value_loss: float) -> None:
        self.episodes.append(episode)
        self.policy_losses.append(policy_loss)
        self.value_losses.append(value_loss)
        self.episodes = self.episodes[-200:]
        self.policy_losses = self.policy_losses[-200:]
        self.value_losses = self.value_losses[-200:]
        self.policy_curve.setData(self.episodes, self.policy_losses)
        self.value_curve.setData(self.episodes, self.value_losses)
        if self.episodes:
            self.plot.setXRange(min(self.episodes), max(self.episodes) or min(self.episodes) + 1, padding=0.04)
