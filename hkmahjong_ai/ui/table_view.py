from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QPoint, QRect, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPixmap
from PySide6.QtWidgets import QWidget

from hkmahjong_ai.env import HKMahjongEnv
from hkmahjong_ai.tiles import names_from_tiles, tile_name

from .assets import AssetManager


@dataclass
class TileHit:
    rect: QRect
    tile: int


class GameTableWidget(QWidget):
    discardRequested = Signal(int)
    tileSelected = Signal(int)

    def __init__(self, assets: AssetManager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.assets = assets
        self.env: HKMahjongEnv | None = None
        self.human_mode = True
        self.show_all_hands = False
        self.selected_tile: int | None = None
        self.tile_hits: list[TileHit] = []
        self.setMinimumSize(920, 610)
        self.setMouseTracking(True)

    def set_game(self, env: HKMahjongEnv, human_mode: bool, show_all_hands: bool = False) -> None:
        self.env = env
        self.human_mode = human_mode
        self.show_all_hands = show_all_hands
        self.update()

    def set_selected_tile(self, tile: int | None) -> None:
        self.selected_tile = tile
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        self.tile_hits = []

        background = self.assets.pixmap(self.assets.ui_path("table_background.svg"), self.size())
        painter.drawPixmap(self.rect(), background)
        self._draw_center_hud(painter)

        if self.env is None:
            return
        self._draw_seats(painter)
        self._draw_discards(painter)
        self._draw_melds(painter)
        self._draw_hands(painter)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if self.env is None or not self.human_mode or self.env.done or self.env.current_player != 0:
            return
        pos = event.position().toPoint()
        for hit in self.tile_hits:
            if hit.rect.contains(pos):
                self.tileSelected.emit(hit.tile)
                return

    def _draw_center_hud(self, painter: QPainter) -> None:
        if self.env is None:
            return
        w, h = self.width(), self.height()
        panel = QRect(w // 2 - 210, h // 2 - 70, 420, 118)
        pixmap = self.assets.pixmap(self.assets.ui_path("status_panel.svg"), panel.size())
        painter.drawPixmap(panel, pixmap)
        painter.setPen(QColor("#f8edc7"))
        painter.setFont(QFont("Microsoft JhengHei", 15, QFont.Weight.Bold))
        title = self.env.result.message if self.env.result else f"輪到 P{self.env.current_player + 1}"
        painter.drawText(panel.adjusted(18, 20, -18, -58), Qt.AlignmentFlag.AlignCenter, title)
        painter.setFont(QFont("Microsoft JhengHei", 10))
        detail = f"牌牆 {len(self.env.wall)}　回合 {self.env.turns}"
        if self.env.result and self.env.result.pattern:
            detail += f"　牌型 {self.env.result.pattern}"
        if self.env.last_discard:
            detail += f"　上張 {tile_name(self.env.last_discard[1])}"
        painter.drawText(panel.adjusted(18, 62, -18, -20), Qt.AlignmentFlag.AlignCenter, detail)

    def _draw_seats(self, painter: QPainter) -> None:
        assert self.env is not None
        w, h = self.width(), self.height()
        rects = [
            QRect(w // 2 - 180, h - 132, 360, 88),
            QRect(24, h // 2 - 70, 270, 82),
            QRect(w // 2 - 180, 34, 360, 88),
            QRect(w - 294, h // 2 - 70, 270, 82),
        ]
        winds = ["東", "南", "西", "北"]
        for pid, rect in enumerate(rects):
            pixmap = self.assets.pixmap(self.assets.ui_path("seat_frame.svg"), rect.size())
            painter.drawPixmap(rect, pixmap)
            active = pid == self.env.current_player and not self.env.done
            painter.setPen(QColor("#fff2c2" if active else "#d8eadf"))
            painter.setFont(QFont("Microsoft JhengHei", 12, QFont.Weight.Bold))
            role = "玩家" if self.human_mode and pid == 0 else "AI"
            painter.drawText(rect.adjusted(76, 14, -12, -44), Qt.AlignmentFlag.AlignLeft, f"P{pid + 1} {role}｜{winds[pid]}")
            painter.setFont(QFont("Microsoft JhengHei", 9))
            melds = len(self.env.players[pid].melds)
            discards = len(self.env.players[pid].discards)
            painter.drawText(rect.adjusted(76, 43, -12, -10), Qt.AlignmentFlag.AlignLeft, f"副露 {melds}　牌河 {discards}")

    def _draw_discards(self, painter: QPainter) -> None:
        assert self.env is not None
        w, h = self.width(), self.height()
        centers = [
            QPoint(w // 2, h // 2 + 112),
            QPoint(w // 2 - 245, h // 2),
            QPoint(w // 2, h // 2 - 152),
            QPoint(w // 2 + 245, h // 2),
        ]
        for pid, center in enumerate(centers):
            tiles = self.env.players[pid].discards[-18:]
            if not tiles:
                continue
            angle = 0 if pid in (0, 2) else 90
            tile_size = QSize(31, 42)
            cols = 6
            rows = (len(tiles) + cols - 1) // cols
            start_x = center.x() - cols * 17
            start_y = center.y() - rows * 23
            for i, tile in enumerate(tiles):
                x = start_x + (i % cols) * 34
                y = start_y + (i // cols) * 45
                self._draw_tile(painter, tile, QRect(x, y, tile_size.width(), tile_size.height()), angle=angle)

    def _draw_melds(self, painter: QPainter) -> None:
        assert self.env is not None
        w, h = self.width(), self.height()
        bases = [
            QPoint(w - 385, h - 188),
            QPoint(36, h // 2 + 58),
            QPoint(w - 385, 138),
            QPoint(w - 146, h // 2 + 58),
        ]
        for pid, base in enumerate(bases):
            x, y = base.x(), base.y()
            for meld in self.env.players[pid].melds:
                for tile in meld.tiles:
                    self._draw_tile(painter, tile, QRect(x, y, 34, 46), angle=0 if pid in (0, 2) else 90)
                    x += 36 if pid in (0, 2) else 0
                    y += 0 if pid in (0, 2) else 36
                x += 10 if pid in (0, 2) else 0
                y += 0 if pid in (0, 2) else 10

    def _draw_hands(self, painter: QPainter) -> None:
        assert self.env is not None
        w, h = self.width(), self.height()
        hands = [player.hand for player in self.env.players]
        self._draw_hand_row(painter, hands[0], QRect(w // 2 - 392, h - 235, 784, 96), face_up=True, pid=0)
        self._draw_hand_row(painter, hands[2], QRect(w // 2 - 330, 138, 660, 76), face_up=self.show_all_hands, pid=2, small=True)
        self._draw_hand_column(painter, hands[1], QRect(74, h // 2 - 238, 74, 476), face_up=self.show_all_hands, pid=1)
        self._draw_hand_column(painter, hands[3], QRect(w - 148, h // 2 - 238, 74, 476), face_up=self.show_all_hands, pid=3)

    def _draw_hand_row(
        self,
        painter: QPainter,
        tiles: list[int],
        area: QRect,
        face_up: bool,
        pid: int,
        small: bool = False,
    ) -> None:
        tile_w = 46 if small else 55
        tile_h = 63 if small else 76
        gap = 4
        total = len(tiles) * (tile_w + gap) - gap
        x = area.center().x() - total // 2
        y = area.top() + (area.height() - tile_h) // 2
        for tile in tiles:
            rect = QRect(x, y, tile_w, tile_h)
            draw_rect = rect
            if pid == 0 and self.human_mode and tile == self.selected_tile:
                draw_rect = rect.translated(0, -12)
            if face_up:
                self._draw_tile(painter, tile, draw_rect)
            else:
                self._draw_tile_back(painter, draw_rect)
            if pid == 0 and self.human_mode and self.env and self.env.current_player == 0 and not self.env.done:
                self.tile_hits.append(TileHit(draw_rect, tile))
            x += tile_w + gap

    def _draw_hand_column(self, painter: QPainter, tiles: list[int], area: QRect, face_up: bool, pid: int) -> None:
        tile_w, tile_h, gap = 44, 60, 3
        total = len(tiles) * (tile_w + gap) - gap
        y = area.center().y() - total // 2
        x = area.left() + (area.width() - tile_h) // 2
        for tile in tiles:
            rect = QRect(x, y, tile_h, tile_w)
            if face_up:
                self._draw_tile(painter, tile, rect, angle=90)
            else:
                self._draw_tile_back(painter, rect, angle=90)
            y += tile_w + gap

    def _draw_tile(self, painter: QPainter, tile: int, rect: QRect, angle: int = 0) -> None:
        pixmap = self.assets.tile_pixmap(tile, QSize(rect.width(), rect.height()))
        self._draw_pixmap(painter, pixmap, rect, angle)

    def _draw_tile_back(self, painter: QPainter, rect: QRect, angle: int = 0) -> None:
        pixmap = self.assets.tile_back(QSize(rect.width(), rect.height()))
        self._draw_pixmap(painter, pixmap, rect, angle)

    def _draw_pixmap(self, painter: QPainter, pixmap: QPixmap, rect: QRect, angle: int = 0) -> None:
        painter.save()
        if angle:
            center = rect.center()
            painter.translate(center)
            painter.rotate(angle)
            draw_rect = QRect(-rect.height() // 2, -rect.width() // 2, rect.height(), rect.width())
            painter.drawPixmap(draw_rect, pixmap)
        else:
            painter.drawPixmap(rect, pixmap)
        painter.restore()
