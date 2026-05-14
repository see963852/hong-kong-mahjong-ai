from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QRectF, QSize, Qt
from PySide6.QtGui import QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSET_ROOT = PROJECT_ROOT / "assets"
TILE_ROOT = ASSET_ROOT / "tiles"
UI_ROOT = ASSET_ROOT / "ui"


class AssetManager:
    def __init__(self) -> None:
        self._pixmaps: dict[tuple[str, int, int], QPixmap] = {}

    def _preferred(self, path: Path) -> Path:
        png = path.with_suffix(".png")
        if png.exists():
            return png
        return path

    def tile_path(self, tile: int) -> Path:
        return self._preferred(TILE_ROOT / f"tile_{tile:02d}.svg")

    def tile_back_path(self) -> Path:
        return self._preferred(TILE_ROOT / "tile_back.svg")

    def ui_path(self, name: str) -> Path:
        return self._preferred(UI_ROOT / name)

    def pixmap(self, path: Path, size: QSize) -> QPixmap:
        key = (str(path), size.width(), size.height())
        cached = self._pixmaps.get(key)
        if cached is not None:
            return cached
        if path.suffix.lower() == ".svg":
            pixmap = QPixmap(size)
            pixmap.fill("transparent")
            renderer = QSvgRenderer(str(path))
            painter = QPainter(pixmap)
            renderer.render(painter, QRectF(0, 0, size.width(), size.height()))
            painter.end()
        else:
            pixmap = QPixmap(str(path)).scaled(
                size,
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        self._pixmaps[key] = pixmap
        return pixmap

    def tile_pixmap(self, tile: int, size: QSize) -> QPixmap:
        return self.pixmap(self.tile_path(tile), size)

    def tile_back(self, size: QSize) -> QPixmap:
        return self.pixmap(self.tile_back_path(), size)
