from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QFont, QFontDatabase


WINDOWS_FONT_CANDIDATES = [
    Path("C:/Windows/Fonts/msjh.ttc"),
    Path("C:/Windows/Fonts/msyh.ttc"),
    Path("C:/Windows/Fonts/mingliu.ttc"),
    Path("C:/Windows/Fonts/simsun.ttc"),
]


def load_cjk_font() -> str:
    for path in WINDOWS_FONT_CANDIDATES:
        if not path.exists():
            continue
        font_id = QFontDatabase.addApplicationFont(str(path))
        if font_id >= 0:
            families = QFontDatabase.applicationFontFamilies(font_id)
            if families:
                return families[0]
    return "Microsoft JhengHei"


def default_cjk_qfont(point_size: int = 10, weight: QFont.Weight = QFont.Weight.Normal) -> QFont:
    family = load_cjk_font()
    return QFont(family, point_size, weight)
