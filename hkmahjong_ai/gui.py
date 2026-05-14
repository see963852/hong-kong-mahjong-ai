from __future__ import annotations


def run_gui() -> None:
    try:
        from hkmahjong_ai.ui.app import run_gui as run_pyside_gui
    except ModuleNotFoundError as exc:
        if exc.name == "PySide6":
            raise SystemExit("PySide6 is required. Run: python -m pip install -r requirements.txt") from exc
        raise
    run_pyside_gui()
