from __future__ import annotations

import os
import random
import sys
import threading
from pathlib import Path

from PySide6.QtCore import QObject, QTimer, Signal, Qt
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from hkmahjong_ai.agents import HeuristicAgent, ModelAgent
from hkmahjong_ai.env import HKMahjongEnv
from hkmahjong_ai.evaluate import evaluate_model
from hkmahjong_ai.model import LinearDiscardModel
from hkmahjong_ai.train import train_model

from .assets import AssetManager, PROJECT_ROOT
from .fonts import load_cjk_font
from .table_view import GameTableWidget


class TaskBridge(QObject):
    finished = Signal(str)


class MahjongWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("香港麻雀 AI 訓練平台")
        self.resize(1280, 820)
        self.setMinimumSize(1040, 720)

        self.assets = AssetManager()
        self.font_family = load_cjk_font()
        self.env = HKMahjongEnv()
        self.human_mode = True
        self.auto_running = False
        self.model_path = str(PROJECT_ROOT / "models" / "model_latest.json")
        self.agents: list[object] = []
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._auto_tick)
        self.bridge = TaskBridge()
        self.bridge.finished.connect(self._task_finished)

        self._build_ui()
        self._apply_style()
        self.new_game(True)

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(14, 12, 14, 14)
        layout.setSpacing(10)

        header = QHBoxLayout()
        self.title_label = QLabel("香港麻雀 AI 牌桌")
        self.title_label.setObjectName("TitleLabel")
        self.status_label = QLabel("準備開始")
        self.status_label.setObjectName("StatusLabel")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        header.addWidget(self.title_label)
        header.addWidget(self.status_label, 1)
        layout.addLayout(header)

        self.table = GameTableWidget(self.assets)
        self.table.discardRequested.connect(self.human_discard)
        layout.addWidget(self.table, 1)

        controls = QFrame()
        controls.setObjectName("ControlBar")
        control_layout = QHBoxLayout(controls)
        control_layout.setContentsMargins(12, 10, 12, 10)
        control_layout.setSpacing(8)

        self.btn_human = self._button("玩家對 AI", lambda: self.new_game(True))
        self.btn_ai = self._button("AI 對 AI", lambda: self.new_game(False))
        self.btn_step = self._button("單步", self.step_ai)
        self.btn_auto = self._button("自動/暫停", self.toggle_auto)
        self.btn_win = self._button("自摸", self.human_win)
        self.btn_train = self._button("訓練 200 局", self.train_quick)
        self.btn_eval = self._button("評估 100 局", self.eval_quick)
        self.btn_load = self._button("載入模型", self.pick_model)
        for button in (
            self.btn_human,
            self.btn_ai,
            self.btn_step,
            self.btn_auto,
            self.btn_win,
            self.btn_train,
            self.btn_eval,
            self.btn_load,
        ):
            control_layout.addWidget(button)

        self.path_edit = QLineEdit(self.model_path)
        self.path_edit.setObjectName("PathEdit")
        self.path_edit.editingFinished.connect(self._sync_model_path)
        control_layout.addWidget(self.path_edit, 1)
        layout.addWidget(controls)

        self.log_label = QLabel("牌局紀錄")
        self.log_label.setObjectName("LogLabel")
        layout.addWidget(self.log_label)

    def _button(self, text: str, callback) -> QPushButton:
        button = QPushButton(text)
        button.clicked.connect(callback)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        return button

    def _apply_style(self) -> None:
        button_asset = self.assets.ui_path("button_green.svg").as_posix()
        self.setStyleSheet(
            f"""
            QMainWindow {{
                background: #061f1a;
            }}
            #TitleLabel {{
                color: #f8edc7;
                font: 700 24px "{self.font_family}";
                letter-spacing: 0;
            }}
            #StatusLabel, #LogLabel {{
                color: #d8eadf;
                font: 500 13px "{self.font_family}";
            }}
            #ControlBar {{
                background: rgba(8, 45, 39, 210);
                border: 2px solid #b89446;
                border-radius: 16px;
            }}
            QPushButton {{
                border-image: url("{button_asset}") 14 28 14 28 stretch stretch;
                color: #fff3ce;
                min-height: 34px;
                min-width: 82px;
                padding: 7px 12px;
                font: 700 12px "{self.font_family}";
            }}
            QPushButton:hover {{
                color: #ffffff;
            }}
            QPushButton:disabled {{
                color: rgba(255, 255, 255, 95);
            }}
            #PathEdit {{
                background: rgba(5, 28, 24, 210);
                border: 1px solid #b89446;
                border-radius: 10px;
                color: #f8edc7;
                padding: 8px 10px;
                font: 12px "Consolas";
            }}
            """
        )

    def _sync_model_path(self) -> None:
        self.model_path = self.path_edit.text().strip()
        self.agents = self._build_agents()

    def _load_model_if_available(self) -> LinearDiscardModel | None:
        path = self.model_path.strip()
        if not path or not os.path.exists(path):
            return None
        try:
            return LinearDiscardModel.load(path)
        except Exception as exc:
            self.status_label.setText(f"模型載入失敗：{exc}")
            return None

    def _build_agents(self) -> list[object]:
        rng = random.Random()
        model = self._load_model_if_available()
        agents: list[object] = []
        for seat in range(4):
            seat_rng = random.Random(rng.randrange(1_000_000_000))
            if model is not None:
                agents.append(ModelAgent(model, seat_rng))
            else:
                agents.append(HeuristicAgent(seat_rng))
        return agents

    def new_game(self, human_mode: bool) -> None:
        self.timer.stop()
        self.human_mode = human_mode
        self.auto_running = not human_mode
        self.env = HKMahjongEnv()
        self.env.reset()
        self.agents = self._build_agents()
        self._refresh()
        if human_mode:
            self._drive_ai_until_human()
        else:
            self.timer.start(260)

    def human_discard(self, tile: int) -> None:
        if not self.human_mode or self.env.current_player != 0 or self.env.done:
            return
        try:
            self.env.discard(tile, self.agents)
        except Exception as exc:
            QMessageBox.warning(self, "出牌失敗", str(exc))
        self._refresh()
        self._drive_ai_until_human()

    def human_win(self) -> None:
        if self.human_mode and self.env.current_player == 0 and self.env.can_self_win(0):
            self.env.declare_self_win(0)
            self._refresh()

    def step_ai(self) -> None:
        if self.env.done:
            return
        if self.human_mode and self.env.current_player == 0:
            self.status_label.setText("輪到玩家，請點選底部手牌")
            return
        self.env.play_ai_turn(self.agents)
        self._refresh()
        self._drive_ai_until_human()

    def toggle_auto(self) -> None:
        self.auto_running = not self.auto_running
        if self.auto_running:
            self.timer.start(260)
        else:
            self.timer.stop()
        self._refresh()

    def _auto_tick(self) -> None:
        if self.env.done:
            self.timer.stop()
            self._refresh()
            return
        if self.human_mode and self.env.current_player == 0:
            self.timer.stop()
            self.auto_running = False
            self._refresh()
            return
        self.env.play_ai_turn(self.agents)
        self._refresh()

    def _drive_ai_until_human(self) -> None:
        if not self.human_mode:
            return
        while not self.env.done and self.env.current_player != 0:
            self.env.play_ai_turn(self.agents)
        self._refresh()

    def pick_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "選擇模型",
            str(PROJECT_ROOT / "models"),
            "JSON model (*.json);;All files (*.*)",
        )
        if path:
            self.model_path = path
            self.path_edit.setText(path)
            self.agents = self._build_agents()
            self.status_label.setText(f"已載入模型：{Path(path).name}")

    def train_quick(self) -> None:
        self._sync_model_path()
        out_path = self.model_path or str(PROJECT_ROOT / "models" / "model_latest.json")
        source = out_path if os.path.exists(out_path) else None
        self._run_background("訓練中...", lambda: train_model(200, out_path, source))

    def eval_quick(self) -> None:
        self._sync_model_path()
        if not os.path.exists(self.model_path):
            QMessageBox.warning(self, "找不到模型", "請先訓練或選擇模型檔")
            return
        self._run_background("評估中...", lambda: evaluate_model(self.model_path, 100))

    def _run_background(self, started: str, job) -> None:
        self.status_label.setText(started)
        self.btn_train.setDisabled(True)
        self.btn_eval.setDisabled(True)

        def worker() -> None:
            try:
                result = job()
            except Exception as exc:
                result = f"失敗：{exc}"
            self.bridge.finished.emit(result)

        threading.Thread(target=worker, daemon=True).start()

    def _task_finished(self, message: str) -> None:
        self.status_label.setText(message)
        self.btn_train.setDisabled(False)
        self.btn_eval.setDisabled(False)
        self.agents = self._build_agents()
        self._refresh()

    def _refresh(self) -> None:
        show_all = not self.human_mode
        self.table.set_game(self.env, self.human_mode, show_all_hands=show_all)
        if self.env.result:
            status = f"{self.env.result.message}｜回合 {self.env.result.turns}｜牌牆 {self.env.result.wall_remaining}"
        else:
            mode = "玩家對 AI" if self.human_mode else "AI 對 AI"
            running = "自動" if self.auto_running else "暫停"
            status = f"{mode}｜{running}｜輪到 P{self.env.current_player + 1}｜牌牆 {len(self.env.wall)}"
        self.status_label.setText(status)
        logs = "　".join(self.env.log[-5:])
        self.log_label.setText(f"牌局紀錄：{logs}")
        self.btn_win.setEnabled(self.human_mode and self.env.current_player == 0 and self.env.can_self_win(0))


def run_gui() -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    window = MahjongWindow()
    window.show()
    app.exec()
