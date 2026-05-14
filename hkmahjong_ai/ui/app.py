from __future__ import annotations

import os
import random
import sys
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from hkmahjong_ai.agents import HeuristicAgent
from hkmahjong_ai.env import HKMahjongEnv
from hkmahjong_ai.rl_agent import RLAgent
from hkmahjong_ai.rl_model import load_rl_checkpoint
from hkmahjong_ai.tiles import names_from_tiles, tile_name

from .assets import AssetManager
from .fonts import load_cjk_font
from .loss_chart import LossChart
from .model_manager import MODELS_DIR, ensure_models_dir, list_pt_models, newest_model_path, read_model_info
from .table_view import GameTableWidget
from .training_worker import TrainingWorker


class HumanGuiAgent:
    def __init__(self, parent: QWidget):
        self.parent = parent

    def choose_discard(self, state: dict[str, Any], legal_discards: list[int]) -> int:
        return legal_discards[0]

    def choose_claim(self, state: dict[str, Any], options: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not options:
            return None
        action_name = {"chow": "食", "pong": "碰", "kong": "槓"}.get(options[0]["kind"], "副露")
        discard = tile_name(state["discard_tile"])
        box = QMessageBox(self.parent)
        box.setWindowTitle(f"是否{action_name}牌")
        box.setText(f"對手打出 {discard}，你可以{action_name}。")
        box.setInformativeText("選擇副露後會輪到你出牌。")
        pass_button = box.addButton("過", QMessageBox.ButtonRole.RejectRole)
        button_map: dict[object, dict[str, Any]] = {}
        for option in options:
            label = f"{action_name} {names_from_tiles(option['tiles'])}"
            button = box.addButton(label, QMessageBox.ButtonRole.AcceptRole)
            button_map[button] = option
        box.exec()
        clicked = box.clickedButton()
        if clicked == pass_button:
            return None
        return button_map.get(clicked)


class MahjongWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        ensure_models_dir()
        self.setWindowTitle("香港跑馬仔麻雀 RL 訓練平台")
        self.resize(1320, 860)
        self.setMinimumSize(1100, 740)

        self.assets = AssetManager()
        self.font_family = load_cjk_font()
        self.env = HKMahjongEnv()
        self.human_mode = True
        self.auto_running = False
        self.selected_tile: int | None = None
        self.result_shown = False
        self.model_path = newest_model_path()
        self.agents: list[object] = []
        self.worker: TrainingWorker | None = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._auto_tick)

        self._build_ui()
        self._apply_style()
        self.refresh_models()
        self.new_game(True)

    def _build_ui(self) -> None:
        tabs = QTabWidget()
        self.setCentralWidget(tabs)
        tabs.addTab(self._build_game_tab(), "遊戲")
        tabs.addTab(self._build_training_tab(), "訓練")
        tabs.addTab(self._build_model_tab(), "模型管理")
        tabs.addTab(self._build_settings_tab(), "設定")

    def _build_game_tab(self) -> QWidget:
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(14, 12, 14, 14)
        layout.setSpacing(10)

        header = QHBoxLayout()
        self.title_label = QLabel("香港跑馬仔麻雀 RL 牌桌")
        self.title_label.setObjectName("TitleLabel")
        self.status_label = QLabel("準備開始")
        self.status_label.setObjectName("StatusLabel")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        header.addWidget(self.title_label)
        header.addWidget(self.status_label, 1)
        layout.addLayout(header)

        self.table = GameTableWidget(self.assets)
        self.table.tileSelected.connect(self.select_tile)
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
        self.btn_discard = self._button("出牌", self.confirm_discard)
        self.btn_win = self._button("自摸", self.human_win)
        self.btn_load_latest = self._button("載入最新模型", self.load_latest_model)
        for button in (
            self.btn_human,
            self.btn_ai,
            self.btn_step,
            self.btn_auto,
            self.btn_discard,
            self.btn_win,
            self.btn_load_latest,
        ):
            control_layout.addWidget(button)

        self.loaded_model_label = QLabel("未載入模型")
        self.loaded_model_label.setObjectName("PathLabel")
        control_layout.addWidget(self.loaded_model_label, 1)
        layout.addWidget(controls)

        self.log_label = QLabel("牌局紀錄：")
        self.log_label.setObjectName("LogLabel")
        layout.addWidget(self.log_label)
        return root

    def _build_training_tab(self) -> QWidget:
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(16, 14, 16, 16)
        layout.setSpacing(10)

        config_box = QGroupBox("訓練設定")
        form = QFormLayout(config_box)
        self.train_out_path = QLineEdit(str(MODELS_DIR / "rl_latest.pt"))
        self.train_source_path = QLineEdit("")
        self.train_episodes = QSpinBox()
        self.train_episodes.setRange(1, 1_000_000)
        self.train_episodes.setValue(1000)
        self.train_batch_size = QSpinBox()
        self.train_batch_size.setRange(8, 4096)
        self.train_batch_size.setValue(64)
        self.train_capacity = QSpinBox()
        self.train_capacity.setRange(500, 2_000_000)
        self.train_capacity.setValue(50_000)
        self.train_lr = QDoubleSpinBox()
        self.train_lr.setDecimals(6)
        self.train_lr.setRange(0.000001, 0.1)
        self.train_lr.setSingleStep(0.0001)
        self.train_lr.setValue(0.0003)
        self.train_entropy = QDoubleSpinBox()
        self.train_entropy.setDecimals(3)
        self.train_entropy.setRange(0.0, 1.0)
        self.train_entropy.setSingleStep(0.01)
        self.train_entropy.setValue(0.05)
        self.train_pool_probability = QDoubleSpinBox()
        self.train_pool_probability.setDecimals(2)
        self.train_pool_probability.setRange(0.0, 1.0)
        self.train_pool_probability.setSingleStep(0.05)
        self.train_pool_probability.setValue(0.15)
        self.train_updates_per_episode = QSpinBox()
        self.train_updates_per_episode.setRange(1, 50)
        self.train_updates_per_episode.setValue(4)
        self.train_update_interval = QSpinBox()
        self.train_update_interval.setRange(1, 5000)
        self.train_update_interval.setValue(1)

        form.addRow("輸出模型", self._path_row(self.train_out_path, self.pick_train_output))
        form.addRow("繼續訓練來源", self._path_row(self.train_source_path, self.pick_train_source))
        form.addRow("Episodes", self.train_episodes)
        form.addRow("Batch size", self.train_batch_size)
        form.addRow("Replay capacity", self.train_capacity)
        form.addRow("Learning rate", self.train_lr)
        form.addRow("Entropy coeff", self.train_entropy)
        form.addRow("Opponent pool 機率", self.train_pool_probability)
        form.addRow("每局更新次數", self.train_updates_per_episode)
        form.addRow("UI 更新間隔", self.train_update_interval)
        layout.addWidget(config_box)

        actions = QHBoxLayout()
        self.btn_start_training = self._button("開始訓練", self.start_training)
        self.btn_pause_training = self._button("暫停", self.pause_training)
        self.btn_resume_training = self._button("繼續", self.resume_training)
        self.btn_stop_training = self._button("停止並保存", self.stop_training)
        for button in (self.btn_start_training, self.btn_pause_training, self.btn_resume_training, self.btn_stop_training):
            actions.addWidget(button)
        actions.addStretch(1)
        layout.addLayout(actions)

        self.training_progress = QProgressBar()
        self.training_progress.setRange(0, 100)
        self.training_progress.setValue(0)
        layout.addWidget(self.training_progress)

        self.training_stats = QLabel("尚未開始訓練")
        self.training_stats.setObjectName("PanelText")
        self.training_stats.setWordWrap(True)
        layout.addWidget(self.training_stats)

        self.loss_chart = LossChart()
        layout.addWidget(self.loss_chart)
        self._set_training_buttons(active=False)
        return root

    def _build_model_tab(self) -> QWidget:
        root = QWidget()
        layout = QHBoxLayout(root)
        layout.setContentsMargins(16, 14, 16, 16)
        layout.setSpacing(12)

        left = QVBoxLayout()
        self.model_list = QListWidget()
        self.model_list.currentItemChanged.connect(self.show_selected_model_metadata)
        left.addWidget(self.model_list, 1)
        model_buttons = QHBoxLayout()
        model_buttons.addWidget(self._button("重新掃描", self.refresh_models))
        model_buttons.addWidget(self._button("載入選取模型", self.load_selected_model))
        left.addLayout(model_buttons)
        layout.addLayout(left, 1)

        right = QVBoxLayout()
        self.model_metadata = QTextEdit()
        self.model_metadata.setReadOnly(True)
        self.model_metadata.setObjectName("MetadataText")
        right.addWidget(QLabel("模型資訊"))
        right.addWidget(self.model_metadata, 1)
        layout.addLayout(right, 1)
        return root

    def _build_settings_tab(self) -> QWidget:
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(16, 14, 16, 16)
        text = QLabel("目前設定：本地單機、PySide6 GUI、PyTorch Actor-Critic RL、136 張牌、不使用花牌、不計番。")
        text.setObjectName("PanelText")
        text.setWordWrap(True)
        layout.addWidget(text)
        layout.addStretch(1)
        return root

    def _path_row(self, line_edit: QLineEdit, callback) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(line_edit, 1)
        layout.addWidget(self._button("選擇", callback))
        return row

    def _button(self, text: str, callback) -> QPushButton:
        button = QPushButton(text)
        button.clicked.connect(callback)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        return button

    def _apply_style(self) -> None:
        button_asset = self.assets.ui_path("button_green.svg").as_posix()
        self.setStyleSheet(
            f"""
            QMainWindow, QWidget {{
                background: #061f1a;
                color: #d8eadf;
                font-family: "{self.font_family}";
            }}
            QTabWidget::pane {{
                border: 1px solid #b89446;
                border-radius: 8px;
                background: #082d27;
            }}
            QTabBar::tab {{
                background: #104537;
                color: #f8edc7;
                padding: 9px 18px;
                border: 1px solid #b89446;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
            }}
            QTabBar::tab:selected {{
                background: #1b6b57;
            }}
            #TitleLabel {{
                color: #f8edc7;
                font: 700 24px "{self.font_family}";
            }}
            #StatusLabel, #LogLabel, #PanelText, #PathLabel {{
                color: #d8eadf;
                font: 500 13px "{self.font_family}";
            }}
            #ControlBar, QGroupBox {{
                background: rgba(8, 45, 39, 220);
                border: 2px solid #b89446;
                border-radius: 14px;
                margin-top: 10px;
                padding: 10px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 14px;
                padding: 0 6px;
                color: #f8edc7;
                font-weight: 700;
            }}
            QPushButton {{
                border-image: url("{button_asset}") 14 28 14 28 stretch stretch;
                color: #fff3ce;
                min-height: 34px;
                min-width: 78px;
                padding: 7px 11px;
                font: 700 12px "{self.font_family}";
            }}
            QPushButton:disabled {{
                color: rgba(255, 255, 255, 95);
            }}
            QLineEdit, QSpinBox, QDoubleSpinBox, QListWidget, QTextEdit {{
                background: rgba(5, 28, 24, 230);
                border: 1px solid #b89446;
                border-radius: 8px;
                color: #f8edc7;
                padding: 6px;
            }}
            QProgressBar {{
                border: 1px solid #b89446;
                border-radius: 8px;
                text-align: center;
                color: #f8edc7;
                background: #082d27;
                min-height: 24px;
            }}
            QProgressBar::chunk {{
                border-radius: 8px;
                background: #1c8f72;
            }}
            """
        )

    def _load_rl_models_if_available(self) -> list[object] | None:
        path = self.model_path.strip()
        if not path or not os.path.exists(path):
            return None
        try:
            return [load_rl_checkpoint(path, "cpu", seat)[0] for seat in range(4)]
        except Exception as exc:
            self.status_label.setText(f"RL 模型載入失敗：{exc}")
            return None

    def _build_agents(self) -> list[object]:
        rng = random.Random()
        rl_models = self._load_rl_models_if_available()
        agents: list[object] = []
        for seat in range(4):
            seat_rng = random.Random(rng.randrange(1_000_000_000))
            if self.human_mode and seat == 0:
                agents.append(HumanGuiAgent(self))
            elif rl_models is not None:
                agents.append(RLAgent(rl_models[seat], device="cpu", rng=seat_rng, deterministic=True))
            else:
                agents.append(HeuristicAgent(seat_rng))
        return agents

    def new_game(self, human_mode: bool) -> None:
        if not self.model_path:
            self.status_label.setText("models/ 目錄沒有 .pt 模型。請先到「訓練」分頁建立模型。")
        self.timer.stop()
        self.human_mode = human_mode
        self.auto_running = not human_mode
        self.selected_tile = None
        self.result_shown = False
        self.env = HKMahjongEnv()
        self.env.reset()
        self.agents = self._build_agents()
        self._refresh()
        if human_mode:
            self._drive_ai_until_human()
        else:
            self.timer.start(260)

    def select_tile(self, tile: int) -> None:
        if not self.human_mode or self.env.done or self.env.current_player != 0:
            return
        if self.selected_tile == tile:
            self.confirm_discard()
            return
        self.selected_tile = tile
        self.table.set_selected_tile(tile)
        self.status_label.setText(f"已選擇 {tile_name(tile)}，按「出牌」確認")
        self._update_buttons()

    def confirm_discard(self) -> None:
        if self.selected_tile is None:
            self.status_label.setText("請先點選一張手牌")
            return
        self.human_discard(self.selected_tile)

    def human_discard(self, tile: int) -> None:
        if not self.human_mode or self.env.current_player != 0 or self.env.done:
            return
        try:
            self.env.discard(tile, self.agents)
        except Exception as exc:
            QMessageBox.warning(self, "出牌失敗", str(exc))
        self.selected_tile = None
        self.table.set_selected_tile(None)
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
            self.status_label.setText("輪到玩家，請點選手牌並按「出牌」")
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

    def load_latest_model(self) -> None:
        path = newest_model_path()
        if not path:
            QMessageBox.information(self, "找不到模型", "models/ 目錄沒有 .pt 模型。")
            return
        self.load_model(path)

    def load_model(self, path: str) -> None:
        self.model_path = path
        self.loaded_model_label.setText(f"已載入：{Path(path).name}")
        self.train_source_path.setText(path)
        self.agents = self._build_agents()
        self._refresh()

    def _refresh(self) -> None:
        show_all = not self.human_mode
        self.table.set_game(self.env, self.human_mode, show_all_hands=show_all)
        self.table.set_selected_tile(self.selected_tile)
        if self.env.result:
            pattern = f"｜牌型 {self.env.result.pattern}" if self.env.result.pattern else ""
            status = f"{self.env.result.message}{pattern}｜回合 {self.env.result.turns}｜牌牆 {self.env.result.wall_remaining}"
        else:
            mode = "玩家對 AI" if self.human_mode else "AI 對 AI"
            running = "自動" if self.auto_running else "手動"
            last = f"｜最新：{self.env.log[-1]}" if self.env.log else ""
            status = f"{mode}｜{running}｜輪到 P{self.env.current_player + 1}｜牌牆 {len(self.env.wall)}{last}"
        self.status_label.setText(status)
        self.loaded_model_label.setText(f"已載入：{Path(self.model_path).name}" if self.model_path else "未載入模型")
        logs = "　".join(self.env.log[-5:])
        self.log_label.setText(f"牌局紀錄：{logs}")
        self._update_buttons()
        self._maybe_show_result()

    def _update_buttons(self) -> None:
        is_human_turn = self.human_mode and not self.env.done and self.env.current_player == 0
        self.btn_discard.setEnabled(is_human_turn and self.selected_tile is not None)
        self.btn_win.setEnabled(is_human_turn and self.env.can_self_win(0))

    def _maybe_show_result(self) -> None:
        if self.result_shown or not self.env.result:
            return
        self.result_shown = True
        result = self.env.result
        if result.winner is None:
            text = "流局"
            detail = f"回合：{result.turns}\n剩餘牌：{result.wall_remaining}"
        else:
            text = result.message
            detail = f"牌型：{result.pattern or '四面一對'}\n回合：{result.turns}\n剩餘牌：{result.wall_remaining}"
        QMessageBox.information(self, "牌局結果", f"{text}\n\n{detail}")

    def pick_train_output(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "選擇輸出模型", str(MODELS_DIR / "rl_latest.pt"), "PyTorch checkpoint (*.pt)")
        if path:
            self.train_out_path.setText(path if path.lower().endswith(".pt") else f"{path}.pt")

    def pick_train_source(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "選擇既有模型", str(MODELS_DIR), "PyTorch checkpoint (*.pt)")
        if path:
            self.train_source_path.setText(path)

    def start_training(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            return
        self.loss_chart.clear()
        self.training_progress.setValue(0)
        self.training_stats.setText("訓練啟動中...")
        self.worker = TrainingWorker(
            episodes=self.train_episodes.value(),
            out_path=self.train_out_path.text().strip(),
            model_path=self.train_source_path.text().strip() or None,
            batch_size=self.train_batch_size.value(),
            capacity=self.train_capacity.value(),
            learning_rate=self.train_lr.value(),
            entropy_coeff=self.train_entropy.value(),
            pool_probability=self.train_pool_probability.value(),
            updates_per_episode=self.train_updates_per_episode.value(),
            update_interval=self.train_update_interval.value(),
        )
        self.worker.progress.connect(self.on_training_progress)
        self.worker.completed.connect(self.on_training_completed)
        self.worker.failed.connect(self.on_training_failed)
        self.worker.start()
        self._set_training_buttons(active=True)

    def pause_training(self) -> None:
        if self.worker is not None:
            self.worker.pause()
            self.training_stats.setText("訓練已暫停")

    def resume_training(self) -> None:
        if self.worker is not None:
            self.worker.resume()
            self.training_stats.setText("訓練繼續中...")

    def stop_training(self) -> None:
        if self.worker is not None:
            self.worker.stop()
            self.training_stats.setText("正在停止並保存 checkpoint...")

    def on_training_progress(self, payload: dict[str, Any]) -> None:
        percent = int(payload.get("percent", 0.0) * 100)
        self.training_progress.setValue(max(0, min(100, percent)))
        episode = int(payload.get("episode", 0))
        total = int(payload.get("total_episodes", 0))
        policy_loss = float(payload.get("policy_loss", 0.0))
        value_loss = float(payload.get("value_loss", 0.0))
        self.loss_chart.add_point(episode, policy_loss, value_loss)
        self.training_stats.setText(
            f"Episode {episode}/{total}｜進度 {percent}%｜總局數 {payload.get('games_trained', 0)}｜"
            f"勝率 {payload.get('win_rate', 0):.2%}｜流局率 {payload.get('draw_rate', 0):.2%}｜"
            f"近局勝率 {payload.get('recent_win_rate', 0):.2%}｜近局流局率 {payload.get('recent_draw_rate', 0):.2%}｜"
            f"平均局長 {payload.get('average_turns', 0):.1f}｜policy {policy_loss:.4f}｜value {value_loss:.4f}｜"
            f"設備 {payload.get('device', '-')}｜儲存 {payload.get('out_path', '-')}"
        )

    def on_training_completed(self, message: str) -> None:
        self._set_training_buttons(active=False)
        self.training_stats.setText(message)
        self.refresh_models()
        out_path = self.train_out_path.text().strip()
        if os.path.exists(out_path):
            self.load_model(out_path)
        QMessageBox.information(self, "訓練完成", message)

    def on_training_failed(self, message: str) -> None:
        self._set_training_buttons(active=False)
        QMessageBox.critical(self, "訓練失敗", message)
        self.training_stats.setText(f"訓練失敗：{message}")

    def _set_training_buttons(self, active: bool) -> None:
        self.btn_start_training.setEnabled(not active)
        self.btn_pause_training.setEnabled(active)
        self.btn_resume_training.setEnabled(active)
        self.btn_stop_training.setEnabled(active)

    def refresh_models(self) -> None:
        self.model_list.clear()
        models = list_pt_models()
        if not models:
            self.model_metadata.setText("models/ 目錄沒有 .pt 模型。請先到「訓練」分頁建立模型。")
            self.model_path = ""
            self.loaded_model_label.setText("未載入模型")
            return
        for model in models:
            suffix = "｜不相容" if model.error else ""
            item = QListWidgetItem(f"{model.name}｜games {model.games_trained}{suffix}")
            item.setData(Qt.ItemDataRole.UserRole, str(model.path))
            self.model_list.addItem(item)
        self.model_list.setCurrentRow(0)
        if not self.model_path:
            self.load_model(str(models[0].path))

    def show_selected_model_metadata(self, current: QListWidgetItem | None) -> None:
        if current is None:
            return
        path = Path(current.data(Qt.ItemDataRole.UserRole))
        info = read_model_info(path)
        if info.error:
            self.model_metadata.setText(f"{info.name}\n\n讀取失敗：{info.error}")
            return
        self.model_metadata.setText(
            f"檔案：{info.name}\n"
            f"路徑：{info.path}\n"
            f"版本：{info.version}\n"
            f"訓練局數：{info.games_trained}\n"
            f"玩家模型數：{info.players}\n"
            f"state_dim：{info.state_dim}\n"
            f"action_size：{info.action_size}\n"
            f"training：{info.training}"
        )

    def load_selected_model(self) -> None:
        item = self.model_list.currentItem()
        if item is None:
            QMessageBox.information(self, "未選擇模型", "請先選擇一個 .pt 模型。")
            return
        self.load_model(item.data(Qt.ItemDataRole.UserRole))


def run_gui() -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    window = MahjongWindow()
    window.show()
    app.exec()
