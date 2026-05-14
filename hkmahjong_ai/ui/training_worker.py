"""Background PySide6 worker for RL self-play training."""

from __future__ import annotations

import threading

from PySide6.QtCore import QThread, Signal

from hkmahjong_ai.rl_train import train_rl


class TrainingWorker(QThread):
    progress = Signal(dict)
    completed = Signal(str)
    failed = Signal(str)

    def __init__(
        self,
        episodes: int,
        out_path: str,
        model_path: str | None,
        batch_size: int,
        capacity: int,
        learning_rate: float,
        pool_probability: float,
        update_interval: int,
        parent=None,
    ):
        super().__init__(parent)
        self.episodes = episodes
        self.out_path = out_path
        self.model_path = model_path
        self.batch_size = batch_size
        self.capacity = capacity
        self.learning_rate = learning_rate
        self.pool_probability = pool_probability
        self.update_interval = update_interval
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()

    def run(self) -> None:
        try:
            result = train_rl(
                episodes=self.episodes,
                out_path=self.out_path,
                model_path=self.model_path or None,
                batch_size=self.batch_size,
                capacity=self.capacity,
                lr=self.learning_rate,
                pool_probability=self.pool_probability,
                progress_callback=self.progress.emit,
                stop_event=self.stop_event,
                pause_event=self.pause_event,
                update_interval=self.update_interval,
            )
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.completed.emit(result)

    def pause(self) -> None:
        self.pause_event.set()

    def resume(self) -> None:
        self.pause_event.clear()

    def stop(self) -> None:
        self.stop_event.set()
        self.pause_event.clear()
