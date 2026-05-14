"""Model discovery and metadata helpers for the PySide6 GUI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hkmahjong_ai.rl_encoder import STATE_DIM
from hkmahjong_ai.rl_train import checkpoint_metadata

from .assets import PROJECT_ROOT


MODELS_DIR = PROJECT_ROOT / "models"


@dataclass(slots=True)
class ModelInfo:
    path: Path
    version: str
    games_trained: int
    state_dim: str
    action_size: str
    players: int
    training: str
    error: str = ""

    @property
    def name(self) -> str:
        return self.path.name


def ensure_models_dir() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def list_pt_models() -> list[ModelInfo]:
    ensure_models_dir()
    models: list[ModelInfo] = []
    for path in sorted(MODELS_DIR.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True):
        models.append(read_model_info(path))
    return models


def read_model_info(path: Path) -> ModelInfo:
    try:
        meta: dict[str, Any] = checkpoint_metadata(str(path))
        if int(meta["state_dim"]) != STATE_DIM:
            raise ValueError(f"state_dim {meta['state_dim']} 不相容，目前需要 {STATE_DIM}")
        return ModelInfo(
            path=path,
            version=str(meta["version"]),
            games_trained=int(meta["games_trained"]),
            state_dim=str(meta["state_dim"]),
            action_size=str(meta["action_size"]),
            players=int(meta["players"]),
            training=str(meta["training"]),
            error="",
        )
    except Exception as exc:
        return ModelInfo(
            path=path,
            version="-",
            games_trained=0,
            state_dim="-",
            action_size="-",
            players=0,
            training="-",
            error=str(exc),
        )


def newest_model_path() -> str:
    for model in list_pt_models():
        if not model.error:
            return str(model.path)
    return ""
