"""Checkpoint path and compatibility helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .model import GPTConfig

CHECKPOINT_FILENAME = "ckpt.pkl"


def checkpoint_path(out_dir: str | Path) -> Path:
    """Return the canonical checkpoint path for an output directory."""
    return Path(out_dir) / CHECKPOINT_FILENAME


def model_config_from_checkpoint(checkpoint: dict[str, Any]) -> GPTConfig:
    """Rebuild a model config from current or legacy checkpoint formats."""
    raw_values = (
        checkpoint.get("model_config") or checkpoint.get("model_args") or checkpoint.get("config")
    )
    if not isinstance(raw_values, dict):
        raise ValueError("Checkpoint does not contain model configuration")

    model_values = {
        key: value for key, value in raw_values.items() if key in GPTConfig.__dataclass_fields__
    }
    if "dropout" in raw_values:
        model_values.setdefault("embd_pdrop", raw_values["dropout"])
        model_values.setdefault("resid_pdrop", raw_values["dropout"])
        model_values.setdefault("attn_pdrop", raw_values["dropout"])
    return GPTConfig(**model_values)
