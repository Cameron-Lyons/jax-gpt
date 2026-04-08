"""Dataset loading helpers for tokenized language-model corpora."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import numpy as np

from .utils import sample_language_model_batch


@dataclass
class TokenDataset:
    """Memory-mapped train/validation token arrays plus optional metadata."""

    train_tokens: np.memmap
    val_tokens: np.memmap
    meta: dict[str, Any] | None
    dataset_dir: Path


def dataset_dir(data_dir: str | Path, dataset: str) -> Path:
    """Return the canonical directory for a named dataset."""
    return Path(data_dir) / dataset


def dataset_split_path(data_dir: str | Path, dataset: str, split: str) -> Path:
    """Return the binary token file path for a dataset split."""
    return dataset_dir(data_dir, dataset) / f"{split}.bin"


def load_dataset_meta(meta_path: str | Path) -> dict[str, Any] | None:
    """Load dataset metadata from a pickle file when present."""
    path = Path(meta_path)
    if not path.exists():
        return None
    with path.open("rb") as handle:
        meta = pickle.load(handle)  # noqa: S301
    if not isinstance(meta, dict):
        raise TypeError(f"Expected metadata dictionary in {path}")
    return meta


def load_token_dataset(data_dir: str | Path, dataset: str) -> TokenDataset:
    """Open train/validation memmaps once and keep them alive for the run."""
    root = dataset_dir(data_dir, dataset)
    train_path = dataset_split_path(data_dir, dataset, "train")
    val_path = dataset_split_path(data_dir, dataset, "val")

    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Validation data not found: {val_path}")

    return TokenDataset(
        train_tokens=np.memmap(train_path, dtype=np.uint16, mode="r"),
        val_tokens=np.memmap(val_path, dtype=np.uint16, mode="r"),
        meta=load_dataset_meta(root / "meta.pkl"),
        dataset_dir=root,
    )


def resolve_train_data_path(data_dir: str | Path, dataset: str) -> Path | None:
    """Resolve the best available training split path for benchmarking."""
    candidate_paths = [
        dataset_split_path(data_dir, dataset, "train"),
        Path(data_dir) / "train.bin",
    ]
    for path in candidate_paths:
        if path.exists():
            return path
    return None


def get_batch(
    tokens: np.memmap | jax.Array,
    *,
    batch_size: int,
    block_size: int,
    rng: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Sample an autoregressive batch from a token sequence."""
    return sample_language_model_batch(
        tokens,
        batch_size=batch_size,
        block_size=block_size,
        rng=rng,
    )
