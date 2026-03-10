"""Integration tests for the train -> resume -> sample workflow."""

from __future__ import annotations

import pickle
from dataclasses import replace
from pathlib import Path

import numpy as np

from configurator import SampleConfig, TrainConfig
from sample import sample_texts
from train import train
from utils import load_training_checkpoint


def write_toy_dataset(root: Path, dataset_name: str = "toy") -> Path:
    """Create a tiny character-level dataset on disk."""
    dataset_dir = root / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    vocab = ["a", "b", "c", "d"]
    stoi = {char: index for index, char in enumerate(vocab)}
    itos = {index: char for index, char in enumerate(vocab)}

    train_tokens = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3], dtype=np.uint16)
    val_tokens = np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.uint16)
    train_tokens.tofile(dataset_dir / "train.bin")
    val_tokens.tofile(dataset_dir / "val.bin")

    with (dataset_dir / "meta.pkl").open("wb") as handle:
        pickle.dump({"vocab_size": len(vocab), "stoi": stoi, "itos": itos}, handle)

    return dataset_dir


def test_train_resume_and_sample_pipeline(tmp_path: Path) -> None:
    write_toy_dataset(tmp_path)
    out_dir = tmp_path / "out"

    initial_config = TrainConfig(
        out_dir=str(out_dir),
        data_dir=str(tmp_path),
        dataset="toy",
        batch_size=2,
        block_size=6,
        n_layer=1,
        n_head=2,
        n_embd=16,
        dropout=0.0,
        learning_rate=1e-3,
        max_iters=2,
        eval_interval=1,
        eval_iters=1,
        log_interval=1,
        always_save_checkpoint=True,
        compile=False,
        device="cpu",
        dtype="float32",
        seed=0,
    )

    initial_summary = train(initial_config)
    checkpoint = load_training_checkpoint(initial_summary["checkpoint_path"])

    assert checkpoint["iter_num"] == 2
    assert checkpoint["dataset"] == "toy"
    assert checkpoint["data_dir"] == str(tmp_path)
    assert checkpoint["opt_state"] is not None
    assert checkpoint["model_config"]["vocab_size"] == 4

    resumed_summary = train(
        replace(
            initial_config,
            init_from="resume",
            max_iters=3,
        )
    )
    resumed_checkpoint = load_training_checkpoint(resumed_summary["checkpoint_path"])

    assert resumed_checkpoint["iter_num"] == 3
    assert resumed_summary["iter_num"] == 3

    samples = sample_texts(
        SampleConfig(
            init_from="resume",
            out_dir=str(out_dir),
            data_dir=str(tmp_path),
            start="ab",
            num_samples=2,
            max_new_tokens=2,
            temperature=0.0,
            top_k=None,
            device="cpu",
            dtype="float32",
        )
    )

    assert len(samples) == 2
    assert all(sample.startswith("ab") for sample in samples)
