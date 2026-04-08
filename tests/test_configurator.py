"""Tests for typed config loading and overrides."""

from pathlib import Path

import pytest

from configurator import parse_sample_config, parse_train_config


def test_parse_train_config_from_python_file(tmp_path: Path) -> None:
    config_path = tmp_path / "train_config.py"
    config_path.write_text(
        "batch_size = 4\ngradient_accumulation_steps = 5 * 8\ncompile = False\n",
        encoding="utf-8",
    )

    config = parse_train_config([str(config_path), "--learning_rate=0.001", "--dataset='toy-set'"])

    assert config.batch_size == 4
    assert config.gradient_accumulation_steps == 40
    assert config.compile is False
    assert config.learning_rate == pytest.approx(0.001)
    assert config.dataset == "toy-set"


def test_parse_sample_config_supports_optional_overrides() -> None:
    config = parse_sample_config(["--top_k=None", "--top_p=0.9", "--temperature=1.25"])

    assert config.top_k is None
    assert config.top_p == pytest.approx(0.9)
    assert config.temperature == pytest.approx(1.25)
