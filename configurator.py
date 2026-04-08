"""Typed configuration loading for training and sampling scripts."""

from __future__ import annotations

import ast
import json
import operator
import sys
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Sequence, TypeVar

from utils import DEFAULT_GPT2_BLOCK_SIZE

ConfigT = TypeVar("ConfigT", bound="BaseConfig")

_ALLOWED_BINOPS: dict[type[ast.operator], Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
}
_ALLOWED_UNARYOPS: dict[type[ast.unaryop], Any] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


@dataclass(frozen=True)
class BaseConfig:
    """Base class for script configuration dataclasses."""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable config mapping."""
        return asdict(self)


@dataclass(frozen=True)
class TrainConfig(BaseConfig):
    """Configuration for `train.py`."""

    out_dir: str = "out"
    data_dir: str = "data"
    models_dir: str = "models"
    eval_interval: int = 2000
    log_interval: int = 1
    eval_iters: int = 200
    eval_only: bool = False
    always_save_checkpoint: bool = True
    init_from: str = "scratch"
    dataset: str = "openwebtext"
    gradient_accumulation_steps: int = 40
    batch_size: int = 12
    block_size: int = DEFAULT_GPT2_BLOCK_SIZE
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
    learning_rate: float = 6e-4
    max_iters: int = 600000
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    decay_lr: bool = True
    warmup_iters: int = 2000
    lr_decay_iters: int = 600000
    min_lr: float = 6e-5
    device: str = "gpu"
    dtype: str = "bfloat16"
    compile: bool = True
    seed: int = 1337


@dataclass(frozen=True)
class SampleConfig(BaseConfig):
    """Configuration for `sample.py`."""

    init_from: str = "resume"
    out_dir: str = "out"
    data_dir: str = "data"
    models_dir: str = "models"
    start: str = "\n"
    num_samples: int = 10
    max_new_tokens: int = 500
    temperature: float = 0.8
    top_k: int | None = 200
    top_p: float | None = None
    seed: int = 1337
    device: str = "gpu"
    dtype: str = "bfloat16"


def _evaluate_expr(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.List):
        return [_evaluate_expr(item) for item in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_evaluate_expr(item) for item in node.elts)
    if isinstance(node, ast.Dict):
        output: dict[Any, Any] = {}
        for key, value in zip(node.keys, node.values):
            if key is None or value is None:
                raise ValueError("Dictionary unpacking is not supported in config files")
            output[_evaluate_expr(key)] = _evaluate_expr(value)
        return output
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        return _ALLOWED_UNARYOPS[type(node.op)](_evaluate_expr(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        return _ALLOWED_BINOPS[type(node.op)](_evaluate_expr(node.left), _evaluate_expr(node.right))
    raise ValueError(f"Unsupported config expression: {ast.dump(node)}")


def _load_python_config(config_path: Path) -> dict[str, Any]:
    module = ast.parse(config_path.read_text(encoding="utf-8"), filename=str(config_path))
    values: dict[str, Any] = {}
    for statement in module.body:
        if isinstance(statement, ast.Assign):
            if len(statement.targets) != 1 or not isinstance(statement.targets[0], ast.Name):
                raise ValueError(f"Unsupported assignment in config file: {config_path}")
            values[statement.targets[0].id] = _evaluate_expr(statement.value)
            continue
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant):
            continue
        raise ValueError(f"Unsupported statement in config file: {config_path}")
    return values


def load_config_file(config_path: str | Path) -> dict[str, Any]:
    """Load config values from a JSON or restricted Python config file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    if path.suffix == ".json":
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise TypeError(f"Expected JSON object in config file: {path}")
        return loaded
    if path.suffix == ".py":
        return _load_python_config(path)
    raise ValueError(f"Unsupported config file type: {path.suffix}")


def _parse_override_value(raw_value: str) -> Any:
    lowered = raw_value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "none":
        return None
    try:
        return ast.literal_eval(raw_value)
    except (ValueError, SyntaxError):
        return raw_value


def _coerce_value(raw_value: Any, current_value: Any) -> Any:
    if raw_value is None:
        return None
    if current_value is None:
        return raw_value
    if isinstance(current_value, bool):
        if not isinstance(raw_value, bool):
            raise TypeError(f"Expected boolean override, got {type(raw_value).__name__}")
        return raw_value
    if isinstance(current_value, int) and not isinstance(current_value, bool):
        if not isinstance(raw_value, int) or isinstance(raw_value, bool):
            raise TypeError(f"Expected integer override, got {type(raw_value).__name__}")
        return raw_value
    if isinstance(current_value, float):
        if not isinstance(raw_value, (int, float)) or isinstance(raw_value, bool):
            raise TypeError(f"Expected float override, got {type(raw_value).__name__}")
        return float(raw_value)
    if isinstance(current_value, str):
        if not isinstance(raw_value, str):
            raise TypeError(f"Expected string override, got {type(raw_value).__name__}")
        return raw_value
    if isinstance(current_value, list):
        if not isinstance(raw_value, list):
            raise TypeError(f"Expected list override, got {type(raw_value).__name__}")
        return raw_value
    if isinstance(current_value, tuple):
        if not isinstance(raw_value, tuple):
            raise TypeError(f"Expected tuple override, got {type(raw_value).__name__}")
        return raw_value
    if not isinstance(raw_value, type(current_value)):
        raise TypeError(
            f"Expected {type(current_value).__name__} override, got {type(raw_value).__name__}"
        )
    return raw_value


def _apply_updates(config: ConfigT, values: dict[str, Any]) -> ConfigT:
    valid_fields = {field.name for field in fields(config)}
    updated = config
    for key, raw_value in values.items():
        if key not in valid_fields:
            raise ValueError(f"Unknown config key: {key}")
        current_value = getattr(updated, key)
        updated = replace(updated, **{key: _coerce_value(raw_value, current_value)})
    return updated


def parse_config_args(config_cls: type[ConfigT], argv: Sequence[str] | None = None) -> ConfigT:
    """Parse the nanoGPT-style config file plus `--key=value` overrides."""
    raw_args = list(sys.argv[1:] if argv is None else argv)
    config = config_cls()
    config_path: str | None = None
    override_values: dict[str, Any] = {}

    for arg in raw_args:
        if arg.startswith("--"):
            if "=" not in arg:
                raise ValueError(f"Overrides must be provided as --key=value, got: {arg}")
            key, raw_value = arg[2:].split("=", 1)
            override_values[key] = _parse_override_value(raw_value)
            continue
        if config_path is not None:
            raise ValueError("Only one config file path may be provided")
        config_path = arg

    if config_path is not None:
        config = _apply_updates(config, load_config_file(config_path))
    return _apply_updates(config, override_values)


def parse_train_config(argv: Sequence[str] | None = None) -> TrainConfig:
    """Parse CLI args into a `TrainConfig`."""
    return parse_config_args(TrainConfig, argv)


def parse_sample_config(argv: Sequence[str] | None = None) -> SampleConfig:
    """Parse CLI args into a `SampleConfig`."""
    return parse_config_args(SampleConfig, argv)
