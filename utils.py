"""Utilities for JAX GPT-2 models."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import AbstractSet, Any, Dict, List, Literal, Protocol, Sequence, cast

import jax
import jax.numpy as jnp
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ModelSize = Literal["124M", "355M", "774M", "1558M"]
DEFAULT_GPT2_VOCAB_SIZE = 50257
DEFAULT_PADDED_VOCAB_SIZE = 50304
DEFAULT_GPT2_BLOCK_SIZE = 1024
DEFAULT_BENCHMARK_PEAK_FLOPS = 312e12
CAUSAL_MASK_FILL_VALUE = -1e9

GPT2_MODEL_SIZE_TO_NAME: dict[ModelSize, str] = {
    "124M": "gpt2",
    "355M": "gpt2-medium",
    "774M": "gpt2-large",
    "1558M": "gpt2-xl",
}
GPT2_MODEL_NAME_TO_SIZE: dict[str, ModelSize] = {
    "gpt2": "124M",
    "gpt2-medium": "355M",
    "gpt2-large": "774M",
    "gpt2-xl": "1558M",
}
GPT2_MODEL_SPECS: dict[str, dict[str, int]] = {
    "gpt2": {"n_layer": 12, "n_head": 12, "n_embd": 768},
    "gpt2-medium": {"n_layer": 24, "n_head": 16, "n_embd": 1024},
    "gpt2-large": {"n_layer": 36, "n_head": 20, "n_embd": 1280},
    "gpt2-xl": {"n_layer": 48, "n_head": 25, "n_embd": 1600},
}


class _TiktokenEncoding(Protocol):
    def encode(
        self, text: str, *, allowed_special: AbstractSet[str] | Literal["all"] = ...
    ) -> list[int]: ...
    def decode(self, tokens: Sequence[int]) -> str: ...


class _TiktokenModule(Protocol):
    def get_encoding(self, encoding_name: str) -> _TiktokenEncoding: ...


def _load_tiktoken() -> _TiktokenModule:
    """Import tiktoken only when GPT-2 tokenization is needed."""
    try:
        import tiktoken
    except ImportError as exc:
        raise ImportError(
            "GPT-2 tokenization requires the 'tokenization' extra: pip install -e '.[tokenization]'"
        ) from exc
    return cast(_TiktokenModule, tiktoken)


class TiktokenEncoder:
    """GPT-2 BPE encoder using tiktoken."""

    def __init__(self) -> None:
        self.enc = _load_tiktoken().get_encoding("gpt2")
        self.n_vocab = DEFAULT_GPT2_VOCAB_SIZE

    def encode(self, text: str) -> List[int]:
        return self.enc.encode(text, allowed_special={"<|endoftext|>"})

    def decode(self, tokens: List[int]) -> str:
        return self.enc.decode(tokens)


def get_gpt2_model_name(model_variant: str | ModelSize) -> str:
    """Normalize a GPT-2 model size or model name to the canonical model name."""
    if model_variant in GPT2_MODEL_NAME_TO_SIZE:
        return cast(str, model_variant)
    if model_variant in GPT2_MODEL_SIZE_TO_NAME:
        return GPT2_MODEL_SIZE_TO_NAME[cast(ModelSize, model_variant)]
    raise ValueError(f"Unknown GPT-2 model variant: {model_variant}")


def get_gpt2_model_size(model_variant: str | ModelSize) -> ModelSize:
    """Normalize a GPT-2 model size or model name to the canonical size label."""
    if model_variant in GPT2_MODEL_SIZE_TO_NAME:
        return cast(ModelSize, model_variant)
    if model_variant in GPT2_MODEL_NAME_TO_SIZE:
        return GPT2_MODEL_NAME_TO_SIZE[model_variant]
    raise ValueError(f"Unknown GPT-2 model variant: {model_variant}")


def get_gpt2_model_spec(model_variant: str | ModelSize) -> dict[str, int]:
    """Return model dimensions for a GPT-2 preset."""
    model_name = get_gpt2_model_name(model_variant)
    return dict(GPT2_MODEL_SPECS[model_name])


def sample_language_model_batch(
    tokens: Any,
    *,
    batch_size: int,
    block_size: int,
    rng: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Sample autoregressive `x`/`y` batches from a token sequence."""
    max_start = len(tokens) - block_size - 1
    if max_start < 0:
        raise ValueError(
            f"Block size {block_size} is too large for dataset with {len(tokens)} tokens"
        )

    start_indices = np.asarray(
        jax.random.randint(rng, (batch_size,), 0, max_start + 1),
        dtype=np.int32,
    ).tolist()
    x = jnp.stack([jnp.asarray(tokens[i : i + block_size], dtype=jnp.int32) for i in start_indices])
    y = jnp.stack(
        [jnp.asarray(tokens[i + 1 : i + 1 + block_size], dtype=jnp.int32) for i in start_indices]
    )
    return x, y


def load_encoder_hparams_and_params(
    model_size: ModelSize, models_dir: str | None = None
) -> tuple[TiktokenEncoder, Dict[str, Any], Dict[str, Any]]:
    """Load encoder, hparams, and params from HuggingFace."""
    try:
        from transformers import GPT2LMHeadModel
    except ImportError:
        raise ImportError("Please install transformers: pip install transformers")

    model_name = get_gpt2_model_name(model_size)
    logger.info(f"Loading HuggingFace model: {model_name}")

    model_hf = GPT2LMHeadModel.from_pretrained(model_name, cache_dir=models_dir)
    sd = model_hf.state_dict()
    config = model_hf.config

    # Build hparams dict
    hparams = {
        "n_layer": config.n_layer,
        "n_head": config.n_head,
        "n_embd": config.n_embd,
        "n_vocab": config.vocab_size,
        "n_ctx": config.n_positions,
    }

    # Convert weights to JAX format
    def to_jax(tensor: Any) -> jax.Array:
        return jnp.array(tensor.numpy())

    blocks: list[Dict[str, Any]] = []
    params = {
        "wte": to_jax(sd["transformer.wte.weight"]),
        "wpe": to_jax(sd["transformer.wpe.weight"]),
        "blocks": blocks,
        "ln_f": {
            "b": to_jax(sd["transformer.ln_f.bias"]),
            "g": to_jax(sd["transformer.ln_f.weight"]),
        },
    }

    for i in range(config.n_layer):
        block = {
            "attn": {
                "c_attn": {
                    "w": to_jax(sd[f"transformer.h.{i}.attn.c_attn.weight"]),
                    "b": to_jax(sd[f"transformer.h.{i}.attn.c_attn.bias"]),
                },
                "c_proj": {
                    "w": to_jax(sd[f"transformer.h.{i}.attn.c_proj.weight"]),
                    "b": to_jax(sd[f"transformer.h.{i}.attn.c_proj.bias"]),
                },
            },
            "mlp": {
                "c_fc": {
                    "w": to_jax(sd[f"transformer.h.{i}.mlp.c_fc.weight"]),
                    "b": to_jax(sd[f"transformer.h.{i}.mlp.c_fc.bias"]),
                },
                "c_proj": {
                    "w": to_jax(sd[f"transformer.h.{i}.mlp.c_proj.weight"]),
                    "b": to_jax(sd[f"transformer.h.{i}.mlp.c_proj.bias"]),
                },
            },
            "ln_1": {
                "b": to_jax(sd[f"transformer.h.{i}.ln_1.bias"]),
                "g": to_jax(sd[f"transformer.h.{i}.ln_1.weight"]),
            },
            "ln_2": {
                "b": to_jax(sd[f"transformer.h.{i}.ln_2.bias"]),
                "g": to_jax(sd[f"transformer.h.{i}.ln_2.weight"]),
            },
        }
        blocks.append(block)

    encoder = TiktokenEncoder()
    return encoder, hparams, params


def resolve_jax_device(device_preference: str) -> jax.Device:
    """Resolve a preferred JAX device, falling back to the default device."""
    try:
        devices = jax.devices(device_preference)
    except RuntimeError:
        devices = []

    if devices:
        return devices[0]

    fallback = jax.devices()[0]
    logger.warning(
        "Requested device '%s' is unavailable; falling back to %s", device_preference, fallback
    )
    return fallback


def save_pickle_payload(payload: Dict[str, Any], filepath: str | Path) -> Path:
    """Save an arbitrary pickle payload to disk."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle)
    return path


def load_pickle_payload(filepath: str | Path) -> Dict[str, Any]:
    """Load a pickled dictionary payload from disk."""
    path = Path(filepath)
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a dictionary payload in {path}")
    return payload


def save_training_checkpoint(checkpoint: Dict[str, Any], filepath: str | Path) -> Path:
    """Save a training checkpoint payload."""
    return save_pickle_payload(checkpoint, filepath)


def load_training_checkpoint(filepath: str | Path) -> Dict[str, Any]:
    """Load a training checkpoint payload."""
    return load_pickle_payload(filepath)


def save_checkpoint(
    params: Dict[str, Any], optimizer_state: Any, step: int, filepath: str | Path
) -> None:
    """Save model checkpoint."""
    checkpoint = {"params": params, "optimizer_state": optimizer_state, "step": step}
    save_pickle_payload(checkpoint, filepath)


def load_checkpoint(filepath: str | Path) -> tuple[Dict[str, Any], Any, int]:
    """Load model checkpoint."""
    checkpoint = load_pickle_payload(filepath)
    return checkpoint["params"], checkpoint["optimizer_state"], checkpoint["step"]
