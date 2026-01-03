"""Utilities for JAX GPT-2 models."""
import json
import logging
from pathlib import Path
from typing import Literal, Dict, Any, Tuple, Optional, Union, List
from dataclasses import dataclass
import pickle

import jax.numpy as jnp
from jax import random
import tiktoken

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ModelSize = Literal["124M", "355M", "774M", "1558M"]


@dataclass
class ModelConfig:
    """Configuration for GPT-2 model variants."""
    n_layer: int
    n_head: int
    n_embd: int
    vocab_size: int
    block_size: int


class TiktokenEncoder:
    """GPT-2 BPE encoder using tiktoken."""

    def __init__(self):
        self.enc = tiktoken.get_encoding("gpt2")
        self.n_vocab = 50257

    def encode(self, text: str) -> List[int]:
        return self.enc.encode(text, allowed_special={"<|endoftext|>"})

    def decode(self, tokens: List[int]) -> str:
        return self.enc.decode(tokens)


def load_encoder_hparams_and_params(
    model_size: ModelSize, models_dir: str = None
) -> Tuple[TiktokenEncoder, Dict[str, Any], Dict[str, Any]]:
    """Load encoder, hparams, and params from HuggingFace."""
    try:
        from transformers import GPT2LMHeadModel
    except ImportError:
        raise ImportError("Please install transformers: pip install transformers")

    model_name_map = {
        "124M": "gpt2",
        "355M": "gpt2-medium",
        "774M": "gpt2-large",
        "1558M": "gpt2-xl",
    }

    model_name = model_name_map.get(model_size, model_size)
    logger.info(f"Loading HuggingFace model: {model_name}")

    model_hf = GPT2LMHeadModel.from_pretrained(model_name)
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
    def to_jax(tensor):
        return jnp.array(tensor.numpy())

    params = {
        "wte": to_jax(sd["transformer.wte.weight"]),
        "wpe": to_jax(sd["transformer.wpe.weight"]),
        "blocks": [],
        "ln_f": {
            "b": to_jax(sd["transformer.ln_f.bias"]),
            "g": to_jax(sd["transformer.ln_f.weight"]),
        }
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
        params["blocks"].append(block)

    encoder = TiktokenEncoder()
    return encoder, hparams, params


def save_checkpoint(params: Dict[str, Any], optimizer_state: Any, step: int,
                    filepath: Union[str, Path]):
    """Save model checkpoint."""
    checkpoint = {
        "params": params,
        "optimizer_state": optimizer_state,
        "step": step
    }
    with open(filepath, 'wb') as f:
        pickle.dump(checkpoint, f)


def load_checkpoint(filepath: Union[str, Path]) -> Tuple[Dict[str, Any], Any, int]:
    """Load model checkpoint."""
    with open(filepath, 'rb') as f:
        checkpoint = pickle.load(f)
    return checkpoint["params"], checkpoint["optimizer_state"], checkpoint["step"]
