"""Parameter conversion between functional (gpt2.py) and Flax (model.py) formats."""

from typing import Any, Dict

import jax
import jax.numpy as jnp

from model import GPT, GPTConfig


def convert_functional_to_flax_params(
    functional_params: Dict[str, Any],
    config: GPTConfig,
) -> Dict[str, Any]:
    """Convert parameters from functional gpt2.py format to model.py GPT Flax format.

    The functional format (from utils.load_encoder_hparams_and_params) uses:
      wte: (vocab_size, n_embd)
      wpe: (block_size, n_embd)
      blocks[i].attn.c_attn.w/b  — w shape (n_embd, 3*n_embd)
      blocks[i].attn.c_proj.w/b
      blocks[i].mlp.c_fc.w/b
      blocks[i].mlp.c_proj.w/b
      blocks[i].ln_1.g/b
      blocks[i].ln_2.g/b
      ln_f.g/b

    The Flax format (model.py GPT) uses:
      wte/embedding: (vocab_size, n_embd)
      wpe/embedding: (block_size, n_embd)
      h_{i}/attn/c_attn/kernel|bias
      h_{i}/attn/c_proj/kernel|bias
      h_{i}/mlp/c_fc/kernel|bias
      h_{i}/mlp/c_proj/kernel|bias
      h_{i}/ln_1/scale|bias
      h_{i}/ln_2/scale|bias
      ln_f/scale|bias
    """
    flax_params: Dict[str, Any] = {}

    flax_params["wte"] = {"embedding": functional_params["wte"]}
    flax_params["wpe"] = {"embedding": functional_params["wpe"]}

    for i, block in enumerate(functional_params["blocks"]):
        block_key = f"h_{i}"
        flax_params[block_key] = {
            "attn": {
                "c_attn": {
                    "kernel": block["attn"]["c_attn"]["w"],
                    "bias": block["attn"]["c_attn"]["b"],
                },
                "c_proj": {
                    "kernel": block["attn"]["c_proj"]["w"],
                    "bias": block["attn"]["c_proj"]["b"],
                },
            },
            "mlp": {
                "c_fc": {
                    "kernel": block["mlp"]["c_fc"]["w"],
                    "bias": block["mlp"]["c_fc"]["b"],
                },
                "c_proj": {
                    "kernel": block["mlp"]["c_proj"]["w"],
                    "bias": block["mlp"]["c_proj"]["b"],
                },
            },
            "ln_1": {
                "scale": block["ln_1"]["g"],
                "bias": block["ln_1"]["b"],
            },
            "ln_2": {
                "scale": block["ln_2"]["g"],
                "bias": block["ln_2"]["b"],
            },
        }

    flax_params["ln_f"] = {
        "scale": functional_params["ln_f"]["g"],
        "bias": functional_params["ln_f"]["b"],
    }

    if not config.tie_word_embeddings:
        flax_params["lm_head"] = {
            "kernel": functional_params["wte"].T,
        }

    return flax_params


def convert_flax_to_functional_params(
    flax_params: Dict[str, Any],
    config: GPTConfig,
) -> Dict[str, Any]:
    """Convert parameters from Flax (model.py GPT) format to functional (gpt2.py) format."""
    functional_params: Dict[str, Any] = {}

    functional_params["wte"] = flax_params["wte"]["embedding"]
    functional_params["wpe"] = flax_params["wpe"]["embedding"]

    blocks = []
    for i in range(config.n_layer):
        block_key = f"h_{i}"
        bp = flax_params[block_key]
        blocks.append(
            {
                "attn": {
                    "c_attn": {
                        "w": bp["attn"]["c_attn"]["kernel"],
                        "b": bp["attn"]["c_attn"]["bias"],
                    },
                    "c_proj": {
                        "w": bp["attn"]["c_proj"]["kernel"],
                        "b": bp["attn"]["c_proj"]["bias"],
                    },
                },
                "mlp": {
                    "c_fc": {
                        "w": bp["mlp"]["c_fc"]["kernel"],
                        "b": bp["mlp"]["c_fc"]["bias"],
                    },
                    "c_proj": {
                        "w": bp["mlp"]["c_proj"]["kernel"],
                        "b": bp["mlp"]["c_proj"]["bias"],
                    },
                },
                "ln_1": {
                    "g": bp["ln_1"]["scale"],
                    "b": bp["ln_1"]["bias"],
                },
                "ln_2": {
                    "g": bp["ln_2"]["scale"],
                    "b": bp["ln_2"]["bias"],
                },
            }
        )

    functional_params["blocks"] = blocks
    functional_params["ln_f"] = {
        "g": flax_params["ln_f"]["scale"],
        "b": flax_params["ln_f"]["bias"],
    }

    return functional_params


def verify_parameter_conversion(
    functional_params: Dict[str, Any],
    flax_params: Dict[str, Any],
    config: GPTConfig,
    dummy_input: jax.Array,
    n_head: int,
) -> bool:
    """Verify that parameter conversion is correct by comparing outputs."""
    from gpt2 import gpt2 as gpt2_fn

    functional_output = gpt2_fn(dummy_input, **functional_params, n_head=n_head)

    model = GPT(config)
    flax_output, _, _ = model.apply({"params": flax_params}, dummy_input[None], training=False)  # type: ignore[misc]
    flax_output = flax_output[0]

    diff = jnp.abs(functional_output - flax_output).max()
    return float(diff) < 1e-4
