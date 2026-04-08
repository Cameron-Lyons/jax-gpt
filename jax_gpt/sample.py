"""Sample text from a trained or pretrained GPT model."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
from jax import random

from .checkpoints import checkpoint_path, model_config_from_checkpoint
from .config import SampleConfig, parse_sample_config
from .data import dataset_dir
from .model import GPT, GPTConfig, generate
from .parameter_converter import convert_functional_to_flax_params
from .utils import (
    TiktokenEncoder,
    get_gpt2_model_size,
    load_encoder_hparams_and_params,
    load_training_checkpoint,
    resolve_jax_device,
)

Tokenizer = tuple[Any, Any]


def load_prompt(start: str) -> str:
    """Load an inline prompt or `FILE:path` prompt source."""
    if start.startswith("FILE:"):
        return Path(start[5:]).read_text(encoding="utf-8")
    return start


def load_meta_tokenizer(meta_path: Path) -> Tokenizer:
    """Load a dataset tokenizer from `meta.pkl`."""
    with meta_path.open("rb") as handle:
        meta = pickle.load(handle)  # noqa: S301
    if not isinstance(meta, dict):
        raise TypeError(f"Expected metadata dictionary in {meta_path}")

    stoi = meta["stoi"]
    itos = meta["itos"]

    def encode_fn(text: str) -> list[int]:
        return [stoi[char] for char in text]

    def decode_fn(tokens: list[int]) -> str:
        return "".join(itos[index] for index in tokens)

    return encode_fn, decode_fn


def load_tiktoken_tokenizer() -> Tokenizer:
    """Load the default GPT-2 tokenizer."""
    encoder = TiktokenEncoder()
    return encoder.encode, encoder.decode


def tokenizer_for_resume(config: SampleConfig, checkpoint: dict[str, Any]) -> Tokenizer:
    """Pick the best tokenizer for a resumed checkpoint."""
    dataset = checkpoint.get("dataset", "openwebtext")
    data_dir = checkpoint.get("data_dir", config.data_dir)
    meta_path = dataset_dir(data_dir, dataset) / "meta.pkl"
    if meta_path.exists():
        print(f"Loading tokenizer metadata from {meta_path}")
        return load_meta_tokenizer(meta_path)
    print("No dataset metadata found; falling back to GPT-2 tokenizer.")
    return load_tiktoken_tokenizer()


def load_sampling_components(
    config: SampleConfig,
) -> tuple[GPT, dict[str, Any], Any, Any]:
    """Load model variables plus encode/decode functions for sampling."""
    if config.init_from == "resume":
        checkpoint = load_training_checkpoint(checkpoint_path(config.out_dir))
        model_config = model_config_from_checkpoint(checkpoint)
        model = GPT(model_config)
        encode_fn, decode_fn = tokenizer_for_resume(config, checkpoint)
        return model, {"params": checkpoint["params"]}, encode_fn, decode_fn

    if config.init_from.startswith("gpt2"):
        model_size = get_gpt2_model_size(config.init_from)
        encoder, hparams, params = load_encoder_hparams_and_params(model_size, config.models_dir)
        model_config = GPTConfig(
            n_layer=int(hparams["n_layer"]),
            n_head=int(hparams["n_head"]),
            n_embd=int(hparams["n_embd"]),
            block_size=int(hparams["n_ctx"]),
            vocab_size=int(hparams["n_vocab"]),
            dtype=config.dtype,
        )
        model = GPT(model_config)
        variables = {"params": convert_functional_to_flax_params(params, model_config)}
        return model, variables, encoder.encode, encoder.decode

    raise ValueError(f"Unknown init_from value: {config.init_from}")


def sample_texts(config: SampleConfig) -> list[str]:
    """Generate one or more text samples and return them as strings."""
    device = resolve_jax_device(config.device)
    prompt = load_prompt(config.start)
    rng = random.PRNGKey(config.seed)

    with jax.default_device(device):
        model, variables, encode, decode = load_sampling_components(config)
        prompt_ids = encode(prompt)
        if not prompt_ids:
            raise ValueError("Prompt must contain at least one token")

        inputs = jnp.asarray([prompt_ids], dtype=jnp.int32)
        samples: list[str] = []
        for _ in range(config.num_samples):
            rng, sample_rng = random.split(rng)
            generated = generate(
                model,
                variables,
                inputs,
                config.max_new_tokens,
                temperature=config.temperature,
                top_k=config.top_k,
                top_p=config.top_p,
                rng=sample_rng,
            )
            samples.append(decode(generated[0].tolist()))
        return samples


def main(argv: list[str] | None = None) -> list[str]:
    """CLI entry point for sampling."""
    config = parse_sample_config(argv)
    samples = sample_texts(config)
    print(f"Generating {len(samples)} sample(s)...")
    print("=" * 50)
    for sample_text in samples:
        print(sample_text)
        print("-" * 50)
    return samples


if __name__ == "__main__":
    main()
