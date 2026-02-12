"""Tests for the generation/sampling pipeline using model.py's GPT."""

import jax
import jax.numpy as jnp
import pytest
from jax import random

from model import GPT, GPTConfig


@pytest.fixture
def tiny_model():
    config = GPTConfig(
        n_layer=1,
        n_head=2,
        n_embd=32,
        vocab_size=64,
        block_size=16,
        embd_pdrop=0.0,
        resid_pdrop=0.0,
        attn_pdrop=0.0,
        dtype="float32",
    )
    model = GPT(config)
    rng = random.PRNGKey(0)
    dummy = jnp.ones((1, 16), dtype=jnp.int32)
    variables = model.init(rng, dummy)
    return model, variables, config


class TestAutoregressive:
    def test_single_token_generation(self, tiny_model):
        model, variables, config = tiny_model
        x = jnp.array([[1, 2, 3]], dtype=jnp.int32)
        logits, _, _ = model.apply(variables, x, training=False)
        next_logits = logits[:, -1, :]
        assert next_logits.shape == (1, config.vocab_size)

    def test_greedy_decode(self, tiny_model):
        model, variables, config = tiny_model
        prompt = jnp.array([[1, 2, 3]], dtype=jnp.int32)
        generated = prompt

        for _ in range(5):
            logits, _, _ = model.apply(variables, generated, training=False)
            next_token = jnp.argmax(logits[:, -1, :], axis=-1, keepdims=True)
            generated = jnp.concatenate([generated, next_token], axis=1)

        assert generated.shape == (1, 8)
        assert generated.dtype == jnp.int32

    def test_temperature_sampling(self, tiny_model):
        model, variables, config = tiny_model
        prompt = jnp.array([[1, 2, 3]], dtype=jnp.int32)

        logits, _, _ = model.apply(variables, prompt, training=False)
        last_logits = logits[:, -1, :]

        low_temp = last_logits / 0.1
        high_temp = last_logits / 2.0

        low_probs = jax.nn.softmax(low_temp, axis=-1)
        high_probs = jax.nn.softmax(high_temp, axis=-1)

        assert jnp.max(low_probs) > jnp.max(high_probs)

    def test_top_k_filtering(self, tiny_model):
        model, variables, config = tiny_model
        prompt = jnp.array([[1, 2, 3]], dtype=jnp.int32)
        logits, _, _ = model.apply(variables, prompt, training=False)
        last_logits = logits[0, -1, :]

        k = 10
        top_k_vals, top_k_idx = jax.lax.top_k(last_logits, k)

        mask = jnp.full_like(last_logits, -float("inf"))
        mask = mask.at[top_k_idx].set(last_logits[top_k_idx])
        filtered_probs = jax.nn.softmax(mask, axis=-1)

        assert jnp.sum(filtered_probs > 0) <= k

    def test_generation_respects_block_size(self, tiny_model):
        model, variables, config = tiny_model
        prompt = jnp.ones((1, config.block_size), dtype=jnp.int32)
        logits, _, _ = model.apply(variables, prompt, training=False)
        assert logits.shape == (1, config.block_size, config.vocab_size)

    def test_deterministic_greedy(self, tiny_model):
        model, variables, config = tiny_model
        prompt = jnp.array([[1, 2, 3]], dtype=jnp.int32)
        logits1, _, _ = model.apply(variables, prompt, training=False)
        logits2, _, _ = model.apply(variables, prompt, training=False)
        assert jnp.allclose(logits1, logits2)
