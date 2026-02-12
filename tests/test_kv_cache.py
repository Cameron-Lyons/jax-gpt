"""Tests for KV-cache, generate(), and top-p sampling."""

import jax.numpy as jnp
import pytest
from jax import random

from model import GPT, GPTConfig, generate, init_kv_cache


@pytest.fixture
def tiny_config():
    return GPTConfig(
        n_layer=2,
        n_head=2,
        n_embd=64,
        vocab_size=128,
        block_size=32,
        embd_pdrop=0.0,
        resid_pdrop=0.0,
        attn_pdrop=0.0,
        bias=True,
        dtype="float32",
        tie_word_embeddings=True,
    )


@pytest.fixture
def tiny_model(tiny_config):
    model = GPT(tiny_config)
    rng = random.PRNGKey(0)
    dummy = jnp.ones((1, tiny_config.block_size), dtype=jnp.int32)
    variables = model.init(rng, dummy)
    return model, variables, tiny_config


class TestKVCacheCorrectness:
    def test_cached_vs_full_logits(self, tiny_model):
        """Full-seq without cache matches prefix+single-token with cache."""
        model, variables, config = tiny_model
        rng = random.PRNGKey(1)
        seq = random.randint(rng, (1, 10), 0, config.vocab_size)

        logits_full, _, _ = model.apply(variables, seq, training=False)

        prefix = seq[:, :-1]
        last_token = seq[:, -1:]

        head_dim = config.n_embd // config.n_head
        caches = init_kv_cache(
            1, config.n_layer, config.n_head, head_dim, config.block_size, jnp.float32
        )

        _, _, caches = model.apply(variables, prefix, training=False, cache=caches)
        logits_cached, _, _ = model.apply(variables, last_token, training=False, cache=caches)

        assert jnp.allclose(logits_full[:, -1, :], logits_cached[:, 0, :], atol=1e-5)

    def test_incremental_vs_full(self, tiny_model):
        """Process tokens one-by-one with cache, compare to full-sequence output."""
        model, variables, config = tiny_model
        rng = random.PRNGKey(2)
        seq_len = 8
        seq = random.randint(rng, (1, seq_len), 0, config.vocab_size)

        logits_full, _, _ = model.apply(variables, seq, training=False)

        head_dim = config.n_embd // config.n_head
        caches = init_kv_cache(
            1, config.n_layer, config.n_head, head_dim, config.block_size, jnp.float32
        )

        for i in range(seq_len):
            token = seq[:, i : i + 1]
            logits_inc, _, caches = model.apply(variables, token, training=False, cache=caches)

        assert jnp.allclose(logits_full[:, -1, :], logits_inc[:, 0, :], atol=1e-5)


class TestKVCacheWithRoPE:
    def test_cached_vs_full_with_rope(self):
        """Same cache correctness test with use_rope=True."""
        config = GPTConfig(
            n_layer=2,
            n_head=2,
            n_embd=64,
            vocab_size=128,
            block_size=32,
            embd_pdrop=0.0,
            resid_pdrop=0.0,
            attn_pdrop=0.0,
            bias=True,
            dtype="float32",
            use_rope=True,
        )
        model = GPT(config)
        rng = random.PRNGKey(0)
        dummy = jnp.ones((1, config.block_size), dtype=jnp.int32)
        variables = model.init(rng, dummy)

        rng = random.PRNGKey(3)
        seq = random.randint(rng, (1, 10), 0, config.vocab_size)

        logits_full, _, _ = model.apply(variables, seq, training=False)

        head_dim = config.n_embd // config.n_head
        caches = init_kv_cache(
            1, config.n_layer, config.n_head, head_dim, config.block_size, jnp.float32
        )

        prefix = seq[:, :-1]
        last_token = seq[:, -1:]
        _, _, caches = model.apply(variables, prefix, training=False, cache=caches)
        logits_cached, _, _ = model.apply(variables, last_token, training=False, cache=caches)

        assert jnp.allclose(logits_full[:, -1, :], logits_cached[:, 0, :], atol=1e-5)


class TestGenerate:
    def test_output_shape(self, tiny_model):
        model, variables, config = tiny_model
        prompt = jnp.array([[1, 2, 3]], dtype=jnp.int32)
        max_new = 5

        result = generate(model, variables, prompt, max_new, rng=random.PRNGKey(0))
        assert result.shape == (1, 3 + max_new)

    def test_greedy_determinism(self, tiny_model):
        model, variables, config = tiny_model
        prompt = jnp.array([[1, 2, 3]], dtype=jnp.int32)

        out1 = generate(model, variables, prompt, 10, temperature=0.0)
        out2 = generate(model, variables, prompt, 10, temperature=0.0)
        assert jnp.array_equal(out1, out2)

    def test_top_k(self, tiny_model):
        model, variables, config = tiny_model
        prompt = jnp.array([[1, 2, 3]], dtype=jnp.int32)
        result = generate(
            model, variables, prompt, 5, temperature=1.0, top_k=10, rng=random.PRNGKey(42)
        )
        assert result.shape == (1, 8)

    def test_top_p(self, tiny_model):
        model, variables, config = tiny_model
        prompt = jnp.array([[1, 2, 3]], dtype=jnp.int32)
        result = generate(
            model, variables, prompt, 5, temperature=1.0, top_p=0.9, rng=random.PRNGKey(42)
        )
        assert result.shape == (1, 8)

    def test_prompt_preserved(self, tiny_model):
        model, variables, config = tiny_model
        prompt = jnp.array([[10, 20, 30]], dtype=jnp.int32)
        result = generate(model, variables, prompt, 3, temperature=0.0)
        assert jnp.array_equal(result[:, :3], prompt)

    def test_batch_generation(self, tiny_model):
        model, variables, config = tiny_model
        prompt = jnp.array([[1, 2, 3], [4, 5, 6]], dtype=jnp.int32)
        result = generate(model, variables, prompt, 4, temperature=0.0)
        assert result.shape == (2, 7)
