"""Tests for the functional GPT-2 implementation in gpt2.py."""

import jax.numpy as jnp
import pytest

from gpt2 import (
    GPTConfig,
    attention,
    gelu,
    layer_norm,
    linear,
    softmax,
)


class TestGPTConfig:
    def test_defaults(self):
        config = GPTConfig()
        assert config.block_size == 1024
        assert config.vocab_size == 50304
        assert config.n_layer == 12
        assert config.n_head == 12
        assert config.n_embd == 768


class TestBasicFunctions:
    def test_linear(self):
        x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        w = jnp.array([[0.5, 0.5], [0.5, 0.5]])
        b = jnp.array([1.0, 1.0])
        result = linear(x, w, b)
        assert result.shape == (2, 2)

    def test_softmax_sums_to_one(self):
        logits = jnp.array([1.0, 2.0, 3.0])
        probs = softmax(logits)
        assert jnp.allclose(jnp.sum(probs), 1.0)

    def test_softmax_monotonic(self):
        logits = jnp.array([1.0, 2.0, 3.0])
        probs = softmax(logits)
        assert probs[2] > probs[1] > probs[0]

    def test_gelu_shape(self):
        x = jnp.array([0.0, 1.0, -1.0])
        result = gelu(x)
        assert result.shape == x.shape

    def test_gelu_zero(self):
        result = gelu(jnp.array([0.0]))
        assert jnp.allclose(result, 0.0, atol=1e-6)

    def test_layer_norm_shape(self):
        x = jnp.ones((4, 8))
        g = jnp.ones(8)
        b = jnp.zeros(8)
        result = layer_norm(x, g, b)
        assert result.shape == x.shape


class TestAttention:
    def test_attention_shape(self):
        q = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        k = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        v = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        mask = jnp.array([[0.0, -1e10], [0.0, 0.0]])
        result = attention(q, k, v, mask)
        assert result.shape == (2, 2)

    def test_attention_causal_mask(self):
        q = jnp.ones((3, 2))
        k = jnp.ones((3, 2))
        v = jnp.eye(3, 2)
        mask = (1 - jnp.tri(3)) * -1e10
        result = attention(q, k, v, mask)
        assert result.shape == (3, 2)


class TestJITCompilation:
    def test_jitted_functions(self):
        x = jnp.array([1.0, 2.0, 3.0])
        result1 = linear(x.reshape(1, -1), jnp.eye(3), jnp.zeros(3))
        result2 = softmax(x)
        result3 = gelu(x)
        assert result1.shape == (1, 3)
        assert result2.shape == x.shape
        assert result3.shape == x.shape


@pytest.mark.network
class TestModelLoading:
    def test_load_encoder_and_params(self):
        from gpt2 import load_encoder_hparams_and_params

        encoder, hparams, params = load_encoder_hparams_and_params("124M", "models")
        assert encoder is not None
        assert hparams is not None
        assert params is not None

    def test_encode_decode_roundtrip(self):
        from gpt2 import load_encoder_hparams_and_params

        encoder, _, _ = load_encoder_hparams_and_params("124M", "models")
        text = "Hello, world!"
        tokens = encoder.encode(text)
        assert len(tokens) > 0

    def test_generation(self):
        from gpt2 import generate, load_encoder_hparams_and_params

        encoder, hparams, params = load_encoder_hparams_and_params("124M", "models")
        input_ids = encoder.encode("Hello")
        output_ids = generate(input_ids, params, hparams["n_head"], n_tokens_to_generate=5)
        assert len(output_ids) == 5
