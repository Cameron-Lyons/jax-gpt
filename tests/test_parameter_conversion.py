import jax
import jax.numpy as jnp
import pytest
from jax import random

from model import GPT, GPTConfig
from parameter_converter import (
    convert_flax_to_functional_params,
    convert_functional_to_flax_params,
)


@pytest.fixture
def small_gpt_config():
    return GPTConfig(
        n_layer=2,
        n_head=2,
        n_embd=64,
        vocab_size=256,
        block_size=32,
        embd_pdrop=0.0,
        resid_pdrop=0.0,
        attn_pdrop=0.0,
        bias=True,
        dtype="float32",
        tie_word_embeddings=True,
    )


@pytest.fixture
def flax_params_and_config(small_gpt_config):
    model = GPT(small_gpt_config)
    rng = random.PRNGKey(0)
    x = jnp.ones((1, 16), dtype=jnp.int32)
    variables = model.init(rng, x)
    return variables["params"], small_gpt_config


class TestFunctionalToFlax:
    def test_roundtrip(self, flax_params_and_config):
        flax_params, config = flax_params_and_config

        functional = convert_flax_to_functional_params(flax_params, config)
        recovered = convert_functional_to_flax_params(functional, config)

        flat_orig = jax.tree.leaves(flax_params)
        flat_recovered = jax.tree.leaves(recovered)
        for o, r in zip(flat_orig, flat_recovered):
            assert jnp.allclose(o, r, atol=1e-6)

    def test_roundtrip_output_match(self, flax_params_and_config):
        flax_params, config = flax_params_and_config
        model = GPT(config)
        x = jnp.ones((1, 16), dtype=jnp.int32)

        logits_orig, _, _ = model.apply({"params": flax_params}, x, training=False)

        functional = convert_flax_to_functional_params(flax_params, config)
        recovered = convert_functional_to_flax_params(functional, config)
        logits_recovered, _, _ = model.apply({"params": recovered}, x, training=False)

        assert jnp.allclose(logits_orig, logits_recovered, atol=1e-5)


class TestFlaxToFunctional:
    def test_structure(self, flax_params_and_config):
        flax_params, config = flax_params_and_config
        functional = convert_flax_to_functional_params(flax_params, config)

        assert "wte" in functional
        assert "wpe" in functional
        assert "blocks" in functional
        assert len(functional["blocks"]) == config.n_layer
        assert "ln_f" in functional

        block = functional["blocks"][0]
        assert "attn" in block
        assert "c_attn" in block["attn"]
        assert "w" in block["attn"]["c_attn"]
        assert "b" in block["attn"]["c_attn"]
        assert "mlp" in block
        assert "ln_1" in block
        assert "ln_2" in block

    def test_shapes(self, flax_params_and_config):
        flax_params, config = flax_params_and_config
        functional = convert_flax_to_functional_params(flax_params, config)

        assert functional["wte"].shape == (config.vocab_size, config.n_embd)
        assert functional["wpe"].shape == (config.block_size, config.n_embd)

        block = functional["blocks"][0]
        assert block["attn"]["c_attn"]["w"].shape == (config.n_embd, 3 * config.n_embd)
        assert block["attn"]["c_attn"]["b"].shape == (3 * config.n_embd,)
        assert block["mlp"]["c_fc"]["w"].shape == (config.n_embd, 4 * config.n_embd)


@pytest.mark.network
class TestPretrainedConversion:
    def test_pretrained_roundtrip(self):
        from utils import load_encoder_hparams_and_params

        encoder, hparams, functional_params = load_encoder_hparams_and_params("124M")

        config = GPTConfig(
            n_layer=hparams["n_layer"],
            n_head=hparams["n_head"],
            n_embd=hparams["n_embd"],
            vocab_size=hparams["n_vocab"],
            block_size=hparams["n_ctx"],
            bias=True,
            dtype="float32",
            embd_pdrop=0.0,
            resid_pdrop=0.0,
            attn_pdrop=0.0,
        )

        flax_params = convert_functional_to_flax_params(functional_params, config)
        recovered = convert_flax_to_functional_params(flax_params, config)

        for leaf_orig, leaf_rec in zip(
            jax.tree.leaves(functional_params), jax.tree.leaves(recovered)
        ):
            assert jnp.allclose(leaf_orig, leaf_rec, atol=1e-6)
