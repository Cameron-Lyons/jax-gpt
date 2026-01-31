import jax
import jax.numpy as jnp
import pytest
from jax import random

from model import GPT, GPTConfig, configure_optimizers, get_model_size_mb, get_num_params


@pytest.fixture
def small_model_and_params(small_config, rng):
    model = GPT(small_config)
    variables = model.init(rng, jnp.ones((1, small_config.block_size), dtype=jnp.int32))
    return model, variables


class TestGPTConfig:
    def test_defaults(self):
        config = GPTConfig()
        assert config.n_layer == 12
        assert config.n_head == 12
        assert config.n_embd == 768
        assert config.vocab_size == 50257
        assert config.block_size == 1024

    def test_custom_values(self):
        config = GPTConfig(n_layer=6, n_head=6, n_embd=384, vocab_size=1000, block_size=128)
        assert config.n_layer == 6
        assert config.n_head == 6

    def test_bias_false(self):
        config = GPTConfig(bias=False)
        assert config.bias is False


class TestGPTForward:
    def test_output_shape(self, small_config, rng):
        model = GPT(small_config)
        x = jnp.ones((2, small_config.block_size), dtype=jnp.int32)
        variables = model.init(rng, x)
        logits, loss = model.apply(variables, x, training=False)
        assert logits.shape == (2, small_config.block_size, small_config.vocab_size)
        assert loss is None

    def test_forward_with_targets(self, small_config, rng):
        model = GPT(small_config)
        x = random.randint(rng, (2, small_config.block_size), 0, small_config.vocab_size)
        variables = model.init(rng, x)
        logits, loss = model.apply(variables, x, targets=x, training=False)
        assert logits.shape == (2, small_config.block_size, small_config.vocab_size)
        assert loss is not None
        assert loss.shape == ()
        assert jnp.isfinite(loss)

    def test_tied_embeddings(self, small_config, rng):
        config = small_config.replace(tie_word_embeddings=True)
        model = GPT(config)
        x = jnp.ones((1, 8), dtype=jnp.int32)
        variables = model.init(rng, x)
        logits, _ = model.apply(variables, x, training=False)
        assert logits.shape == (1, 8, config.vocab_size)

    def test_untied_embeddings(self, small_config, rng):
        config = small_config.replace(tie_word_embeddings=False)
        model = GPT(config)
        x = jnp.ones((1, 8), dtype=jnp.int32)
        variables = model.init(rng, x)
        logits, _ = model.apply(variables, x, training=False)
        assert logits.shape == (1, 8, config.vocab_size)
        assert "lm_head" in variables["params"]

    def test_different_model_sizes(self, rng):
        for n_layer in [1, 2, 4]:
            config = GPTConfig(
                n_layer=n_layer,
                n_head=2,
                n_embd=64,
                vocab_size=128,
                block_size=16,
                embd_pdrop=0.0,
                resid_pdrop=0.0,
                attn_pdrop=0.0,
                dtype="float32",
            )
            model = GPT(config)
            x = jnp.ones((1, 16), dtype=jnp.int32)
            variables = model.init(rng, x)
            logits, _ = model.apply(variables, x, training=False)
            assert logits.shape == (1, 16, 128)


class TestFlashAttention:
    def test_flash_vs_manual_match(self, rng):
        base_config = GPTConfig(
            n_layer=1,
            n_head=2,
            n_embd=64,
            vocab_size=128,
            block_size=16,
            embd_pdrop=0.0,
            resid_pdrop=0.0,
            attn_pdrop=0.0,
            dtype="float32",
            use_flash_attention=False,
        )
        flash_config = base_config.replace(use_flash_attention=True)

        model_manual = GPT(base_config)
        model_flash = GPT(flash_config)

        x = jnp.ones((1, 16), dtype=jnp.int32)
        variables = model_manual.init(rng, x)

        logits_manual, _ = model_manual.apply(variables, x, training=False)
        logits_flash, _ = model_flash.apply(variables, x, training=False)

        assert jnp.allclose(logits_manual, logits_flash, atol=1e-4)


class TestGradientCheckpointing:
    def test_checkpointing_forward_match(self, rng):
        base_config = GPTConfig(
            n_layer=2,
            n_head=2,
            n_embd=64,
            vocab_size=128,
            block_size=16,
            embd_pdrop=0.0,
            resid_pdrop=0.0,
            attn_pdrop=0.0,
            dtype="float32",
            gradient_checkpointing=False,
        )
        ckpt_config = base_config.replace(gradient_checkpointing=True)

        model_base = GPT(base_config)
        model_ckpt = GPT(ckpt_config)

        x = jnp.ones((1, 16), dtype=jnp.int32)
        variables = model_base.init(rng, x)

        logits_base, _ = model_base.apply(variables, x, training=False)
        logits_ckpt, _ = model_ckpt.apply(variables, x, training=False)

        assert jnp.allclose(logits_base, logits_ckpt, atol=1e-5)

    def test_checkpointing_grad_match(self, rng):
        base_config = GPTConfig(
            n_layer=2,
            n_head=2,
            n_embd=64,
            vocab_size=128,
            block_size=16,
            embd_pdrop=0.0,
            resid_pdrop=0.0,
            attn_pdrop=0.0,
            dtype="float32",
            gradient_checkpointing=False,
        )
        ckpt_config = base_config.replace(gradient_checkpointing=True)

        model_base = GPT(base_config)
        model_ckpt = GPT(ckpt_config)

        x = random.randint(rng, (2, 16), 0, 128)
        variables = model_base.init(rng, x)

        def loss_fn(params, model):
            logits, loss = model.apply({"params": params}, x, targets=x, training=False)
            return loss

        grads_base = jax.grad(loss_fn)(variables["params"], model_base)
        grads_ckpt = jax.grad(loss_fn)(variables["params"], model_ckpt)

        flat_base = jax.tree.leaves(grads_base)
        flat_ckpt = jax.tree.leaves(grads_ckpt)
        for gb, gc in zip(flat_base, flat_ckpt):
            assert jnp.allclose(gb, gc, atol=1e-5)


class TestLayerScaling:
    def test_inverse_layer_idx_scaling(self, rng):
        config_no_scale = GPTConfig(
            n_layer=2,
            n_head=2,
            n_embd=64,
            vocab_size=128,
            block_size=16,
            embd_pdrop=0.0,
            resid_pdrop=0.0,
            attn_pdrop=0.0,
            dtype="float32",
            scale_attn_by_inverse_layer_idx=False,
        )
        config_scale = config_no_scale.replace(scale_attn_by_inverse_layer_idx=True)

        model_no_scale = GPT(config_no_scale)
        model_scale = GPT(config_scale)

        x = jnp.ones((1, 16), dtype=jnp.int32)
        variables = model_no_scale.init(rng, x)

        logits_no_scale, _ = model_no_scale.apply(variables, x, training=False)
        logits_scale, _ = model_scale.apply(variables, x, training=False)

        assert not jnp.allclose(logits_no_scale, logits_scale, atol=1e-5)


class TestUpcastAttn:
    def test_reorder_and_upcast(self, rng):
        config = GPTConfig(
            n_layer=1,
            n_head=2,
            n_embd=64,
            vocab_size=128,
            block_size=16,
            embd_pdrop=0.0,
            resid_pdrop=0.0,
            attn_pdrop=0.0,
            dtype="float32",
            reorder_and_upcast_attn=True,
        )
        model = GPT(config)
        x = jnp.ones((1, 16), dtype=jnp.int32)
        variables = model.init(rng, x)
        logits, _ = model.apply(variables, x, training=False)
        assert logits.shape == (1, 16, 128)


class TestStandaloneFunctions:
    def test_get_num_params(self, small_model_and_params):
        _, variables = small_model_and_params
        n = get_num_params(variables["params"])
        assert n > 0

    def test_get_model_size_mb(self, small_model_and_params):
        _, variables = small_model_and_params
        mb = get_model_size_mb(variables["params"])
        assert mb > 0.0

    def test_configure_optimizers(self, small_model_and_params):
        _, variables = small_model_and_params
        tx = configure_optimizers(
            variables["params"],
            learning_rate=1e-3,
            weight_decay=0.1,
        )
        opt_state = tx.init(variables["params"])
        assert opt_state is not None


class TestNamedSubmodules:
    def test_param_names(self, small_config, rng):
        model = GPT(small_config)
        x = jnp.ones((1, small_config.block_size), dtype=jnp.int32)
        variables = model.init(rng, x)
        params = variables["params"]

        assert "wte" in params
        assert "wpe" in params
        assert "ln_f" in params
        assert "h_0" in params
        assert "h_1" in params

        block_params = params["h_0"]
        assert "attn" in block_params
        assert "ln_1" in block_params
        assert "ln_2" in block_params
        assert "mlp" in block_params
