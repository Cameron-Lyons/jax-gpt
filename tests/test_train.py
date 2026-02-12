"""Smoke tests for train.py functions (not the top-level script)."""

import jax
import jax.numpy as jnp
import optax
import pytest
from jax import random

from model import GPT, GPTConfig


@pytest.fixture
def tiny_model_and_data():
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
    x = jnp.ones((2, 16), dtype=jnp.int32)
    variables = model.init(rng, x)
    params = variables["params"]
    return model, params, config


class TestTrainStep:
    def test_single_step(self, tiny_model_and_data):
        model, params, config = tiny_model_and_data

        optimizer = optax.adamw(learning_rate=1e-3, weight_decay=0.01)
        opt_state = optimizer.init(params)

        rng = random.PRNGKey(1)
        x = random.randint(rng, (2, 16), 0, 64)
        y = random.randint(rng, (2, 16), 0, 64)

        def loss_fn(params):
            logits, loss, _ = model.apply(
                {"params": params}, x, targets=y, training=True, rngs={"dropout": rng}
            )
            return loss, logits

        (loss, _), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        assert jnp.isfinite(loss)
        assert loss > 0

        flat_old = jax.tree.leaves(params)
        flat_new = jax.tree.leaves(new_params)
        changed = any(not jnp.allclose(a, b) for a, b in zip(flat_old, flat_new))
        assert changed

    def test_gradient_accumulation(self, tiny_model_and_data):
        model, params, config = tiny_model_and_data
        rng = random.PRNGKey(42)

        grad_accum_steps = 4
        accumulated_grads = None

        for step in range(grad_accum_steps):
            rng, batch_rng, step_rng = random.split(rng, 3)
            x = random.randint(batch_rng, (2, 16), 0, 64)
            y = random.randint(batch_rng, (2, 16), 0, 64)

            def loss_fn(params):
                logits, loss, _ = model.apply(
                    {"params": params},
                    x,
                    targets=y,
                    training=True,
                    rngs={"dropout": step_rng},
                )
                return loss, logits

            (loss, _), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            if accumulated_grads is None:
                accumulated_grads = grads
            else:
                accumulated_grads = jax.tree.map(lambda a, b: a + b, accumulated_grads, grads)

        avg_grads = jax.tree.map(lambda g: g / grad_accum_steps, accumulated_grads)

        flat_grads = jax.tree.leaves(avg_grads)
        assert all(jnp.isfinite(g).all() for g in flat_grads)


class TestEvalStep:
    def test_eval_no_grad(self, tiny_model_and_data):
        model, params, config = tiny_model_and_data
        rng = random.PRNGKey(0)
        x = random.randint(rng, (2, 16), 0, 64)
        y = random.randint(rng, (2, 16), 0, 64)

        logits, loss, _ = model.apply({"params": params}, x, targets=y, training=False)
        assert jnp.isfinite(loss)
        assert logits.shape == (2, 16, 64)
