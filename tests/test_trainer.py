"""Tests for the ModernTrainer."""

import jax.numpy as jnp
import pytest
from jax import random

from model import GPT, GPTConfig
from trainer import DataLoader, MetricsTracker, ModernTrainer, TrainingConfig


@pytest.fixture
def tiny_config():
    return GPTConfig(
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


@pytest.fixture
def dummy_data():
    rng = random.PRNGKey(0)
    data = random.randint(rng, (1000,), 0, 64).astype(jnp.int32)
    return data


class TestDataLoader:
    def test_get_batch_shapes(self, dummy_data):
        loader = DataLoader(dummy_data, batch_size=4, block_size=16)
        x, y = loader.get_batch("train")
        assert x.shape == (4, 16)
        assert y.shape == (4, 16)

    def test_get_batch_advances_rng(self, dummy_data):
        loader = DataLoader(dummy_data, batch_size=4, block_size=16)
        x1, _ = loader.get_batch("train")
        x2, _ = loader.get_batch("train")
        assert not jnp.array_equal(x1, x2)


class TestMetricsTracker:
    def test_update_and_get(self):
        tracker = MetricsTracker()
        tracker.update(loss=1.0, lr=0.001)
        assert tracker.get_latest("loss") == 1.0
        assert tracker.get_latest("lr") == 0.001

    def test_avg(self):
        tracker = MetricsTracker()
        for i in range(10):
            tracker.update(loss=float(i))
        avg = tracker.get_avg("loss", window=5)
        assert avg == pytest.approx(7.0)  # mean of [5, 6, 7, 8, 9]


class TestModernTrainer:
    def test_init(self, tiny_config, dummy_data):
        training_config = TrainingConfig(
            batch_size=4,
            block_size=16,
            vocab_size=64,
            max_iters=10,
            learning_rate=1e-3,
            warmup_iters=2,
            lr_decay_iters=10,
            eval_interval=5,
            eval_iters=2,
            save_interval=100,
            log_interval=1,
            compile=False,
            save_dir="/tmp/test_trainer_ckpt",
        )
        model = GPT(tiny_config)
        trainer = ModernTrainer(training_config, model, dummy_data, dummy_data)
        assert trainer.state is not None
        assert trainer.state.params is not None

    def test_train_step(self, tiny_config, dummy_data):
        training_config = TrainingConfig(
            batch_size=4,
            block_size=16,
            vocab_size=64,
            max_iters=2,
            learning_rate=1e-3,
            warmup_iters=1,
            lr_decay_iters=2,
            eval_interval=100,
            eval_iters=1,
            save_interval=100,
            log_interval=1,
            compile=False,
            save_dir="/tmp/test_trainer_step_ckpt",
        )
        model = GPT(tiny_config)
        trainer = ModernTrainer(training_config, model, dummy_data, dummy_data)

        batch = trainer.train_loader.get_batch("train")
        rng = random.PRNGKey(0)
        new_state, metrics = trainer._train_step(trainer.state, batch, rng)

        assert jnp.isfinite(metrics["loss"])
        assert new_state.step == trainer.state.step + 1

    def test_eval_step(self, tiny_config, dummy_data):
        training_config = TrainingConfig(
            batch_size=4,
            block_size=16,
            vocab_size=64,
            max_iters=2,
            learning_rate=1e-3,
            warmup_iters=1,
            lr_decay_iters=2,
            eval_interval=100,
            eval_iters=1,
            save_interval=100,
            log_interval=1,
            compile=False,
            save_dir="/tmp/test_trainer_eval_ckpt",
        )
        model = GPT(tiny_config)
        trainer = ModernTrainer(training_config, model, dummy_data, dummy_data)

        batch = trainer.val_loader.get_batch("val")
        metrics = trainer._eval_step(trainer.state, batch)
        assert jnp.isfinite(metrics["loss"])
