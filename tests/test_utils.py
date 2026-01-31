"""Tests for utils.py (TiktokenEncoder, checkpoint save/load)."""

import tempfile
from pathlib import Path

import jax.numpy as jnp

from utils import TiktokenEncoder, load_checkpoint, save_checkpoint


class TestTiktokenEncoder:
    def test_encode_nonempty(self):
        enc = TiktokenEncoder()
        tokens = enc.encode("Hello, world!")
        assert len(tokens) > 0

    def test_decode_roundtrip(self):
        enc = TiktokenEncoder()
        text = "The quick brown fox."
        tokens = enc.encode(text)
        decoded = enc.decode(tokens)
        assert decoded == text

    def test_special_token(self):
        enc = TiktokenEncoder()
        tokens = enc.encode("<|endoftext|>")
        assert len(tokens) == 1

    def test_vocab_size(self):
        enc = TiktokenEncoder()
        assert enc.n_vocab == 50257


class TestCheckpoint:
    def test_save_and_load(self):
        params = {
            "layer": {
                "kernel": jnp.ones((4, 8)),
                "bias": jnp.zeros(8),
            }
        }
        opt_state = {"count": 42}
        step = 100

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            filepath = f.name

        save_checkpoint(params, opt_state, step, filepath)
        loaded_params, loaded_opt_state, loaded_step = load_checkpoint(filepath)

        assert loaded_step == step
        assert loaded_opt_state["count"] == 42
        assert jnp.allclose(loaded_params["layer"]["kernel"], params["layer"]["kernel"])
        assert jnp.allclose(loaded_params["layer"]["bias"], params["layer"]["bias"])

        Path(filepath).unlink()

    def test_save_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "ckpt.pkl"
            save_checkpoint({"w": jnp.array(1.0)}, None, 0, filepath)
            assert filepath.exists()
