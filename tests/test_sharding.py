"""Tests for sharding utilities.

Run with:
  XLA_FLAGS=--xla_force_host_platform_device_count=4 pytest tests/test_sharding.py -v
"""

import os

import jax
import jax.numpy as jnp
from jax import random

from sharding import (
    create_mesh,
    data_sharding,
    get_num_devices,
    replicated_sharding,
    shard_batch,
    shard_params,
)

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")


class TestMeshCreation:
    def test_create_mesh(self):
        mesh = create_mesh()
        assert mesh is not None
        assert len(mesh.devices.flat) == jax.device_count()

    def test_mesh_axis_name(self):
        mesh = create_mesh("dp")
        assert "dp" in mesh.axis_names


class TestSharding:
    def test_data_sharding(self):
        mesh = create_mesh()
        ds = data_sharding(mesh)
        assert ds is not None

    def test_replicated_sharding(self):
        mesh = create_mesh()
        rs = replicated_sharding(mesh)
        assert rs is not None

    def test_shard_batch(self):
        mesh = create_mesh()
        n_devices = get_num_devices()
        batch_size = 4 * n_devices
        rng = random.PRNGKey(0)
        x = random.randint(rng, (batch_size, 16), 0, 100)
        y = random.randint(rng, (batch_size, 16), 0, 100)

        sx, sy = shard_batch((x, y), mesh)
        assert sx.shape == x.shape
        assert sy.shape == y.shape

    def test_shard_params(self):
        mesh = create_mesh()
        params = {
            "layer": {
                "kernel": jnp.ones((32, 64)),
                "bias": jnp.zeros((64,)),
            }
        }
        sharded = shard_params(params, mesh)
        assert sharded["layer"]["kernel"].shape == (32, 64)


class TestWithModel:
    def test_sharded_forward(self):
        from model import GPT, GPTConfig

        mesh = create_mesh()
        n_devices = get_num_devices()

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
        x = jnp.ones((n_devices * 2, 16), dtype=jnp.int32)
        variables = model.init(rng, x)

        params = shard_params(variables["params"], mesh)
        (sx,) = shard_batch((x,), mesh)

        @jax.jit
        def forward(params, x):
            return model.apply({"params": params}, x, training=False)

        logits, loss, _ = forward(params, sx)
        assert logits.shape == (n_devices * 2, 16, 64)
