"""SPMD data-parallel sharding utilities for JAX GPT-2 training.

Uses jax.sharding with NamedSharding + PartitionSpec for modern
data-parallel training.  Data is sharded along the batch dimension;
parameters are replicated across all devices.

Falls back gracefully to a single device when only one is available.
"""

from typing import Any, Dict

import jax
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P


def create_mesh(axis_name: str = "data") -> Mesh:
    """Create a 1-D device mesh for data parallelism."""
    devices = jax.devices()
    return Mesh(devices, axis_names=(axis_name,))


def data_sharding(mesh: Mesh, axis_name: str = "data") -> NamedSharding:
    """Sharding for data tensors: shard batch dim, replicate sequence dim."""
    return NamedSharding(mesh, P(axis_name, None))


def replicated_sharding(mesh: Mesh) -> NamedSharding:
    """Sharding for replicated tensors (e.g. parameters)."""
    return NamedSharding(mesh, P())


def shard_params(params: Dict[str, Any], mesh: Mesh) -> Dict[str, Any]:
    """Replicate parameters across all devices."""
    rep = replicated_sharding(mesh)
    return jax.tree.map(lambda x: jax.device_put(x, rep), params)  # type: ignore[no-any-return]


def shard_batch(
    batch: tuple[jax.Array, ...], mesh: Mesh, axis_name: str = "data"
) -> tuple[jax.Array, ...]:
    """Shard a (x, y) batch along the batch dimension."""
    ds = data_sharding(mesh, axis_name)
    return tuple(jax.device_put(x, ds) for x in batch)


def get_num_devices() -> int:
    """Return the number of available JAX devices."""
    return jax.device_count()
