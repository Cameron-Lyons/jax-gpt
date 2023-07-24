"""
A much shorter version of train.py for benchmarking
"""
import os
from contextlib import nullcontext
import jax.numpy as jnp
import jax
import optax
import time
from gpt2 import GPTConfig, gpt2
from typing import Literal

batch_size: int = 12
block_size: int = 1024
bias: bool = False
real_data: bool = True
seed: int = 0
key = jax.random.PRNGKey(seed)
device: Literal["cpu", "gpu", "tpu"] = "gpu"
dtype: Literal["bfloat16", "float16", "float"] = "bfloat16"
compile: bool = True
profile: bool = False
exec(open("configurator.py").read())  # overrides from command line or config file

if real_data:
    dataset = "openwebtext"
    data_dir = os.path.join("data", dataset)
    train_data = jnp.memmap(
        os.path.join(data_dir, "train.bin"), dtype=jnp.uint16, mode="r"
    )

    def get_batch(split):
        data = train_data  # note ignore split in benchmarking script
        ix = jax.random.randint(key, (batch_size,), 0, len(data) - block_size)
        x = torch.stack(
            [torch.from_numpy((data[i : i + block_size]).astype(jnp.int64)) for i in ix]
        )
        y = torch.stack(
            [
                torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64))
                for i in ix
            ]
        )
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
            device, non_blocking=True
        )
        return x, y

else:
    # alternatively, if fixed data is desired to not care about data loading
    x = jax.random.randint(
        key=key,
        shape=(batch_size, block_size),
        minval=0,
        maxval=50304,
    )
    y = jax.random.randint(
        key=key,
        shape=(batch_size, block_size),
        minval=0,
        maxval=50304,
    )
    get_batch = lambda split: (x, y)

# model init
gptconf = GPTConfig(
    block_size=block_size,  # how far back does the model look? i.e. context size
    n_layer=12,
    n_head=12,
    n_embd=768,  # size of the model
    dropout=0,  # for determinism
    bias=bias,
)
model = gpt2(gptconf)

optimizer = optax.adamw(learning_rate=1e-4)
opt_state = optimizer.init(model)
