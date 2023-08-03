"""
A much shorter version of train.py for benchmarking
"""
import os
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
    train_data = jnp.fromfile(data_dir, dtype=jnp.uint16)

    def get_batch():
        data = train_data  # note ignore split in benchmarking script
        ix = jax.random.randint(key, (batch_size,), 0, len(data) - block_size)
        x = jnp.stack([(data[i : i + block_size]).astype(jnp.int64) for i in ix])
        y = jnp.stack(
            [(data[i + 1 : i + 1 + block_size]).astype(jnp.int64) for i in ix]
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

if compile:
    model = jax.jit(model)

for stage, num_steps in enumerate([10, 20]):  # burnin, then benchmark
    t0 = time.time()
    X, Y = get_batch("train")
    for k in range(num_steps):
        logits, loss = model(X, Y)
        X, Y = get_batch("train")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        lossf = loss.item()
        print(f"{k}/{num_steps} loss: {lossf:.4f}")
    t1 = time.time()
    dt = t1 - t0
    mfu = model.estimate_mfu(batch_size * 1 * num_steps, dt)
    if stage == 1:
        print(f"time per iteration: {dt/num_steps*1000:.4f}ms, MFU: {mfu*100:.2f}%")
