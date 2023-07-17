"""Script to train Jax GPT-2 model"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import jax.numpy as jnp
import jax
from typing import Literal

# I/O
out_dir: str = "out"
eval_interval: int = 2000
log_interval: int = 1
eval_iters: int = 200
eval_only: bool = False  # if True, script exits right after the first eval
always_save_checkpoint: bool = True  # if True, always save a checkpoint after each eval
init_from: Literal["scratch", "resume", "gpt2"] = "scratch"

# data
dataset: str = "openwebtext"
gradient_accumulation_steps: int = 5 * 8  # used to simulate larger batch sizes
batch_size: int = 12  # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size: int = 1024

# model
n_layer: int = 12
n_head: int = 12
n_embd: int = 768
dropout: float = 0.0  # for pretraining 0 is good, for finetuning try 0.1+
bias: bool = False  # do we use bias inside LayerNorm and Linear layers?

# adamw optimizer
learning_rate: float = 6e-4  # max learning rate
max_iters: int = 600000  # total number of training iterations
weight_decay: float = 1e-1
beta1: float = 0.9
beta2: float = 0.95
grad_clip: float = 1.0  # clip gradients at this value, or disable if == 0.0

# learning rate decay settings
decay_lr: bool = True  # whether to decay the learning rate
warmup_iters: int = 2000  # how many steps to warm up for
lr_decay_iters: int = 600000
min_lr: float = 6e-5
# system
device: Literal["cpu", "gpu", "tpu"] = jax.default_backend()
dtype: Literal["bfloat16", "float16", "float"] = "bfloat16"

config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
exec(open("configurator.py").read())  # overrides from command line or config file
config = {k: globals()[k] for k in config_keys}  # will be useful for logging

random_key = jax.random.PRNGKey(0)

data_dir = os.path.join("data", dataset)
train_data: jax.Array = jnp.load(
    os.path.join(data_dir, "train.bin"), dtype=jnp.uint16, mode="r"
)
val_data: jax.Array = jnp.load(
    os.path.join(data_dir, "val.bin"), dtype=jnp.uint16, mode="r"
)


def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = jax.random.randint(random_key, (), 0, data.shape[0])
    x = jnp.stack([data[i : i + block_size].astype(jnp.int64) for i in ix], axis=-1)
    y = jnp.stack(
        [data[i + 1 : i + 1 + block_size].astype(jnp.int64) for i in ix], axis=-1
    )
    return x, y


iter_num: int = 0
best_val_loss: float = 1e9

meta_path: str = os.path.join(data_dir, "meta.pkl")
meta_vocab_size: int = 0
if os.path.exists(meta_path):
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    meta_vocab_size = meta["vocab_size"]
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

model_args = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=block_size,
    bias=bias,
    vocab_size=None,
    dropout=dropout,
)  # start with model_args from command line
