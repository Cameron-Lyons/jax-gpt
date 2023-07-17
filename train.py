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
