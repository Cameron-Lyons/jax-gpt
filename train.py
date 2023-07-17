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
