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
