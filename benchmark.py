"""
A much shorter version of train.py for benchmarking
"""
import os
from contextlib import nullcontext
import jax.numpy as jnp
import time
import torch
from gpt2 import GPTConfig, gpt2

batch_size: int = 12
block_size: int = 1024
bias: bool = False
real_data: bool = True
seed: int = 0
device = "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)  # 'float32' or 'bfloat16' or 'float16'
compile: bool = True
profile: bool = False
exec(open("configurator.py").read())  # overrides from command line or config file
