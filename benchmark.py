"""
A much shorter version of train.py for benchmarking
"""
import os
from contextlib import nullcontext
import jax.numpy as jnp
import time
import torch
from gpt2 import GPTConfig, gpt2
from typing import Literal


batch_size: int = 12
block_size: int = 1024
bias: bool = False
real_data: bool = True
seed: int = 0
device: Literal["cpu", "gpu", "tpu"] = "gpu"
dtype: Literal["bfloat16", "float16", "float"] = "bfloat16"
compile: bool = True
profile: bool = False
exec(open("configurator.py").read())  # overrides from command line or config file
