"""
A much shorter version of train.py for benchmarking
"""
import os
from contextlib import nullcontext
import jax.numpy as jnp
import time
import torch
from gpt2 import GPTConfig, gpt2
