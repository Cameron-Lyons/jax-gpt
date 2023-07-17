"""Script to train Jax GPT-2 model"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import jax.numpy as jnp
import jax
