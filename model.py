import jax.numpy as jnp
from flax import linen as nn

class NewGELU(nn.Module):
    def apply(self, x):
        return 0.5 * x * (1.0 + jnp.tanh(jnp.sqrt(2.0 / jnp.pi) * (x + 0.044715 * jnp.power(x, 3.0))))

