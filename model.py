import jax.numpy as jnp
from flax import linen as nn
from jax.scipy.special import logsumexp
import math


class NewGELU(nn.Module):
    def apply(self, x):
        return (
            0.5
            * x
            * (
                1.0
                + jnp.tanh(jnp.sqrt(2.0 / jnp.pi) * (x + 0.044715 * jnp.power(x, 3.0)))
            )
        )


class CausalSelfAttention(nn.Module):
    n_embd: int
    n_head: int
    block_size: int
    attn_pdrop: float
    resid_pdrop: float

    def setup(self):
        assert self.n_embd % self.n_head == 0
        self.c_attn = nn.Dense(
            3 * self.n_embd, kernel_init=nn.initializers.xavier_uniform()
        )
        self.c_proj = nn.Dense(
            self.n_embd, kernel_init=nn.initializers.xavier_uniform()
        )
        self.bias = jnp.tril(jnp.ones((self.block_size, self.block_size)))

    def __call__(self, x):
        B, T, C = x.shape

        q, k, v = jnp.split(self.c_attn(x), 3, axis=-1)
        k = k.reshape((B, T, self.n_head, C // self.n_head)).transpose((0, 2, 1, 3))
        q = q.reshape((B, T, self.n_head, C // self.n_head)).transpose((0, 2, 1, 3))
        v = v.reshape((B, T, self.n_head, C // self.n_head)).transpose((0, 2, 1, 3))

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.shape[-1]))
        att = jnp.where(self.bias[:T, :T] == 0, float("-inf"), att)
        att = jnp.exp(att - logsumexp(att, axis=-1, keepdims=True))
        att = nn.Dropout(self.attn_pdrop)(att)
        y = att @ v
        y = y.transpose((0, 2, 1, 3)).reshape((B, T, C))

        y = nn.Dropout(self.resid_pdrop)(self.c_proj(y))
        return y


class Block(nn.Module):
    n_embd: int
    resid_pdrop: float

    def setup(self):
        self.ln_1 = nn.LayerNorm()
        self.attn = CausalSelfAttention(
            n_embd=self.n_embd,
            n_head=self.n_head,
            block_size=self.block_size,
            attn_pdrop=self.attn_pdrop,
            resid_pdrop=self.resid_pdrop,
        )
        self.ln_2 = nn.LayerNorm()

        self.c_fc = nn.Dense(
            4 * self.n_embd, kernel_init=nn.initializers.xavier_uniform()
        )
        self.c_proj = nn.Dense(
            self.n_embd, kernel_init=nn.initializers.xavier_uniform()
        )
        self.act = NewGELU()
        self.dropout = nn.Dropout(rate=self.resid_pdrop)

    def mlpf(self, x):
        return self.dropout(self.c_proj(self.act(self.c_fc(x))))

    def __call__(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x
