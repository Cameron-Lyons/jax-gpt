import jax.numpy as jnp
from jax.scipy.special import logsumexp
from flax import linen as nn
from flax.struct import dataclass
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


@dataclass
class GPTConfig:
    model_type: str = "gpt"
    n_layer: int = None
    n_head: int = None
    n_embd: int = None
    vocab_size: int = None
    block_size: int = None
    embd_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    attn_pdrop: float = 0.1


class GPT(nn.Module):
    config: GPTConfig

    def setup(self):
        config = self.config
        assert config.vocab_size is not None
        assert config.block_size is not None

        if config.model_type:
            model_configs = {
                # GPT-1
                "openai-gpt": dict(n_layer=12, n_head=12, n_embd=768),  # 117M params
                # GPT-2 configs
                "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
                "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
                "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
                "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
                # Gophers
                "gopher-44m": dict(n_layer=8, n_head=16, n_embd=512),
                # Smaller custom models
                "gpt-mini": dict(n_layer=6, n_head=6, n_embd=192),
                "gpt-micro": dict(n_layer=4, n_head=4, n_embd=128),
                "gpt-nano": dict(n_layer=3, n_head=3, n_embd=48),
            }

            config = GPTConfig(**model_configs[config.model_type])

        self.wte = nn.Embed(vocab_size=config.vocab_size, features=config.n_embd)
        self.wpe = nn.Embed(vocab_size=config.block_size, features=config.n_embd)
        self.drop = nn.Dropout(rate=config.embd_pdrop)
        self.h = [Block(config) for _ in range(config.n_layer)]
        self.ln_f = nn.LayerNorm()
        self.lm_head = nn.Dense(
            config.vocab_size,
            kernel_init=nn.initializers.xavier_uniform(),
            use_bias=False,
        )
