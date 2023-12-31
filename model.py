import jax.numpy as jnp
from jax import random
from jax.scipy.special import logsumexp
import flax
from flax import linen as nn
from flax.struct import dataclass
import optax
from transformers import FlaxGPT2LMHeadModel
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
        x += self.attn(self.ln_1(x))
        x += self.mlpf(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    model_type: str = "gpt2"
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

        self.wte = nn.Embed(num_embeddings=config.vocab_size, features=config.n_embd)
        self.wpe = nn.Embed(num_embeddings=config.block_size, features=config.n_embd)
        self.drop = nn.Dropout(rate=config.embd_pdrop)
        self.h = [Block(config) for _ in range(config.n_layer)]
        self.ln_f = nn.LayerNorm()
        self.lm_head = nn.Dense(
            config.vocab_size,
            kernel_init=nn.initializers.xavier_uniform(),
            use_bias=False,
        )

    def _init_weights(self, rng, shape, module_name, is_bias=False):
        if module_name in ["Dense", "Embed"]:
            # For weight initialization
            if not is_bias:
                std = 0.02
                return random.normal(rng, shape) * std
            # For bias initialization
            else:
                return jnp.zeros(shape)
        elif module_name == "LayerNorm":
            # For weight initialization
            if not is_bias:
                return jnp.ones(shape)
            # For bias initialization
            else:
                return jnp.zeros(shape)
        else:
            raise ValueError(f"Unknown module name: {module_name}")

    @classmethod
    def from_pretrained(cls, model_type):
        """
        Initialize a pretrained GPT model by copying over the weights
        from a huggingface/transformers checkpoint.
        """
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}

        config = cls.get_default_config()
        config.model_type = model_type
        config.vocab_size = 50257  # openai's model vocabulary
        config.block_size = 1024  # openai's model block_size
        model = GPT(config)

        model_hf = FlaxGPT2LMHeadModel.from_pretrained(model_type)
        return model

    def configure_optimizers(self, train_config):
        """
        Set up the optimizer for the JAX/Flax model.
        """

        decay_params = []
        no_decay_params = []

        whitelist_weight_modules = (nn.Dense,)
        blacklist_weight_modules = (
            nn.LayerNorm,
            nn.Embed,
        )

        params = flax.traverse_util.flatten_dict(self.params)

        for (module_name, param_name), _ in params.items():
            fpn = f"{module_name}.{param_name}"

            if param_name.endswith("bias"):
                no_decay_params.append(fpn)
            elif param_name.endswith("weight") and isinstance(
                getattr(self, module_name), whitelist_weight_modules
            ):
                decay_params.append(fpn)
            elif param_name.endswith("weight") and isinstance(
                getattr(self, module_name), blacklist_weight_modules
            ):
                no_decay_params.append(fpn)

        # Ensure no overlap and coverage of parameters
        inter_params = set(decay_params) & set(no_decay_params)
        union_params = set(decay_params) | set(no_decay_params)
        assert (
            len(inter_params) == 0
        ), f"Parameters {inter_params} made it into both decay/no_decay sets!"
        assert (
            len(set(params.keys()) - union_params) == 0
        ), f"Parameters {set(params.keys()) - union_params} were not separated into either decay/no_decay set!"

        weight_decay = optax.adamw(train_config.weight_decay)
        no_weight_decay = optax.adamw(0.0)

        optimizer = optax.chain(
            optax.masked(weight_decay, mask=decay_params),
            optax.masked(no_weight_decay, mask=no_decay_params),
            optax.scale_by_adam(b1=train_config.betas[0], b2=train_config.betas[1]),
            optax.scale(-train_config.learning_rate),
        )

        return optimizer

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.shape
        assert (
            t <= self.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = jnp.arange(0, t, dtype=jnp.int32).reshape(1, -1)  # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(
            pos
        )  # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            flat_logits = logits.reshape(-1, logits.shape[-1])
            flat_targets = targets.reshape(-1)
            loss = optax.softmax_cross_entropy(flat_logits, flat_targets).mean()

        return logits, loss

    def generate(
        self,
        idx,
        max_new_tokens,
        temperature=1.0,
        do_sample=False,
        top_k=None,
        rng=None,
    ):
        """
        Take a conditioning sequence of indices idx (of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        """
        if rng is None:
            rng = random.PRNGKey(0)

        for _ in range(max_new_tokens):
            idx_cond = (
                idx if idx.shape[1] <= self.block_size else idx[:, -self.block_size :]
            )

            logits, _ = self(idx_cond)

            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v = jnp.sort(logits, axis=-1)[:, -top_k:]
            probs = nn.softmax(logits)

            if do_sample:
                idx_next = random.categorical(rng, logits, axis=-1)
            else:
                idx_next = jnp.argmax(probs, axis=-1)

            idx = jnp.concatenate((idx, idx_next[:, None]), axis=1)

        return idx
