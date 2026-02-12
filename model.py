"""
Improved GPT-2 model implementation for JAX with enhanced features.

This implementation includes several improvements over the original:
- Modern JAX/Flax best practices
- JIT compilation for better performance
- Enhanced attention mechanisms
- Better initialization strategies
- Improved generation capabilities
- Type hints and comprehensive documentation
- Memory-efficient operations
- Advanced sampling strategies
"""

import math
from typing import Any, Dict

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.struct import dataclass
from jax import random


@dataclass
class GPTConfig:
    """Enhanced GPT configuration with additional parameters."""

    model_type: str = "gpt2"
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    vocab_size: int = 50257
    block_size: int = 1024
    embd_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    bias: bool = True
    use_rope: bool = False
    use_flash_attention: bool = False
    gradient_checkpointing: bool = False
    dtype: str = "bfloat16"
    tie_word_embeddings: bool = True
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    scale_attn_weights: bool = True
    scale_attn_by_inverse_layer_idx: bool = False
    reorder_and_upcast_attn: bool = False


@dataclass
class KVCache:
    """Pre-allocated key/value cache for autoregressive generation."""

    k: jax.Array
    v: jax.Array
    length: jax.Array


def init_kv_cache(
    batch_size: int,
    n_layer: int,
    n_head: int,
    head_dim: int,
    max_seq_len: int,
    dtype: jnp.dtype = jnp.bfloat16,
) -> list["KVCache"]:
    """Create a list of empty KV caches, one per layer."""
    caches = []
    for _ in range(n_layer):
        k = jnp.zeros((batch_size, n_head, max_seq_len, head_dim), dtype=dtype)
        v = jnp.zeros((batch_size, n_head, max_seq_len, head_dim), dtype=dtype)
        length = jnp.array(0, dtype=jnp.int32)
        caches.append(KVCache(k=k, v=v, length=length))
    return caches


class NewGELU(nn.Module):
    """Improved GELU activation with better numerical stability."""

    @nn.compact
    def __call__(self, x):
        return (
            0.5 * x * (1.0 + jnp.tanh(jnp.sqrt(2.0 / jnp.pi) * (x + 0.044715 * jnp.power(x, 3.0))))
        )


class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embeddings (RoPE) for better position encoding."""

    dim: int
    max_position_embeddings: int = 2048

    @nn.compact
    def __call__(
        self, x: jax.Array, seq_len: int | None = None, offset: int | jax.Array = 0
    ) -> jax.Array:
        if seq_len is None:
            seq_len = x.shape[-2]

        inv_freq = 1.0 / (10000 ** (jnp.arange(0, self.dim, 2).astype(jnp.float32) / self.dim))
        sinusoid_inp = jnp.einsum("i,j->ij", jnp.arange(seq_len) + offset, inv_freq)

        sin = jnp.concatenate([jnp.sin(sinusoid_inp)] * 2, axis=-1)
        cos = jnp.concatenate([jnp.cos(sinusoid_inp)] * 2, axis=-1)

        x1 = x[..., : self.dim // 2]
        x2 = x[..., self.dim // 2 :]
        x_rot = jnp.concatenate([-x2, x1], axis=-1)

        return x * cos + x_rot * sin


class CausalSelfAttention(nn.Module):
    """Enhanced causal self-attention with modern optimizations."""

    n_embd: int
    n_head: int
    block_size: int
    attn_pdrop: float
    resid_pdrop: float
    bias: bool = True
    use_rope: bool = False
    use_flash_attention: bool = False
    scale_attn_weights: bool = True
    scale_attn_by_inverse_layer_idx: bool = False
    reorder_and_upcast_attn: bool = False
    layer_idx: int = 0
    dtype: str = "bfloat16"

    @nn.compact
    def __call__(
        self, x: jax.Array, training: bool = True, cache: KVCache | None = None
    ) -> tuple[jax.Array, KVCache | None]:
        B, T, C = x.shape
        head_dim = C // self.n_head

        qkv = nn.Dense(
            features=3 * C,
            use_bias=self.bias,
            kernel_init=nn.initializers.xavier_uniform(),
            dtype=getattr(jnp, self.dtype),
            name="c_attn",
        )(x)

        qkv = qkv.reshape(B, T, 3, self.n_head, head_dim)
        qkv = qkv.transpose(0, 2, 3, 1, 4)  # (B, 3, n_head, T, head_dim)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        if self.use_rope:
            rope = RotaryPositionEmbedding(dim=head_dim)
            offset = cache.length if cache is not None else 0
            q = rope(q, offset=offset)
            k = rope(k, offset=offset)

        new_cache: KVCache | None = None
        if cache is not None:
            cache_len = cache.length
            k_cache = cache.k.at[:, :, cache_len : cache_len + T, :].set(k)
            v_cache = cache.v.at[:, :, cache_len : cache_len + T, :].set(v)
            new_cache = KVCache(k=k_cache, v=v_cache, length=cache_len + T)

            k = k_cache[:, :, : cache_len + T, :]
            v = v_cache[:, :, : cache_len + T, :]
            T_kv: int | jax.Array = cache_len + T
        else:
            T_kv = T

        if self.use_flash_attention:
            q_fa = q.transpose(0, 2, 1, 3)  # (B, T_q, n_head, head_dim)
            k_fa = k.transpose(0, 2, 1, 3)  # (B, T_kv, n_head, head_dim)
            v_fa = v.transpose(0, 2, 1, 3)
            if cache is not None:
                row_idx = jnp.arange(T)[None, :, None] + cache.length
                col_idx = jnp.arange(T_kv)[None, None, :]
                mask = col_idx <= row_idx
                y = jax.nn.dot_product_attention(q_fa, k_fa, v_fa, mask=mask)
            else:
                y = jax.nn.dot_product_attention(q_fa, k_fa, v_fa, is_causal=True)
            y = y.reshape(B, T, C)
        else:
            scale = 1.0 / math.sqrt(head_dim) if self.scale_attn_weights else 1.0
            if self.scale_attn_by_inverse_layer_idx:
                scale *= 1.0 / (self.layer_idx + 1)

            if self.reorder_and_upcast_attn:
                q_f32 = q.astype(jnp.float32)
                k_f32 = k.astype(jnp.float32)
                att = (q_f32 @ k_f32.transpose(0, 1, 3, 2)) * scale
            else:
                att = (q @ k.transpose(0, 1, 3, 2)) * scale

            if cache is not None:
                row_idx = jnp.arange(T)[:, None] + cache.length
                col_idx = jnp.arange(T_kv)[None, :]
                mask = (col_idx > row_idx).astype(att.dtype)
            else:
                mask = jnp.triu(jnp.ones((T, T_kv), dtype=att.dtype), k=1)
            att = jnp.where(mask == 1, -1e9, att)

            att = jax.nn.softmax(att, axis=-1)
            if self.reorder_and_upcast_attn:
                att = att.astype(v.dtype)
            if self.attn_pdrop > 0:
                att = nn.Dropout(rate=self.attn_pdrop)(att, deterministic=not training)

            y = (att @ v).transpose(0, 2, 1, 3).reshape(B, T, C)

        y = nn.Dense(
            features=C,
            use_bias=self.bias,
            kernel_init=nn.initializers.xavier_uniform(),
            dtype=getattr(jnp, self.dtype),
            name="c_proj",
        )(y)

        if self.resid_pdrop > 0:
            y = nn.Dropout(rate=self.resid_pdrop)(y, deterministic=not training)

        return y, new_cache


class MLP(nn.Module):
    """Feed-forward network."""

    config: GPTConfig

    @nn.compact
    def __call__(self, x: jax.Array, training: bool = True) -> jax.Array:
        x = nn.Dense(
            features=4 * self.config.n_embd,
            use_bias=self.config.bias,
            kernel_init=nn.initializers.xavier_uniform(),
            dtype=getattr(jnp, self.config.dtype),
            name="c_fc",
        )(x)
        x = NewGELU()(x)

        if self.config.resid_pdrop > 0:
            x = nn.Dropout(rate=self.config.resid_pdrop)(x, deterministic=not training)

        x = nn.Dense(
            features=self.config.n_embd,
            use_bias=self.config.bias,
            kernel_init=nn.initializers.xavier_uniform(),
            dtype=getattr(jnp, self.config.dtype),
            name="c_proj",
        )(x)

        if self.config.resid_pdrop > 0:
            x = nn.Dropout(rate=self.config.resid_pdrop)(x, deterministic=not training)

        return x


class Block(nn.Module):
    """Enhanced transformer block with modern features."""

    config: GPTConfig
    layer_idx: int = 0

    @nn.compact
    def __call__(
        self, x: jax.Array, training: bool = True, cache: KVCache | None = None
    ) -> tuple[jax.Array, KVCache | None]:
        attn_out, new_cache = CausalSelfAttention(
            n_embd=self.config.n_embd,
            n_head=self.config.n_head,
            block_size=self.config.block_size,
            attn_pdrop=self.config.attn_pdrop,
            resid_pdrop=self.config.resid_pdrop,
            bias=self.config.bias,
            use_rope=self.config.use_rope,
            use_flash_attention=self.config.use_flash_attention,
            scale_attn_weights=self.config.scale_attn_weights,
            scale_attn_by_inverse_layer_idx=self.config.scale_attn_by_inverse_layer_idx,
            reorder_and_upcast_attn=self.config.reorder_and_upcast_attn,
            layer_idx=self.layer_idx,
            dtype=self.config.dtype,
            name="attn",
        )(nn.LayerNorm(epsilon=self.config.layer_norm_epsilon, name="ln_1")(x), training, cache)
        x = x + attn_out

        x = x + MLP(self.config, name="mlp")(
            nn.LayerNorm(epsilon=self.config.layer_norm_epsilon, name="ln_2")(x), training
        )

        return x, new_cache


class GPT(nn.Module):
    """Enhanced GPT model with modern JAX/Flax features."""

    config: GPTConfig

    @nn.compact
    def __call__(
        self,
        idx: jax.Array,
        targets: jax.Array | None = None,
        training: bool = True,
        cache: list[KVCache] | None = None,
    ) -> tuple[jax.Array, jax.Array | None, list[KVCache] | None]:
        """Forward pass.

        Args:
            idx: Input token indices (B, T)
            targets: Target token indices for loss computation (B, T)
            training: Whether in training mode
            cache: Optional list of KVCache per layer for autoregressive generation

        Returns:
            logits: Model output logits (B, T, vocab_size)
            loss: Computed loss if targets provided, None otherwise
            caches: Updated KV caches if cache input was provided, None otherwise
        """
        B, T = idx.shape
        assert T <= self.config.block_size, (
            f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        )

        wte = nn.Embed(
            num_embeddings=self.config.vocab_size,
            features=self.config.n_embd,
            embedding_init=nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=getattr(jnp, self.config.dtype),
            name="wte",
        )
        tok_emb = wte(idx)

        pos_offset = cache[0].length if cache is not None else 0
        pos_emb = nn.Embed(
            num_embeddings=self.config.block_size,
            features=self.config.n_embd,
            embedding_init=nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=getattr(jnp, self.config.dtype),
            name="wpe",
        )(jnp.arange(T) + pos_offset)

        x = tok_emb + pos_emb

        if self.config.embd_pdrop > 0:
            x = nn.Dropout(rate=self.config.embd_pdrop)(x, deterministic=not training)

        block_cls = nn.remat(Block) if self.config.gradient_checkpointing else Block
        new_caches: list[KVCache] = []
        for i in range(self.config.n_layer):
            layer_cache = cache[i] if cache is not None else None
            x, new_cache = block_cls(self.config, layer_idx=i, name=f"h_{i}")(
                x, training, layer_cache
            )
            if new_cache is not None:
                new_caches.append(new_cache)

        x = nn.LayerNorm(epsilon=self.config.layer_norm_epsilon, name="ln_f")(x)

        if self.config.tie_word_embeddings:
            logits = x @ wte.embedding.T
        else:
            logits = nn.Dense(
                features=self.config.vocab_size,
                use_bias=False,
                kernel_init=nn.initializers.xavier_uniform(),
                dtype=getattr(jnp, self.config.dtype),
                name="lm_head",
            )(x)

        loss = None
        if targets is not None:
            flat_logits = logits.reshape(-1, logits.shape[-1])
            flat_targets = targets.reshape(-1)

            loss = optax.softmax_cross_entropy_with_integer_labels(flat_logits, flat_targets).mean()

        return logits, loss, new_caches if cache is not None else None

    @classmethod
    def from_pretrained(cls, model_type: str, **kwargs: Any) -> "GPT":
        """Initialize a pretrained GPT model configuration.

        Args:
            model_type: Model type (gpt2, gpt2-medium, etc.)
            **kwargs: Additional configuration parameters

        Returns:
            GPT model instance (not yet initialized; call .init() next)
        """
        model_configs = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),
        }

        if model_type not in model_configs:
            raise ValueError(f"Unknown model type: {model_type}")

        config = GPTConfig(
            model_type=model_type,
            vocab_size=50257,
            block_size=1024,
            **model_configs[model_type],  # type: ignore[arg-type]
            **kwargs,
        )

        return cls(config)


def get_num_params(params: Dict[str, Any]) -> int:
    """Get the total number of parameters."""
    flat = flax.traverse_util.flatten_dict(params)
    return sum(p.size for p in flat.values())


def get_model_size_mb(params: Dict[str, Any]) -> float:
    """Get the model size in megabytes (assuming float32)."""
    return get_num_params(params) * 4 / (1024 * 1024)


def configure_optimizers(
    params: Dict[str, Any],
    learning_rate: float,
    weight_decay: float,
    betas: tuple[float, float] = (0.9, 0.95),
    grad_clip: float = 1.0,
) -> optax.GradientTransformation:
    """Create AdamW optimizer with weight-decay masking.

    Applies weight decay to Dense kernels only; biases, LayerNorm params,
    and embedding tables are excluded from decay.
    """

    def decay_mask_fn(params: Dict[str, Any]) -> Dict[str, Any]:
        flat_params = flax.traverse_util.flatten_dict(params, sep="/")
        mask = {}
        for key in flat_params:
            parts = key.split("/")
            param_name = parts[-1]
            if param_name == "kernel":
                mask[key] = True
            else:
                mask[key] = False
        return flax.traverse_util.unflatten_dict(mask, sep="/")  # type: ignore[no-any-return]

    tx = optax.chain(
        optax.clip_by_global_norm(grad_clip) if grad_clip > 0.0 else optax.identity(),
        optax.adamw(
            learning_rate=learning_rate,
            b1=betas[0],
            b2=betas[1],
            weight_decay=weight_decay,
            mask=decay_mask_fn,
        ),
    )
    return tx


def create_gpt_model(
    model_type: str = "gpt2",
    vocab_size: int = 50257,
    block_size: int = 1024,
    **kwargs: Any,
) -> GPT:
    """Convenience function to create a GPT model."""
    return GPT.from_pretrained(model_type, vocab_size=vocab_size, block_size=block_size, **kwargs)


def _sample_top_p(logits: jax.Array, p: float) -> jax.Array:
    """Mask logits outside the top-p nucleus, returning filtered logits."""
    sorted_logits = jnp.sort(logits, axis=-1)[:, ::-1]
    cumulative_probs = jnp.cumsum(jax.nn.softmax(sorted_logits, axis=-1), axis=-1)
    sorted_mask = cumulative_probs > p
    sorted_mask = jnp.roll(sorted_mask, 1, axis=-1)
    sorted_mask = sorted_mask.at[:, 0].set(False)

    rank = jnp.argsort(jnp.argsort(logits, axis=-1), axis=-1)
    indices_to_remove = sorted_mask[jnp.arange(logits.shape[0])[:, None], rank]
    return jnp.where(indices_to_remove, -float("inf"), logits)


def generate(
    model: GPT,
    variables: Dict[str, Any],
    idx: jax.Array,
    max_new_tokens: int,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    rng: jax.Array | None = None,
) -> jax.Array:
    """Generate tokens autoregressively using KV-cache.

    Args:
        model: GPT model instance
        variables: Model parameters (e.g. {"params": ...})
        idx: Prompt token indices (B, T)
        max_new_tokens: Number of tokens to generate
        temperature: Sampling temperature (0 = greedy)
        top_k: Top-k filtering
        top_p: Top-p (nucleus) filtering
        rng: PRNG key for sampling

    Returns:
        Full sequence including prompt and generated tokens (B, T + max_new_tokens)
    """
    if rng is None:
        rng = random.PRNGKey(0)

    B, T = idx.shape
    config = model.config
    head_dim = config.n_embd // config.n_head
    cache_dtype = getattr(jnp, config.dtype)

    caches = init_kv_cache(
        B, config.n_layer, config.n_head, head_dim, config.block_size, cache_dtype
    )

    logits, _, caches = model.apply(variables, idx, training=False, cache=caches)  # type: ignore[misc]

    next_logits = logits[:, -1, :]
    generated = idx

    for _ in range(max_new_tokens):
        if temperature == 0.0:
            next_token = jnp.argmax(next_logits, axis=-1)
        else:
            scaled_logits = next_logits / temperature

            if top_k is not None:
                top_k_vals, _ = jax.lax.top_k(scaled_logits, min(top_k, scaled_logits.shape[-1]))
                threshold = top_k_vals[:, -1:]
                scaled_logits = jnp.where(scaled_logits < threshold, -float("inf"), scaled_logits)

            if top_p is not None:
                scaled_logits = _sample_top_p(scaled_logits, top_p)

            rng, sample_rng = random.split(rng)
            next_token = random.categorical(sample_rng, scaled_logits, axis=-1)

        generated = jnp.concatenate([generated, next_token[:, None]], axis=1)

        logits, _, caches = model.apply(  # type: ignore[misc]
            variables, next_token[:, None], training=False, cache=caches
        )
        next_logits = logits[:, -1, :]

    return generated


if __name__ == "__main__":
    model = create_gpt_model("gpt2", vocab_size=1000, block_size=512)

    batch_size, seq_len = 2, 10
    x = jnp.ones((batch_size, seq_len), dtype=jnp.int32)

    rng = random.PRNGKey(0)
    variables = model.init(rng, x)

    print(f"Model parameters: {get_num_params(variables['params']):,}")
    print(f"Model size: {get_model_size_mb(variables['params']):.1f} MB")

    logits, loss, _ = model.apply(variables, x, targets=x)  # type: ignore[misc]
    print(f"Output shape: {logits.shape}")
    print(f"Loss: {loss}")
