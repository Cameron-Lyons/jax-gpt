"""GPT-2 model. For text generation."""

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import jit, random

from utils import load_encoder_hparams_and_params


@dataclass
class GPTConfig:
    """GPT-2 model configuration."""

    block_size: int = 1024
    vocab_size: int = (
        50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    )
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = (
        True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    )


@jit
def lm_loss(params, inputs, n_head) -> jax.Array:
    """Cross-entropy loss for language models."""
    x, y = inputs[:-1], inputs[1:]
    output = gpt2(x, **params, n_head=n_head)
    loss = jnp.mean(-jnp.log(output[y]))
    return loss


@jit
def linear(x: jax.Array, w: jax.Array, b: jax.Array) -> jax.Array:
    """Linear layer. Multiplies input by weight matrix and adds bias.
    For use in attention layers."""
    return x @ w + b


@jit
def softmax(x: jax.Array) -> jax.Array:
    """Softmax activation function."""
    exp_x = jnp.exp(x - jnp.max(x, axis=-1, keepdims=True))
    return exp_x / jnp.sum(exp_x, axis=-1, keepdims=True)


@jit
def gelu(x: jax.Array) -> jax.Array:
    """Gaussian Error Linear Unit activation function.
    See https://arxiv.org/abs/1606.08415.
    Improves learning over ReLU at near zero values
    """
    return 0.5 * x * (1 + jnp.tanh(jnp.sqrt(2 / jnp.pi) * (x + 0.044715 * x**3)))


@jit
def layer_norm(x: jax.Array, g: jax.Array, b: jax.Array, eps: float = 1e-5) -> jax.Array:
    """Layer normalization.
    for consistent range to speed up training."""
    mean = jnp.mean(x, axis=-1, keepdims=True)
    std = jnp.std(x, axis=-1, keepdims=True)
    return g * (x - mean) / (std + eps) + b


@jit
def attention(q: jax.Array, k: jax.Array, v: jax.Array, mask: jax.Array) -> jax.Array:
    """Scaled dot-product attention."""
    return softmax(q @ k.T / jnp.sqrt(q.shape[-1]) + mask) @ v  # type: ignore[no-any-return]


@jit
def causal_self_attention(x: jax.Array, c_attn: Dict, c_proj: Dict):
    """Attention layer with a causal mask to prevent attending to future tokens."""
    x = linear(x, **c_attn)

    q, k, v = jnp.split(x, 3, axis=1)
    mask: jax.Array = (1 - jnp.tri(x.shape[0], dtype=x.dtype)) * -1e10
    x = attention(q, k, v, mask)

    return linear(x, **c_proj)


@jit
def multihead_attn(x: jax.Array, c_attn: Dict, c_proj: Dict, n_head: int) -> jax.Array:
    """Multi-head attention layer.
    Splits input into n_head chunks and runs attention on each chunk."""
    x = linear(x, **c_attn)
    qkv = jnp.split(x, 3, axis=-1)
    qkv_heads = list(map(lambda x: jnp.split(x, n_head, axis=1), qkv))
    causal_mask = (1 - jnp.tri(x.shape[0], dtype=x.dtype)) * -1e10
    out_heads = [attention(q, k, v, causal_mask) for q, k, v in zip(*qkv_heads)]
    x = jnp.hstack(out_heads)
    return linear(x, **c_proj)  # type: ignore[no-any-return]


@jit
def feed_forward_network(x, c_fc, c_proj):
    """Feed-forward network for each position."""
    a = gelu(linear(x, **c_fc))
    x = linear(a, **c_proj)
    return x


@jit
def transformer_block(x, mlp, attn, ln_1, ln_2, n_head):
    """Transformer block.
    Consists of a causal self-attention layer and a feed-forward network.
    """
    x = x + multihead_attn(layer_norm(x, **ln_1), **attn, n_head=n_head)
    x = x + feed_forward_network(layer_norm(x, **ln_2), **mlp)

    return x


@jit
def gpt2(inputs: jax.Array, wte: jax.Array, wpe, blocks: Dict, ln_f, n_head: int):
    """GPT-2 model. Consists of an embedding layer and a stack of transformer blocks."""
    x = wte[inputs] + wpe[jnp.arange(len(inputs))]

    for block in blocks:
        x = transformer_block(x, **block, n_head=n_head)

    x = layer_norm(x, **ln_f)
    return x @ wte.T


def sample_top_k(logits: jax.Array, k: int, rng: jax.Array) -> jax.Array:
    """Sample from top-k logits."""
    top_k_logits, top_k_indices = jax.lax.top_k(logits, k)
    probs = jax.nn.softmax(top_k_logits)
    sampled_index = jax.random.categorical(rng, probs)
    return top_k_indices[sampled_index]


def sample_top_p(logits: jax.Array, p: float, rng: jax.Array) -> jax.Array:
    """Sample from top-p (nucleus) logits."""
    sorted_logits = jnp.sort(logits)[::-1]
    cumulative_probs = jnp.cumsum(jax.nn.softmax(sorted_logits))
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove = jnp.roll(sorted_indices_to_remove, 1)
    sorted_indices_to_remove = sorted_indices_to_remove.at[0].set(False)

    indices_to_remove = sorted_indices_to_remove[jnp.argsort(jnp.argsort(logits))]
    filtered_logits = jnp.where(indices_to_remove, -float("inf"), logits)
    probs = jax.nn.softmax(filtered_logits)
    return jax.random.categorical(rng, probs)


@jit
def generate_step(
    inputs: jax.Array,
    params: Dict[str, Any],
    n_head: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    rng: Optional[jax.Array] = None,
) -> Tuple[jax.Array, jax.Array]:
    """Generate one token from the model."""
    logits = gpt2(inputs, **params, n_head=n_head)
    logits = logits[-1] / temperature

    if top_k is not None:
        next_id = sample_top_k(logits, top_k, rng)  # type: ignore[arg-type]
    elif top_p is not None:
        next_id = sample_top_p(logits, top_p, rng)  # type: ignore[arg-type]
    else:
        next_id = jnp.argmax(logits)

    return next_id, logits


def generate(
    inputs: List[int],
    params: Dict[str, Any],
    n_head: int,
    n_tokens_to_generate: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    rng: Optional[jax.Array] = None,
) -> List[int]:
    """Generate text from a prompt with sampling controls."""
    if rng is None:
        rng = random.PRNGKey(0)

    input_array = jnp.array(inputs)
    generated_tokens = []

    for _ in range(n_tokens_to_generate):
        rng, step_rng = random.split(rng)
        next_id, _ = generate_step(input_array, params, n_head, temperature, top_k, top_p, step_rng)
        next_id = int(next_id)
        generated_tokens.append(next_id)
        input_array = jnp.append(input_array, next_id)

    return generated_tokens


def generate_with_stopping(
    inputs: List[int],
    params: Dict[str, Any],
    n_head: int,
    max_tokens: int,
    stop_tokens: Optional[List[int]] = None,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    rng: Optional[jax.Array] = None,
) -> List[int]:
    """Generate text with stopping conditions."""
    if rng is None:
        rng = random.PRNGKey(0)

    if stop_tokens is None:
        stop_tokens = []

    input_array = jnp.array(inputs)
    generated_tokens = []

    for _ in range(max_tokens):
        rng, step_rng = random.split(rng)
        next_id, _ = generate_step(input_array, params, n_head, temperature, top_k, top_p, step_rng)
        next_id = int(next_id)

        if next_id in stop_tokens:
            break

        generated_tokens.append(next_id)
        input_array = jnp.append(input_array, next_id)

    return generated_tokens


def main(
    prompt: str,
    n_tokens_to_generate: int = 40,
    model_size: Literal["124M", "355M", "774M", "1558M"] = "124M",
    models_dir: str = "models",
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    seed: int = 42,
):
    """Generate text from a prompt using GPT-2."""
    encoder, hparams, params = load_encoder_hparams_and_params(model_size, models_dir)
    input_ids = encoder.encode(prompt)
    assert len(input_ids) + n_tokens_to_generate < hparams["n_ctx"]

    rng = random.PRNGKey(seed)
    output_ids = generate(
        input_ids, params, hparams["n_head"], n_tokens_to_generate, temperature, top_k, top_p, rng
    )
    output_text = encoder.decode(output_ids)

    return output_text


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Prompt for the model to complete.",
    )
    parser.add_argument(
        "--n_tokens_to_generate",
        type=int,
        default=40,
        help="Number of tokens to generate.",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="124M",
        help="Size of the GPT-2 model to use.",
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        default="models",
        help="Directory where the GPT-2 model files are stored.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for sampling (higher = more random).",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Top-k sampling parameter.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=None,
        help="Top-p (nucleus) sampling parameter.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for generation.",
    )
    args = parser.parse_args()

    print(
        main(
            args.prompt,
            args.n_tokens_to_generate,
            args.model_size,
            args.models_dir,
            args.temperature,
            args.top_k,
            args.top_p,
            args.seed,
        )
    )
