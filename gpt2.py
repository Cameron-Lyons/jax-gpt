"""GPT-2 model. For text generation."""
from typing import List, Dict, Any, Literal
import jax.numpy as jnp
import jax
from utils import load_encoder_hparams_and_params
from dataclasses import dataclass


@dataclass
class GPTConfig:
    """GPT-2 model configuration."""

    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster


def lm_loss(params, inputs, n_head) -> jax.Array:
    """Cross-entropy loss for language models."""
    x, y = inputs[:-1], inputs[1:]
    output = gpt2(x, **params, n_head=n_head)
    loss = jnp.mean(-jnp.log(output[y]))
    return loss


def linear(x: jax.Array, w: jax.Array, b: jax.Array) -> jax.Array:
    """Linear layer. Multiplies input by weight matrix and adds bias.
    For use in attention layers."""
    return x @ w + b


def softmax(x: jax.Array) -> jax.Array:
    """Softmax activation function."""
    exp_x = jnp.exp(x - jnp.max(x, axis=-1, keepdims=True))
    return exp_x / jnp.sum(exp_x, axis=-1, keepdims=True)


def gelu(x: jax.Array) -> jax.Array:
    """Gaussian Error Linear Unit activation function.
    See https://arxiv.org/abs/1606.08415.
    Improves learning over ReLU at near zero values
    """
    return 0.5 * x * (1 + jnp.tanh(jnp.sqrt(2 / jnp.pi) * (x + 0.044715 * x**3)))


def layer_norm(
    x: jax.Array, g: jax.Array, b: jax.Array, eps: float = 1e-5
) -> jax.Array:
    """Layer normalization.
    for consistent range to speed up training."""
    mean = jnp.mean(x, axis=-1, keepdims=True)
    std = jnp.std(x, axis=-1, keepdims=True)
    return g * (x - mean) / (std + eps) + b


def attention(q: jax.Array, k: jax.Array, v: jax.Array, mask: jax.Array) -> jax.Array:
    """Scaled dot-product attention."""
    return softmax(q @ k.T / jnp.sqrt(q.shape[-1]) + mask) @ v


def causal_self_attention(x: jax.Array, c_attn: jax.Array, c_proj: jax.Array):
    """Attention layer with a causal mask to prevent attending to future tokens."""
    x = linear(x, **c_attn)

    q, k, v = jnp.split(x, 3, axis=1)
    mask: jax.Array = (1 - jnp.tri(x.shape[0], dtype=x.dtype)) * -1e10
    x = attention(q, k, v, mask)

    return linear(x, **c_proj)


def multihead_attn(
    x: jax.Array, c_attn: jax.Array, c_proj: jax.Array, n_head: int
) -> jax.Array:
    """Multi-head attention layer.
    Splits input into n_head chunks and runs attention on each chunk."""
    x = linear(x, **c_attn)
    qkv = jnp.split(x, 3, axis=-1)
    qkv_heads = list(map(lambda x: jnp.split(x, n_head, axis=1), qkv))
    causal_mask = (1 - jnp.tri(x.shape[0], dtype=x.dtype)) * -1e10
    out_heads = [attention(q, k, v, causal_mask) for q, k, v in zip(*qkv_heads)]
    x = jnp.hstack(out_heads)
    return linear(x, **c_proj)


def feed_forward_network(x, c_fc, c_proj):
    """Feed-forward network for each position."""
    a = gelu(linear(x, **c_fc))
    x = linear(a, **c_proj)
    return x


def transformer_block(x, mlp, attn, ln_1, ln_2, n_head):
    """Transformer block.
    Consists of a causal self-attention layer and a feed-forward network.
    """
    x = x + multihead_attn(layer_norm(x, **ln_1), **attn, n_head=n_head)
    x = x + feed_forward_network(layer_norm(x, **ln_2), **mlp)

    return x


def gpt2(inputs: List[int], wte: jax.Array, wpe, blocks: Dict, ln_f, n_head: int):
    """GPT-2 model. Consists of an embedding layer and a stack of transformer blocks."""
    x = wte[inputs] + wpe(range(len(inputs)))

    for block in blocks:
        x = transformer_block(x, **block, n_head=n_head)

    x = layer_norm(x, **ln_f)
    return x @ wte.T


def generate(
    inputs: List[int], params: Dict[str, Any], n_head: int, n_tokens_to_generate: int
):
    """Generate text from a prompt."""
    for _ in range(n_tokens_to_generate):
        logits = gpt2(inputs, **params, n_head=n_head)
        next_id = jnp.argmax(logits[-1])
        inputs.append(int(next_id))
    return inputs[len(inputs) - n_tokens_to_generate:]


def main(
    prompt: str,
    n_tokens_to_generate: int = 40,
    model_size: Literal["124M", "355M", "774M", "1558M"] = "124M",
    models_dir: str = "models",
):
    """Generate text from a prompt using GPT-2."""
    encoder, hparams, params = load_encoder_hparams_and_params(model_size, models_dir)
    input_ids = encoder.encode(prompt)
    assert len(input_ids) + n_tokens_to_generate < hparams["n_ctx"]

    output_ids = generate(input_ids, params, hparams["n_head"], n_tokens_to_generate)
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
    args = parser.parse_args()

    print(
        main(
            args.prompt,
            args.n_tokens_to_generate,
            args.model_size,
            args.models_dir,
        )
    )
