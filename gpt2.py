import fire
import jax.numpy as jnp
from typing import List, Dict, Any


def lm_loss(params, inputs, n_head) -> float:
    x, y = inputs[:-1], inputs[1:]
    output = gpt2(x, **params, n_head=n_head)
    loss = jnp.mean(-jnp.log(output[y]))
    return loss


def linear(x: jnp.array, w: jnp.array, b: jnp.array) -> jnp.array:
    return x @ w + b


def softmax(x: jnp.array) -> jnp.array:
    exp_x = jnp.exp(x - jnp.max(x, axis=-1, keepdims=True))
    return exp_x / jnp.sum(exp_x, axis=-1, keepdims=True)


def gelu(x: jnp.array) -> jnp.array:
    return 0.5 * x * (1 + jnp.tanh(jnp.sqrt(2 / jnp.pi) * (x + 0.044715 * x**3)))


def attention(q: jnp.array, k: jnp.array, v: jnp.array, mask: jnp.array) -> jnp.array:
    return softmax(q @ k.T / jnp.sqrt(q.shape[-1]) + mask) @ v


def causal_self_attention(x: jnp.array, c_attn: jnp.array, c_proj: jnp.array):
    x = linear(x, **c_attn)

    q, k, v = jnp.split(x, 3, axis=1)

    x = attention(q, k, v)

    return linear(x, **c_proj)


def multihead_attn(
    x: jnp.array, c_attn: jnp.array, c_proj: jnp.array, n_head: int
) -> jnp.array:
    x = linear(x, **c_attn)
    qkv = jnp.split(x, 3, axis=-1)
    qkv_heads = list(map(lambda x: jnp.split(x, n_head, axis=1), qkv))
    causal_mask = (1 - jnp.tri(x.shape[0], dtype=x.dtype)) * -1e10
    out_heads = [attention(q, k, v, causal_mask) for q, k, v in zip(*qkv_heads)]
    x = jnp.hstack(out_heads)
    return linear(x, **c_proj)


def ffn(x, c_fc, c_proj):
    a = gelu(linear(x, **c_fc))
    x = linear(a, **c_proj)
    return x


def transformer_block(x, mlp, attn, ln_1, ln_2, n_head):
    x = x + mha(layer_norm(x, **ln_1), **attn, n_head=n_head)
    x = x + ffn(layer_norm(x, **ln_2), **mlp)

    return x


def gpt2(inputs: List[int], wte: jnp.array, wpe, blocks: Dict, ln_f, n_head: int):
    x = wte[inputs] + wpe(range(len(inputs)))

    for block in blocks:
        x = transformer_block(x, **block, n_head=n_head)

    x = layer_norm(x, **ln_f)
    return x @ wte.T


def generate(inputs: List[int], eos_id: int, max_seq_len: int) -> List[int]:
    promt_len = len(inputs)
    while inputs[-1] != eos_id and len(inputs) < max_seq_len:
        output = gpt2(inputs)
        next_id = jnp.argmax(output[-1])
        inputs.append(int(next_id))
    return inputs[promt_len:]


def main(
    prompt: str,
    n_tokens_to_generate: int = 40,
    model_size: str = "124M",
    models_dir: str = "models",
):
    from utils import load_encoder_hparams_and_params

    encoder, hparams, params = load_encoder_hparams_and_params(model_size, models_dir)
    input_ids = encoder.encode(prompt)
    assert len(input_ids) + n_tokens_to_generate < hparams["n_ctx"]

    output_ids = generate(input_ids, params, hparams["n_head"], n_tokens_to_generate)
    output_text = encoder.decode(output_ids)

    return output_text


if __name__ == "__main__":
    fire.Fire(main)
