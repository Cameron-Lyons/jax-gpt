import fire
import jax.numpy as jnp
from typing import List, Dict, Any


def attention(q: jnp.array, k: jnp.array, v: jnp.array) -> jnp.array:
    return softmax(q @ k.T / jnp.sqrt(q.shape[-1])) @ v


def self_attention(x: jnp.array,
                   w_fc: jnp.array,
                   w_proj: jnp.array):
    x = x @ w_fc
    q, k, v = jnp.split(x, 3, axis=1)
    x = attention(q, k, v)

    return x @ w_proj


def ffn(x, c_fc, c_proj):
    a = gelu(linear(x, **c_fc))
    x = linear(a, **c_proj)
    return x


def transformer_block(x, mlp, attn, ln_1, ln_2, n_head):
    x = x + mha(layer_norm(x, **ln_1), **attn, n_head=n_head)
    x = x + ffn(layer_norm(x, **ln_2), **mlp)

    return x


def gpt2(
        inputs: List[int],
        wte: jnp.array,
        wpe,
        blocks: Dict,
        ln_f,
        n_head: int
):
    x = wte[inputs] + wpe(range(len(inputs)))

    for block in blocks:
        x = transformer_block(x, **block, n_head=n_head)

    x = layer_norm(x, **ln_f)
    return x @ wte.T


def generate(
        inputs: List[int],
        params: Dict[Any, Any],
        n_head: int,
        n_tokens_to_generate: int
) -> List[int]:
    from tqdm import tqdm

    for _ in tqdm(range(n_tokens_to_generate), "generating"):
        logits = gpt2(inputs, **params, n_head=n_head)
        next_id = jnp.argmax(logits[-1])
        inputs.append(int(next_id))

    return inputs[len(inputs)-n_tokens_to_generate:]


def main(prompt: str,
         n_tokens_to_generate: int = 40,
         model_size: str = '124M',
         models_dir: str = 'models'):
    from utils import load_encoder_hparams_and_params

    encoder, hparams, params = load_encoder_hparams_and_params(
        model_size, models_dir)
    input_ids = encoder.encode(prompt)
    assert len(input_ids) + n_tokens_to_generate < hparams['n_ctx']

    output_ids = generate(
        input_ids, params, hparams['n_head'], n_tokens_to_generate)
    output_text = encoder.decode(output_ids)

    return output_text


if __name__ == '__main__'

fire.Fire(main)
