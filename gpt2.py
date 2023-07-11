import jax.numpy as jnp
from typing import List, Dict, Any


def generate(
        inputs: List[str],
        params: Dict[Any, Any],
        n_head: int,
        n_tokens_to_generate: int
) -> List[str]:
    from tqdm import tqdm

    for _ in tqdm(range(n_tokens_to_generate), "generating"):
        logits = gpt2(inputs, **params, n_head=n_head)
        next_id = jnp.argmax(logits[-1])
        inputs.append(int(next_id))

    return inputs[len(inputs)-n_tokens_to_generate:]
