"""
Sample from a trained model.
"""

import os
import pickle

import jax
import jax.numpy as jnp
from jax import random

from model import GPT, GPTConfig

# -----------------------------------------------------------------------------
init_from = "resume"  # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = "out"  # ignored if init_from is not 'resume'
start = "\n"  # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 10  # number of samples to draw
max_new_tokens = 500  # number of tokens generated in each sample
temperature = 0.8  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200  # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = "gpu"  # 'cpu', 'gpu', or 'tpu'
dtype = "bfloat16"  # 'float32', 'bfloat16', or 'float16'
# -----------------------------------------------------------------------------

exec(open("configurator.py").read())  # overrides from command line or config file

# -----------------------------------------------------------------------------

# Set random seed
rng = random.PRNGKey(seed)

# model
if init_from == "resume":
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, "ckpt.pkl")
    with open(ckpt_path, "rb") as f:
        checkpoint = pickle.load(f)

    # Load model config from checkpoint
    model_config = checkpoint.get("config", None)
    if model_config is None:
        # Fallback to default config
        model_config = GPTConfig(
            n_layer=12, n_head=12, n_embd=768, block_size=1024, vocab_size=50304
        )

    model = GPT(model_config)
    params = checkpoint["params"]

    # Check for meta.pkl in the data directory to get the encoder
    meta_path = os.path.join("data", checkpoint.get("dataset", "openwebtext"), "meta.pkl")
    load_meta = os.path.exists(meta_path)

elif init_from.startswith("gpt2"):
    # init from a given GPT-2 model
    from utils import load_encoder_hparams_and_params

    model_size_map = {
        "gpt2": "124M",
        "gpt2-medium": "355M",
        "gpt2-large": "774M",
        "gpt2-xl": "1558M",
    }
    model_size = model_size_map.get(init_from, "124M")

    encoder, hparams, params = load_encoder_hparams_and_params(model_size, "models")  # type: ignore[arg-type]

    model_config = GPTConfig(
        n_layer=hparams["n_layer"],
        n_head=hparams["n_head"],
        n_embd=hparams["n_embd"],
        block_size=hparams["n_ctx"],
        vocab_size=hparams["n_vocab"],
    )
    model = GPT(model_config)
    load_meta = False
else:
    raise ValueError(f"Unknown init_from: {init_from}")

# look for the meta pickle in case it is available in the dataset folder
if init_from == "resume":
    if load_meta:
        print(f"Loading meta from {meta_path}...")
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        stoi, itos = meta["stoi"], meta["itos"]

        def encode(s):
            return [stoi[c] for c in s]

        def decode(tokens):
            return "".join([itos[i] for i in tokens])
    else:
        # Use tiktoken for GPT-2 BPE encoding
        print("No meta.pkl found, using tiktoken GPT-2 encoding...")
        import tiktoken

        enc = tiktoken.get_encoding("gpt2")

        def encode(s):
            return enc.encode(s, allowed_special={"<|endoftext|>"})

        def decode(tokens):
            return enc.decode(tokens)
else:
    # For pretrained models, use the encoder we loaded
    encode = encoder.encode
    decode = encoder.decode

# encode the beginning of the prompt
if start.startswith("FILE:"):
    with open(start[5:], "r", encoding="utf-8") as f:  # type: ignore[assignment]
        start = f.read()  # type: ignore[assignment]
start_ids = encode(start)
x = jnp.array([start_ids])

# Initialize model with dummy input to get the apply function ready
rng, init_rng = random.split(rng)
if init_from == "resume":
    # For resumed training, params are already loaded
    variables = {"params": params}
else:
    # For pretrained models, convert params format
    from parameter_converter import convert_functional_to_flax_params

    flax_params = convert_functional_to_flax_params(params, model_config)
    variables = {"params": flax_params}

# run generation
print(f"Generating {num_samples} samples...")
print("=" * 50)

for k in range(num_samples):
    rng, sample_rng = random.split(rng)

    # Generate tokens
    generated = x
    for _ in range(max_new_tokens):
        # Crop to block size if needed
        idx_cond = (
            generated
            if generated.shape[1] <= model_config.block_size
            else generated[:, -model_config.block_size :]
        )

        # Forward pass
        logits, _ = model.apply(variables, idx_cond, training=False)  # type: ignore[misc, assignment]

        # Get logits for last position and apply temperature
        logits = logits[:, -1, :] / temperature

        # Apply top-k filtering
        if top_k is not None:
            top_k_logits, top_k_indices = jax.lax.top_k(logits, min(top_k, logits.shape[-1]))
            # Zero out all logits not in top_k
            mask = jnp.zeros_like(logits).at[0, top_k_indices[0]].set(1)
            logits = jnp.where(mask == 1, logits, -float("inf"))

        # Sample from the distribution
        probs = jax.nn.softmax(logits, axis=-1)
        sample_rng, cat_rng = random.split(sample_rng)
        next_token = random.categorical(cat_rng, logits, axis=-1)

        # Append to sequence
        generated = jnp.concatenate([generated, next_token[:, None]], axis=1)

    # Decode and print
    tokens = generated[0].tolist()
    text = decode(tokens)
    print(text)
    print("-" * 50)
