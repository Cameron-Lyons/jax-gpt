"""Script to train Jax GPT-2 model"""

import os
import time
import math
import pickle
import jax.numpy as jnp
import jax
import optax
from typing import Literal, Dict
from gpt2 import GPTConfig, gpt2, lm_loss

# I/O
out_dir: str = "out"
eval_interval: int = 2000
log_interval: int = 1
eval_iters: int = 200
eval_only: bool = False  # if True, script exits right after the first eval
always_save_checkpoint: bool = True  # if True, always save a checkpoint after each eval
init_from: Literal["scratch", "resume", "gpt2"] = "scratch"
assert init_from in ["scratch", "resume", "gpt2"]

# data
dataset: str = "openwebtext"
gradient_accumulation_steps: int = 5 * 8  # used to simulate larger batch sizes
batch_size: int = 12  # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size: int = 1024

# model
n_layer: int = 12
n_head: int = 12
n_embd: int = 768
dropout: float = 0.0  # for pretraining 0 is good, for finetuning try 0.1+
bias: bool = False  # do we use bias inside LayerNorm and Linear layers?

# adamw optimizer
learning_rate: float = 6e-4  # max learning rate
max_iters: int = 600000  # total number of training iterations
weight_decay: float = 1e-1
beta1: float = 0.9
beta2: float = 0.95
grad_clip: float = 1.0  # clip gradients at this value, or disable if == 0.0

# learning rate decay settings
decay_lr: bool = True  # whether to decay the learning rate
warmup_iters: int = 2000  # how many steps to warm up for
lr_decay_iters: int = 600000
min_lr: float = 6e-5
# system
device: Literal["cpu", "gpu", "tpu"] = jax.default_backend()
dtype: Literal["bfloat16", "float16", "float"] = "bfloat16"

config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
exec(open("configurator.py").read())  # overrides from command line or config file
config = {k: globals()[k] for k in config_keys}  # will be useful for logging

random_key = jax.random.PRNGKey(0)

data_dir = os.path.join("data", dataset)
train_data: jax.Array = jnp.load(
    os.path.join(data_dir, "train.bin"), dtype=jnp.uint16, mode="r"
)
val_data: jax.Array = jnp.load(
    os.path.join(data_dir, "val.bin"), dtype=jnp.uint16, mode="r"
)


def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = jax.random.randint(random_key, (), 0, data.shape[0])
    x = jnp.stack([data[i : i + block_size].astype(jnp.int64) for i in ix], axis=-1)
    y = jnp.stack(
        [data[i + 1 : i + 1 + block_size].astype(jnp.int64) for i in ix], axis=-1
    )
    return x, y


iter_num: int = 0
best_val_loss: float = 1e9

meta_path: str = os.path.join(data_dir, "meta.pkl")
meta_vocab_size: int = 0
if os.path.exists(meta_path):
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    meta_vocab_size = meta["vocab_size"]
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

model_args = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=block_size,
    bias=bias,
    vocab_size=None,
    dropout=dropout,
)  # start with model_args from command line

if init_from == "scratch":
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print(
            "defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)"
        )
    model_args["vocab_size"] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = gpt2(gptconf)

elif init_from == "resume":
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint["model_args"]
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = gpt2(gptconf)
    state_dict = checkpoint["model"]
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint["iter_num"]
    best_val_loss = checkpoint["best_val_loss"]

else:
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = gpt2.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
        model_args[k] = getattr(model.config, k)

if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args[
        "block_size"
    ] = block_size  # so that the checkpoint will have the right value
model.to(device)

optimizer = optax.adamw(
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    b1=beta1,
    b2=beta2,
    eps=1e-8,
)

checkpoint = None

if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = jax.jit(model)


def estimate_loss() -> Dict[str, float]:
    out = {}
    for split in ["train", "val"]:
        losses = jnp.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    return out


def get_lr(it: int) -> float:
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


X, Y = get_batch("train")
t0 = time.time()
local_iter_num: int = 0
raw_model = model
running_mfu: float = -1.0
state = optimizer.init(model)


def train(texts: list[list[str]], params) -> float:
    for text in texts:
        inputs = tokenizer.encode(text)
        loss = lm_loss(inputs, params)
        gradients = compute_gradients_via_backpropagation(loss, params)
        params = gradient_descent_update_step(gradients, params)
    return params
