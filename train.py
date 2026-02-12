"""
Training script for GPT-2 models in JAX.

To run on a single GPU:
$ python train.py config/train_shakespeare_char.py

To override config values:
$ python train.py config/train_shakespeare_char.py --batch_size=32
"""

import os
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from jax import random

from model import GPT, GPTConfig

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = "out"
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = "scratch"  # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False
wandb_project = "jax-gpt"
wandb_run_name = "gpt2"
# data
dataset = "openwebtext"
gradient_accumulation_steps = 5 * 8
batch_size = 12
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = True
# adamw optimizer
learning_rate = 6e-4
max_iters = 600000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
# learning rate decay settings
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 600000
min_lr = 6e-5
# system
device = "gpu"
dtype = "bfloat16"
compile = True
seed = 1337
# -----------------------------------------------------------------------------
config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
exec(open("configurator.py").read())  # noqa: S102
config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

master_rng = random.PRNGKey(seed)

if wandb_log:
    import wandb

    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

data_dir = os.path.join("data", dataset)


def get_batch(split: str, rng: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Load a batch from memory-mapped data."""
    if split == "train":
        data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
    else:
        data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")

    ix = random.randint(rng, (batch_size,), 0, len(data) - block_size)
    ix = np.array(ix)  # type: ignore[assignment]

    x = jnp.stack([jnp.array(data[i : i + block_size].astype(np.int32)) for i in ix])
    y = jnp.stack([jnp.array(data[i + 1 : i + 1 + block_size].astype(np.int32)) for i in ix])

    return x, y


# init these up here, can override if init_from='resume'
iter_num = 0
best_val_loss = 1e9

meta_path = os.path.join(data_dir, "meta.pkl")
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)  # noqa: S301
    meta_vocab_size = meta["vocab_size"]
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

model_args: dict = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=block_size,
    bias=bias,
    vocab_size=None,
    embd_pdrop=dropout,
    resid_pdrop=dropout,
    attn_pdrop=dropout,
)

if init_from == "scratch":
    print("Initializing a new model from scratch")
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args["vocab_size"] = meta_vocab_size if meta_vocab_size is not None else 50304

    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

    master_rng, init_rng = random.split(master_rng)
    dummy_input = jnp.ones((batch_size, block_size), dtype=jnp.int32)
    variables = model.init(init_rng, dummy_input, training=False)
    params = variables["params"]

elif init_from == "resume":
    print(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, "ckpt.pkl")
    with open(ckpt_path, "rb") as f:
        checkpoint = pickle.load(f)  # noqa: S301
    checkpoint_model_args = checkpoint["model_args"]
    for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
        model_args[k] = checkpoint_model_args[k]
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    params = checkpoint["params"]
    iter_num = checkpoint["iter_num"]
    best_val_loss = checkpoint["best_val_loss"]

elif init_from.startswith("gpt2"):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    from parameter_converter import convert_functional_to_flax_params
    from utils import load_encoder_hparams_and_params

    model_size_map = {
        "gpt2": "124M",
        "gpt2-medium": "355M",
        "gpt2-large": "774M",
        "gpt2-xl": "1558M",
    }
    model_size = model_size_map.get(init_from, "124M")

    encoder, hparams, pretrained_params = load_encoder_hparams_and_params(model_size, "models")  # type: ignore[arg-type]

    model_args["n_layer"] = hparams["n_layer"]
    model_args["n_head"] = hparams["n_head"]
    model_args["n_embd"] = hparams["n_embd"]
    model_args["block_size"] = hparams["n_ctx"]
    model_args["vocab_size"] = hparams["n_vocab"]
    model_args["bias"] = True

    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

    params = convert_functional_to_flax_params(pretrained_params, gptconf)

else:
    raise ValueError(f"Unknown init_from: {init_from}")

print(f"Model config: {gptconf}")

# Learning rate schedule
lr_schedule = (
    optax.join_schedules(
        schedules=[
            optax.linear_schedule(0.0, learning_rate, warmup_iters),
            optax.cosine_decay_schedule(
                learning_rate, lr_decay_iters - warmup_iters, min_lr / learning_rate
            ),
        ],
        boundaries=[warmup_iters],
    )
    if decay_lr
    else optax.constant_schedule(learning_rate)
)


def create_optimizer(
    learning_rate_fn: optax.Schedule,
) -> optax.GradientTransformation:
    """Create AdamW optimizer with learning rate schedule."""
    return optax.chain(
        optax.clip_by_global_norm(grad_clip) if grad_clip > 0.0 else optax.identity(),
        optax.adamw(
            learning_rate=learning_rate_fn,
            b1=beta1,
            b2=beta2,
            weight_decay=weight_decay,
        ),
    )


optimizer = create_optimizer(lr_schedule)
opt_state = optimizer.init(params)

# Orbax checkpoint manager
os.makedirs(out_dir, exist_ok=True)
ckpt_mgr = ocp.CheckpointManager(
    out_dir,
    options=ocp.CheckpointManagerOptions(max_to_keep=3, create=True),
)


@jax.jit
def train_step(
    params: dict, opt_state: optax.OptState, x: jax.Array, y: jax.Array, rng: jax.Array
) -> tuple[dict, optax.OptState, jax.Array]:
    """Single training step with dropout RNG."""

    def loss_fn(params: dict) -> tuple[jax.Array, jax.Array]:
        logits, loss, _ = model.apply(
            {"params": params}, x, targets=y, training=True, rngs={"dropout": rng}
        )
        return loss, logits  # type: ignore[return-value]

    (loss, _logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    return new_params, new_opt_state, loss


@jax.jit
def accumulate_gradients(
    params: dict, x: jax.Array, y: jax.Array, rng: jax.Array
) -> tuple[jax.Array, dict]:
    """Compute gradients for a single micro-batch (used in gradient accumulation)."""

    def loss_fn(params: dict) -> tuple[jax.Array, jax.Array]:
        logits, loss, _ = model.apply(
            {"params": params}, x, targets=y, training=True, rngs={"dropout": rng}
        )
        return loss, logits  # type: ignore[return-value]

    (loss, _logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    return loss, grads


@jax.jit
def eval_step(params: dict, x: jax.Array, y: jax.Array) -> jax.Array:
    """Single evaluation step."""
    _logits, loss, _ = model.apply({"params": params}, x, targets=y, training=False)
    return loss  # type: ignore[return-value]


def estimate_loss(params: dict, rng: jax.Array) -> dict[str, float]:
    """Estimate loss on train and validation sets."""
    out: dict[str, float] = {}
    for split in ["train", "val"]:
        losses = []
        for _k in range(eval_iters):
            rng, batch_rng = random.split(rng)
            X, Y = get_batch(split, batch_rng)
            loss = eval_step(params, X, Y)
            losses.append(float(loss))
        out[split] = float(np.mean(losses))
    return out


# training loop
print(f"Starting training from iteration {iter_num}")
print(
    f"Training config: batch_size={batch_size}, block_size={block_size}, "
    f"n_layer={n_layer}, n_head={n_head}, n_embd={n_embd}"
)
print(f"Learning rate: {learning_rate}, weight_decay={weight_decay}, warmup_iters={warmup_iters}")

t0 = time.time()
local_iter_num = 0

while True:
    lr = float(lr_schedule(iter_num))

    if iter_num % eval_interval == 0:
        losses = estimate_loss(params, master_rng)
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        if wandb_log:
            wandb.log(
                {
                    "iter": iter_num,
                    "train/loss": losses["train"],
                    "val/loss": losses["val"],
                    "lr": lr,
                }
            )

        if losses["val"] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses["val"]
            if iter_num > 0:
                checkpoint = {
                    "params": params,
                    "model_args": model_args,
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                    "config": config,
                    "dataset": dataset,
                }
                print(f"saving checkpoint to {out_dir}")
                with open(os.path.join(out_dir, "ckpt.pkl"), "wb") as f:  # type: ignore[assignment]
                    pickle.dump(checkpoint, f)

    if iter_num == 0 and eval_only:
        break

    # Gradient accumulation: accumulate grads across micro-batches, then apply once
    if gradient_accumulation_steps == 1:
        master_rng, batch_rng, step_rng = random.split(master_rng, 3)
        X, Y = get_batch("train", batch_rng)
        params, opt_state, loss = train_step(params, opt_state, X, Y, step_rng)
    else:
        total_loss = 0.0
        accumulated_grads = None
        for micro_step in range(gradient_accumulation_steps):
            master_rng, batch_rng, step_rng = random.split(master_rng, 3)
            X, Y = get_batch("train", batch_rng)
            micro_loss, micro_grads = accumulate_gradients(params, X, Y, step_rng)
            total_loss += float(micro_loss)
            if accumulated_grads is None:
                accumulated_grads = micro_grads
            else:
                accumulated_grads = jax.tree.map(lambda a, b: a + b, accumulated_grads, micro_grads)

        accumulated_grads = jax.tree.map(
            lambda g: g / gradient_accumulation_steps, accumulated_grads
        )
        updates, opt_state = optimizer.update(accumulated_grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        loss = total_loss / gradient_accumulation_steps

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1

    if iter_num % log_interval == 0:
        lossf = float(loss)
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms, lr {lr:.2e}")

    iter_num += 1
    local_iter_num += 1

    if iter_num > max_iters:
        break

print("Training complete!")
