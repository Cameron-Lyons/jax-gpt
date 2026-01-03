"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (DDP).

To run on a single GPU, example:
$ python train.py config/train_shakespeare_char.py

To run with DDP on 4 gpus on 1 node, example:
$ python train.py config/train_gpt2.py
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import optax
from flax.training import train_state

from model import GPT, GPTConfig

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False  # if True, script exits right after the first eval
always_save_checkpoint = True  # if True, always save a checkpoint after each eval
init_from = 'scratch'  # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False  # disabled by default
wandb_project = 'jax-gpt'
wandb_run_name = 'gpt2'  # 'run' + str(time.time())
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8  # used to simulate larger batch sizes
batch_size = 12  # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0  # for pretraining 0 is good, for finetuning try 0.1+
bias = True  # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4  # max learning rate
max_iters = 600000  # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True  # whether to decay the learning rate
warmup_iters = 2000  # how many steps to warm up for
lr_decay_iters = 600000  # should be ~= max_iters per Chinchilla
min_lr = 6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# system
device = 'gpu'  # 'cpu', 'gpu', or 'tpu'
dtype = 'bfloat16'  # 'float32', 'float16', or 'bfloat16'
compile = True  # use JAX JIT compilation (default True for performance)
seed = 1337
# -----------------------------------------------------------------------------
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read())  # overrides from command line or config file
config = {k: globals()[k] for k in config_keys}  # will be useful for logging
# -----------------------------------------------------------------------------

# set random seeds
master_rng = random.PRNGKey(seed)

# logging
if wandb_log:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# data loading
data_dir = os.path.join('data', dataset)


def get_batch(split, rng):
    """Load data from disk and return a batch."""
    # We recreate np.memmap every batch to avoid a memory leak
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

    ix = random.randint(rng, (batch_size,), 0, len(data) - block_size)
    ix = np.array(ix)  # Convert to numpy for indexing

    x = jnp.stack([jnp.array(data[i:i + block_size].astype(np.int32)) for i in ix])
    y = jnp.stack([jnp.array(data[i + 1:i + 1 + block_size].astype(np.int32)) for i in ix])

    return x, y


# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, embd_pdrop=dropout, resid_pdrop=dropout,
                  attn_pdrop=dropout)  # start with model_args from command line

if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304

    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

    # Initialize model parameters
    master_rng, init_rng = random.split(master_rng)
    dummy_input = jnp.ones((batch_size, block_size), dtype=jnp.int32)
    variables = model.init(init_rng, dummy_input, training=False)
    params = variables['params']

elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint
    ckpt_path = os.path.join(out_dir, 'ckpt.pkl')
    with open(ckpt_path, 'rb') as f:
        checkpoint = pickle.load(f)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    params = checkpoint['params']
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']

elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    from utils import load_encoder_hparams_and_params
    from parameter_converter import convert_functional_to_flax_params

    model_size_map = {
        'gpt2': '124M',
        'gpt2-medium': '355M',
        'gpt2-large': '774M',
        'gpt2-xl': '1558M'
    }
    model_size = model_size_map.get(init_from, '124M')

    encoder, hparams, pretrained_params = load_encoder_hparams_and_params(model_size, 'models')

    # Override model args from pretrained model
    model_args['n_layer'] = hparams['n_layer']
    model_args['n_head'] = hparams['n_head']
    model_args['n_embd'] = hparams['n_embd']
    model_args['block_size'] = hparams['n_ctx']
    model_args['vocab_size'] = hparams['n_vocab']
    model_args['bias'] = True

    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

    # Convert pretrained params to Flax format
    params = convert_functional_to_flax_params(pretrained_params, gptconf)

else:
    raise ValueError(f"Unknown init_from: {init_from}")

print(f"Model config: {gptconf}")

# optimizer
def get_lr(step):
    """Learning rate schedule with warmup and cosine decay."""
    # 1) linear warmup for warmup_iters steps
    if step < warmup_iters:
        return learning_rate * step / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if step > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (step - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


def create_optimizer(learning_rate_fn):
    """Create AdamW optimizer with learning rate schedule."""
    return optax.chain(
        optax.clip_by_global_norm(grad_clip) if grad_clip > 0.0 else optax.identity(),
        optax.adamw(
            learning_rate=learning_rate_fn,
            b1=beta1,
            b2=beta2,
            weight_decay=weight_decay
        )
    )


# Create learning rate schedule
lr_schedule = optax.join_schedules(
    schedules=[
        optax.linear_schedule(0.0, learning_rate, warmup_iters),
        optax.cosine_decay_schedule(learning_rate, lr_decay_iters - warmup_iters, min_lr / learning_rate)
    ],
    boundaries=[warmup_iters]
) if decay_lr else optax.constant_schedule(learning_rate)

optimizer = create_optimizer(lr_schedule)
opt_state = optimizer.init(params)


# Training and evaluation functions
@jax.jit
def train_step(params, opt_state, x, y, rng):
    """Single training step."""
    def loss_fn(params):
        logits, loss = model.apply({'params': params}, x, targets=y, training=True, rngs={'dropout': rng})
        return loss, logits

    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    return params, opt_state, loss


@jax.jit
def eval_step(params, x, y):
    """Single evaluation step."""
    logits, loss = model.apply({'params': params}, x, targets=y, training=False)
    return loss


def estimate_loss(params, rng):
    """Estimate loss on train and validation sets."""
    out = {}
    for split in ['train', 'val']:
        losses = []
        for k in range(eval_iters):
            rng, batch_rng = random.split(rng)
            X, Y = get_batch(split, batch_rng)
            loss = eval_step(params, X, Y)
            losses.append(float(loss))
        out[split] = np.mean(losses)
    return out


# training loop
print(f"Starting training from iteration {iter_num}")
print(f"Training config: batch_size={batch_size}, block_size={block_size}, n_layer={n_layer}, n_head={n_head}, n_embd={n_embd}")
print(f"Learning rate: {learning_rate}, weight_decay={weight_decay}, warmup_iters={warmup_iters}")

X, Y = get_batch('train', master_rng)  # fetch the very first batch
t0 = time.time()
local_iter_num = 0  # number of iterations in the lifetime of this process
running_mfu = -1.0

while True:
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0:
        losses = estimate_loss(params, master_rng)
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
            })

        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'params': params,
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                    'dataset': dataset,
                }
                print(f"saving checkpoint to {out_dir}")
                os.makedirs(out_dir, exist_ok=True)
                with open(os.path.join(out_dir, 'ckpt.pkl'), 'wb') as f:
                    pickle.dump(checkpoint, f)

    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    for micro_step in range(gradient_accumulation_steps):
        master_rng, batch_rng, step_rng = random.split(master_rng, 3)
        X, Y = get_batch('train', batch_rng)
        params, opt_state, loss = train_step(params, opt_state, X, Y, step_rng)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1

    if iter_num % log_interval == 0:
        # get loss as float
        lossf = float(loss) * gradient_accumulation_steps
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms, lr {lr:.2e}")

    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

print("Training complete!")
