"""Training script for GPT-style language models in JAX."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, replace
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import random

from .checkpoints import checkpoint_path, model_config_from_checkpoint
from .config import TrainConfig, parse_train_config
from .data import TokenDataset, get_batch, load_token_dataset
from .model import GPT, GPTConfig, configure_optimizers
from .parameter_converter import convert_functional_to_flax_params
from .utils import (
    DEFAULT_PADDED_VOCAB_SIZE,
    get_gpt2_model_size,
    load_encoder_hparams_and_params,
    load_training_checkpoint,
    resolve_jax_device,
    save_training_checkpoint,
)


@dataclass
class TrainingArtifacts:
    """Mutable state for an in-progress training run."""

    model: GPT
    model_config: GPTConfig
    params: Any
    opt_state: optax.OptState
    iter_num: int
    best_val_loss: float
    rng: jax.Array


def prepare_resume_config(config: TrainConfig) -> tuple[TrainConfig, dict[str, Any] | None]:
    """Hydrate resume config values from checkpoint metadata when available."""
    if config.init_from != "resume":
        return config, None

    checkpoint = load_training_checkpoint(checkpoint_path(config.out_dir))
    updates: dict[str, Any] = {}
    if isinstance(checkpoint.get("dataset"), str):
        updates["dataset"] = checkpoint["dataset"]
    if isinstance(checkpoint.get("data_dir"), str):
        updates["data_dir"] = checkpoint["data_dir"]
    if updates:
        config = replace(config, **updates)
    return config, checkpoint


def create_lr_schedule(config: TrainConfig) -> optax.Schedule:
    """Build the training learning-rate schedule."""
    if not config.decay_lr:
        return optax.constant_schedule(config.learning_rate)

    if config.warmup_iters > 0:
        warmup = optax.linear_schedule(
            init_value=0.0,
            end_value=config.learning_rate,
            transition_steps=max(config.warmup_iters, 1),
        )
        decay = optax.cosine_decay_schedule(
            init_value=config.learning_rate,
            decay_steps=max(config.lr_decay_iters - config.warmup_iters, 1),
            alpha=config.min_lr / config.learning_rate,
        )
        return optax.join_schedules([warmup, decay], boundaries=[config.warmup_iters])

    return optax.cosine_decay_schedule(
        init_value=config.learning_rate,
        decay_steps=max(config.lr_decay_iters, 1),
        alpha=config.min_lr / config.learning_rate,
    )


def build_model_config(
    config: TrainConfig,
    *,
    vocab_size: int,
    n_layer: int | None = None,
    n_head: int | None = None,
    n_embd: int | None = None,
    block_size: int | None = None,
    bias: bool | None = None,
) -> GPTConfig:
    """Build a `GPTConfig` from a training config plus optional overrides."""
    return GPTConfig(
        n_layer=config.n_layer if n_layer is None else n_layer,
        n_head=config.n_head if n_head is None else n_head,
        n_embd=config.n_embd if n_embd is None else n_embd,
        vocab_size=vocab_size,
        block_size=config.block_size if block_size is None else block_size,
        embd_pdrop=config.dropout,
        resid_pdrop=config.dropout,
        attn_pdrop=config.dropout,
        bias=config.bias if bias is None else bias,
        dtype=config.dtype,
    )


def create_step_functions(
    model: GPT,
    optimizer: optax.GradientTransformation,
    *,
    compile_enabled: bool,
) -> tuple[Any, Any, Any]:
    """Create train and eval step functions, optionally JIT compiling them."""

    def loss_and_grads(
        params: Any,
        x: jax.Array,
        y: jax.Array,
        rng: jax.Array,
    ) -> tuple[jax.Array, Any]:
        def loss_fn(model_params: Any) -> tuple[jax.Array, jax.Array]:
            logits, loss, _ = cast(
                tuple[jax.Array, jax.Array, Any],
                model.apply(
                    {"params": model_params},
                    x,
                    targets=y,
                    training=True,
                    rngs={"dropout": rng},
                ),
            )
            return loss, logits

        (loss, _logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        return loss, grads

    def train_step(
        params: Any,
        opt_state: optax.OptState,
        x: jax.Array,
        y: jax.Array,
        rng: jax.Array,
    ) -> tuple[Any, optax.OptState, jax.Array]:
        loss, grads = loss_and_grads(params, x, y, rng)
        updates, next_opt_state = optimizer.update(grads, opt_state, params)
        next_params = optax.apply_updates(params, updates)
        return next_params, next_opt_state, loss

    def accumulate_gradients(
        params: Any,
        x: jax.Array,
        y: jax.Array,
        rng: jax.Array,
    ) -> tuple[jax.Array, Any]:
        return loss_and_grads(params, x, y, rng)

    def eval_step(params: Any, x: jax.Array, y: jax.Array) -> jax.Array:
        _logits, loss, _ = cast(
            tuple[jax.Array, jax.Array, Any],
            model.apply({"params": params}, x, targets=y, training=False),
        )
        return loss

    if not compile_enabled:
        return train_step, accumulate_gradients, eval_step
    return jax.jit(train_step), jax.jit(accumulate_gradients), jax.jit(eval_step)


def estimate_loss(
    *,
    params: Any,
    dataset: TokenDataset,
    batch_size: int,
    block_size: int,
    eval_iters: int,
    eval_step: Any,
    rng: jax.Array,
) -> tuple[dict[str, float], jax.Array]:
    """Estimate train/validation loss using fresh batches each time."""
    losses: dict[str, float] = {}
    for split_name, tokens in (("train", dataset.train_tokens), ("val", dataset.val_tokens)):
        split_losses: list[float] = []
        for _ in range(eval_iters):
            rng, batch_rng = random.split(rng)
            x, y = get_batch(
                tokens,
                batch_size=batch_size,
                block_size=block_size,
                rng=batch_rng,
            )
            split_losses.append(float(eval_step(params, x, y)))
        losses[split_name] = float(np.mean(split_losses))
    return losses, rng


def run_training_step(
    *,
    config: TrainConfig,
    dataset: TokenDataset,
    artifacts: TrainingArtifacts,
    optimizer: optax.GradientTransformation,
    train_step: Any,
    accumulate_gradients: Any,
) -> float:
    """Run one optimizer step, including optional gradient accumulation."""
    if config.gradient_accumulation_steps == 1:
        artifacts.rng, batch_rng, step_rng = random.split(artifacts.rng, 3)
        x, y = get_batch(
            dataset.train_tokens,
            batch_size=config.batch_size,
            block_size=artifacts.model_config.block_size,
            rng=batch_rng,
        )
        artifacts.params, artifacts.opt_state, loss = train_step(
            artifacts.params,
            artifacts.opt_state,
            x,
            y,
            step_rng,
        )
        return float(loss)

    total_loss = 0.0
    accumulated_grads: Any | None = None
    for _ in range(config.gradient_accumulation_steps):
        artifacts.rng, batch_rng, step_rng = random.split(artifacts.rng, 3)
        x, y = get_batch(
            dataset.train_tokens,
            batch_size=config.batch_size,
            block_size=artifacts.model_config.block_size,
            rng=batch_rng,
        )
        micro_loss, micro_grads = accumulate_gradients(artifacts.params, x, y, step_rng)
        total_loss += float(micro_loss)
        if accumulated_grads is None:
            accumulated_grads = micro_grads
        else:
            accumulated_grads = jax.tree_util.tree_map(
                lambda left, right: left + right,
                accumulated_grads,
                micro_grads,
            )

    assert accumulated_grads is not None
    averaged_grads = jax.tree_util.tree_map(
        lambda value: value / config.gradient_accumulation_steps,
        accumulated_grads,
    )
    updates, artifacts.opt_state = optimizer.update(
        averaged_grads,
        artifacts.opt_state,
        artifacts.params,
    )
    artifacts.params = optax.apply_updates(artifacts.params, updates)
    return total_loss / config.gradient_accumulation_steps


def maybe_evaluate_and_checkpoint(
    *,
    config: TrainConfig,
    dataset: TokenDataset,
    artifacts: TrainingArtifacts,
    eval_step: Any,
) -> bool:
    """Run evaluation and persist the current checkpoint when configured."""
    losses, artifacts.rng = estimate_loss(
        params=artifacts.params,
        dataset=dataset,
        batch_size=config.batch_size,
        block_size=artifacts.model_config.block_size,
        eval_iters=config.eval_iters,
        eval_step=eval_step,
        rng=artifacts.rng,
    )
    print(
        f"step {artifacts.iter_num}: "
        f"train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
    )

    if losses["val"] < artifacts.best_val_loss:
        artifacts.best_val_loss = losses["val"]

    should_save = artifacts.iter_num > 0 and (
        losses["val"] <= artifacts.best_val_loss or config.always_save_checkpoint
    )
    if should_save:
        save_checkpoint_state(config, artifacts)
    return should_save


def initialize_training(
    config: TrainConfig,
    dataset: TokenDataset,
    optimizer: optax.GradientTransformation,
    *,
    resume_checkpoint: dict[str, Any] | None = None,
) -> TrainingArtifacts:
    """Initialize a model, params, optimizer state, and RNG."""
    run_rng = random.PRNGKey(config.seed)

    if config.init_from == "resume":
        checkpoint = resume_checkpoint or load_training_checkpoint(checkpoint_path(config.out_dir))
        model_config = model_config_from_checkpoint(checkpoint)
        model = GPT(model_config)
        params = checkpoint["params"]
        opt_state = checkpoint.get("opt_state") or checkpoint.get("optimizer_state")
        if opt_state is None:
            print("Checkpoint has no optimizer state; starting optimizer from scratch.")
            opt_state = optimizer.init(params)
        run_rng = checkpoint.get("rng", run_rng)
        return TrainingArtifacts(
            model=model,
            model_config=model_config,
            params=params,
            opt_state=opt_state,
            iter_num=int(checkpoint.get("iter_num", checkpoint.get("step", 0))),
            best_val_loss=float(checkpoint.get("best_val_loss", float("inf"))),
            rng=run_rng,
        )

    if config.init_from == "scratch":
        vocab_size = dataset.meta.get("vocab_size") if dataset.meta is not None else None
        if vocab_size is None:
            print(
                "No vocab_size found in dataset metadata; "
                "defaulting to GPT-2 padded vocab size 50304."
            )
        model_config = build_model_config(
            config, vocab_size=int(vocab_size or DEFAULT_PADDED_VOCAB_SIZE)
        )
        model = GPT(model_config)
        run_rng, init_rng = random.split(run_rng)
        dummy_input = jnp.ones((config.batch_size, model_config.block_size), dtype=jnp.int32)
        params = model.init(init_rng, dummy_input, training=False)["params"]
        return TrainingArtifacts(
            model=model,
            model_config=model_config,
            params=params,
            opt_state=optimizer.init(params),
            iter_num=0,
            best_val_loss=float("inf"),
            rng=run_rng,
        )

    if config.init_from.startswith("gpt2"):
        model_size = get_gpt2_model_size(config.init_from)
        _encoder, hparams, pretrained_params = load_encoder_hparams_and_params(
            model_size,
            config.models_dir,
        )
        model_config = build_model_config(
            config,
            vocab_size=int(hparams["n_vocab"]),
            n_layer=int(hparams["n_layer"]),
            n_head=int(hparams["n_head"]),
            n_embd=int(hparams["n_embd"]),
            block_size=int(hparams["n_ctx"]),
            bias=True,
        )
        model = GPT(model_config)
        params = convert_functional_to_flax_params(pretrained_params, model_config)
        return TrainingArtifacts(
            model=model,
            model_config=model_config,
            params=params,
            opt_state=optimizer.init(params),
            iter_num=0,
            best_val_loss=float("inf"),
            rng=run_rng,
        )

    raise ValueError(f"Unknown init_from value: {config.init_from}")


def save_checkpoint_state(config: TrainConfig, artifacts: TrainingArtifacts) -> Path:
    """Persist the latest training state to disk."""
    path = checkpoint_path(config.out_dir)
    payload = {
        "version": 2,
        "params": artifacts.params,
        "opt_state": artifacts.opt_state,
        "iter_num": artifacts.iter_num,
        "best_val_loss": artifacts.best_val_loss,
        "model_config": asdict(artifacts.model_config),
        "model_args": asdict(artifacts.model_config),
        "train_config": config.to_dict(),
        "config": config.to_dict(),
        "dataset": config.dataset,
        "data_dir": config.data_dir,
        "rng": artifacts.rng,
    }
    print(f"Saving checkpoint to {path}")
    return save_training_checkpoint(payload, path)


def train(config: TrainConfig) -> dict[str, Any]:
    """Run training and return a small summary payload."""
    if config.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be at least 1")

    config, resume_checkpoint = prepare_resume_config(config)
    dataset = load_token_dataset(config.data_dir, config.dataset)
    device = resolve_jax_device(config.device)

    with jax.default_device(device):
        lr_schedule = create_lr_schedule(config)
        optimizer = configure_optimizers(
            {},
            learning_rate=lr_schedule,
            weight_decay=config.weight_decay,
            betas=(config.beta1, config.beta2),
            grad_clip=config.grad_clip,
        )
        artifacts = initialize_training(
            config,
            dataset,
            optimizer,
            resume_checkpoint=resume_checkpoint,
        )
        train_step, accumulate_gradients, eval_step = create_step_functions(
            artifacts.model,
            optimizer,
            compile_enabled=config.compile,
        )

        print(f"Using device: {device}")
        print(f"Loading dataset from: {dataset.dataset_dir}")
        print(f"Model config: {artifacts.model_config}")
        print(f"Starting training from iteration {artifacts.iter_num}")

        t0 = time.time()
        last_checkpoint_iter = -1
        while artifacts.iter_num < config.max_iters:
            current_lr = float(lr_schedule(artifacts.iter_num))

            if artifacts.iter_num % config.eval_interval == 0:
                if maybe_evaluate_and_checkpoint(
                    config=config,
                    dataset=dataset,
                    artifacts=artifacts,
                    eval_step=eval_step,
                ):
                    last_checkpoint_iter = artifacts.iter_num

                if config.eval_only:
                    break

            loss = run_training_step(
                config=config,
                dataset=dataset,
                artifacts=artifacts,
                optimizer=optimizer,
                train_step=train_step,
                accumulate_gradients=accumulate_gradients,
            )

            t1 = time.time()
            dt = t1 - t0
            t0 = t1

            if artifacts.iter_num % config.log_interval == 0:
                print(
                    f"iter {artifacts.iter_num}: "
                    f"loss {loss:.4f}, time {dt * 1000:.2f}ms, lr {current_lr:.2e}"
                )

            artifacts.iter_num += 1

        if artifacts.iter_num > 0 and last_checkpoint_iter != artifacts.iter_num:
            save_checkpoint_state(config, artifacts)

    print("Training complete!")
    return {
        "checkpoint_path": str(checkpoint_path(config.out_dir)),
        "iter_num": artifacts.iter_num,
        "best_val_loss": artifacts.best_val_loss,
        "model_config": asdict(artifacts.model_config),
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    """CLI entry point for training."""
    return train(parse_train_config(argv))


if __name__ == "__main__":
    main()
