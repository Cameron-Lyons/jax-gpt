"""Modern JAX trainer for GPT-2 with advanced features and optimizations."""

import json
import logging
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from flax.training import train_state
from jax import jit, random, value_and_grad


@dataclass
class TrainingConfig:
    """Configuration for training."""

    model_size: Literal["124M", "355M", "774M", "1558M"] = "124M"
    block_size: int = 1024
    vocab_size: int = 50304

    batch_size: int = 32
    max_iters: int = 100000
    learning_rate: float = 6e-4
    weight_decay: float = 0.1
    warmup_iters: int = 2000
    lr_decay_iters: int = 100000
    min_lr: float = 6e-5
    grad_clip: float = 1.0

    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8

    eval_interval: int = 1000
    eval_iters: int = 200
    save_interval: int = 5000

    seed: int = 42
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    compile: bool = True

    log_interval: int = 10
    use_wandb: bool = False
    project_name: str = "jax-gpt2"
    save_dir: str = "checkpoints"
    max_checkpoints: int = 5

    def __post_init__(self) -> None:
        self.save_path = Path(self.save_dir)
        self.save_path.mkdir(exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MetricsTracker:
    """Track and log training metrics."""

    def __init__(self) -> None:
        self.metrics: Dict[str, List[float]] = defaultdict(list)
        self.start_time = time.time()

    def update(self, **kwargs: float) -> None:
        for key, value in kwargs.items():
            self.metrics[key].append(float(value))

    def get_latest(self, key: str) -> Optional[float]:
        return self.metrics[key][-1] if self.metrics[key] else None

    def get_avg(self, key: str, window: int = 100) -> Optional[float]:
        values = self.metrics[key]
        if not values:
            return None
        recent = values[-window:] if len(values) > window else values
        return sum(recent) / len(recent)

    def log_to_wandb(self, step: int) -> None:
        try:
            import wandb

            if not wandb.run:
                return
        except ImportError:
            return

        log_dict: Dict[str, float] = {}
        for key, values in self.metrics.items():
            if values:
                log_dict[f"train/{key}"] = values[-1]
                if len(values) > 1:
                    avg = self.get_avg(key)
                    if avg is not None:
                        log_dict[f"train/{key}_avg"] = avg

        log_dict["train/iter_per_sec"] = step / (time.time() - self.start_time)
        wandb.log(log_dict, step=step)

    def print_summary(self, step: int) -> None:
        loss = self.get_latest("loss")
        lr = self.get_latest("lr")
        grad_norm = self.get_latest("grad_norm")
        print(f"Step {step:6d} | Loss: {loss:.4f} | LR: {lr:.2e} | Grad Norm: {grad_norm:.4f}")


class DataLoader:
    """Efficient data loader for JAX training."""

    def __init__(self, data: jax.Array, batch_size: int, block_size: int, seed: int = 42) -> None:
        self.data = data
        self.batch_size = batch_size
        self.block_size = block_size
        self.rng = random.PRNGKey(seed)
        self.n_samples = len(data) - block_size

    def get_batch(self, split: Literal["train", "val"] = "train") -> Tuple[jax.Array, jax.Array]:
        self.rng, split_rng = random.split(self.rng)
        split_rng = random.fold_in(split_rng, 0 if split == "train" else 1)

        start_indices = random.randint(split_rng, (self.batch_size,), 0, self.n_samples)

        x = jnp.stack([self.data[i : i + self.block_size] for i in start_indices])
        y = jnp.stack([self.data[i + 1 : i + 1 + self.block_size] for i in start_indices])

        return x, y

    def get_eval_batches(
        self, n_batches: int, split: Literal["train", "val"] = "val"
    ) -> List[Tuple[jax.Array, jax.Array]]:
        batches = []
        for _ in range(n_batches):
            batches.append(self.get_batch(split))
        return batches


class CheckpointManager:
    """Manage model checkpoints."""

    def __init__(self, save_dir: str, max_checkpoints: int = 5) -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.checkpoint_manager = ocp.CheckpointManager(
            self.save_dir,
            options=ocp.CheckpointManagerOptions(max_to_keep=max_checkpoints, create=True),
        )

    def save(self, state: train_state.TrainState, step: int, metrics: Dict[str, Any]) -> None:
        self.checkpoint_manager.save(step, args=ocp.args.StandardSave(state))

        metadata = {"step": step, "metrics": metrics, "timestamp": time.time()}
        metadata_path = self.save_dir / f"metadata_{step}.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        print(f"Saved checkpoint at step {step}")

    def load_latest(
        self,
    ) -> Optional[Tuple[train_state.TrainState, int, Dict[str, Any]]]:
        latest_step = self.checkpoint_manager.latest_step()
        if latest_step is None:
            return None

        state = self.checkpoint_manager.restore(latest_step)

        metadata_path = self.save_dir / f"metadata_{latest_step}.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        else:
            metadata = {"step": latest_step, "metrics": {}}

        return state, latest_step, metadata

    def cleanup_old_metadata(self) -> None:
        for metadata_file in self.save_dir.glob("metadata_*.json"):
            step = int(metadata_file.stem.split("_")[1])
            if step not in self.checkpoint_manager.all_steps():
                metadata_file.unlink()


class ModernTrainer:
    """Modern JAX trainer with advanced features."""

    def __init__(
        self,
        config: TrainingConfig,
        model_fn: Any,
        train_data: jax.Array,
        val_data: jax.Array,
    ) -> None:
        self.config = config
        self.model_fn = model_fn
        self.train_data = train_data
        self.val_data = val_data

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.rng = random.PRNGKey(config.seed)
        self.metrics = MetricsTracker()
        self.checkpoint_manager = CheckpointManager(config.save_dir, config.max_checkpoints)

        self.train_loader = DataLoader(
            train_data, config.batch_size, config.block_size, config.seed
        )
        self.val_loader = DataLoader(
            val_data, config.batch_size, config.block_size, config.seed + 1
        )

        if config.use_wandb:
            import wandb

            wandb.init(project=config.project_name, config=config.to_dict())

        self.lr_schedule = self._create_lr_schedule()
        self.optimizer = self._create_optimizer()
        self.state = self._initialize_training_state()

        if config.compile:
            self.train_step = jit(self._train_step)
            self.eval_step = jit(self._eval_step)
        else:
            self.train_step = self._train_step  # type: ignore[assignment]
            self.eval_step = self._eval_step  # type: ignore[assignment]

    def _create_lr_schedule(self) -> Any:
        def lr_schedule(step: jax.Array) -> jax.Array:
            warmup_lr = self.config.learning_rate * step / self.config.warmup_iters
            decay_ratio = (step - self.config.warmup_iters) / (
                self.config.lr_decay_iters - self.config.warmup_iters
            )
            decay_ratio = jnp.clip(decay_ratio, 0.0, 1.0)
            coeff = 0.5 * (1.0 + jnp.cos(jnp.pi * decay_ratio))
            decay_lr = self.config.min_lr + coeff * (self.config.learning_rate - self.config.min_lr)
            return jnp.where(step < self.config.warmup_iters, warmup_lr, decay_lr)

        return lr_schedule

    def _create_optimizer(self) -> optax.GradientTransformation:
        optimizer = optax.chain(
            optax.clip_by_global_norm(self.config.grad_clip),
            optax.adamw(
                learning_rate=self.lr_schedule,
                b1=self.config.beta1,
                b2=self.config.beta2,
                eps=self.config.eps,
                weight_decay=self.config.weight_decay,
            ),
        )
        return optimizer

    def _initialize_training_state(self) -> train_state.TrainState:
        dummy_x, dummy_y = self.train_loader.get_batch()

        self.rng, init_rng = random.split(self.rng)
        variables = self.model_fn.init(init_rng, dummy_x, training=False)
        params = variables["params"]

        state: train_state.TrainState = train_state.TrainState.create(
            apply_fn=self.model_fn.apply,
            params=params,
            tx=self.optimizer,
        )

        return state

    def _train_step(
        self,
        state: train_state.TrainState,
        batch: Tuple[jax.Array, jax.Array],
        rng: jax.Array,
    ) -> Tuple[train_state.TrainState, Dict[str, jax.Array]]:
        x, y = batch

        def loss_fn(params: dict) -> Tuple[jax.Array, jax.Array]:
            logits, loss = state.apply_fn(
                {"params": params},
                x,
                targets=y,
                training=True,
                rngs={"dropout": rng},
            )
            return loss, logits

        (loss, logits), grads = value_and_grad(loss_fn, has_aux=True)(state.params)

        state = state.apply_gradients(grads=grads)

        grad_norm = optax.global_norm(grads)
        lr = self.lr_schedule(state.step)

        metrics = {
            "loss": loss,
            "grad_norm": grad_norm,
            "lr": lr,
        }

        return state, metrics

    def _eval_step(
        self,
        state: train_state.TrainState,
        batch: Tuple[jax.Array, jax.Array],
    ) -> Dict[str, jax.Array]:
        x, y = batch

        logits, loss = state.apply_fn({"params": state.params}, x, targets=y, training=False)

        return {"loss": loss}

    def evaluate(self) -> Dict[str, float]:
        eval_metrics: Dict[str, list] = defaultdict(list)

        for _ in range(self.config.eval_iters):
            batch = self.val_loader.get_batch("val")
            metrics = self.eval_step(self.state, batch)

            for key, value in metrics.items():
                eval_metrics[key].append(value)

        avg_metrics = {
            key: float(jnp.mean(jnp.array(values))) for key, values in eval_metrics.items()
        }

        return avg_metrics

    def train(self) -> None:
        self.logger.info("Starting training...")

        checkpoint_data = self.checkpoint_manager.load_latest()
        if checkpoint_data:
            self.state, start_step, metadata = checkpoint_data
            self.logger.info(f"Resumed from checkpoint at step {start_step}")
        else:
            start_step = 0
            self.logger.info("Starting from scratch")

        for step in range(start_step, self.config.max_iters):
            self.rng, step_rng = random.split(self.rng)
            batch = self.train_loader.get_batch("train")

            self.state, metrics = self.train_step(self.state, batch, step_rng)

            self.metrics.update(**{k: float(v) for k, v in metrics.items()})

            if step % self.config.log_interval == 0:
                self.metrics.print_summary(step)
                if self.config.use_wandb:
                    self.metrics.log_to_wandb(step)

            if step % self.config.eval_interval == 0 and step > 0:
                eval_metrics = self.evaluate()
                self.logger.info(f"Step {step} | Val Loss: {eval_metrics['loss']:.4f}")

                if self.config.use_wandb:
                    import wandb

                    wandb.log({"val/loss": eval_metrics["loss"]}, step=step)

            if step % self.config.save_interval == 0 and step > 0:
                current_metrics = {
                    key: self.metrics.get_latest(key) for key in ["loss", "grad_norm"]
                }
                self.checkpoint_manager.save(self.state, step, current_metrics)
                self.checkpoint_manager.cleanup_old_metadata()

        self.logger.info("Training completed!")

        final_metrics = {key: self.metrics.get_latest(key) for key in ["loss", "grad_norm"]}
        self.checkpoint_manager.save(self.state, self.config.max_iters, final_metrics)

    def get_model_params(self) -> Any:
        return self.state.params


def create_trainer_from_config(
    config_path: str,
    model_fn: Any,
    train_data: jax.Array,
    val_data: jax.Array,
) -> ModernTrainer:
    """Create a trainer from a configuration file."""
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    config = TrainingConfig(**config_dict)
    return ModernTrainer(config, model_fn, train_data, val_data)
