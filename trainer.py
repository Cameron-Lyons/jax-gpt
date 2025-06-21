"""
Modern JAX trainer for GPT-2 with advanced features and optimizations.
"""

import time
import json
import math
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Literal
from dataclasses import dataclass, asdict
from collections import defaultdict
import logging

import jax
import jax.numpy as jnp
import optax
from jax import jit, value_and_grad, random
import orbax.checkpoint as ocp
from flax.training import train_state
from flax.training import orbax_utils as orbax_utils
import wandb


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

    def __post_init__(self):
        self.save_dir = Path(self.save_dir)
        self.save_dir.mkdir(exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MetricsTracker:
    """Track and log training metrics."""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.start_time = time.time()
    
    def update(self, **kwargs):
        """Update metrics with new values."""
        for key, value in kwargs.items():
            self.metrics[key].append(value)
    
    def get_latest(self, key: str) -> Optional[float]:
        """Get the latest value for a metric."""
        return self.metrics[key][-1] if self.metrics[key] else None
    
    def get_avg(self, key: str, window: int = 100) -> Optional[float]:
        """Get average of recent values for a metric."""
        values = self.metrics[key]
        if not values:
            return None
        recent = values[-window:] if len(values) > window else values
        return sum(recent) / len(recent)
    
    def log_to_wandb(self, step: int):
        """Log metrics to wandb."""
        if not wandb.run:
            return
        
        log_dict = {}
        for key, values in self.metrics.items():
            if values:
                log_dict[f"train/{key}"] = values[-1]
                if len(values) > 1:
                    log_dict[f"train/{key}_avg"] = self.get_avg(key)
        
        log_dict["train/iter_per_sec"] = step / (time.time() - self.start_time)
        wandb.log(log_dict, step=step)
    
    def print_summary(self, step: int):
        """Print a summary of current metrics."""
        loss = self.get_latest("loss")
        lr = self.get_latest("lr")
        grad_norm = self.get_latest("grad_norm")
        
        print(f"Step {step:6d} | Loss: {loss:.4f} | LR: {lr:.2e} | Grad Norm: {grad_norm:.4f}")


class DataLoader:
    """Efficient data loader for JAX training."""
    
    def __init__(self, data: jax.Array, batch_size: int, block_size: int, seed: int = 42):
        self.data = data
        self.batch_size = batch_size
        self.block_size = block_size
        self.rng = random.PRNGKey(seed)
        self.n_samples = len(data) - block_size
    
    def get_batch(self, split: Literal["train", "val"] = "train") -> Tuple[jax.Array, jax.Array]:
        """Get a batch of data."""
        split_rng = random.fold_in(self.rng, 0 if split == "train" else 1)
        
        start_indices = random.randint(
            split_rng, 
            (self.batch_size,), 
            0, 
            self.n_samples
        )
        
        x = jnp.stack([self.data[i:i + self.block_size] for i in start_indices])
        y = jnp.stack([self.data[i + 1:i + 1 + self.block_size] for i in start_indices])
        
        return x, y
    
    def get_eval_batches(self, n_batches: int, split: Literal["train", "val"] = "val") -> List[Tuple[jax.Array, jax.Array]]:
        """Get multiple batches for evaluation."""
        batches = []
        for _ in range(n_batches):
            batches.append(self.get_batch(split))
        return batches


class CheckpointManager:
    """Manage model checkpoints."""
    
    def __init__(self, save_dir: str, max_checkpoints: int = 5):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.checkpoint_manager = ocp.CheckpointManager(
            self.save_dir,
            options=ocp.CheckpointManagerOptions(
                max_to_keep=max_checkpoints,
                create=True
            )
        )
    
    def save(self, state: train_state.TrainState, step: int, metrics: Dict[str, Any]):
        """Save a checkpoint."""
        save_args = orbax_utils.save_args_from_target(state)
        self.checkpoint_manager.save(step, state, save_kwargs={"save_args": save_args})
        
        metadata = {
            "step": step,
            "metrics": metrics,
            "timestamp": time.time()
        }
        metadata_path = self.save_dir / f"metadata_{step}.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved checkpoint at step {step}")
    
    def load_latest(self) -> Optional[Tuple[train_state.TrainState, int, Dict[str, Any]]]:
        """Load the latest checkpoint."""
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
    
    def cleanup_old_metadata(self):
        """Remove old metadata files."""
        for metadata_file in self.save_dir.glob("metadata_*.json"):
            step = int(metadata_file.stem.split("_")[1])
            if step not in self.checkpoint_manager.all_steps():
                metadata_file.unlink()


class ModernTrainer:
    """Modern JAX trainer with advanced features."""
    
    def __init__(self, config: TrainingConfig, model_fn, train_data: jax.Array, val_data: jax.Array):
        self.config = config
        self.model_fn = model_fn
        self.train_data = train_data
        self.val_data = val_data
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.rng = random.PRNGKey(config.seed)
        self.metrics = MetricsTracker()
        self.checkpoint_manager = CheckpointManager(config.save_dir, config.max_checkpoints)
        
        self.train_loader = DataLoader(train_data, config.batch_size, config.block_size, config.seed)
        self.val_loader = DataLoader(val_data, config.batch_size, config.block_size, config.seed + 1)
        
        if config.use_wandb:
            wandb.init(project=config.project_name, config=config.to_dict())
        
        self.optimizer = self._create_optimizer()
        
        self.state = self._initialize_training_state()

        if config.compile:
            self.train_step = jit(self._train_step)
            self.eval_step = jit(self._eval_step)
        else:
            self.train_step = self._train_step
            self.eval_step = self._eval_step
    
    def _create_optimizer(self) -> optax.GradientTransformation:
        """Create the optimizer with learning rate schedule."""
        def lr_schedule(step):
            if step < self.config.warmup_iters:
                return self.config.learning_rate * step / self.config.warmup_iters
            elif step > self.config.lr_decay_iters:
                return self.config.min_lr
            else:
                decay_ratio = (step - self.config.warmup_iters) / (self.config.lr_decay_iters - self.config.warmup_iters)
                coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
                return self.config.min_lr + coeff * (self.config.learning_rate - self.config.min_lr)
        
        optimizer = optax.chain(
            optax.clip_by_global_norm(self.config.grad_clip),
            optax.adamw(
                learning_rate=lr_schedule,
                b1=self.config.beta1,
                b2=self.config.beta2,
                eps=self.config.eps,
                weight_decay=self.config.weight_decay
            )
        )
        
        return optimizer
    
    def _initialize_training_state(self) -> train_state.TrainState:
        """Initialize the training state."""
        dummy_x, dummy_y = self.train_loader.get_batch()
        
        self.rng, init_rng = random.split(self.rng)
        variables = self.model_fn.init(init_rng, dummy_x, training=True)
        params = variables["params"]
        
        state = train_state.TrainState.create(
            apply_fn=self.model_fn.apply,
            params=params,
            tx=self.optimizer
        )
        
        return state
    
    def _train_step(self, state: train_state.TrainState, batch: Tuple[jax.Array, jax.Array]) -> Tuple[train_state.TrainState, Dict[str, jax.Array]]:
        """Single training step."""
        x, y = batch
        
        def loss_fn(params):
            logits = state.apply_fn({"params": params}, x, training=True)
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
            return loss, logits
        
        (loss, logits), grads = value_and_grad(loss_fn, has_aux=True)(state.params)
        
        state = state.apply_gradients(grads=grads)
        
        grad_norm = optax.global_norm(grads)
        accuracy = (logits.argmax(axis=-1) == y).mean()
        
        metrics = {
            "loss": loss,
            "accuracy": accuracy,
            "grad_norm": grad_norm,
            "lr": self.optimizer.learning_rate(state.step)
        }
        
        return state, metrics
    
    def _eval_step(self, state: train_state.TrainState, batch: Tuple[jax.Array, jax.Array]) -> Dict[str, jax.Array]:
        """Single evaluation step."""
        x, y = batch
        
        logits = state.apply_fn({"params": state.params}, x, training=False)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
        accuracy = (logits.argmax(axis=-1) == y).mean()
        
        return {
            "loss": loss,
            "accuracy": accuracy
        }
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model on validation set."""
        self.state = self.state.replace(training=False)
        
        eval_metrics = defaultdict(list)
        
        for _ in range(self.config.eval_iters):
            batch = self.val_loader.get_batch("val")
            metrics = self.eval_step(self.state, batch)
            
            for key, value in metrics.items():
                eval_metrics[key].append(value)
        
        avg_metrics = {key: jnp.mean(values) for key, values in eval_metrics.items()}
        
        self.state = self.state.replace(training=True)
        return avg_metrics
    
    def train(self):
        """Main training loop."""
        self.logger.info("Starting training...")
        
        checkpoint_data = self.checkpoint_manager.load_latest()
        if checkpoint_data:
            self.state, start_step, metadata = checkpoint_data
            self.logger.info(f"Resumed from checkpoint at step {start_step}")
        else:
            start_step = 0
            self.logger.info("Starting from scratch")
        
        for step in range(start_step, self.config.max_iters):
            batch = self.train_loader.get_batch("train")
            
            self.state, metrics = self.train_step(self.state, batch)
            
            self.metrics.update(**metrics)
            
            if step % self.config.log_interval == 0:
                self.metrics.print_summary(step)
                if self.config.use_wandb:
                    self.metrics.log_to_wandb(step)

            if step % self.config.eval_interval == 0 and step > 0:
                eval_metrics = self.evaluate()
                self.logger.info(f"Step {step} | Val Loss: {eval_metrics['loss']:.4f} | Val Acc: {eval_metrics['accuracy']:.4f}")
                
                if self.config.use_wandb:
                    wandb.log({
                        "val/loss": eval_metrics["loss"],
                        "val/accuracy": eval_metrics["accuracy"]
                    }, step=step)
            
            if step % self.config.save_interval == 0 and step > 0:
                current_metrics = {
                    key: self.metrics.get_latest(key) 
                    for key in ["loss", "accuracy", "grad_norm"]
                }
                self.checkpoint_manager.save(self.state, step, current_metrics)
                self.checkpoint_manager.cleanup_old_metadata()
        
        self.logger.info("Training completed!")
        
        final_metrics = {
            key: self.metrics.get_latest(key) 
            for key in ["loss", "accuracy", "grad_norm"]
        }
        self.checkpoint_manager.save(self.state, self.config.max_iters, final_metrics)
    
    def get_model_params(self):
        """Get the current model parameters."""
        return self.state.params


def create_trainer_from_config(config_path: str, model_fn, train_data: jax.Array, val_data: jax.Array) -> ModernTrainer:
    """Create a trainer from a configuration file."""
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    
    config = TrainingConfig(**config_dict)
    return ModernTrainer(config, model_fn, train_data, val_data)
