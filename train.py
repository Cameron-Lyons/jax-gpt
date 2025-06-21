"""
Modern training script for JAX GPT-2 using the improved trainer.
"""

import pickle
import argparse
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
from jax import random

from trainer import ModernTrainer, TrainingConfig
from flax_gpt2 import create_gpt2_model, get_model_config, print_model_summary
from utils import load_encoder_hparams_and_params
from parameter_converter import initialize_model_with_pretrained_weights


def load_training_data(data_dir: str) -> tuple[jax.Array, jax.Array]:
    """Load training and validation data."""
    data_path = Path(data_dir)
    
    train_data = jnp.load(str(data_path / "train.bin"), dtype=jnp.uint16, mode="r")
    val_data = jnp.load(str(data_path / "val.bin"), dtype=jnp.uint16, mode="r")
    
    print(f"Loaded training data: {len(train_data):,} tokens")
    print(f"Loaded validation data: {len(val_data):,} tokens")
    
    return train_data, val_data


def get_vocab_size(data_dir: str) -> int:
    """Get vocabulary size from metadata."""
    meta_path = Path(data_dir) / "meta.pkl"
    if meta_path.exists():
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        vocab_size = meta["vocab_size"]
        print(f"Found vocab_size = {vocab_size} in {meta_path}")
        return vocab_size
    else:
        print("No meta.pkl found, using default vocab_size = 50304")
        return 50304


def create_training_config(
    model_size: Literal["124M", "355M", "774M", "1558M"] = "124M",
    batch_size: int = 32,
    max_iters: int = 100000,
    learning_rate: float = 6e-4,
    weight_decay: float = 0.1,
    warmup_iters: int = 2000,
    eval_interval: int = 2000,
    save_interval: int = 5000,
    use_wandb: bool = False,
    out_dir: str = "out"
) -> TrainingConfig:
    """Create training configuration."""
    
    return TrainingConfig(
        model_size=model_size,
        block_size=1024,
        vocab_size=50304,  # Will be updated based on data
        batch_size=batch_size,
        max_iters=max_iters,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_iters=warmup_iters,
        lr_decay_iters=max_iters,
        min_lr=learning_rate * 0.1,
        grad_clip=1.0,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        eval_interval=eval_interval,
        eval_iters=200,
        save_interval=save_interval,
        seed=42,
        dtype="bfloat16",
        compile=True,
        log_interval=10,
        use_wandb=use_wandb,
        project_name="jax-gpt2",
        save_dir=out_dir,
        max_checkpoints=5
    )


def train_from_scratch(
    config: TrainingConfig,
    train_data: jax.Array,
    val_data: jax.Array,
    vocab_size: int
):
    """Train a model from scratch."""
    print("🚀 Training GPT-2 from scratch")
    print("=" * 50)
    
    config.vocab_size = vocab_size
    
    model_config = get_model_config(config.model_size)
    model_config.vocab_size = vocab_size
    model = create_gpt2_model(model_config)
    
    print_model_summary(model, model_config)
    
    trainer = ModernTrainer(config, model, train_data, val_data)
    
    trainer.train()
    
    return trainer.get_model_params()


def train_from_pretrained(
    config: TrainingConfig,
    train_data: jax.Array,
    val_data: jax.Array,
    vocab_size: int
):
    """Fine-tune a pretrained model."""
    print("🔄 Fine-tuning pretrained GPT-2")
    print("=" * 50)
    
    config.vocab_size = vocab_size
    config.learning_rate = 1e-5
    config.warmup_iters = 100
    
    print("Loading pretrained GPT-2 weights...")
    encoder, hparams, pretrained_params = load_encoder_hparams_and_params(config.model_size, "models")
    
    model_config = get_model_config(config.model_size)
    model_config.vocab_size = vocab_size
    model = create_gpt2_model(model_config)
    
    print_model_summary(model, model_config)
    
    dummy_input = jnp.ones((config.batch_size, config.block_size), dtype=jnp.int32)
    
    print("Converting pretrained parameters to Flax format...")
    pretrained_state = initialize_model_with_pretrained_weights(
        model, pretrained_params, dummy_input
    )
    
    trainer = ModernTrainer(config, model, train_data, val_data)
    
    trainer.state = pretrained_state.replace(tx=trainer.optimizer)
    
    print("Starting fine-tuning with pretrained weights...")
    trainer.train()
    
    return trainer.get_model_params()


def resume_training(
    config: TrainingConfig,
    train_data: jax.Array,
    val_data: jax.Array,
    vocab_size: int
):
    """Resume training from a checkpoint."""
    print("📂 Resuming training from checkpoint")
    print("=" * 50)
    
    config.vocab_size = vocab_size
    
    model_config = get_model_config(config.model_size)
    model_config.vocab_size = vocab_size
    model = create_gpt2_model(model_config)
    
    print_model_summary(model, model_config)
    
    trainer = ModernTrainer(config, model, train_data, val_data)
    
    trainer.train()
    
    return trainer.get_model_params()


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train JAX GPT-2 model")
    
    parser.add_argument("--data_dir", type=str, default="data/openwebtext", 
                       help="Directory containing training data")
    parser.add_argument("--dataset", type=str, default="openwebtext",
                       help="Dataset name")

    parser.add_argument("--model_size", type=str, default="124M",
                       choices=["124M", "355M", "774M", "1558M"],
                       help="Model size to train")
    parser.add_argument("--init_from", type=str, default="scratch",
                       choices=["scratch", "resume", "gpt2"],
                       help="Initialization method")
    
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for training")
    parser.add_argument("--max_iters", type=int, default=100000,
                       help="Maximum training iterations")
    parser.add_argument("--learning_rate", type=float, default=6e-4,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.1,
                       help="Weight decay")
    parser.add_argument("--warmup_iters", type=int, default=2000,
                       help="Warmup iterations")
    
    parser.add_argument("--eval_interval", type=int, default=2000,
                       help="Evaluation interval")
    parser.add_argument("--save_interval", type=int, default=5000,
                       help="Checkpoint save interval")
    
    parser.add_argument("--out_dir", type=str, default="out",
                       help="Output directory")
    parser.add_argument("--use_wandb", action="store_true",
                       help="Use Weights & Biases for logging")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    random.PRNGKey(args.seed)
    
    print("🎯 JAX GPT-2 Training")
    print("=" * 60)
    print(f"Model size: {args.model_size}")
    print(f"Initialization: {args.init_from}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max iterations: {args.max_iters}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Output directory: {args.out_dir}")
    print()
    
    print("📊 Loading training data...")
    try:
        train_data, val_data = load_training_data(args.data_dir)
        vocab_size = get_vocab_size(args.data_dir)
    except FileNotFoundError:
        print(f"❌ Data not found in {args.data_dir}")
        print("Please ensure the data directory contains train.bin, val.bin, and meta.pkl")
        return
    
    config = create_training_config(
        model_size=args.model_size,
        batch_size=args.batch_size,
        max_iters=args.max_iters,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_iters=args.warmup_iters,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        use_wandb=args.use_wandb,
        out_dir=args.out_dir
    )
    
    try:
        if args.init_from == "scratch":
            final_params = train_from_scratch(config, train_data, val_data, vocab_size)
        elif args.init_from == "resume":
            final_params = resume_training(config, train_data, val_data, vocab_size)
        elif args.init_from == "gpt2":
            final_params = train_from_pretrained(config, train_data, val_data, vocab_size)
        else:
            raise ValueError(f"Unknown initialization method: {args.init_from}")
        
        print("✅ Training completed successfully!")
        print(f"Final model parameters saved to {args.out_dir}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        print("Checkpoint should be available for resuming")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
