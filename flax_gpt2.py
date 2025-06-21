"""
Flax-based GPT-2 model for training with the modern trainer.
"""

from typing import Dict, Any
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.struct import dataclass
import math


@dataclass
class GPTConfig:
    """GPT-2 model configuration."""
    vocab_size: int = 50304
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention layer."""
    config: GPTConfig
    
    @nn.compact
    def __call__(self, x, training: bool = True):
        B, T, C = x.shape
        assert C % self.config.n_head == 0
        
        # Linear projections for Q, K, V
        qkv = nn.Dense(
            features=3 * C,
            use_bias=self.config.bias,
            kernel_init=nn.initializers.xavier_uniform()
        )(x)
        
        # Reshape to separate heads
        qkv = qkv.reshape(B, T, 3, self.config.n_head, C // self.config.n_head)
        qkv = qkv.transpose(0, 2, 3, 1, 4)  # (B, 3, n_head, T, head_dim)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        
        # Compute attention
        scale = 1.0 / math.sqrt(k.shape[-1])
        att = (q @ k.transpose(0, 1, 3, 2)) * scale
        
        # Apply causal mask
        mask = jnp.triu(jnp.ones((T, T), dtype=att.dtype), k=1)
        att = jnp.where(mask == 1, -1e9, att)
        
        # Apply softmax and dropout
        att = jax.nn.softmax(att, axis=-1)
        if training and self.config.dropout > 0:
            att = nn.Dropout(rate=self.config.dropout)(att, deterministic=not training)
        
        # Apply attention to values
        y = (att @ v).transpose(0, 2, 1, 3).reshape(B, T, C)
        
        # Output projection
        y = nn.Dense(
            features=C,
            use_bias=self.config.bias,
            kernel_init=nn.initializers.xavier_uniform()
        )(y)
        
        if training and self.config.dropout > 0:
            y = nn.Dropout(rate=self.config.dropout)(y, deterministic=not training)
        
        return y


class FeedForward(nn.Module):
    """Feed-forward network."""
    config: GPTConfig
    
    @nn.compact
    def __call__(self, x, training: bool = True):
        # First linear layer with GELU activation
        x = nn.Dense(
            features=4 * self.config.n_embd,
            use_bias=self.config.bias,
            kernel_init=nn.initializers.xavier_uniform()
        )(x)
        x = jax.nn.gelu(x)
        
        if training and self.config.dropout > 0:
            x = nn.Dropout(rate=self.config.dropout)(x, deterministic=not training)
        
        # Second linear layer
        x = nn.Dense(
            features=self.config.n_embd,
            use_bias=self.config.bias,
            kernel_init=nn.initializers.xavier_uniform()
        )(x)
        
        if training and self.config.dropout > 0:
            x = nn.Dropout(rate=self.config.dropout)(x, deterministic=not training)
        
        return x


class TransformerBlock(nn.Module):
    """Transformer block with attention and feed-forward layers."""
    config: GPTConfig
    
    @nn.compact
    def __call__(self, x, training: bool = True):
        # Self-attention with residual connection
        x = x + MultiHeadAttention(self.config)(nn.LayerNorm()(x), training)
        
        # Feed-forward with residual connection
        x = x + FeedForward(self.config)(nn.LayerNorm()(x), training)
        
        return x


class GPT2(nn.Module):
    """GPT-2 model."""
    config: GPTConfig
    
    @nn.compact
    def __call__(self, x, training: bool = True):
        B, T = x.shape
        
        # Token and position embeddings
        tok_emb = nn.Embed(
            num_embeddings=self.config.vocab_size,
            features=self.config.n_embd,
            embedding_init=nn.initializers.normal(stddev=0.02)
        )(x)
        
        pos_emb = nn.Embed(
            num_embeddings=self.config.block_size,
            features=self.config.n_embd,
            embedding_init=nn.initializers.normal(stddev=0.02)
        )(jnp.arange(T))
        
        x = tok_emb + pos_emb
        
        if training and self.config.dropout > 0:
            x = nn.Dropout(rate=self.config.dropout)(x, deterministic=not training)
        
        # Apply transformer blocks
        for _ in range(self.config.n_layer):
            x = TransformerBlock(self.config)(x, training)
        
        # Final layer norm
        x = nn.LayerNorm()(x)
        
        # Language modeling head
        logits = nn.Dense(
            features=self.config.vocab_size,
            use_bias=False,
            kernel_init=nn.initializers.xavier_uniform()
        )(x)
        
        return logits


def create_gpt2_model(config: GPTConfig) -> GPT2:
    """Create a GPT-2 model with the given configuration."""
    return GPT2(config)


def get_model_config(model_size: str) -> GPTConfig:
    """Get model configuration for different sizes."""
    configs = {
        "124M": GPTConfig(
            vocab_size=50304,
            block_size=1024,
            n_layer=12,
            n_head=12,
            n_embd=768,
            dropout=0.0,
            bias=True
        ),
        "355M": GPTConfig(
            vocab_size=50304,
            block_size=1024,
            n_layer=24,
            n_head=16,
            n_embd=1024,
            dropout=0.0,
            bias=True
        ),
        "774M": GPTConfig(
            vocab_size=50304,
            block_size=1024,
            n_layer=36,
            n_head=20,
            n_embd=1280,
            dropout=0.0,
            bias=True
        ),
        "1558M": GPTConfig(
            vocab_size=50304,
            block_size=1024,
            n_layer=48,
            n_head=25,
            n_embd=1600,
            dropout=0.0,
            bias=True
        )
    }
    
    if model_size not in configs:
        raise ValueError(f"Unknown model size: {model_size}")
    
    return configs[model_size]


def count_parameters(params: Dict[str, Any]) -> int:
    """Count the number of parameters in the model."""
    total_params = 0
    for param in jax.tree_util.tree_leaves(params):
        total_params += param.size
    return total_params


def print_model_summary(model: GPT2, config: GPTConfig):
    """Print a summary of the model."""
    print(f"GPT-2 Model Summary ({config.n_layer}L, {config.n_head}H, {config.n_embd}D)")
    print(f"Vocabulary size: {config.vocab_size}")
    print(f"Block size: {config.block_size}")
    print(f"Total parameters: {config.n_layer * (12 * config.n_embd**2 + 2 * config.n_embd * config.vocab_size):,}")
    print(f"Model size: {config.n_layer * (12 * config.n_embd**2 + 2 * config.n_embd * config.vocab_size) * 4 / 1e6:.1f} MB") 