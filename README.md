# JAX GPT-2 Implementation

A complete implementation of GPT-2 using JAX, featuring efficient text generation, multiple sampling strategies, modern training infrastructure, and fine-tuning capabilities.

## Features

✅ **Core GPT-2 Architecture**
- Multi-head self-attention with causal masking
- Transformer blocks with residual connections
- Layer normalization and GELU activation
- Positional embeddings

✅ **Advanced Generation**
- Temperature-controlled sampling
- Top-k and Top-p (nucleus) sampling
- Stopping conditions
- Batch generation support

✅ **Modern Training Infrastructure**
- JAX/Flax-based trainer with JIT compilation
- Automatic checkpointing with Orbax
- Comprehensive metrics tracking
- Learning rate scheduling and gradient clipping
- Weights & Biases integration
- Type-safe configuration

✅ **Fine-tuning Capabilities**
- Parameter conversion between functional and Flax formats
- Fine-tuning from pretrained GPT-2 weights
- Resume training from checkpoints
- Support for custom datasets

✅ **JAX Optimizations**
- JIT compilation for performance
- Functional programming paradigm
- Efficient memory usage
- GPU/TPU acceleration support

✅ **Model Loading**
- Direct loading from OpenAI's TensorFlow checkpoints
- Support for all GPT-2 model sizes (124M, 355M, 774M, 1558M)
- Automatic model downloading

## Installation

1. **Install JAX** (choose your platform):
   ```bash
   # CPU only
   pip install --upgrade pip
   pip install --upgrade "jax[cpu]"

   # GPU (CUDA)
   pip install --upgrade pip
   pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

   # TPU
   pip install --upgrade pip
   pip install --upgrade "jax[tpu]" -f https://storage.googleapis.com/jax-releases/jax_tpu_releases.html
   ```

2. **Install other dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

### Basic Text Generation

```python
from gpt2 import main

# Generate text with default settings
result = main(
    prompt="The future of artificial intelligence is",
    n_tokens_to_generate=50,
    model_size="124M"
)
print(result)
```

### Advanced Generation with Sampling Controls

```python
from gpt2 import main

# Creative generation with temperature and top-p sampling
result = main(
    prompt="Once upon a time in a magical forest",
    n_tokens_to_generate=100,
    model_size="124M",
    temperature=1.2,
    top_p=0.9
)
print(result)
```

### Command Line Usage

```bash
# Basic generation
python gpt2.py --prompt "Hello, world!" --n_tokens_to_generate 50

# Creative generation
python gpt2.py --prompt "A robot walks into a bar" --temperature 1.2 --top_k 10

# Large model
python gpt2.py --prompt "The meaning of life is" --model_size 774M --top_p 0.9
```

## Architecture Overview

### Core Components

1. **Attention Mechanism**
   ```python
   def multihead_attn(x, c_attn, c_proj, n_head):
       # Split input into query, key, value
       qkv = jnp.split(linear(x, **c_attn), 3, axis=-1)
       # Apply causal attention with masking
       # Combine heads and project
   ```

2. **Transformer Block**
   ```python
   def transformer_block(x, mlp, attn, ln_1, ln_2, n_head):
       # Self-attention with residual connection
       x = x + multihead_attn(layer_norm(x, **ln_1), **attn, n_head=n_head)
       # Feed-forward network with residual connection
       x = x + feed_forward_network(layer_norm(x, **ln_2), **mlp)
   ```

3. **GPT-2 Model**
   ```python
   def gpt2(inputs, wte, wpe, blocks, ln_f, n_head):
       # Token and position embeddings
       x = wte[inputs] + wpe(jnp.arange(len(inputs)))
       # Stack of transformer blocks
       for block in blocks:
           x = transformer_block(x, **block, n_head=n_head)
       # Final layer norm and projection to vocab
       return layer_norm(x, **ln_f) @ wte.T
   ```

### Sampling Strategies

1. **Temperature Sampling**
   - Controls randomness: lower = more deterministic
   - `temperature = 0.0` → greedy decoding
   - `temperature = 1.0` → standard sampling

2. **Top-k Sampling**
   - Only consider top-k most likely tokens
   - Reduces repetition and improves quality
   - Typical values: k = 10-50

3. **Top-p (Nucleus) Sampling**
   - Consider tokens until cumulative probability reaches p
   - More adaptive than top-k
   - Typical values: p = 0.9-0.95

## Performance Optimizations

### JIT Compilation
All core functions are JIT-compiled for maximum performance:

```python
@jit
def gpt2(inputs, wte, wpe, blocks, ln_f, n_head):
    # JIT-compiled for optimal performance
    pass
```

### Memory Efficiency
- Uses JAX arrays for efficient memory management
- Functional approach avoids mutable state
- Automatic memory optimization

### GPU/TPU Support
- Automatically utilizes available accelerators
- Scales efficiently across multiple devices
- Optimized for modern hardware

## Examples

### Interactive Generation

Run the interactive example:
```bash
python example_usage.py
```

This demonstrates:
- Different sampling strategies
- Performance benchmarking
- Batch generation
- Stopping conditions

### Custom Generation Loop

```python
from gpt2 import generate, generate_with_stopping
from utils import load_encoder_hparams_and_params

# Load model
encoder, hparams, params = load_encoder_hparams_and_params("124M", "models")

# Generate with custom parameters
input_ids = encoder.encode("The best way to learn programming is")
output_ids = generate(
    input_ids,
    params,
    hparams["n_head"],
    n_tokens_to_generate=50,
    temperature=0.8,
    top_k=10
)

# Decode and print
output_text = encoder.decode(output_ids)
print(output_text)
```

### Training Integration

The implementation is designed to work with JAX training loops:

```python
import optax
from gpt2 import lm_loss

# Define optimizer
optimizer = optax.adamw(learning_rate=1e-4)

# Training step
def train_step(params, batch, optimizer_state):
    loss, grads = jax.value_and_grad(lm_loss)(params, batch, n_head=12)
    updates, optimizer_state = optimizer.update(grads, optimizer_state)
    params = optax.apply_updates(params, updates)
    return params, optimizer_state, loss
```

## Model Sizes

| Model | Parameters | Layers | Heads | Embedding Dim |
|-------|------------|--------|-------|---------------|
| 124M  | 124M       | 12     | 12    | 768           |
| 355M  | 355M       | 24     | 16    | 1024          |
| 774M  | 774M       | 36     | 20    | 1280          |
| 1558M | 1558M      | 48     | 25    | 1600          |

## Advanced Features

### Stopping Conditions
```python
# Stop at specific tokens
stop_tokens = [encoder.encode(".")[0], encoder.encode("\n")[0]]
output_ids = generate_with_stopping(
    input_ids, params, hparams["n_head"], 
    max_tokens=100, stop_tokens=stop_tokens
)
```

### Batch Processing
```python
# Generate multiple sequences
prompts = ["Hello", "Goodbye", "How are you"]
results = []
for prompt in prompts:
    input_ids = encoder.encode(prompt)
    output_ids = generate(input_ids, params, hparams["n_head"], 20)
    results.append(encoder.decode(output_ids))
```

### Custom Sampling
```python
# Implement custom sampling strategies
def custom_sampling(logits, temperature=1.0, top_k=None):
    if top_k:
        top_k_logits, top_k_indices = jax.lax.top_k(logits, top_k)
        probs = jax.nn.softmax(top_k_logits / temperature)
        return jax.random.categorical(rng, probs)
    return jax.random.categorical(rng, jax.nn.softmax(logits / temperature))
```

## Training and Fine-tuning

### Training from Scratch

Train a new GPT-2 model from scratch:

```bash
# Basic training
python train.py --init_from scratch --model_size 124M --batch_size 32

# With custom parameters
python train.py \
    --init_from scratch \
    --model_size 355M \
    --batch_size 16 \
    --learning_rate 3e-4 \
    --max_iters 50000 \
    --use_wandb
```

### Fine-tuning from Pretrained Weights

Fine-tune a pretrained GPT-2 model:

```bash
# Fine-tune with lower learning rate
python train.py \
    --init_from gpt2 \
    --model_size 124M \
    --learning_rate 1e-5 \
    --batch_size 8 \
    --max_iters 10000
```

### Resuming Training

Resume training from a checkpoint:

```bash
python train.py --init_from resume --out_dir out
```

### Parameter Conversion

The implementation includes utilities for converting between functional and Flax parameter formats:

```python
from parameter_converter import (
    convert_functional_to_flax_params,
    convert_flax_to_functional_params,
    initialize_model_with_pretrained_weights
)

# Convert pretrained functional parameters to Flax format
flax_params = convert_functional_to_flax_params(
    functional_params, model, dummy_input
)

# Initialize model with pretrained weights
pretrained_state = initialize_model_with_pretrained_weights(
    model, pretrained_params, dummy_input
)
```

### Testing Parameter Conversion

Test the parameter conversion functionality:

```bash
python test_parameter_conversion.py
```

This will:
- Test conversion between functional and Flax formats
- Verify that model outputs match after conversion
- Test loading pretrained weights (if available)

### Training Configuration

The trainer supports extensive configuration options:

```python
from trainer import TrainingConfig

config = TrainingConfig(
    model_size="124M",
    batch_size=32,
    max_iters=100000,
    learning_rate=6e-4,
    weight_decay=0.1,
    warmup_iters=2000,
    eval_interval=2000,
    save_interval=5000,
    use_wandb=True,
    project_name="jax-gpt2"
)
```

### Data Preparation

Prepare your training data in the required format:

```python
# Tokenize and save data
from utils import encode_dataset

# Encode text data
train_data = encode_dataset(train_texts, encoder)
val_data = encode_dataset(val_texts, encoder)

# Save as binary files
train_data.tofile("data/train.bin")
val_data.tofile("data/val.bin")

# Save metadata
import pickle
meta = {"vocab_size": encoder.n_vocab}
with open("data/meta.pkl", "wb") as f:
    pickle.dump(meta, f)
```

### Monitoring Training

The trainer provides comprehensive logging:

```python
# Console output
Iter 1000: train_loss=2.345, val_loss=2.123, lr=0.000600
Iter 2000: train_loss=2.123, val_loss=2.001, lr=0.000600
...

# Weights & Biases integration
# Automatically logs metrics, gradients, and model checkpoints
```

### Checkpointing

Automatic checkpointing with Orbax:

```python
# Checkpoints are saved automatically
# Resume from latest checkpoint
trainer = ModernTrainer(config, model, train_data, val_data)
# Automatically loads latest checkpoint if available
```

## Troubleshooting

### Common Issues

1. **JAX Installation**
   - Ensure you have the correct JAX version for your platform
   - Check CUDA compatibility for GPU usage

2. **Memory Issues**
   - Reduce batch size or sequence length
   - Use smaller model size (124M instead of 1558M)

3. **Performance**
   - Ensure JIT compilation is working
   - Check if GPU/TPU is being utilized
   - Profile with `jax.profiler`

### Debugging

```python
# Enable JAX debugging
import jax
jax.config.update('jax_debug_nans', True)
jax.config.update('jax_debug_infs', True)

# Profile specific functions
from jax.profiler import trace
with trace("gpt2_inference"):
    result = gpt2(inputs, **params, n_head=12)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This implementation is provided for educational and research purposes. Please respect OpenAI's terms of service when using GPT-2 models.

## Acknowledgments

- OpenAI for the original GPT-2 architecture
- JAX team for the excellent framework
- The open-source community for various utilities and tools 