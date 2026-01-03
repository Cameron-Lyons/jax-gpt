# jax-gpt

The simplest, fastest repository for training/finetuning medium-sized GPTs using JAX. A JAX/Flax port of [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy.

This is a rewrite of nanoGPT that uses JAX instead of PyTorch, keeping the same simple structure and functionality.

## install

```bash
pip install -r requirements.txt
```

Dependencies include JAX, Flax, Optax, tiktoken, datasets, wandb, numpy, and tqdm.

For GPU support, install JAX with CUDA:
```bash
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## quick start

If you want to train on the Shakespeare character-level dataset:

```bash
python data/shakespeare_char/prepare.py
python train.py config/train_shakespeare_char.py
```

This will download the tiny shakespeare dataset, tokenize it at the character level, and train a small GPT model. On a GPU this trains in a few minutes. The best validation loss is about 1.47. After training, you can sample from the model:

```bash
python sample.py --out_dir=out-shakespeare-char
```

This generates text samples based on the trained model.

## reproducing GPT-2

A more serious deep learning professional may be interested in reproducing GPT-2 results. To do so, first tokenize the OpenWebText dataset:

```bash
python data/openwebtext/prepare.py
```

This downloads and tokenizes the [OpenWebText](https://huggingface.co/datasets/openwebtext) dataset. It will create `train.bin` and `val.bin` in the `data/openwebtext` folder, which hold the GPT-2 BPE token ids as raw uint16 bytes.

To train GPT-2 (124M) on OpenWebText:

```bash
python train.py config/train_gpt2.py
```

The training script will automatically use gradient accumulation to simulate larger batch sizes.

## finetuning

Finetuning is easy. For example, to finetune GPT-2 on Shakespeare:

```bash
python data/shakespeare/prepare.py
python train.py config/finetune_shakespeare.py
```

This will load the GPT-2 (XL by default) weights and finetune on the Shakespeare dataset. The finetuning config uses a lower learning rate and fewer iterations.

## sampling / inference

Use `sample.py` to generate text from a trained model:

```bash
# sample from a custom trained model
python sample.py --out_dir=out-shakespeare-char

# sample from pretrained GPT-2 (downloads weights automatically)
python sample.py --init_from=gpt2
python sample.py --init_from=gpt2-medium
python sample.py --init_from=gpt2-large
python sample.py --init_from=gpt2-xl

# specify the starting prompt
python sample.py --init_from=gpt2-xl --start="What is the meaning of life?"

# load prompt from file
python sample.py --init_from=gpt2-xl --start="FILE:prompt.txt"
```

## baselines

OpenAI GPT-2 checkpoints can be loaded for comparison. We provide evaluation configs:

```bash
python sample.py config/eval_gpt2.py
python sample.py config/eval_gpt2_medium.py
python sample.py config/eval_gpt2_large.py
python sample.py config/eval_gpt2_xl.py
```

## config

Configuration is handled similarly to nanoGPT. You can:

1. Pass a config file as the first argument:
   ```bash
   python train.py config/train_shakespeare_char.py
   ```

2. Override specific parameters:
   ```bash
   python train.py config/train_shakespeare_char.py --batch_size=32 --learning_rate=1e-4
   ```

All configuration happens through the `configurator.py` file which is exec'd by both `train.py` and `sample.py`.

## files

```
├── train.py           # main training script (~300 lines)
├── model.py           # GPT model definition with Flax
├── sample.py          # sampling/inference script
├── configurator.py    # config override utility
├── gpt2.py            # pure functional GPT-2 implementation
├── data/
│   ├── shakespeare_char/
│   │   └── prepare.py   # character-level shakespeare
│   ├── shakespeare/
│   │   └── prepare.py   # BPE tokenized shakespeare
│   └── openwebtext/
│       └── prepare.py   # OpenWebText dataset
├── config/
│   ├── train_shakespeare_char.py
│   ├── train_gpt2.py
│   ├── finetune_shakespeare.py
│   ├── eval_gpt2.py
│   ├── eval_gpt2_medium.py
│   ├── eval_gpt2_large.py
│   └── eval_gpt2_xl.py
├── flax_gpt2.py       # alternative Flax GPT-2 model
├── trainer.py         # advanced trainer with checkpointing
├── utils.py           # utilities for model loading
├── parameter_converter.py  # convert between param formats
├── encoder.py         # GPT-2 tokenizer
├── bpe.py             # BPE implementation
├── benchmark.py       # benchmarking utilities
└── test_*.py          # test files
```

## model sizes

| Model | Parameters | Layers | Heads | Embedding Dim |
|-------|------------|--------|-------|---------------|
| 124M  | 124M       | 12     | 12    | 768           |
| 355M  | 355M       | 24     | 16    | 1024          |
| 774M  | 774M       | 36     | 20    | 1280          |
| 1558M | 1558M      | 48     | 25    | 1600          |

## training configuration

Key training parameters (see `train.py` for full list):

| Parameter | Default | Description |
|-----------|---------|-------------|
| batch_size | 12 | micro batch size |
| block_size | 1024 | context length |
| n_layer | 12 | number of transformer layers |
| n_head | 12 | number of attention heads |
| n_embd | 768 | embedding dimension |
| learning_rate | 6e-4 | max learning rate |
| max_iters | 600000 | total training iterations |
| warmup_iters | 2000 | warmup steps |
| weight_decay | 0.1 | AdamW weight decay |
| grad_clip | 1.0 | gradient clipping |
| dropout | 0.0 | dropout rate |
| decay_lr | True | use cosine LR decay |

## differences from nanoGPT

This implementation uses JAX/Flax instead of PyTorch:

- Uses `jax.jit` instead of `torch.compile`
- Uses `optax` for optimization instead of PyTorch optimizers
- Uses Flax's `nn.Module` instead of `torch.nn.Module`
- Uses JAX's random number handling (explicit PRNG keys)
- No DDP/FSDP (JAX uses different parallelism strategies like pmap/pjit)

The training script structure and config system are designed to match nanoGPT as closely as possible.

## advanced: pure functional implementation

For inference or educational purposes, `gpt2.py` contains a pure functional implementation:

```python
from gpt2 import main

result = main(
    prompt="The future of AI is",
    n_tokens_to_generate=50,
    model_size="124M",
    temperature=0.8,
    top_k=40
)
print(result)
```

This loads OpenAI's pretrained GPT-2 weights and generates text using pure JAX functions.

## license

MIT
