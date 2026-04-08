# jax-gpt

Train, finetune, benchmark, and sample GPT-style language models with JAX/Flax.

This project keeps the small-script feel of nanoGPT, but now uses typed configs,
importable entrypoints, and resumable checkpoints with optimizer state.

## install

Use the package metadata as the source of truth:

```bash
pip install -e .
```

Add extras only for the features you need:

```bash
pip install -e ".[data]"
pip install -e ".[pretrained]"
```

- `.[tokenization]`: GPT-2 BPE tokenization without the full pretrained/data stack
- `.[data]`: dataset download and preprocessing scripts
- `.[pretrained]`: Hugging Face GPT-2 weight loading for pretrained init and sampling

For a full development environment:

```bash
pip install -e ".[all,dev]"
```

If you prefer a pinned all-features-plus-dev environment, `requirements.txt` is still available.

For GPU support, install a JAX build that matches your CUDA setup.

## quick start

Install `.[data]` if you want to run the dataset preparation scripts.

Prepare the tiny Shakespeare character dataset:

```bash
python data/shakespeare_char/prepare.py
python -m jax_gpt.train config/train_shakespeare_char.py
```

You can also use the installed console script:

```bash
jax-gpt-train config/train_shakespeare_char.py
```

Sample from the latest checkpoint:

```bash
python -m jax_gpt.sample --out_dir=out-shakespeare-char --start="To be"
```

## training

Train GPT-2 124M on OpenWebText:

```bash
python data/openwebtext/prepare.py
python -m jax_gpt.train config/train_gpt2.py
```

Finetune from pretrained GPT-2 weights:

```bash
python data/shakespeare/prepare.py
python -m jax_gpt.train config/finetune_shakespeare.py
```

Resume a previous run with optimizer state intact:

```bash
python -m jax_gpt.train --init_from=resume --out_dir=out-shakespeare-char
```

## sampling

Sample from a local checkpoint:

```bash
python -m jax_gpt.sample --init_from=resume --out_dir=out-shakespeare-char --start="FILE:prompt.txt"
```

Sample from pretrained GPT-2 variants:

```bash
python -m jax_gpt.sample --init_from=gpt2 --start="The future of AI is"
python -m jax_gpt.sample --init_from=gpt2-medium
python -m jax_gpt.sample --init_from=gpt2-large
python -m jax_gpt.sample --init_from=gpt2-xl
```

`jax_gpt.sample` supports both `top_k` and `top_p` controls:

```bash
python -m jax_gpt.sample --init_from=gpt2 --top_k=None --top_p=0.9 --temperature=0.7
```

## configuration

`jax_gpt.train` and `jax_gpt.sample` accept:

1. A config file as the first argument.
2. `--key=value` overrides after that.

Examples:

```bash
python -m jax_gpt.train config/train_shakespeare_char.py --batch_size=32 --learning_rate=1e-4
python -m jax_gpt.sample config/eval_gpt2.py --top_p=0.95
```

Supported config file types:

- `.py` files with simple assignments and literal arithmetic like `5 * 8`
- `.json` files

The old `exec`-based configuration flow is gone, so config files are now safer and easier to test.

## benchmarking

Run the benchmark suite:

```bash
python -m jax_gpt.benchmark --model_size 124M --device cpu --use_synthetic_data --no_save
```

Or with the installed entrypoint:

```bash
jax-gpt-benchmark --model_size 124M --device cpu --use_synthetic_data --no_save
```

Real-data benchmarking reads `train.bin` from `data/<dataset>/train.bin`.

## python api

The scripts are also importable:

```python
from jax_gpt.config import SampleConfig, TrainConfig
from jax_gpt.sample import sample_texts
from jax_gpt.train import train

summary = train(TrainConfig(dataset="shakespeare_char", out_dir="out-shakespeare-char"))
samples = sample_texts(SampleConfig(out_dir="out-shakespeare-char", start="Hello"))
```

## project layout

```text
.
├── config/
├── data/
├── jax_gpt/
├── tests/
└── README.md
```

## testing

```bash
ruff check --fix
ruff format
mypy .
pytest
```

## model sizes

| Model | Parameters | Layers | Heads | Embedding Dim |
|-------|------------|--------|-------|---------------|
| 124M  | 124M       | 12     | 12    | 768           |
| 355M  | 355M       | 24     | 16    | 1024          |
| 774M  | 774M       | 36     | 20    | 1280          |
| 1558M | 1558M      | 48     | 25    | 1600          |

## license

MIT
