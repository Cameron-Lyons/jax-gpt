"""Modern utilities for JAX GPT-2 models with enhanced functionality."""
import json
import re
import logging
from pathlib import Path
from typing import Literal, Dict, Any, Tuple, Optional, Union, List
from dataclasses import dataclass
from functools import lru_cache
import pickle
from ast import literal_eval
import sys

import jax.numpy as jnp
from jax import random, jit
import requests
import tensorflow as tf
from tqdm import tqdm

from encoder import get_encoder, Encoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ModelSize = Literal["124M", "355M", "774M", "1558M"]
Array = jnp.ndarray
PRNGKey = jnp.ndarray

    
@dataclass
class ModelConfig:
    """Configuration for GPT-2 model variants."""
    n_layer: int
    n_head: int
    n_embd: int
    vocab_size: int
    block_size: int
    embd_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    use_bias: bool = True
    dtype: str = "bfloat16"
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ModelConfig":
        """Create ModelConfig from dictionary."""
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "n_layer": self.n_layer,
            "n_head": self.n_head,
            "n_embd": self.n_embd,
            "vocab_size": self.vocab_size,
            "block_size": self.block_size,
            "embd_pdrop": self.embd_pdrop,
            "resid_pdrop": self.resid_pdrop,
            "attn_pdrop": self.attn_pdrop,
            "layer_norm_epsilon": self.layer_norm_epsilon,
            "initializer_range": self.initializer_range,
            "use_bias": self.use_bias,
            "dtype": self.dtype,
        }


@dataclass
class DownloadConfig:
    """Configuration for model downloads."""
    chunk_size: int = 8192
    timeout: int = 30
    retries: int = 3
    verify_ssl: bool = True
    user_agent: str = "JAX-GPT/1.0"
    
    def get_session(self) -> requests.Session:
        """Create configured requests session."""
        session = requests.Session()
        session.headers.update({"User-Agent": self.user_agent})
        return session


class ModelManager:
    """Manages GPT-2 model downloads, caching, and parameter loading."""
    
    def __init__(self, models_dir: str = "models", download_config: Optional[DownloadConfig] = None):
        self.models_dir = Path(models_dir)
        self.download_config = download_config or DownloadConfig()
        self.models_dir.mkdir(exist_ok=True)
        
        # Model URLs and file lists
        self.model_urls = {
            "124M": "https://openaipublic.blob.core.windows.net/gpt-2/models/124M",
            "355M": "https://openaipublic.blob.core.windows.net/gpt-2/models/355M",
            "774M": "https://openaipublic.blob.core.windows.net/gpt-2/models/774M",
            "1558M": "https://openaipublic.blob.core.windows.net/gpt-2/models/1558M",
        }
        
        self.required_files = [
            "checkpoint",
            "encoder.json",
            "hparams.json",
            "model.ckpt.data-00000-of-00001",
            "model.ckpt.index",
            "model.ckpt.meta",
            "vocab.bpe",
        ]
    
    def get_model_path(self, model_size: ModelSize) -> Path:
        """Get path for specific model size."""
        return self.models_dir / model_size
    
    def is_model_downloaded(self, model_size: ModelSize) -> bool:
        """Check if model is already downloaded."""
        model_path = self.get_model_path(model_size)
        return all((model_path / filename).exists() for filename in self.required_files)
    
    def download_file(self, url: str, filepath: Path, desc: str) -> None:
        """Download a single file with progress bar and error handling."""
        session = self.download_config.get_session()
        
        for attempt in range(self.download_config.retries):
            try:
                response = session.get(
                    url, 
                    stream=True, 
                    timeout=self.download_config.timeout,
                    verify=self.download_config.verify_ssl
                )
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                
                with open(filepath, 'wb') as f:
                    with tqdm(
                        total=total_size,
                        unit='B',
                        unit_scale=True,
                        desc=desc,
                        ncols=100
                    ) as pbar:
                        for chunk in response.iter_content(chunk_size=self.download_config.chunk_size):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                return
                
            except (requests.RequestException, IOError) as e:
                logger.warning(f"Download attempt {attempt + 1} failed: {e}")
                if attempt == self.download_config.retries - 1:
                    raise RuntimeError(f"Failed to download {url} after {self.download_config.retries} attempts")
                continue
    
    def download_model(self, model_size: ModelSize) -> None:
        """Download complete GPT-2 model."""
        if self.is_model_downloaded(model_size):
            logger.info(f"Model {model_size} already downloaded")
            return
        
        logger.info(f"Downloading GPT-2 {model_size} model...")
        model_path = self.get_model_path(model_size)
        model_path.mkdir(exist_ok=True)
        
        base_url = self.model_urls[model_size]
        
        for filename in self.required_files:
            url = f"{base_url}/{filename}"
            filepath = model_path / filename
            self.download_file(url, filepath, f"Downloading {filename}")
        
        logger.info(f"Successfully downloaded GPT-2 {model_size} model")
    
    @lru_cache(maxsize=4)
    def load_hparams(self, model_size: ModelSize) -> ModelConfig:
        """Load and cache model hyperparameters."""
        model_path = self.get_model_path(model_size)
        hparams_file = model_path / "hparams.json"
        
        if not hparams_file.exists():
            self.download_model(model_size)
        
        with open(hparams_file, 'r') as f:
            hparams_dict = json.load(f)
        
        return ModelConfig.from_dict(hparams_dict)
    
    def load_encoder(self, model_size: ModelSize) -> Encoder:
        """Load model encoder."""
        if not self.is_model_downloaded(model_size):
            self.download_model(model_size)
        
        return get_encoder(model_size, str(self.models_dir))


class ParameterConverter:
    """Converts TensorFlow checkpoints to JAX/Flax parameters."""
    
    def __init__(self):
        self.param_mapping = {
            "wte": "token_embedding",
            "wpe": "position_embedding",
            
            "attn/c_attn": "attention/query_key_value",
            "attn/c_proj": "attention/output",
            "attn/c_attn/b": "attention/query_key_value_bias",
            "attn/c_proj/b": "attention/output_bias",
            
            "mlp/c_fc": "mlp/fc1",
            "mlp/c_proj": "mlp/fc2",
            "mlp/c_fc/b": "mlp/fc1_bias",
            "mlp/c_proj/b": "mlp/fc2_bias",
            
            "ln_1": "layer_norm1",
            "ln_2": "layer_norm2",
            "ln_1/b": "layer_norm1_bias",
            "ln_2/b": "layer_norm2_bias",
            "ln_1/g": "layer_norm1_scale",
            "ln_2/g": "layer_norm2_scale",
            
            "ln_f": "final_layer_norm",
            "ln_f/b": "final_layer_norm_bias",
            "ln_f/g": "final_layer_norm_scale",
        }
    
    def convert_tf_to_jax_params(self, tf_ckpt_path: str, hparams: ModelConfig) -> Dict[str, Any]:
        """Convert TensorFlow checkpoint to JAX parameters."""
        logger.info(f"Converting TensorFlow checkpoint: {tf_ckpt_path}")
        
        params = {"blocks": [{} for _ in range(hparams.n_layer)]}
        
        for name, _ in tf.train.list_variables(tf_ckpt_path):
            array = tf.train.load_variable(tf_ckpt_path, name)
            array = jnp.array(array)
            
            name = name[len("model/"):]
            
            if name.startswith("h"):
                m = re.match(r"h([0-9]+)/(.*)", name)
                if m:
                    n = int(m[1])
                    sub_name = m[2]
                    self._set_nested_dict(params["blocks"][n], sub_name.split("/"), array)
            else:
                self._set_nested_dict(params, name.split("/"), array)
        
        return self._convert_to_flax_format(params, hparams)
    
    def _set_nested_dict(self, d: Dict, keys: List[str], val: Array) -> None:
        """Set value in nested dictionary."""
        if not keys:
            return
        if keys[0] not in d:
            d[keys[0]] = {}
        if len(keys) == 1:
            d[keys[0]] = val
        else:
            self._set_nested_dict(d[keys[0]], keys[1:], val)
    
    def _convert_to_flax_format(self, params: Dict[str, Any], hparams: ModelConfig) -> Dict[str, Any]:
        """Convert parameters to Flax format."""
        flax_params = {
            "token_embedding": {"embedding": params["wte"]},
            "position_embedding": {"embedding": params["wpe"]},
            "blocks": [],
            "final_layer_norm": {
                "bias": params["ln_f"]["b"],
                "scale": params["ln_f"]["g"]
            }
        }
        
        for block in params["blocks"]:
            flax_block = {
                "attention": {
                    "query_key_value": {
                        "kernel": block["attn"]["c_attn"]["w"].T,
                        "bias": block["attn"]["c_attn"]["b"]
                    },
                    "output": {
                        "kernel": block["attn"]["c_proj"]["w"].T,
                        "bias": block["attn"]["c_proj"]["b"]
                    }
                },
                "mlp": {
                    "fc1": {
                        "kernel": block["mlp"]["c_fc"]["w"].T,
                        "bias": block["mlp"]["c_fc"]["b"]
                    },
                    "fc2": {
                        "kernel": block["mlp"]["c_proj"]["w"].T,
                        "bias": block["mlp"]["c_proj"]["b"]
                    }
                },
                "layer_norm1": {
                    "bias": block["ln_1"]["b"],
                    "scale": block["ln_1"]["g"]
                },
                "layer_norm2": {
                    "bias": block["ln_2"]["b"],
                    "scale": block["ln_2"]["g"]
                }
            }
            flax_params["blocks"].append(flax_block)
        
        return flax_params


class DataUtils:
    """Utilities for data processing and tokenization."""
    
    @staticmethod
    @jit
    def create_causal_mask(seq_len: int, dtype: str = "bfloat16") -> Array:
        """Create causal attention mask."""
        mask = jnp.triu(jnp.ones((seq_len, seq_len), dtype=dtype), k=1)
        return mask
    
    @staticmethod
    @jit
    def create_padding_mask(attention_mask: Array) -> Array:
        """Create padding mask from attention mask."""
        return attention_mask[:, :, None] * attention_mask[:, None, :]
    
    @staticmethod
    def tokenize_batch(texts: List[str], encoder: Encoder, max_length: Optional[int] = None) -> Dict[str, Array]:
        """Tokenize a batch of texts."""
        if max_length is None:
            max_length = max(len(encoder.encode(text)) for text in texts)
        
        input_ids = []
        attention_mask = []
        
        for text in texts:
            tokens = encoder.encode(text)
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
            
            padding_length = max_length - len(tokens)
            input_ids.append(tokens + [0] * padding_length)
            attention_mask.append([1] * len(tokens) + [0] * padding_length)
        
        return {
            "input_ids": jnp.array(input_ids, dtype=jnp.int32),
            "attention_mask": jnp.array(attention_mask, dtype=jnp.int32)
        }
    
    @staticmethod
    def create_text_dataset(texts: List[str], encoder: Encoder, block_size: int, 
                          stride: Optional[int] = None) -> Array:
        """Create training dataset from texts."""
        if stride is None:
            stride = block_size // 2
        
        all_tokens = []
        for text in texts:
            tokens = encoder.encode(text)
            all_tokens.extend(tokens)
        
        sequences = []
        for i in range(0, len(all_tokens) - block_size + 1, stride):
            sequences.append(all_tokens[i:i + block_size])
        
        return jnp.array(sequences, dtype=jnp.int32)
    
    @staticmethod
    @jit
    def get_batch(data: Array, batch_size: int, block_size: int, rng: PRNGKey) -> Tuple[Array, Array]:
        """Get random batch from dataset."""
        rng, batch_rng = random.split(rng)
        
        ix = random.randint(batch_rng, (batch_size,), 0, data.shape[0] - block_size)
        
        x = jnp.stack([data[i:i + block_size] for i in ix])
        y = jnp.stack([data[i + 1:i + block_size + 1] for i in ix])
        
        return x, y


class MetricsTracker:
    """Track and compute training metrics."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.metrics = {
            "loss": [],
            "accuracy": [],
            "learning_rate": [],
            "gradient_norm": [],
            "param_norm": []
        }
    
    def update(self, **kwargs):
        """Update metrics with new values."""
        for key, value in kwargs.items():
            if key in self.metrics:
                self.metrics[key].append(float(value))
    
    def get_latest(self, metric: str) -> Optional[float]:
        """Get latest value for a metric."""
        if metric in self.metrics and self.metrics[metric]:
            return self.metrics[metric][-1]
        return None
    
    def get_average(self, metric: str, window: int = 100) -> Optional[float]:
        """Get average of recent values for a metric."""
        if metric in self.metrics and self.metrics[metric]:
            values = self.metrics[metric][-window:]
            return sum(values) / len(values)
        return None
    
    def to_dict(self) -> Dict[str, List[float]]:
        """Convert to dictionary."""
        return self.metrics.copy()


class ConfigManager:
    """Modern configuration management with validation and serialization."""
    
    def __init__(self, **kwargs):
        self._config = {}
        self._frozen = False
        self.update(**kwargs)
    
    def update(self, **kwargs):
        """Update configuration."""
        if self._frozen:
            raise RuntimeError("Configuration is frozen")
        self._config.update(kwargs)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set configuration value."""
        if self._frozen:
            raise RuntimeError("Configuration is frozen")
        self._config[key] = value
    
    def freeze(self):
        """Freeze configuration to prevent further changes."""
        self._frozen = True
    
    def unfreeze(self):
        """Unfreeze configuration."""
        self._frozen = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self._config.copy()
    
    def save(self, filepath: Union[str, Path]):
        """Save configuration to file."""
        with open(filepath, 'w') as f:
            json.dump(self._config, f, indent=2)
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "ConfigManager":
        """Load configuration from file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    def merge_from_dict(self, config_dict: Dict[str, Any]):
        """Merge configuration from dictionary."""
        if self._frozen:
            raise RuntimeError("Configuration is frozen")
        self._config.update(config_dict)
    
    def merge_from_args(self, args: List[str]):
        """Merge configuration from command line arguments."""
        if self._frozen:
            raise RuntimeError("Configuration is frozen")
        
        for arg in args:
            if not arg.startswith('--'):
                continue
            
            try:
                key, val = arg[2:].split('=', 1)
                keys = key.split('.')
                
                obj = self._config
                for k in keys[:-1]:
                    if k not in obj:
                        obj[k] = {}
                    obj = obj[k]
                
                try:
                    val = literal_eval(val)
                except (ValueError, SyntaxError):
                    pass
                
                obj[keys[-1]] = val
                logger.info(f"Set config {key} = {val}")
                
            except ValueError as e:
                logger.warning(f"Invalid argument format: {arg}, error: {e}")


def setup_logging(work_dir: Union[str, Path], level: str = "INFO"):
    """Setup logging configuration."""
    work_dir = Path(work_dir)
    work_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(work_dir / 'training.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    with open(work_dir / 'args.txt', 'w') as f:
        f.write(' '.join(sys.argv))


def count_parameters(params: Dict[str, Any]) -> Dict[str, int]:
    """Count parameters in a Flax model."""
    total_params = 0
    trainable_params = 0
    
    def count_dict(d):
        nonlocal total_params, trainable_params
        for key, value in d.items():
            if isinstance(value, dict):
                count_dict(value)
            elif isinstance(value, jnp.ndarray):
                param_count = value.size
                total_params += param_count
                trainable_params += param_count
    
    count_dict(params)
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_size_mb": total_params * 4 / (1024 * 1024)  # Assuming float32
    }


def save_checkpoint(params: Dict[str, Any], optimizer_state: Any, step: int, 
                   filepath: Union[str, Path]):
    """Save model checkpoint."""
    checkpoint = {
        "params": params,
        "optimizer_state": optimizer_state,
        "step": step
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(checkpoint, f)


def load_checkpoint(filepath: Union[str, Path]) -> Tuple[Dict[str, Any], Any, int]:
    """Load model checkpoint."""
    with open(filepath, 'rb') as f:
        checkpoint = pickle.load(f)
    
    return checkpoint["params"], checkpoint["optimizer_state"], checkpoint["step"]


def download_gpt2_files(model_size: ModelSize, model_dir: str):
    """Download GPT-2 model files (legacy function)."""
    manager = ModelManager(model_dir)
    manager.download_model(model_size)


def load_gpt2_params_from_tf_ckpt(tf_ckpt_path: str, hparams: Dict[str, Any]) -> Dict[str, Any]:
    """Load GPT-2 params from TF checkpoint (legacy function)."""
    converter = ParameterConverter()
    config = ModelConfig.from_dict(hparams)
    return converter.convert_tf_to_jax_params(tf_ckpt_path, config)


def load_encoder_hparams_and_params(
    model_size: ModelSize, models_dir: str
) -> Tuple[Encoder, Dict[str, Any], Dict[str, Any]]:
    """Load encoder, hparams, and params (legacy function)."""
    manager = ModelManager(models_dir)
    converter = ParameterConverter()
    
    manager.download_model(model_size)
    
    encoder = manager.load_encoder(model_size)
    hparams = manager.load_hparams(model_size)
    model_path = manager.get_model_path(model_size)
    
    tf_ckpt_path = tf.train.latest_checkpoint(str(model_path))
    params = converter.convert_tf_to_jax_params(tf_ckpt_path, hparams)
    
    return encoder, hparams.to_dict(), params


class CfgNode:
    """Legacy configuration class for backward compatibility."""
    
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    
    def __str__(self):
        return self._str_helper(0)
    
    def _str_helper(self, indent):
        parts = []
        for k, v in self.__dict__.items():
            if isinstance(v, CfgNode):
                parts.append("%s:\n" % k)
                parts.append(v._str_helper(indent + 1))
            else:
                parts.append("%s: %s\n" % (k, v))
        parts = [' ' * (indent * 4) + p for p in parts]
        return "".join(parts)
    
    def to_dict(self):
        return {k: v.to_dict() if isinstance(v, CfgNode) else v 
                for k, v in self.__dict__.items()}
    
    def merge_from_dict(self, d):
        self.__dict__.update(d)
    
    def merge_from_args(self, args):
        for arg in args:
            if not arg.startswith('--'):
                continue
            
            try:
                key, val = arg[2:].split('=', 1)
                keys = key.split('.')
                obj = self
                
                for k in keys[:-1]:
                    obj = getattr(obj, k)
                
                try:
                    val = literal_eval(val)
                except ValueError:
                    pass
                
                setattr(obj, keys[-1], val)
                print(f"command line overwriting config attribute {key} with {val}")
                
            except (ValueError, AttributeError) as e:
                print(f"Error processing argument {arg}: {e}")


if __name__ == "__main__":
    manager = ModelManager()
    config = manager.load_hparams("124M")
    print(f"GPT-2 124M config: {config}")
    
    from model import create_gpt_model
    model = create_gpt_model("gpt2", vocab_size=50257, block_size=1024)
    rng = random.PRNGKey(0)
    variables = model.init(rng, jnp.ones((1, 10), dtype=jnp.int32))
    stats = count_parameters(variables["params"])
    print(f"Model statistics: {stats}")
