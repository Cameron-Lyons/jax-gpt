"""
Parameter conversion utilities for mapping between functional and Flax GPT-2 parameter formats.
"""

from typing import Dict, Any
import jax
import jax.numpy as jnp
from flax.training import train_state
from flax_gpt2 import GPT2


def convert_functional_to_flax_params(
    functional_params: Dict[str, Any], 
    model: GPT2,
    dummy_input: jax.Array
) -> Dict[str, Any]:
    """
    Convert parameters from functional format to Flax format.
    
    Args:
        functional_params: Parameters in functional format (from utils.load_encoder_hparams_and_params)
        model: Flax GPT-2 model instance
        dummy_input: Dummy input for model initialization
    
    Returns:
        Flax-formatted parameters
    """
    
    variables = model.init(jax.random.PRNGKey(0), dummy_input, training=False)
    flax_params = variables["params"]
    
    converted_params = {}
    
    if "wte" in functional_params:
        converted_params["Embed_0"] = {
            "embedding": functional_params["wte"]
        }
    
    if "wpe" in functional_params:
        converted_params["Embed_1"] = {
            "embedding": functional_params["wpe"]
        }
    
    if "blocks" in functional_params:
        for i, block in enumerate(functional_params["blocks"]):
            block_key = f"TransformerBlock_{i}"
            converted_params[block_key] = {}
            
            if "ln_1" in block:
                converted_params[block_key]["LayerNorm_0"] = {
                    "scale": block["ln_1"]["g"],
                    "bias": block["ln_1"]["b"]
                }
            
            if "attn" in block:
                attn_key = f"{block_key}_MultiHeadAttention_0"
                converted_params[attn_key] = {}
                
                if "c_attn" in block["attn"]:
                    qkv_w = block["attn"]["c_attn"]["w"]
                    qkv_b = block["attn"]["c_attn"]["b"]
                    
                    dim = qkv_w.shape[1] // 3
                    converted_params[attn_key]["Dense_0"] = {
                        "kernel": qkv_w,
                        "bias": qkv_b
                    }
                
                if "c_proj" in block["attn"]:
                    converted_params[attn_key]["Dense_1"] = {
                        "kernel": block["attn"]["c_proj"]["w"],
                        "bias": block["attn"]["c_proj"]["b"]
                    }
            
            if "ln_2" in block:
                converted_params[block_key]["LayerNorm_1"] = {
                    "scale": block["ln_2"]["g"],
                    "bias": block["ln_2"]["b"]
                }
            
            if "mlp" in block:
                ff_key = f"{block_key}_FeedForward_0"
                converted_params[ff_key] = {}
                
                if "c_fc" in block["mlp"]:
                    converted_params[ff_key]["Dense_0"] = {
                        "kernel": block["mlp"]["c_fc"]["w"],
                        "bias": block["mlp"]["c_fc"]["b"]
                    }
                
                if "c_proj" in block["mlp"]:
                    converted_params[ff_key]["Dense_1"] = {
                        "kernel": block["mlp"]["c_proj"]["w"],
                        "bias": block["mlp"]["c_proj"]["b"]
                    }
    
    if "ln_f" in functional_params:
        converted_params["LayerNorm_0"] = {
            "scale": functional_params["ln_f"]["g"],
            "bias": functional_params["ln_f"]["b"]
        }
    
    if "wte" in functional_params:
        converted_params["Dense_0"] = {
            "kernel": functional_params["wte"].T  # Transpose for output projection
        }
    
    return converted_params


def convert_flax_to_functional_params(
    flax_params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Convert parameters from Flax format to functional format.
    
    Args:
        flax_params: Parameters in Flax format
    
    Returns:
        Functional-formatted parameters
    """
    
    functional_params = {}
    
    if "Embed_0" in flax_params:
        functional_params["wte"] = flax_params["Embed_0"]["embedding"]
    
    if "Embed_1" in flax_params:
        functional_params["wpe"] = flax_params["Embed_1"]["embedding"]
    
    blocks = []
    block_indices = []
    
    for key in flax_params.keys():
        if key.startswith("TransformerBlock_"):
            block_idx = int(key.split("_")[1])
            block_indices.append(block_idx)
    
    block_indices.sort()
    
    for block_idx in block_indices:
        block_key = f"TransformerBlock_{block_idx}"
        block = {}
        
        if f"{block_key}_LayerNorm_0" in flax_params:
            ln_0 = flax_params[f"{block_key}_LayerNorm_0"]
            block["ln_1"] = {
                "g": ln_0["scale"],
                "b": ln_0["bias"]
            }
        
        attn_key = f"{block_key}_MultiHeadAttention_0"
        if attn_key in flax_params:
            attn = {}

            if f"{attn_key}_Dense_0" in flax_params:
                dense_0 = flax_params[f"{attn_key}_Dense_0"]
                attn["c_attn"] = {
                    "w": dense_0["kernel"],
                    "b": dense_0["bias"]
                }
            
            if f"{attn_key}_Dense_1" in flax_params:
                dense_1 = flax_params[f"{attn_key}_Dense_1"]
                attn["c_proj"] = {
                    "w": dense_1["kernel"],
                    "b": dense_1["bias"]
                }
            
            block["attn"] = attn
        
        if f"{block_key}_LayerNorm_1" in flax_params:
            ln_1 = flax_params[f"{block_key}_LayerNorm_1"]
            block["ln_2"] = {
                "g": ln_1["scale"],
                "b": ln_1["bias"]
            }
        
        ff_key = f"{block_key}_FeedForward_0"
        if ff_key in flax_params:
            mlp = {}
            
            if f"{ff_key}_Dense_0" in flax_params:
                dense_0 = flax_params[f"{ff_key}_Dense_0"]
                mlp["c_fc"] = {
                    "w": dense_0["kernel"],
                    "b": dense_0["bias"]
                }
            
            if f"{ff_key}_Dense_1" in flax_params:
                dense_1 = flax_params[f"{ff_key}_Dense_1"]
                mlp["c_proj"] = {
                    "w": dense_1["kernel"],
                    "b": dense_1["bias"]
                }
            
            block["mlp"] = mlp
        
        blocks.append(block)
    
    functional_params["blocks"] = blocks
    
    if "LayerNorm_0" in flax_params:
        ln_f = flax_params["LayerNorm_0"]
        functional_params["ln_f"] = {
            "g": ln_f["scale"],
            "b": ln_f["bias"]
        }
    
    return functional_params


def initialize_model_with_pretrained_weights(
    model: GPT2,
    pretrained_params: Dict[str, Any],
    dummy_input: jax.Array
) -> train_state.TrainState:
    """
    Initialize a Flax model with pretrained weights.
    
    Args:
        model: Flax GPT-2 model
        pretrained_params: Pretrained parameters in functional format
        dummy_input: Dummy input for model initialization
    
    Returns:
        TrainState with pretrained parameters
    """
    
    flax_params = convert_functional_to_flax_params(pretrained_params, model, dummy_input)
    
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=flax_params,
        tx=None
    )
    
    return state


def verify_parameter_conversion(
    functional_params: Dict[str, Any],
    flax_params: Dict[str, Any],
    model: GPT2,
    dummy_input: jax.Array
) -> bool:
    """
    Verify that parameter conversion is correct by comparing outputs.
    
    Args:
        functional_params: Original functional parameters
        flax_params: Converted Flax parameters
        model: Flax model
        dummy_input: Test input
    
    Returns:
        True if outputs match, False otherwise
    """
    
    from gpt2 import gpt2
    functional_output = gpt2(
        dummy_input, 
        **functional_params, 
        n_head=12
    )
    
    flax_output = model.apply({"params": flax_params}, dummy_input, training=False)
    
    diff = jnp.abs(functional_output - flax_output).max()
    print(f"Maximum difference between outputs: {diff}")
    
    return diff < 1e-5


def print_parameter_summary(params: Dict[str, Any], format_name: str):
    """Print a summary of parameters."""
    print(f"\n{format_name} Parameter Summary:")
    print("=" * 40)
    
    total_params = 0
    for key, value in params.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                if hasattr(subvalue, 'shape'):
                    param_count = subvalue.size
                    total_params += param_count
                    print(f"  {key}.{subkey}: {subvalue.shape} ({param_count:,} params)")
        elif hasattr(value, 'shape'):
            param_count = value.size
            total_params += param_count
            print(f"  {key}: {value.shape} ({param_count:,} params)")
    
    print(f"Total parameters: {total_params:,}")
    print(f"Model size: {total_params * 4 / 1e6:.1f} MB") 
