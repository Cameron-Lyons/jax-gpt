#!/usr/bin/env python3
"""
Test script for parameter conversion between functional and Flax formats.
"""

import jax
import jax.numpy as jnp
from flax_gpt2 import create_gpt2_model, get_model_config
from parameter_converter import (
    convert_functional_to_flax_params,
    convert_flax_to_functional_params,
    print_parameter_summary
)


def test_parameter_conversion():
    """Test parameter conversion between formats."""
    print("🧪 Testing Parameter Conversion")
    print("=" * 50)
    
    config = get_model_config("124M")
    config.vocab_size = 1000
    model = create_gpt2_model(config)
    
    dummy_input = jnp.ones((2, 64), dtype=jnp.int32)
    
    variables = model.init(jax.random.PRNGKey(0), dummy_input, training=False)
    flax_params = variables["params"]
    
    print("✅ Flax model initialized successfully")
    print_parameter_summary(flax_params, "Flax")
    
    print("\n🔄 Converting Flax to functional format...")
    functional_params = convert_flax_to_functional_params(flax_params)
    print_parameter_summary(functional_params, "Functional")
    
    print("\n🔄 Converting back to Flax format...")
    converted_flax_params = convert_functional_to_flax_params(
        functional_params, model, dummy_input
    )
    print_parameter_summary(converted_flax_params, "Converted Flax")
    
    print("\n🧪 Testing model outputs...")
    
    original_output = model.apply({"params": flax_params}, dummy_input, training=False)
    
    converted_output = model.apply({"params": converted_flax_params}, dummy_input, training=False)
    
    diff = jnp.abs(original_output - converted_output).max()
    print(f"Maximum difference between outputs: {diff}")
    
    if diff < 1e-5:
        print("✅ Parameter conversion successful! Outputs match.")
    else:
        print("❌ Parameter conversion failed! Outputs don't match.")
    
    return diff < 1e-5


def test_pretrained_loading():
    """Test loading pretrained weights (if available)."""
    print("\n🔍 Testing Pretrained Weight Loading")
    print("=" * 50)
    
    try:
        from utils import load_encoder_hparams_and_params
        
        _, hparams, pretrained_params = load_encoder_hparams_and_params("124M", "models")
        
        print("✅ Pretrained weights loaded successfully")
        print(f"Model hyperparameters: {hparams}")
        
        config = get_model_config("124M")
        model = create_gpt2_model(config)
        
        dummy_input = jnp.ones((1, 64), dtype=jnp.int32)
        
        print("🔄 Converting pretrained parameters to Flax format...")
        flax_params = convert_functional_to_flax_params(
            pretrained_params, model, dummy_input
        )
        
        print("✅ Parameter conversion completed")
        print_parameter_summary(flax_params, "Converted Pretrained")
        
        return True
        
    except FileNotFoundError:
        print("⚠️  Pretrained weights not found in 'models' directory")
        print("   This is expected if you haven't downloaded the weights yet")
        return False
    except Exception as e:
        print(f"❌ Error loading pretrained weights: {e}")
        return False


if __name__ == "__main__":
    print("🚀 JAX GPT-2 Parameter Conversion Test")
    print("=" * 60)
    
    success = test_parameter_conversion()
    
    pretrained_success = test_pretrained_loading()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")
    
    if pretrained_success:
        print("✅ Pretrained weight loading works!")
    else:
        print("⚠️  Pretrained weight loading not tested (weights not available)") 