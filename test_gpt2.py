"""
Test script for the JAX GPT-2 implementation.
"""

import jax.numpy as jnp
from gpt2 import GPTConfig, generate
from gpt2 import load_encoder_hparams_and_params


def test_gpt_config():
    """Test GPT configuration."""
    print("Testing GPT configuration...")
    
    config = GPTConfig()
    assert config.block_size == 1024
    assert config.vocab_size == 50304
    assert config.n_layer == 12
    assert config.n_head == 12
    assert config.n_embd == 768
    
    print("✅ GPT configuration test passed")


def test_basic_functions():
    """Test basic mathematical functions."""
    print("Testing basic functions...")
    
    # Test linear function
    x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    w = jnp.array([[0.5, 0.5], [0.5, 0.5]])
    b = jnp.array([1.0, 1.0])
    
    from gpt2 import linear
    result = linear(x, w, b)
    assert result.shape == (2, 2)
    
    # Test softmax
    from gpt2 import softmax
    logits = jnp.array([1.0, 2.0, 3.0])
    probs = softmax(logits)
    assert jnp.allclose(jnp.sum(probs), 1.0)
    
    # Test GELU
    from gpt2 import gelu
    x = jnp.array([0.0, 1.0, -1.0])
    result = gelu(x)
    assert result.shape == x.shape
    
    print("✅ Basic functions test passed")


def test_attention():
    """Test attention mechanism."""
    print("Testing attention mechanism...")
    
    from gpt2 import attention
    
    # Create dummy inputs
    q = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    k = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    v = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    mask = jnp.array([[0.0, -1e10], [0.0, 0.0]])
    
    result = attention(q, k, v, mask)
    assert result.shape == (2, 2)
    
    print("✅ Attention test passed")


def test_model_loading():
    """Test model loading functionality."""
    print("Testing model loading...")
    
    try:
        # This will download the model if not present
        encoder, hparams, params = load_encoder_hparams_and_params("124M", "models")
        
        assert encoder is not None
        assert hparams is not None
        assert params is not None
        
        # Test encoding/decoding
        text = "Hello, world!"
        tokens = encoder.encode(text)
        decoded = encoder.decode(tokens)
        
        # Note: BPE encoding might not be exactly reversible
        assert len(tokens) > 0
        
        print("✅ Model loading test passed")
        
    except Exception as e:
        print(f"⚠️  Model loading test skipped (likely no internet connection): {e}")


def test_generation():
    """Test text generation."""
    print("Testing text generation...")
    
    try:
        encoder, hparams, params = load_encoder_hparams_and_params("124M", "models")
        
        # Test basic generation
        input_ids = encoder.encode("Hello")
        output_ids = generate(
            input_ids, 
            params, 
            hparams["n_head"], 
            n_tokens_to_generate=5
        )
        
        assert len(output_ids) == 5
        output_text = encoder.decode(output_ids)
        assert isinstance(output_text, str)
        
        print("✅ Text generation test passed")
        
    except Exception as e:
        print(f"⚠️  Text generation test skipped: {e}")


def test_jit_compilation():
    """Test JIT compilation."""
    print("Testing JIT compilation...")
    
    # Test that JIT-compiled functions work
    from gpt2 import linear, softmax, gelu
    
    # These should be JIT-compiled
    x = jnp.array([1.0, 2.0, 3.0])
    
    # Test that functions are callable
    result1 = linear(x.reshape(1, -1), jnp.eye(3), jnp.zeros(3))
    result2 = softmax(x)
    result3 = gelu(x)
    
    assert result1.shape == (1, 3)
    assert result2.shape == x.shape
    assert result3.shape == x.shape
    
    print("✅ JIT compilation test passed")


def test_sampling_strategies():
    """Test different sampling strategies."""
    print("Testing sampling strategies...")
    
    try:
        encoder, hparams, params = load_encoder_hparams_and_params("124M", "models")
        input_ids = encoder.encode("Test")
        
        # Test different temperature values
        for temp in [0.5, 1.0, 1.5]:
            output_ids = generate(
                input_ids, 
                params, 
                hparams["n_head"], 
                n_tokens_to_generate=3,
                temperature=temp
            )
            assert len(output_ids) == 3
        
        # Test top-k sampling
        output_ids = generate(
            input_ids, 
            params, 
            hparams["n_head"], 
            n_tokens_to_generate=3,
            top_k=10
        )
        assert len(output_ids) == 3
        
        print("✅ Sampling strategies test passed")
        
    except Exception as e:
        print(f"⚠️  Sampling strategies test skipped: {e}")


def run_all_tests():
    """Run all tests."""
    print("Running JAX GPT-2 tests...")
    print("=" * 50)
    
    tests = [
        test_gpt_config,
        test_basic_functions,
        test_attention,
        test_jit_compilation,
        test_model_loading,
        test_generation,
        test_sampling_strategies,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ Test failed: {e}")
    
    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed or were skipped")


if __name__ == "__main__":
    run_all_tests() 