"""Tests for the benchmark module."""

import jax.numpy as jnp
import pytest

from benchmark import BenchmarkConfig, BenchmarkRunner


@pytest.fixture
def small_benchmark_config():
    return BenchmarkConfig(
        model_size="124M",
        batch_size=2,
        block_size=16,
        vocab_size=64,
        warmup_steps=1,
        benchmark_steps=2,
        num_runs=1,
        device="cpu",
        dtype="float32",
        compile=False,
        use_real_data=False,
        measure_memory=False,
        save_results=False,
        verbose=False,
    )


class TestBenchmarkConfig:
    def test_defaults(self):
        config = BenchmarkConfig()
        assert config.model_size == "124M"
        assert config.batch_size == 8
        assert config.peak_flops is None

    def test_peak_flops_override(self):
        config = BenchmarkConfig(peak_flops=200e12)
        assert config.peak_flops == 200e12


class TestBenchmarkRunner:
    def test_init(self, small_benchmark_config):
        runner = BenchmarkRunner(small_benchmark_config)
        assert runner.variables is not None
        assert runner.count_parameters() > 0

    def test_get_batch(self, small_benchmark_config):
        runner = BenchmarkRunner(small_benchmark_config)
        x, y = runner.get_batch()
        assert x.shape == (2, 16)
        assert y.shape == (2, 16)

    def test_get_batch_fresh_each_call(self, small_benchmark_config):
        runner = BenchmarkRunner(small_benchmark_config)
        x1, _ = runner.get_batch()
        x2, _ = runner.get_batch()
        assert not jnp.array_equal(x1, x2)

    def test_count_parameters(self, small_benchmark_config):
        runner = BenchmarkRunner(small_benchmark_config)
        n_params = runner.count_parameters()
        assert n_params > 0


class TestMFU:
    def test_mfu_returns_float(self, small_benchmark_config):
        runner = BenchmarkRunner(small_benchmark_config)
        mfu = runner.calculate_mfu(1.0, forward_only=False)
        assert isinstance(mfu, float)
        assert mfu >= 0

    def test_forward_vs_training_mfu(self, small_benchmark_config):
        runner = BenchmarkRunner(small_benchmark_config)
        forward_mfu = runner.calculate_mfu(1.0, forward_only=True)
        training_mfu = runner.calculate_mfu(1.0, forward_only=False)
        assert training_mfu == pytest.approx(3.0 * forward_mfu)

    def test_mfu_scales_with_throughput(self, small_benchmark_config):
        runner = BenchmarkRunner(small_benchmark_config)
        mfu1 = runner.calculate_mfu(1.0)
        mfu2 = runner.calculate_mfu(2.0)
        assert mfu2 == pytest.approx(2.0 * mfu1)

    def test_custom_peak_flops(self):
        config = BenchmarkConfig(
            model_size="124M",
            batch_size=2,
            block_size=16,
            vocab_size=64,
            device="cpu",
            dtype="float32",
            compile=False,
            use_real_data=False,
            save_results=False,
            verbose=False,
            peak_flops=100e12,
        )
        runner = BenchmarkRunner(config)
        mfu = runner.calculate_mfu(1.0)
        assert mfu > 0


class TestBenchmarkForward:
    def test_forward_pass_benchmark(self, small_benchmark_config):
        runner = BenchmarkRunner(small_benchmark_config)
        results = runner.benchmark_forward_pass()
        assert "latency_ms" in results
        assert "throughput_samples_per_sec" in results
        assert results["latency_ms"]["mean"] > 0
        assert results["throughput_samples_per_sec"]["mean"] > 0


class TestBenchmarkTraining:
    def test_training_step_benchmark(self, small_benchmark_config):
        runner = BenchmarkRunner(small_benchmark_config)
        results = runner.benchmark_training_step()
        assert "latency_ms" in results
        assert "throughput_samples_per_sec" in results
        assert results["latency_ms"]["mean"] > 0
