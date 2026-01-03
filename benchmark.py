"""
Comprehensive benchmarking suite for JAX GPT-2 models.
Supports multiple scenarios, performance analysis, and detailed metrics.
"""
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Literal
from dataclasses import dataclass
from contextlib import contextmanager
import statistics

import jax
import jax.numpy as jnp
from jax import random, jit, grad
import optax

from model import GPTConfig, GPT

# Model configurations for different sizes
MODEL_CONFIGS = {
    "124M": {"n_layer": 12, "n_head": 12, "n_embd": 768},
    "355M": {"n_layer": 24, "n_head": 16, "n_embd": 1024},
    "774M": {"n_layer": 36, "n_head": 20, "n_embd": 1280},
    "1558M": {"n_layer": 48, "n_head": 25, "n_embd": 1600},
}


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking runs."""
    model_size: Literal["124M", "355M", "774M", "1558M"] = "124M"
    batch_size: int = 8
    block_size: int = 1024
    vocab_size: int = 50257
    
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 10
    benchmark_steps: int = 50
    
    device: Literal["cpu", "gpu", "tpu"] = "gpu"
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    compile: bool = True
    profile: bool = False
    
    use_real_data: bool = True
    data_dir: str = "data"
    dataset: str = "openwebtext"
    
    measure_memory: bool = True
    measure_throughput: bool = True
    measure_latency: bool = True
    measure_mfu: bool = True
    num_runs: int = 3
    
    output_dir: str = "benchmarks"
    save_results: bool = True
    verbose: bool = True


class BenchmarkRunner:
    """Comprehensive benchmarking runner for GPT-2 models."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results = {}
        
        self.setup_device()
        
        self.setup_model()
        self.setup_data()
        self.setup_optimizer()
        
        if self.config.save_results:
            self.output_dir = Path(self.config.output_dir)
            self.output_dir.mkdir(exist_ok=True)
    
    def setup_device(self):
        """Setup JAX device configuration."""
        if self.config.device == "gpu":
            # Ensure GPU is available
            if not jax.devices("gpu"):
                print("Warning: GPU not available, falling back to CPU")
                self.config.device = "cpu"
        
        # Set default device
        self.device = jax.devices(self.config.device)[0]
        print(f"Using device: {self.device}")
        
        # Set dtype
        self.dtype = getattr(jnp, self.config.dtype)
        print(f"Using dtype: {self.config.dtype}")
    
    def setup_model(self):
        """Initialize the GPT-2 model."""
        hparams = MODEL_CONFIGS[self.config.model_size]

        self.model_config = GPTConfig(
            n_layer=hparams["n_layer"],
            n_head=hparams["n_head"],
            n_embd=hparams["n_embd"],
            vocab_size=self.config.vocab_size,
            block_size=self.config.block_size,
            embd_pdrop=0.0,
            resid_pdrop=0.0,
            attn_pdrop=0.0,
            use_bias=True,
            dtype=self.config.dtype,
        )
        
        self.model = GPT(self.model_config)
        
        self.rng = random.PRNGKey(0)
        self.rng, init_rng = random.split(self.rng)
        
        dummy_input = jnp.ones((1, self.config.block_size), dtype=jnp.int32)
        self.variables = self.model.init(init_rng, dummy_input)
        
        print(f"Model initialized: {self.config.model_size}")
        print(f"Parameters: {self.count_parameters():,}")
    
    def setup_data(self):
        """Setup data loading for benchmarking."""
        if self.config.use_real_data:
            self.setup_real_data()
        else:
            self.setup_synthetic_data()
    
    def setup_real_data(self):
        """Setup real data loading."""
        data_path = Path(self.config.data_dir) / self.config.dataset
        
        if data_path.exists():
            self.train_data = jnp.fromfile(data_path, dtype=jnp.uint16)
            print(f"Loaded real data: {len(self.train_data):,} tokens")
        else:
            print("Warning: Real data not found, falling back to synthetic data")
            self.config.use_real_data = False
            self.setup_synthetic_data()
    
    def setup_synthetic_data(self):
        """Setup synthetic data generation."""
        self.rng, data_rng = random.split(self.rng)
        
        self.train_data = random.randint(
            data_rng,
            shape=(1000000,),
            minval=0,
            maxval=self.config.vocab_size,
            dtype=jnp.uint16
        )
        print("Using synthetic data for benchmarking")
    
    def setup_optimizer(self):
        """Setup optimizer for training benchmarks."""
        self.optimizer = optax.adamw(
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        self.opt_state = self.optimizer.init(self.variables["params"])
        
        self.create_training_step()
    
    def create_training_step(self):
        """Create JIT-compiled training step function."""
        def loss_fn(params, batch):
            x, y = batch
            logits, loss = self.model.apply(
                {"params": params}, x, targets=y, training=True
            )
            return loss, logits
        
        def train_step(params, opt_state, batch):
            (loss, logits), grads = grad(loss_fn, has_aux=True)(params, batch)
            updates, new_opt_state = self.optimizer.update(grads, opt_state)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_state, loss
        
        if self.config.compile:
            self.train_step = jit(train_step)
        else:
            self.train_step = train_step
    
    def get_batch(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Get a random batch of data."""
        self.rng, batch_rng = random.split(self.rng)
        
        ix = random.randint(
            batch_rng,
            (self.config.batch_size,),
            0,
            len(self.train_data) - self.config.block_size
        )
        
        x = jnp.stack([
            self.train_data[i:i + self.config.block_size].astype(jnp.int32)
            for i in ix
        ])
        y = jnp.stack([
            self.train_data[i + 1:i + 1 + self.config.block_size].astype(jnp.int32)
            for i in ix
        ])
        
        return x, y
    
    def count_parameters(self) -> int:
        """Count model parameters."""
        total_params = 0
        
        def count_dict(d):
            nonlocal total_params
            for value in d.values():
                if isinstance(value, dict):
                    count_dict(value)
                elif isinstance(value, jnp.ndarray):
                    total_params += value.size
        
        count_dict(self.variables["params"])
        return total_params
    
    @contextmanager
    def measure_time(self, name: str):
        """Context manager for measuring execution time."""
        start_time = time.time()
        start_memory = self.get_memory_usage() if self.config.measure_memory else None
        
        yield
        
        end_time = time.time()
        end_memory = self.get_memory_usage() if self.config.measure_memory else None
        
        duration = end_time - start_time
        memory_used = end_memory - start_memory if start_memory and end_memory else None
        
        self.results[f"{name}_time"] = duration
        if memory_used:
            self.results[f"{name}_memory"] = memory_used
        
        if self.config.verbose:
            print(f"{name}: {duration:.4f}s")
            if memory_used:
                print(f"{name} memory: {memory_used:.2f}MB")
    
    def get_memory_usage(self) -> Optional[float]:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return None
    
    def benchmark_forward_pass(self) -> Dict[str, Any]:
        """Benchmark forward pass performance."""
        print("\n=== Forward Pass Benchmark ===")
        
        latencies = []
        throughputs = []
        
        for run in range(self.config.num_runs):
            for _ in range(self.config.warmup_steps):
                batch = self.get_batch()
                _ = self.model.apply({"params": self.variables["params"]}, batch[0])
            
            batch = self.get_batch()
            
            with self.measure_time(f"forward_pass_run_{run}"):
                for _ in range(self.config.benchmark_steps):
                    start_time = time.time()
                    _ = self.model.apply({"params": self.variables["params"]}, batch[0])
                    end_time = time.time()
                    
                    latency = (end_time - start_time) * 1000
                    throughput = self.config.batch_size / (end_time - start_time)
                    
                    latencies.append(latency)
                    throughputs.append(throughput)
            
            batch = self.get_batch()
        
        results = {
            "latency_ms": {
                "mean": statistics.mean(latencies),
                "median": statistics.median(latencies),
                "std": statistics.stdev(latencies) if len(latencies) > 1 else 0,
                "min": min(latencies),
                "max": max(latencies)
            },
            "throughput_samples_per_sec": {
                "mean": statistics.mean(throughputs),
                "median": statistics.median(throughputs),
                "std": statistics.stdev(throughputs) if len(throughputs) > 1 else 0,
                "min": min(throughputs),
                "max": max(throughputs)
            }
        }
        
        if self.config.verbose:
            print(f"Latency: {results['latency_ms']['mean']:.2f} ± {results['latency_ms']['std']:.2f} ms")
            print(f"Throughput: {results['throughput_samples_per_sec']['mean']:.2f} samples/sec")
        
        return results
    
    def benchmark_training_step(self) -> Dict[str, Any]:
        """Benchmark training step performance."""
        print("\n=== Training Step Benchmark ===")
        
        latencies = []
        throughputs = []
        
        params = self.variables["params"]
        opt_state = self.opt_state
        
        for run in range(self.config.num_runs):
            for _ in range(self.config.warmup_steps):
                batch = self.get_batch()
                params, opt_state, _ = self.train_step(params, opt_state, batch)
            
            batch = self.get_batch()
            
            with self.measure_time(f"training_step_run_{run}"):
                for _ in range(self.config.benchmark_steps):
                    start_time = time.time()
                    params, opt_state, loss = self.train_step(params, opt_state, batch)
                    end_time = time.time()
                    
                    latency = (end_time - start_time) * 1000
                    throughput = self.config.batch_size / (end_time - start_time)
                    
                    latencies.append(latency)
                    throughputs.append(throughput)
            
            batch = self.get_batch()
        
        results = {
            "latency_ms": {
                "mean": statistics.mean(latencies),
                "median": statistics.median(latencies),
                "std": statistics.stdev(latencies) if len(latencies) > 1 else 0,
                "min": min(latencies),
                "max": max(latencies)
            },
            "throughput_samples_per_sec": {
                "mean": statistics.mean(throughputs),
                "median": statistics.median(throughputs),
                "std": statistics.stdev(throughputs) if len(throughputs) > 1 else 0,
                "min": min(throughputs),
                "max": max(throughputs)
            }
        }
        
        if self.config.verbose:
            print(f"Latency: {results['latency_ms']['mean']:.2f} ± {results['latency_ms']['std']:.2f} ms")
            print(f"Throughput: {results['throughput_samples_per_sec']['mean']:.2f} samples/sec")
        
        return results
    
    def benchmark_memory_usage(self) -> Dict[str, Any]:
        """Benchmark memory usage patterns."""
        if not self.config.measure_memory:
            return {}
        
        print("\n=== Memory Usage Benchmark ===")
        
        memory_usage = []
        
        baseline_memory = self.get_memory_usage()
        
        for _ in range(self.config.benchmark_steps):
            batch = self.get_batch()
            current_memory = self.get_memory_usage()
            _ = self.model.apply({"params": self.variables["params"]}, batch[0])
            peak_memory = self.get_memory_usage()
            
            memory_usage.append({
                "baseline": baseline_memory,
                "current": current_memory,
                "peak": peak_memory,
                "delta": peak_memory - current_memory if current_memory and peak_memory else 0
            })
        
        deltas = [m["delta"] for m in memory_usage if m["delta"] is not None]
        
        results = {
            "baseline_memory_mb": baseline_memory,
            "peak_memory_mb": max(m["peak"] for m in memory_usage if m["peak"]),
            "memory_delta_mb": {
                "mean": statistics.mean(deltas) if deltas else 0,
                "median": statistics.median(deltas) if deltas else 0,
                "std": statistics.stdev(deltas) if len(deltas) > 1 else 0,
                "max": max(deltas) if deltas else 0
            }
        }
        
        if self.config.verbose:
            print(f"Baseline memory: {results['baseline_memory_mb']:.2f} MB")
            print(f"Peak memory: {results['peak_memory_mb']:.2f} MB")
            print(f"Memory delta: {results['memory_delta_mb']['mean']:.2f} ± {results['memory_delta_mb']['std']:.2f} MB")
        
        return results
    
    def calculate_mfu(self, throughput: float) -> float:
        """Calculate Model FLOPs Utilization (MFU)."""
        n_layer = self.model_config.n_layer
        n_head = self.model_config.n_head
        n_embd = self.model_config.n_embd
        vocab_size = self.model_config.vocab_size
        block_size = self.config.block_size
        
        flops_per_token = (
            2 * n_layer * (12 * n_embd * n_embd + 2 * n_embd * vocab_size)
        )
        
        peak_flops = 312e12
        
        actual_flops = flops_per_token * throughput
        
        mfu = actual_flops / peak_flops
        
        return mfu
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive benchmarking suite."""
        print(f"\n{'='*60}")
        print(f"Comprehensive Benchmark: {self.config.model_size}")
        print(f"Batch size: {self.config.batch_size}, Block size: {self.config.block_size}")
        print(f"Device: {self.config.device}, Dtype: {self.config.dtype}")
        print(f"Compile: {self.config.compile}")
        print(f"{'='*60}")
        
        forward_results = self.benchmark_forward_pass()
        training_results = self.benchmark_training_step()
        memory_results = self.benchmark_memory_usage()
        
        if self.config.measure_mfu:
            forward_mfu = self.calculate_mfu(forward_results["throughput_samples_per_sec"]["mean"])
            training_mfu = self.calculate_mfu(training_results["throughput_samples_per_sec"]["mean"])
            
            mfu_results = {
                "forward_mfu": forward_mfu,
                "training_mfu": training_mfu
            }
            
            if self.config.verbose:
                print(f"\nModel FLOPs Utilization (MFU):")
                print(f"Forward pass: {forward_mfu*100:.2f}%")
                print(f"Training step: {training_mfu*100:.2f}%")
        else:
            mfu_results = {}
        
        self.results = {
            "config": self.config.__dict__,
            "model_info": {
                "model_size": self.config.model_size,
                "parameters": self.count_parameters(),
                "parameters_mb": self.count_parameters() * 4 / (1024 * 1024)
            },
            "forward_pass": forward_results,
            "training_step": training_results,
            "memory_usage": memory_results,
            "mfu": mfu_results,
            "timestamp": time.time()
        }
        
        if self.config.save_results:
            self.save_results()
        
        return self.results
    
    def save_results(self):
        """Save benchmark results to file."""
        timestamp = int(time.time())
        filename = f"benchmark_{self.config.model_size}_{timestamp}.json"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nResults saved to: {filepath}")


def main():
    """Main benchmarking function."""
    parser = argparse.ArgumentParser(description="Benchmark JAX GPT-2 models")
    
    parser.add_argument("--model_size", type=str, default="124M",
                       choices=["124M", "355M", "774M", "1558M"],
                       help="GPT-2 model size")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for benchmarking")
    parser.add_argument("--block_size", type=int, default=1024,
                       help="Sequence length for benchmarking")
    
    parser.add_argument("--device", type=str, default="gpu",
                       choices=["cpu", "gpu", "tpu"],
                       help="Device to run benchmarks on")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                       choices=["bfloat16", "float16", "float32"],
                       help="Data type for computation")
    parser.add_argument("--compile", action="store_true", default=True,
                       help="Use JIT compilation")
    parser.add_argument("--no_compile", dest="compile", action="store_false",
                       help="Disable JIT compilation")
    
    parser.add_argument("--warmup_steps", type=int, default=10,
                       help="Number of warmup steps")
    parser.add_argument("--benchmark_steps", type=int, default=50,
                       help="Number of benchmark steps")
    parser.add_argument("--num_runs", type=int, default=3,
                       help="Number of benchmark runs")
    
    parser.add_argument("--use_real_data", action="store_true", default=True,
                       help="Use real data for benchmarking")
    parser.add_argument("--use_synthetic_data", dest="use_real_data", action="store_false",
                       help="Use synthetic data for benchmarking")
    parser.add_argument("--data_dir", type=str, default="data",
                       help="Directory containing training data")
    
    parser.add_argument("--output_dir", type=str, default="benchmarks",
                       help="Directory to save benchmark results")
    parser.add_argument("--no_save", dest="save_results", action="store_false",
                       help="Don't save benchmark results")
    parser.add_argument("--verbose", action="store_true", default=True,
                       help="Verbose output")
    parser.add_argument("--quiet", dest="verbose", action="store_false",
                       help="Quiet output")
    
    args = parser.parse_args()
    
    config = BenchmarkConfig(
        model_size=args.model_size,
        batch_size=args.batch_size,
        block_size=args.block_size,
        device=args.device,
        dtype=args.dtype,
        compile=args.compile,
        warmup_steps=args.warmup_steps,
        benchmark_steps=args.benchmark_steps,
        num_runs=args.num_runs,
        use_real_data=args.use_real_data,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        save_results=args.save_results,
        verbose=args.verbose
    )
    
    runner = BenchmarkRunner(config)
    results = runner.run_comprehensive_benchmark()
    
    if config.verbose:
        print(f"\n{'='*60}")
        print("BENCHMARK SUMMARY")
        print(f"{'='*60}")
        print(f"Model: {config.model_size}")
        print(f"Parameters: {results['model_info']['parameters']:,}")
        print(f"Forward latency: {results['forward_pass']['latency_ms']['mean']:.2f} ms")
        print(f"Training latency: {results['training_step']['latency_ms']['mean']:.2f} ms")
        print(f"Forward throughput: {results['forward_pass']['throughput_samples_per_sec']['mean']:.2f} samples/sec")
        print(f"Training throughput: {results['training_step']['throughput_samples_per_sec']['mean']:.2f} samples/sec")
        
        if "mfu" in results and results["mfu"]:
            print(f"Forward MFU: {results['mfu']['forward_mfu']*100:.2f}%")
            print(f"Training MFU: {results['mfu']['training_mfu']*100:.2f}%")


if __name__ == "__main__":
    main()
