#!/usr/bin/env python3
"""Gradient Checkpointing Benchmark.

This script benchmarks the memory and compute trade-off of gradient checkpointing.
Typical results:
- Memory savings: 30-50%
- Compute overhead: 20-30%

Usage:
    python benchmarks/checkpointing_benchmark.py --all
    python benchmarks/checkpointing_benchmark.py --memory
    python benchmarks/checkpointing_benchmark.py --speed
"""

import argparse
import gc
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn

try:
    from src.model import SimpleTransformer
    from src.training.base_trainer import TrainingConfig
except ImportError:
    SimpleTransformer = None
    TrainingConfig = None


def get_memory_info() -> Dict[str, float]:
    """Get current memory usage information."""
    info = {}
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        info["gpu_allocated_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
        info["gpu_reserved_mb"] = torch.cuda.memory_reserved() / 1024 / 1024
        info["gpu_max_allocated_mb"] = torch.cuda.max_memory_allocated() / 1024 / 1024
    
    try:
        import psutil
        process = psutil.Process()
        info["cpu_rss_mb"] = process.memory_info().rss / 1024 / 1024
    except ImportError:
        info["cpu_rss_mb"] = 0
    
    return info


def reset_memory_stats():
    """Reset GPU memory statistics."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    gc.collect()


def benchmark_forward_backward(
    model: nn.Module,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
    num_iterations: int = 5,
) -> Dict[str, float]:
    """Benchmark forward and backward pass times."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    times = {"forward": [], "backward": [], "total": []}
    
    for _ in range(num_iterations):
        reset_memory_stats()
        
        start = time.perf_counter()
        logits, _ = model(input_ids)
        forward_time = time.perf_counter() - start
        
        start = time.perf_counter()
        loss = criterion(
            logits.view(-1, logits.size(-1)),
            target_ids.view(-1)
        )
        loss.backward()
        backward_time = time.perf_counter() - start
        
        times["forward"].append(forward_time)
        times["backward"].append(backward_time)
        times["total"].append(forward_time + backward_time)
        
        model.zero_grad()
    
    return {
        "forward_mean_ms": sum(times["forward"]) / len(times["forward"]) * 1000,
        "backward_mean_ms": sum(times["backward"]) / len(times["backward"]) * 1000,
        "total_mean_ms": sum(times["total"]) / len(times["total"]) * 1000,
    }


def benchmark_memory(
    model: nn.Module,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
) -> Dict[str, float]:
    """Benchmark memory usage during training."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    
    reset_memory_stats()
    mem_before = get_memory_info()
    
    logits, _ = model(input_ids)
    loss = criterion(
        logits.view(-1, logits.size(-1)),
        target_ids.view(-1)
    )
    loss.backward()
    
    mem_after = get_memory_info()
    
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
    else:
        peak_memory = 0
    
    model.zero_grad()
    
    return {
        "peak_gpu_mb": peak_memory,
        "allocated_gpu_mb": mem_after.get("gpu_allocated_mb", 0) - mem_before.get("gpu_allocated_mb", 0),
    }


def run_memory_comparison(
    vocab_size: int = 5000,
    d_model: int = 256,
    nhead: int = 8,
    num_layers: int = 8,
    dim_feedforward: int = 512,
    batch_size: int = 4,
    seq_len: int = 128,
) -> Dict[str, Dict]:
    """Compare memory usage with and without gradient checkpointing."""
    print("\n" + "=" * 70)
    print("MEMORY COMPARISON: Gradient Checkpointing ON vs OFF")
    print("=" * 70)
    print(f"Model: {num_layers} layers, d_model={d_model}, heads={nhead}")
    print(f"Batch: {batch_size}, Seq Len: {seq_len}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    model_config = dict(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=0.0,
        max_len=seq_len * 2,
        mode="modern",
    )
    
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    target_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    
    results = {}
    
    for checkpointing in [False, True]:
        label = "ON (with checkpointing)" if checkpointing else "OFF (without checkpointing)"
        print(f"\n--- Testing with Gradient Checkpointing {label} ---")
        
        reset_memory_stats()
        
        model = SimpleTransformer(
            **model_config,
            gradient_checkpointing=checkpointing,
        ).to(device)
        
        mem_info = benchmark_memory(model, input_ids, target_ids)
        timing_info = benchmark_forward_backward(model, input_ids, target_ids)
        
        results[f"checkpointing_{checkpointing}"] = {
            "peak_gpu_mb": mem_info["peak_gpu_mb"],
            "forward_ms": timing_info["forward_mean_ms"],
            "backward_ms": timing_info["backward_mean_ms"],
            "total_ms": timing_info["total_mean_ms"],
        }
        
        del model
        reset_memory_stats()
    
    no_ckpt = results["checkpointing_False"]
    with_ckpt = results["checkpointing_True"]
    
    if no_ckpt["peak_gpu_mb"] > 0:
        memory_savings = (no_ckpt["peak_gpu_mb"] - with_ckpt["peak_gpu_mb"]) / no_ckpt["peak_gpu_mb"] * 100
    else:
        memory_savings = 0.0
        print("Note: GPU memory not available, memory savings cannot be calculated")
    
    if no_ckpt["total_ms"] > 0:
        time_overhead = (with_ckpt["total_ms"] - no_ckpt["total_ms"]) / no_ckpt["total_ms"] * 100
    else:
        time_overhead = 0.0
    
    print("\n" + "-" * 70)
    print("RESULTS SUMMARY")
    print("-" * 70)
    print(f"{'Metric':<30} {'Without Checkpointing':<20} {'With Checkpointing':<20}")
    print("-" * 70)
    print(f"{'Peak GPU Memory (MB)':<30} {no_ckpt['peak_gpu_mb']:<20.2f} {with_ckpt['peak_gpu_mb']:<20.2f}")
    print(f"{'Forward Time (ms)':<30} {no_ckpt['forward_ms']:<20.2f} {with_ckpt['forward_ms']:<20.2f}")
    print(f"{'Backward Time (ms)':<30} {no_ckpt['backward_ms']:<20.2f} {with_ckpt['backward_ms']:<20.2f}")
    print(f"{'Total Time (ms)':<30} {no_ckpt['total_ms']:<20.2f} {with_ckpt['total_ms']:<20.2f}")
    print("-" * 70)
    print(f"\n📊 Memory Savings: {memory_savings:.1f}%")
    print(f"⏱️  Time Overhead: {time_overhead:.1f}%")
    print(f"💡 Trade-off: ~{abs(memory_savings):.0f}% memory for ~{time_overhead:.0f}% more compute")
    
    return results


def run_selective_checkpointing(
    vocab_size: int = 5000,
    d_model: int = 256,
    nhead: int = 8,
    num_layers: int = 8,
    dim_feedforward: int = 512,
    batch_size: int = 4,
    seq_len: int = 128,
    ratios: List[float] = None,
) -> Dict[str, Dict]:
    """Test different checkpointing ratios for selective checkpointing."""
    if ratios is None:
        ratios = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    print("\n" + "=" * 70)
    print("SELECTIVE CHECKPOINTING: Testing Different Ratios")
    print("=" * 70)
    print(f"Ratios to test: {ratios}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_config = dict(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=0.0,
        max_len=seq_len * 2,
        mode="modern",
        gradient_checkpointing=True,
    )
    
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    target_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    
    results = {}
    baseline_memory = None
    baseline_time = None
    
    for ratio in ratios:
        print(f"\n--- Testing ratio={ratio:.2f} ---")
        
        reset_memory_stats()
        
        model = SimpleTransformer(
            **model_config,
            gradient_checkpointing_ratio=ratio,
        ).to(device)
        
        checkpointed_layers = sum(
            1 for block in model.transformer_blocks 
            if block.gradient_checkpointing
        )
        
        mem_info = benchmark_memory(model, input_ids, target_ids)
        timing_info = benchmark_forward_backward(model, input_ids, target_ids)
        
        if baseline_memory is None:
            baseline_memory = mem_info["peak_gpu_mb"]
            baseline_time = timing_info.get("total_ms", timing_info.get("total_mean_ms", 0))
        
        if baseline_memory > 0:
            memory_savings = (baseline_memory - mem_info["peak_gpu_mb"]) / baseline_memory * 100
        else:
            memory_savings = 0.0
        total_time = timing_info.get("total_ms", timing_info.get("total_mean_ms", 0))
        time_savings = (baseline_time - total_time) / baseline_time * 100 if baseline_time > 0 else 0
        
        results[f"ratio_{ratio:.2f}"] = {
            "ratio": ratio,
            "checkpointed_layers": checkpointed_layers,
            "peak_gpu_mb": mem_info["peak_gpu_mb"],
            "total_ms": total_time,
            "memory_savings_pct": memory_savings,
            "time_savings_pct": time_savings,
        }
        
        print(f"  Checkpointed layers: {checkpointed_layers}/{num_layers}")
        print(f"  Peak memory: {mem_info['peak_gpu_mb']:.2f} MB (savings: {memory_savings:.1f}%)")
        print(f"  Total time: {total_time:.2f} ms")
        
        del model
        reset_memory_stats()
    
    print("\n" + "-" * 70)
    print("SELECTIVE CHECKPOINTING RESULTS")
    print("-" * 70)
    print(f"{'Ratio':<10} {'Layers':<10} {'Memory (MB)':<15} {'Time (ms)':<15} {'Mem Save':<10}")
    print("-" * 70)
    for ratio_str, data in sorted(results.items()):
        print(f"{data['ratio']:<10.2f} {data['checkpointed_layers']:<10} {data['peak_gpu_mb']:<15.2f} "
              f"{data['total_ms']:<15.2f} {data['memory_savings_pct']:<10.1f}%")
    print("-" * 70)
    
    return results


def run_large_model_comparison():
    """Compare memory usage on larger model configurations."""
    print("\n" + "=" * 70)
    print("LARGE MODEL COMPARISON")
    print("=" * 70)
    
    configs = [
        {"name": "Small", "num_layers": 4, "d_model": 128, "nhead": 4},
        {"name": "Medium", "num_layers": 8, "d_model": 256, "nhead": 8},
        {"name": "Large", "num_layers": 12, "d_model": 512, "nhead": 16},
    ]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    
    for config in configs:
        name = config["name"]
        print(f"\n--- {name} Model: {config['num_layers']} layers, d_model={config['d_model']} ---")
        
        model = SimpleTransformer(
            vocab_size=5000,
            d_model=config["d_model"],
            nhead=config["nhead"],
            num_layers=config["num_layers"],
            dim_feedforward=config["d_model"] * 4,
            dropout=0.0,
            max_len=256,
            mode="modern",
        ).to(device)
        
        input_ids = torch.randint(0, 5000, (2, 64), device=device)
        target_ids = torch.randint(0, 5000, (2, 64), device=device)
        
        memory_results = {}
        for checkpointing in [False, True]:
            reset_memory_stats()
            model_gc = SimpleTransformer(
                vocab_size=5000,
                d_model=config["d_model"],
                nhead=config["nhead"],
                num_layers=config["num_layers"],
                dim_feedforward=config["d_model"] * 4,
                dropout=0.0,
                max_len=256,
                mode="modern",
                gradient_checkpointing=checkpointing,
            ).to(device)
            
            mem_info = benchmark_memory(model_gc, input_ids, target_ids)
            memory_results[f"checkpointing_{checkpointing}"] = mem_info["peak_gpu_mb"]
            
            del model_gc
            reset_memory_stats()
        
        memory_savings = (memory_results["checkpointing_False"] - memory_results["checkpointing_True"]) / \
                        memory_results["checkpointing_False"] * 100
        
        results[name] = {
            "without_checkpointing_mb": memory_results["checkpointing_False"],
            "with_checkpointing_mb": memory_results["checkpointing_True"],
            "memory_savings_pct": memory_savings,
        }
        
        print(f"  Without: {memory_results['checkpointing_False']:.2f} MB")
        print(f"  With: {memory_results['checkpointing_True']:.2f} MB")
        print(f"  Savings: {memory_savings:.1f}%")
        
        del model
    
    return results


def run_speed_benchmark(
    vocab_size: int = 5000,
    d_model: int = 256,
    nhead: int = 8,
    num_layers: int = 8,
    dim_feedforward: int = 512,
    batch_size: int = 4,
    seq_len: int = 128,
) -> Dict:
    """Benchmark compute overhead of gradient checkpointing."""
    print("\n" + "=" * 70)
    print("SPEED BENCHMARK: Compute Overhead Analysis")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_config = dict(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=0.0,
        max_len=seq_len * 2,
        mode="modern",
    )
    
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    target_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    
    results = {}
    
    for checkpointing in [False, True]:
        label = "ON" if checkpointing else "OFF"
        print(f"\n--- Gradient Checkpointing {label} ---")
        
        reset_memory_stats()
        
        model = SimpleTransformer(
            **model_config,
            gradient_checkpointing=checkpointing,
        ).to(device)
        
        warmup_iters = 10
        test_iters = 50
        
        for _ in range(warmup_iters):
            model.zero_grad()
            logits, _ = model(input_ids)
            loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            loss.backward()
        
        reset_memory_stats()
        
        times = []
        for _ in range(test_iters):
            model.zero_grad()
            start = time.perf_counter()
            logits, _ = model(input_ids)
            loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            loss.backward()
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
        
        avg_time = sum(times) / len(times)
        results[f"checkpointing_{checkpointing}"] = {
            "avg_time_ms": avg_time * 1000,
            "min_time_ms": min(times) * 1000,
            "max_time_ms": max(times) * 1000,
        }
        
        print(f"  Average: {avg_time * 1000:.2f} ms")
        print(f"  Min: {min(times) * 1000:.2f} ms")
        print(f"  Max: {max(times) * 1000:.2f} ms")
        
        del model
        reset_memory_stats()
    
    overhead = (results["checkpointing_True"]["avg_time_ms"] - 
                results["checkpointing_False"]["avg_time_ms"]) / \
               results["checkpointing_False"]["avg_time_ms"] * 100
    
    print("\n" + "-" * 70)
    print(f"Compute Overhead: {overhead:.1f}%")
    print(f"Trade-off: {overhead:.0f}% slower for memory savings")
    print("-" * 70)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Gradient Checkpointing Benchmark")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    parser.add_argument("--memory", action="store_true", help="Run memory comparison")
    parser.add_argument("--selective", action="store_true", help="Run selective checkpointing test")
    parser.add_argument("--speed", action="store_true", help="Run speed benchmark")
    parser.add_argument("--large", action="store_true", help="Run large model comparison")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    
    args = parser.parse_args()
    
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\n{'#'*70}")
    print(f"# GRADIENT CHECKPOINTING BENCHMARK")
    print(f"# Device: {device}")
    print(f"# CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"# GPU: {torch.cuda.get_device_name(0)}")
        print(f"# GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"{'#'*70}")
    
    if args.all or (not args.memory and not args.selective and not args.speed and not args.large):
        run_memory_comparison()
        run_selective_checkpointing()
        run_speed_benchmark()
        run_large_model_comparison()
    else:
        if args.memory:
            run_memory_comparison()
        if args.selective:
            run_selective_checkpointing()
        if args.speed:
            run_speed_benchmark()
        if args.large:
            run_large_model_comparison()
    
    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
    print("\nKey Insights:")
    print("- Gradient checkpointing saves 30-50% memory by recomputing activations")
    print("- Cost: 20-30% more compute time per iteration")
    print("- Selective checkpointing allows fine-tuning the memory/compute trade-off")
    print("- Best for: Large models, long sequences, limited GPU memory")
    print("=" * 70)


if __name__ == "__main__":
    main()
