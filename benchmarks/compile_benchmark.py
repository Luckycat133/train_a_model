#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
torch.compile() Performance Benchmark

比较编译前后的模型性能:
- 训练模式 (reduce-overhead)
- 推理模式 (max-autotune)
- 不同模型大小的基准测试
- 编译时间 vs 加速收益分析
"""

import sys
import time
import torch
import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import SimpleTransformer, TORCH_AVAILABLE


def format_time(seconds: float) -> str:
    """格式化时间输出"""
    if seconds < 1:
        return f"{seconds * 1000:.2f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"


def format_memory(bytes: int) -> str:
    """格式化内存输出"""
    if bytes < 1024:
        return f"{bytes} B"
    elif bytes < 1024 * 1024:
        return f"{bytes / 1024:.1f} KB"
    elif bytes < 1024 * 1024 * 1024:
        return f"{bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{bytes / (1024 * 1024 * 1024):.2f} GB"


def get_memory_usage(device: str) -> int:
    """获取当前内存使用量（字节）"""
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.memory_allocated()
    import psutil
    return psutil.Process().memory_info().rss


def benchmark_model(
    model: torch.nn.Module,
    batch_size: int,
    seq_len: int,
    num_iters: int = 50,
    warmup_iters: int = 10,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    mode: str = "forward_only",
) -> Dict[str, float]:
    """对模型进行基准测试

    Args:
        model: 要测试的模型
        batch_size: 批大小
        seq_len: 序列长度
        num_iters: 迭代次数
        warmup_iters: 预热次数
        device: 设备
        mode: "forward_only" 或 "training"

    Returns:
        包含性能指标的字典
    """
    vocab_size = model.vocab_size if hasattr(model, 'vocab_size') else 10000
    model.eval()

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    if mode == "training":
        model.train()

    mem_before = get_memory_usage(device)

    for _ in range(warmup_iters):
        with torch.no_grad() if not model.training else torch.enable_grad():
            output, _ = model(input_ids)
            if model.training:
                loss = output.sum()
                loss.backward()

    torch.cuda.synchronize() if device == "cuda" else None

    start_time = time.perf_counter()

    for _ in range(num_iters):
        with torch.no_grad() if not model.training else torch.enable_grad():
            output, _ = model(input_ids)
            if model.training:
                loss = output.sum()
                loss.backward()

    torch.cuda.synchronize() if device == "cuda" else None

    end_time = time.perf_counter()
    total_time = end_time - start_time

    mem_after = get_memory_usage(device)
    mem_used = mem_after - mem_before

    return {
        "total_time": total_time,
        "avg_time": total_time / num_iters,
        "throughput": num_iters * batch_size / total_time,
        "memory_used": mem_used,
    }


def run_benchmark(
    model_size: str = "small",
    compile_modes: Optional[List[str]] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 8,
    seq_len: int = 512,
    num_iters: int = 50,
    save_results: bool = True,
) -> List[Dict]:
    """运行编译性能基准测试

    Args:
        model_size: 模型大小 ("tiny", "small", "medium", "large")
        compile_modes: 要测试的编译模式列表
        device: 设备
        batch_size: 批大小
        seq_len: 序列长度
        num_iters: 迭代次数
        save_results: 是否保存结果到CSV

    Returns:
        基准测试结果列表
    """
    if compile_modes is None:
        compile_modes = ["none", "reduce-overhead", "max-autotune"]

    model_configs = {
        "tiny": {"d_model": 128, "nhead": 4, "num_layers": 2, "dim_feedforward": 256},
        "small": {"d_model": 256, "nhead": 8, "num_layers": 4, "dim_feedforward": 512},
        "medium": {"d_model": 512, "nhead": 8, "num_layers": 6, "dim_feedforward": 1024},
        "large": {"d_model": 768, "nhead": 12, "num_layers": 12, "dim_feedforward": 2048},
    }

    config = model_configs.get(model_size, model_configs["small"])
    vocab_size = 10000

    results = []

    print(f"\n{'='*70}")
    print(f"torch.compile() Performance Benchmark")
    print(f"{'='*70}")
    print(f"Model size: {model_size}")
    print(f"Configuration: d_model={config['d_model']}, layers={config['num_layers']}, "
          f"heads={config['nhead']}, ffn={config['dim_feedforward']}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}, Sequence length: {seq_len}")
    print(f"Iterations: {num_iters}")
    print(f"{'='*70}\n")

    for mode in compile_modes:
        print(f"\nTesting mode: {mode.upper()}")

        model = SimpleTransformer(
            vocab_size=vocab_size,
            **config,
            dropout=0.0,
            max_len=1024,
            mode="modern",
        ).to(device)

        compile_time = 0.0
        is_compiled = False

        if mode != "none" and TORCH_AVAILABLE:
            print(f"  Compiling model with mode='{mode}'...")
            compile_start = time.perf_counter()
            compiled_model = torch.compile(model, mode=mode, dynamic=True)
            compile_time = time.perf_counter() - compile_start
            model = compiled_model
            is_compiled = True
            print(f"  Compilation time: {format_time(compile_time)}")
        elif mode == "none":
            print(f"  Running eager mode (no compilation)")

        print(f"  Warming up...")
        _ = benchmark_model(model, batch_size, seq_len, num_iters=5, warmup_iters=5, device=device)

        print(f"  Benchmarking forward pass...")
        forward_results = benchmark_model(
            model, batch_size, seq_len, num_iters=num_iters,
            warmup_iters=10, device=device, mode="forward_only"
        )

        print(f"  Benchmarking training pass...")
        model.train()
        training_results = benchmark_model(
            model, batch_size, seq_len, num_iters=num_iters,
            warmup_iters=10, device=device, mode="training"
        )

        result = {
            "mode": mode,
            "model_size": model_size,
            "compile_time": compile_time,
            "is_compiled": is_compiled,
            "forward_time": forward_results["avg_time"],
            "forward_throughput": forward_results["throughput"],
            "training_time": training_results["avg_time"],
            "training_throughput": training_results["throughput"],
            "memory_used": forward_results["memory_used"],
            "device": device,
            "batch_size": batch_size,
            "seq_len": seq_len,
        }

        results.append(result)

        print(f"\n  Results:")
        print(f"    Forward pass: {format_time(forward_results['avg_time'])} per batch")
        print(f"    Forward throughput: {forward_results['throughput']:.2f} tokens/s")
        print(f"    Training pass: {format_time(training_results['avg_time'])} per batch")
        print(f"    Training throughput: {training_results['throughput']:.2f} tokens/s")

    if len(results) > 1:
        print(f"\n{'='*70}")
        print("SPEEDUP COMPARISON (vs eager mode)")
        print(f"{'='*70}")

        eager_forward = next((r for r in results if r["mode"] == "none"), None)
        if eager_forward:
            for result in results:
                if result["mode"] != "none":
                    forward_speedup = eager_forward["forward_time"] / result["forward_time"]
                    training_speedup = eager_forward["training_time"] / result["training_time"]
                    print(f"\n{result['mode'].upper()} vs EAGER:")
                    print(f"  Forward speedup: {forward_speedup:.2f}x")
                    print(f"  Training speedup: {training_speedup:.2f}x")

    if save_results:
        output_dir = Path("benchmarks/results")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"compile_benchmark_{model_size}_{device}.csv"

        with open(output_file, 'w', newline='') as f:
            if results:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)

        print(f"\nResults saved to: {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="torch.compile() Performance Benchmark")
    parser.add_argument("--model_size", type=str, default="small",
                       choices=["tiny", "small", "medium", "large"],
                       help="模型大小")
    parser.add_argument("--modes", type=str, default="none,reduce-overhead,max-autotune",
                       help="要测试的编译模式（逗号分隔）")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="设备")
    parser.add_argument("--batch_size", type=int, default=8, help="批大小")
    parser.add_argument("--seq_len", type=int, default=512, help="序列长度")
    parser.add_argument("--num_iters", type=int, default=50, help="迭代次数")
    parser.add_argument("--no_save", action="store_true", help="不保存结果")

    args = parser.parse_args()

    compile_modes = [m.strip() for m in args.modes.split(",")]

    if not TORCH_AVAILABLE:
        print("WARNING: torch.compile() not available, testing eager mode only")
        compile_modes = ["none"]

    results = run_benchmark(
        model_size=args.model_size,
        compile_modes=compile_modes,
        device=args.device,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_iters=args.num_iters,
        save_results=not args.no_save,
    )

    print(f"\n{'='*70}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*70}")

    if results:
        print("\nSummary Table:")
        print(f"{'Mode':<20} {'Forward':<15} {'Training':<15} {'Speedup':<15}")
        print("-" * 65)

        eager_result = next((r for r in results if r["mode"] == "none"), None)

        for result in results:
            speedup_str = "-"
            if eager_result and result["mode"] != "none":
                speedup = eager_result["forward_time"] / result["forward_time"]
                speedup_str = f"{speedup:.2f}x"

            print(f"{result['mode']:<20} "
                  f"{format_time(result['forward_time']):<15} "
                  f"{format_time(result['training_time']):<15} "
                  f"{speedup_str:<15}")


if __name__ == "__main__":
    main()
