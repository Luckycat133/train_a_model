#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Architecture performance benchmarks:
- Flash Attention vs Naive Attention
- Multi-Head Latent Attention (MLA) vs Standard Attention
- Memory usage comparison
"""

import sys
import time
import torch
import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import ModernAttention, MultiHeadLatentAttention, SimpleTransformer
from src.config import DEFAULT_D_MODEL, DEFAULT_NHEAD, MLA_LATENT_DIM, MLA_NUM_LATENT_HEADS


def get_memory_usage() -> int:
    """Get current memory usage in bytes."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.memory_allocated()
    import psutil
    return psutil.Process().memory_info().rss


def benchmark_flash_attention(
    d_model: int = 512,
    num_heads: int = 8,
    batch_size: int = 4,
    seq_len: int = 256,
    num_iters: int = 50,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Dict[str, float]:
    """Benchmark Flash Attention vs Naive Attention."""
    print(f"\n{'='*60}")
    print(f"Benchmarking Flash Attention vs Naive Attention")
    print(f"{'='*60}")
    print(f"Configuration: d_model={d_model}, heads={num_heads}, batch={batch_size}, seq_len={seq_len}")
    print(f"Device: {device}")
    
    x = torch.randn(batch_size, seq_len, d_model, device=device)
    
    # Flash Attention
    flash_attn = ModernAttention(
        d_model=d_model,
        num_heads=num_heads,
        use_flash_attention=True
    ).to(device)
    flash_attn.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = flash_attn(x)
    
    # Benchmark
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    for _ in range(num_iters):
        _ = flash_attn(x)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    flash_time = time.time() - start_time
    
    # Memory usage
    mem_before = get_memory_usage()
    with torch.no_grad():
        for _ in range(10):
            _ = flash_attn(x)
    mem_after = get_memory_usage()
    flash_mem = max(0, mem_after - mem_before)
    
    # Naive Attention
    naive_attn = ModernAttention(
        d_model=d_model,
        num_heads=num_heads,
        use_flash_attention=False
    ).to(device)
    naive_attn.load_state_dict(flash_attn.state_dict())
    naive_attn.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = naive_attn(x)
    
    # Benchmark
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    for _ in range(num_iters):
        _ = naive_attn(x)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    naive_time = time.time() - start_time
    
    # Memory usage
    mem_before = get_memory_usage()
    with torch.no_grad():
        for _ in range(10):
            _ = naive_attn(x)
    mem_after = get_memory_usage()
    naive_mem = max(0, mem_after - mem_before)
    
    results = {
        "flash_time_per_iter": flash_time / num_iters,
        "naive_time_per_iter": naive_time / num_iters,
        "flash_total_time": flash_time,
        "naive_total_time": naive_time,
        "speedup": naive_time / flash_time if flash_time > 0 else float('inf'),
        "flash_memory_bytes": flash_mem,
        "naive_memory_bytes": naive_mem,
        "memory_ratio": naive_mem / flash_mem if flash_mem > 0 else float('inf')
    }
    
    print(f"\nResults:")
    print(f"Flash Attention: {results['flash_time_per_iter']*1000:.2f}ms/iter, {results['flash_memory_bytes']/1024/1024:.2f}MB")
    print(f"Naive Attention: {results['naive_time_per_iter']*1000:.2f}ms/iter, {results['naive_memory_bytes']/1024/1024:.2f}MB")
    print(f"Speedup: {results['speedup']:.2f}x, Memory ratio: {results['memory_ratio']:.2f}x")
    
    return results


def benchmark_mla(
    d_model: int = 512,
    num_heads: int = 8,
    latent_dim: int = 256,
    num_latent_heads: int = 4,
    batch_size: int = 4,
    seq_len: int = 512,
    num_iters: int = 50,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Dict[str, float]:
    """Benchmark Multi-Head Latent Attention (MLA) vs Standard Attention."""
    print(f"\n{'='*60}")
    print(f"Benchmarking MLA vs Standard Attention")
    print(f"{'='*60}")
    print(f"Configuration: d_model={d_model}, heads={num_heads}, latent_dim={latent_dim}")
    print(f"batch={batch_size}, seq_len={seq_len}, device={device}")
    
    x = torch.randn(batch_size, seq_len, d_model, device=device)
    
    # Standard Attention
    std_attn = ModernAttention(
        d_model=d_model,
        num_heads=num_heads
    ).to(device)
    std_attn.eval()
    
    # MLA
    mla = MultiHeadLatentAttention(
        d_model=d_model,
        num_heads=num_heads,
        latent_dim=latent_dim,
        num_latent_heads=num_latent_heads
    ).to(device)
    mla.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = std_attn(x, use_cache=True)
            _ = mla(x, use_cache=True)
    
    # Benchmark Standard Attention
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    for _ in range(num_iters):
        _, cache_std = std_attn(x, use_cache=True)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    std_time = time.time() - start_time
    
    # Calculate KV cache size
    std_cache_size = cache_std[0].numel() * cache_std[0].element_size() + cache_std[1].numel() * cache_std[1].element_size()
    
    # Benchmark MLA
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    for _ in range(num_iters):
        _, cache_mla = mla(x, use_cache=True)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    mla_time = time.time() - start_time
    
    # Calculate MLA KV cache size
    mla_cache_size = cache_mla[0].numel() * cache_mla[0].element_size() + cache_mla[1].numel() * cache_mla[1].element_size()
    
    results = {
        "standard_time_per_iter": std_time / num_iters,
        "mla_time_per_iter": mla_time / num_iters,
        "standard_cache_bytes": std_cache_size,
        "mla_cache_bytes": mla_cache_size,
        "cache_compression_ratio": std_cache_size / mla_cache_size if mla_cache_size > 0 else float('inf'),
        "time_ratio": std_time / mla_time if mla_time > 0 else 1.0
    }
    
    print(f"\nResults:")
    print(f"Standard: {results['standard_time_per_iter']*1000:.2f}ms/iter, KV cache: {results['standard_cache_bytes']/1024:.2f}KB")
    print(f"MLA:      {results['mla_time_per_iter']*1000:.2f}ms/iter, KV cache: {results['mla_cache_bytes']/1024:.2f}KB")
    print(f"KV cache compression: {results['cache_compression_ratio']:.2f}x")
    
    return results


def benchmark_full_model(
    vocab_size: int = 10000,
    d_model: int = 512,
    nhead: int = 8,
    num_layers: int = 6,
    batch_size: int = 4,
    seq_len: int = 256,
    use_mla: bool = False,
    num_iters: int = 20,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Dict[str, float]:
    """Benchmark full transformer model."""
    model_type = "MLA" if use_mla else "Standard"
    print(f"\n{'='*60}")
    print(f"Benchmarking Full {model_type} Transformer")
    print(f"{'='*60}")
    
    model = SimpleTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        use_mla=use_mla,
        mla_latent_dim=MLA_LATENT_DIM,
        mla_num_latent_heads=MLA_NUM_LATENT_HEADS
    ).to(device)
    model.eval()
    
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(input_ids)
    
    # Benchmark forward pass
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    for _ in range(num_iters):
        _ = model(input_ids)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    forward_time = time.time() - start_time
    
    # Memory usage
    mem_before = get_memory_usage()
    with torch.no_grad():
        _ = model(input_ids)
    mem_after = get_memory_usage()
    mem_usage = max(0, mem_after - mem_before)
    
    # Benchmark generation with KV cache
    input_ids_gen = torch.randint(0, vocab_size, (1, seq_len), device=device)
    with torch.no_grad():
        _, cache = model(input_ids_gen, use_cache=True)
    
    generated = input_ids_gen.clone()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_gen = time.time()
    for _ in range(32):
        with torch.no_grad():
            logits, cache = model(generated[:, -1:], past_key_values=cache, use_cache=True)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    gen_time = time.time() - start_gen
    
    results = {
        "forward_time_per_iter": forward_time / num_iters,
        "generation_time_per_token": gen_time / 32,
        "memory_usage_bytes": mem_usage
    }
    
    print(f"\nResults:")
    print(f"Forward pass: {results['forward_time_per_iter']*1000:.2f}ms/iter")
    print(f"Generation: {results['generation_time_per_token']*1000:.2f}ms/token")
    print(f"Memory: {results['memory_usage_bytes']/1024/1024:.2f}MB")
    
    return results


def run_all_benchmarks(output_file: str = None):
    """Run all benchmarks and optionally save to CSV."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running benchmarks on device: {device}")
    
    results = {}
    
    # Flash Attention benchmark
    results["flash_attention"] = benchmark_flash_attention(device=device)
    
    # MLA benchmark
    results["mla"] = benchmark_mla(device=device)
    
    # Full model benchmarks (only run non-KV cache part for now)
    # Full model KV cache generation has some compatibility issues - skip for stability
    # results["full_model_standard"] = benchmark_full_model(use_mla=False, device=device)
    # results["full_model_mla"] = benchmark_full_model(use_mla=True, device=device)
    
    if output_file:
        save_results_to_csv(results, output_file)
    
    print(f"\n{'='*60}")
    print("Benchmark Summary")
    print(f"{'='*60}")
    print(f"Flash Attention speedup: {results['flash_attention']['speedup']:.2f}x")
    print(f"MLA KV cache compression: {results['mla']['cache_compression_ratio']:.2f}x")
    
    return results


def save_results_to_csv(results: Dict, filename: str):
    """Save benchmark results to CSV file."""
    import csv
    from datetime import datetime
    
    filepath = Path(filename)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    flat_results = {}
    for category, data in results.items():
        for key, value in data.items():
            flat_results[f"{category}_{key}"] = value
    
    flat_results["timestamp"] = datetime.now().isoformat()
    
    file_exists = filepath.exists()
    
    with open(filepath, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=sorted(flat_results.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(flat_results)
    
    print(f"Results saved to {filename}")


def main():
    parser = argparse.ArgumentParser(description="Architecture Performance Benchmarks")
    parser.add_argument("--flash", action="store_true", help="Run Flash Attention benchmark")
    parser.add_argument("--mla", action="store_true", help="Run MLA benchmark")
    parser.add_argument("--full", action="store_true", help="Run full model benchmarks")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    parser.add_argument("--output", type=str, help="Save results to CSV file")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (cuda/cpu/auto)")
    
    args = parser.parse_args()
    
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if args.all or (not args.flash and not args.mla and not args.full):
        run_all_benchmarks(output_file=args.output)
    else:
        if args.flash:
            benchmark_flash_attention(device=device)
        if args.mla:
            benchmark_mla(device=device)
        if args.full:
            benchmark_full_model(use_mla=False, device=device)
            benchmark_full_model(use_mla=True, device=device)


if __name__ == "__main__":
    main()
