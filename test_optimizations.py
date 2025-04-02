#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
综合性能测试脚本
用于验证所有优化后的功能是否正常工作
"""

import os
import time
import logging
import numpy as np
from pathlib import Path
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("性能测试")

# 导入优化后的组件
from tokenizer_optimized import ClassicalTokenizer
from processor import MemoryPool
from train_model import LMDataset

def test_classical_tokenizer():
    """测试优化后的分词器性能"""
    logger.info("开始测试 ClassicalTokenizer 性能...")
    
    # 初始化分词器
    start_time = time.time()
    tokenizer = ClassicalTokenizer()
    init_time = time.time() - start_time
    logger.info(f"分词器初始化耗时: {init_time:.4f} 秒")
    
    # 准备测试文本
    test_texts = [
        "道可道，非常道。名可名，非常名。",
        "无名，天地之始，有名，万物之母。",
        "故常无欲，以观其妙；常有欲，以观其徼。",
        "此两者，同出而异名，同谓之玄，玄之又玄，众妙之门。"
    ]
    
    # 单个文本分词测试
    start_time = time.time()
    tokens = tokenizer.tokenize(test_texts[0])
    single_time = time.time() - start_time
    logger.info(f"单个文本分词耗时: {single_time:.4f} 秒, 结果: {tokens[:10]}...")
    
    # 批量分词测试
    start_time = time.time()
    batch_tokens = tokenizer.batch_tokenize(test_texts)
    batch_time = time.time() - start_time
    logger.info(f"批量分词耗时: {batch_time:.4f} 秒, 处理了 {len(test_texts)} 个文本")
    
    # 缓存命中测试
    start_time = time.time()
    tokens_cached = tokenizer.tokenize(test_texts[0])
    cached_time = time.time() - start_time
    cache_speedup = single_time / cached_time if cached_time > 0 else float('inf')
    logger.info(f"缓存命中分词耗时: {cached_time:.6f} 秒, 加速比: {cache_speedup:.2f}x")
    
    return True

def test_memory_pool():
    """测试优化后的内存池性能"""
    logger.info("开始测试 MemoryPool 性能...")
    
    # 初始化内存池
    start_time = time.time()
    memory_pool = MemoryPool(max_size=100 * 1024 * 1024)  # 100MB
    init_time = time.time() - start_time
    logger.info(f"内存池初始化耗时: {init_time:.4f} 秒")
    
    # 使用 acquire 方法
    start_time = time.time()
    with memory_pool.acquire():
        # 在内存池中执行一些操作
        data = [i for i in range(1000)]
    acquire_time = time.time() - start_time
    logger.info(f"使用内存池 acquire 耗时: {acquire_time:.6f} 秒")
    
    # 清理测试
    start_time = time.time()
    memory_pool.cleanup()
    cleanup_time = time.time() - start_time
    logger.info(f"内存池清理耗时: {cleanup_time:.4f} 秒")
    
    # 获取统计信息
    stats = memory_pool.get_stats()
    logger.info(f"内存池统计信息: {stats}")
    
    return True

def test_lm_dataset():
    """测试优化后的数据集加载性能"""
    logger.info("开始测试 LMDataset 性能...")
    
    # 检查测试文件是否存在
    test_file = Path("dataset/test_sample.jsonl")
    if not test_file.exists():
        # 创建测试样本文件
        logger.info("创建测试样本文件...")
        os.makedirs(test_file.parent, exist_ok=True)
        with open(test_file, "w", encoding="utf-8") as f:
            for i in range(1000):
                f.write(f'{{"text": "这是测试文本 {i}，用于验证数据集加载性能优化。"}}\n')
    
    # 初始化数据集
    start_time = time.time()
    dataset = LMDataset(
        data_path=str(test_file),
        context_length=128,
        stride=64,
        max_chunks=10
    )
    init_time = time.time() - start_time
    logger.info(f"数据集初始化耗时: {init_time:.4f} 秒, 样本数: {len(dataset)}")
    
    # 测试数据获取性能
    start_time = time.time()
    batch_ids = []
    for i in range(min(10, len(dataset))):
        item = dataset[i]
        if isinstance(item, dict) and 'input_ids' in item:
            batch_ids.append(item['input_ids'])
        elif isinstance(item, tuple) and len(item) > 0:
            batch_ids.append(item[0])
    get_time = time.time() - start_time
    logger.info(f"获取 {len(batch_ids)} 个数据样本耗时: {get_time:.4f} 秒")
    
    # 内存使用情况
    peak_memory = dataset.peak_memory_mb if hasattr(dataset, 'peak_memory_mb') else "未知"
    logger.info(f"数据集峰值内存使用: {peak_memory}MB")
    
    return True

def run_all_tests():
    """运行所有性能测试"""
    logger.info("开始运行所有性能测试...")
    
    tests = [
        ("ClassicalTokenizer", test_classical_tokenizer),
        ("MemoryPool", test_memory_pool),
        ("LMDataset", test_lm_dataset)
    ]
    
    results = {}
    all_passed = True
    
    for name, test_func in tests:
        logger.info(f"\n{'=' * 50}\n测试 {name}\n{'=' * 50}")
        start_time = time.time()
        try:
            passed = test_func()
            duration = time.time() - start_time
            status = "通过" if passed else "失败"
            results[name] = {
                "status": status,
                "duration": duration
            }
            logger.info(f"{name} 测试{status}，耗时: {duration:.4f} 秒")
            if not passed:
                all_passed = False
        except Exception as e:
            logger.error(f"{name} 测试出错: {e}")
            results[name] = {
                "status": "错误",
                "error": str(e)
            }
            all_passed = False
    
    # 汇总测试结果
    logger.info("\n" + "=" * 50)
    logger.info("性能测试汇总:")
    for name, result in results.items():
        status = result["status"]
        if status == "通过":
            logger.info(f"✅ {name}: {status}, 耗时: {result['duration']:.4f} 秒")
        else:
            error_msg = result.get("error", "未知错误")
            logger.info(f"❌ {name}: {status}, 错误: {error_msg}")
    
    overall = "全部通过" if all_passed else "部分失败"
    logger.info(f"测试结果: {overall}")
    logger.info("=" * 50)
    
    return all_passed

if __name__ == "__main__":
    run_all_tests() 