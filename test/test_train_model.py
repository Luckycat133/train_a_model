#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练模型(train_model.py)测试脚本
用于测试模型训练系统的功能和性能
"""

import os
import sys
import time
import logging
import tempfile
import shutil
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("训练模型测试")

# 导入训练模型组件
sys.path.append(str(Path(__file__).resolve().parent.parent))
from train_model import LMDataset, format_memory_size, format_time, plot_training_stats

def setup_test_environment():
    """设置测试环境，创建必要的目录和文件"""
    logger.info("设置测试环境...")
    
    # 创建临时测试目录
    test_dir = Path(tempfile.mkdtemp(prefix="train_model_test_"))
    logger.info(f"创建临时测试目录: {test_dir}")
    
    # 创建必要的子目录
    dataset_dir = test_dir / "dataset"
    logs_dir = test_dir / "logs"
    model_dir = test_dir / "model_weights"
    
    for d in [dataset_dir, logs_dir, model_dir]:
        d.mkdir(exist_ok=True)
    
    # 创建测试数据文件
    data_file = dataset_dir / "test_sample.jsonl"
    with open(data_file, "w", encoding="utf-8") as f:
        for i in range(100):
            sample = {
                "text": f"这是测试文本 {i}，用于验证模型训练功能。这段文本需要足够长，以便测试上下文窗口滑动功能。" * 5
            }
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    
    return test_dir

def cleanup_test_environment(test_dir):
    """清理测试环境"""
    logger.info(f"清理测试环境: {test_dir}")
    try:
        shutil.rmtree(test_dir)
    except Exception as e:
        logger.error(f"清理测试环境时出错: {e}")

def test_lm_dataset(test_dir):
    """测试语言模型数据集功能"""
    logger.info("测试语言模型数据集功能...")
    
    try:
        # 准备测试数据
        data_file = test_dir / "dataset" / "test_sample.jsonl"
        
        # 测试不同上下文长度和步长的组合
        test_configs = [
            {"context_length": 64, "stride": 32},
            {"context_length": 128, "stride": 64},
            {"context_length": 256, "stride": 128}
        ]
        
        results = []
        for config in test_configs:
            context_length = config["context_length"]
            stride = config["stride"]
            
            logger.info(f"测试配置: 上下文长度={context_length}, 步长={stride}")
            
            # 初始化数据集
            start_time = time.time()
            dataset = LMDataset(
                data_path=str(data_file),
                context_length=context_length,
                stride=stride,
                max_chunks=10
            )
            init_time = time.time() - start_time
            
            # 测试数据获取
            start_time = time.time()
            sample_count = min(10, len(dataset))
            samples = [dataset[i] for i in range(sample_count)]
            access_time = time.time() - start_time
            
            # 记录结果
            results.append({
                "context_length": context_length,
                "stride": stride,
                "sample_count": len(dataset),
                "init_time": init_time,
                "access_time": access_time,
                "peak_memory_mb": getattr(dataset, "peak_memory_mb", 0)
            })
            
            logger.info(f"  样本数量: {len(dataset)}")
            logger.info(f"  初始化时间: {init_time:.4f} 秒")
            logger.info(f"  数据访问时间: {access_time:.4f} 秒")
            logger.info(f"  峰值内存: {getattr(dataset, 'peak_memory_mb', 0):.2f} MB")
        
        # 输出比较结果
        logger.info("\n数据集配置比较:")
        for result in results:
            logger.info(f"上下文长度: {result['context_length']}, 步长: {result['stride']}")
            logger.info(f"  样本数量: {result['sample_count']}")
            logger.info(f"  初始化时间: {result['init_time']:.4f} 秒")
            logger.info(f"  数据访问时间: {result['access_time']:.4f} 秒")
            logger.info(f"  峰值内存: {result['peak_memory_mb']:.2f} MB")
        
        # 找出最佳配置
        best_config = min(results, key=lambda x: x["init_time"])
        logger.info(f"\n初始化最快的配置: 上下文长度={best_config['context_length']}, 步长={best_config['stride']}")
        
        logger.info("语言模型数据集功能测试通过")
        return True
    except Exception as e:
        logger.error(f"语言模型数据集功能测试失败: {e}")
        return False

def test_utility_functions():
    """测试工具函数"""
    logger.info("测试工具函数...")
    
    try:
        # 测试内存大小格式化
        memory_sizes = [1024, 1024 * 1024, 1024 * 1024 * 1024, None]
        expected_formats = ["1.00 KB", "1.00 MB", "1.00 GB", "未知"]
        
        for size, expected in zip(memory_sizes, expected_formats):
            formatted = format_memory_size(size)
            logger.info(f"格式化内存大小 {size} -> {formatted}")
            assert formatted == expected, f"内存格式化错误: {formatted} != {expected}"
        
        # 测试时间格式化
        time_seconds = [30, 90, 3600, 7200]
        expected_formats = ["30.0秒", "1.5分钟", "1.0小时0.0分钟", "2.0小时0.0分钟"]
        
        for seconds, expected in zip(time_seconds, expected_formats):
            formatted = format_time(seconds)
            logger.info(f"格式化时间 {seconds}秒 -> {formatted}")
            assert formatted == expected, f"时间格式化错误: {formatted} != {expected}"
        
        logger.info("工具函数测试通过")
        return True
    except Exception as e:
        logger.error(f"工具函数测试失败: {e}")
        return False

def test_stats_plotting(test_dir):
    """测试统计图表绘制功能"""
    logger.info("测试统计图表绘制功能...")
    
    try:
        # 创建模拟训练统计数据
        stats = {
            "losses": [2.5, 2.3, 2.1, 1.9, 1.8, 1.7, 1.65, 1.6],
            "learning_rates": [0.001, 0.0009, 0.0008, 0.0007, 0.0006, 0.0005, 0.0004, 0.0003],
            "epoch_times": [120, 118, 119, 117, 116, 115, 114, 113],
            "gpu_memory_usage": [1024 * 1024 * 1024 * i for i in range(1, 9)]
        }
        
        # 测试绘图功能
        save_dir = test_dir / "logs"
        start_time = time.time()
        plot_training_stats(stats, str(save_dir))
        plot_time = time.time() - start_time
        
        logger.info(f"绘制统计图表耗时: {plot_time:.4f} 秒")
        
        # 验证图表文件是否生成
        plot_files = list(save_dir.glob("training_stats_*.png"))
        assert len(plot_files) > 0, "未生成统计图表文件"
        
        logger.info(f"生成的图表文件: {[f.name for f in plot_files]}")
        logger.info("统计图表绘制功能测试通过")
        return True
    except Exception as e:
        logger.error(f"统计图表绘制功能测试失败: {e}")
        return False

def collect_performance_data(test_dir):
    """收集性能数据"""
    logger.info("开始收集训练模型性能数据...")
    
    try:
        # 准备测试数据
        data_file = test_dir / "dataset" / "test_sample.jsonl"
        
        # 测试不同配置下的性能
        context_lengths = [64, 128, 256]
        strides = [32, 64, 128]
        
        performance_data = []
        
        for context_length in context_lengths:
            for stride in strides:
                logger.info(f"测试配置: 上下文长度={context_length}, 步长={stride}")
                
                # 测量初始化时间
                start_time = time.time()
                dataset = LMDataset(
                    data_path=str(data_file),
                    context_length=context_length,
                    stride=stride,
                    max_chunks=10
                )
                init_time = time.time() - start_time
                
                # 测量数据访问时间
                start_time = time.time()
                for i in range(min(10, len(dataset))):
                    _ = dataset[i]
                access_time = time.time() - start_time
                
                # 记录性能数据
                performance_data.append({
                    "context_length": context_length,
                    "stride": stride,
                    "sample_count": len(dataset),
                    "init_time": init_time,
                    "access_time": access_time,
                    "peak_memory_mb": getattr(dataset, "peak_memory_mb", 0)
                })
        
        # 输出性能数据
        logger.info("\n性能数据汇总:")
        for data in performance_data:
            logger.info(f"上下文长度: {data['context_length']}, 步长: {data['stride']}")
            logger.info(f"  样本数量: {data['sample_count']}")
            logger.info(f"  初始化时间: {data['init_time']:.4f} 秒")
            logger.info(f"  数据访问时间: {data['access_time']:.4f} 秒")
            logger.info(f"  峰值内存: {data['peak_memory_mb']:.2f} MB")
        
        # 找出最佳配置
        best_init = min(performance_data, key=lambda x: x["init_time"])
        best_access = min(performance_data, key=lambda x: x["access_time"])
        best_memory = min(performance_data, key=lambda x: x["peak_memory_mb"])
        
        logger.info(f"\n最佳初始化时间配置: 上下文长度={best_init['context_length']}, 步长={best_init['stride']}")
        logger.info(f"最佳数据访问时间配置: 上下文长度={best_access['context_length']}, 步长={best_access['stride']}")
        logger.info(f"最佳内存使用配置: 上下文长度={best_memory['context_length']}, 步长={best_memory['stride']}")
        
        return performance_data
    except Exception as e:
        logger.error(f"收集性能数据时出错: {e}")
        return None

def run_all_tests():
    """运行所有测试"""
    logger.info("开始运行所有训练模型测试...")
    
    # 设置测试环境
    test_dir = setup_test_environment()
    
    tests = [
        ("语言模型数据集", lambda: test_lm_dataset(test_dir)),
        ("工具函数", test_utility_functions),
        ("统计图表绘制", lambda: test_stats_plotting(test_dir))
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
    logger.info("训练模型测试汇总:")
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
    
    # 清理测试环境
    cleanup_test_environment(test_dir)
    
    return all_passed

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="训练模型测试脚本")
    parser.add_argument("--all", action="store_true", help="运行所有测试")
    parser.add_argument("--dataset", action="store_true", help="测试数据集功能")
    parser.add_argument("--utils", action="store_true", help="测试工具函数")
    parser.add_argument("--plot", action="store_true", help="测试统计图表绘制")
    parser.add_argument("--perf", action="store_true", help="收集性能数据")
    
    args = parser.parse_args()
    
    # 如果没有指定任何参数，则运行所有测试
    if not any(vars(args).values()):
        args.all = True
    
    if args.all:
        run_all_tests()
    else:
        test_dir = setup_test_environment()
        try:
            if args.dataset:
                test_lm_dataset(test_dir)
            if args.utils:
                test_utility_functions()
            if args.plot:
                test_stats_plotting(test_dir)
            if args.perf:
                collect_performance_data(test_dir)
        finally:
            cleanup_test_environment(test_dir)

if __name__ == "__main__":
    main()