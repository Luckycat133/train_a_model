#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
处理器(processor.py)测试脚本
用于测试数据处理系统的功能和性能
"""

import os
import sys
import time
import logging
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("处理器测试")

# 导入处理器组件
sys.path.append(str(Path(__file__).resolve().parent.parent))
from processor import ProcessorMain, MemoryPool
from processors.data_processor import DataProcessor
from processors.text_cleaner import TextCleaner
from processors.structure_organizer import StructureOrganizer

def setup_test_environment():
    """设置测试环境，创建必要的目录和文件"""
    logger.info("设置测试环境...")
    
    # 创建临时测试目录
    test_dir = Path(tempfile.mkdtemp(prefix="processor_test_"))
    logger.info(f"创建临时测试目录: {test_dir}")
    
    # 创建必要的子目录
    config_dir = test_dir / "config"
    dataset_dir = test_dir / "dataset"
    logs_dir = test_dir / "logs"
    temp_dir = test_dir / "temp_data"
    
    for d in [config_dir, dataset_dir, logs_dir, temp_dir]:
        d.mkdir(exist_ok=True)
    
    # 创建测试配置文件
    config_file = config_dir / "config.yaml"
    with open(config_file, "w", encoding="utf-8") as f:
        f.write("""# 测试配置文件
processor:
  max_workers: 2
  batch_size: 10
  memory_limit: 104857600  # 100MB

tokenizer:
  vocab_size: 1000
  special_tokens: ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
""")
    
    # 创建测试数据文件
    for i in range(3):
        data_file = dataset_dir / f"test_data_{i}.txt"
        with open(data_file, "w", encoding="utf-8") as f:
            f.write(f"这是测试数据文件 {i}\n")
            f.write("春风又绿江南岸，明月何时照我还。\n")
            f.write("落霞与孤鹜齐飞，秋水共长天一色。\n")
    
    return test_dir

def cleanup_test_environment(test_dir):
    """清理测试环境"""
    logger.info(f"清理测试环境: {test_dir}")
    try:
        shutil.rmtree(test_dir)
    except Exception as e:
        logger.error(f"清理测试环境时出错: {e}")

def test_processor_initialization(test_dir):
    """测试处理器初始化"""
    logger.info("测试处理器初始化...")
    
    start_time = time.time()
    config_path = test_dir / "config" / "config.yaml"
    
    try:
        processor = ProcessorMain(config_path=str(config_path))
        init_time = time.time() - start_time
        logger.info(f"处理器初始化耗时: {init_time:.4f} 秒")
        
        # 验证关键组件是否正确初始化
        assert processor.text_cleaner is not None, "文本清洗器未初始化"
        assert processor.structure_organizer is not None, "结构组织器未初始化"
        assert processor.data_processor is not None, "数据处理器未初始化"
        
        logger.info("处理器初始化测试通过")
        return True
    except Exception as e:
        logger.error(f"处理器初始化测试失败: {e}")
        return False

def test_memory_pool():
    """测试内存池功能"""
    logger.info("测试内存池功能...")
    
    try:
        # 创建内存池
        memory_limit = 10 * 1024 * 1024  # 10MB
        memory_pool = MemoryPool(max_size=memory_limit)
        
        # 测试内存获取和释放
        start_time = time.time()
        with memory_pool.acquire() as mem:
            # 分配一些内存
            data = [i for i in range(100000)]
            # 模拟处理
            time.sleep(0.1)
        
        acquire_time = time.time() - start_time
        logger.info(f"内存池获取和释放耗时: {acquire_time:.4f} 秒")
        
        # 测试内存清理
        start_time = time.time()
        memory_pool.cleanup()
        cleanup_time = time.time() - start_time
        logger.info(f"内存池清理耗时: {cleanup_time:.4f} 秒")
        
        # 获取统计信息
        stats = memory_pool.get_stats()
        logger.info(f"内存池统计信息: {stats}")
        
        logger.info("内存池功能测试通过")
        return True
    except Exception as e:
        logger.error(f"内存池功能测试失败: {e}")
        return False

def test_data_processor(test_dir):
    """测试数据处理功能"""
    logger.info("测试数据处理功能...")
    
    try:
        config_path = test_dir / "config" / "config.yaml"
        processor = ProcessorMain(config_path=str(config_path))
        
        # 修改数据目录为测试目录
        processor.dataset_dir = test_dir / "dataset"
        
        # 测试数据处理
        start_time = time.time()
        result = processor.process_data(force=True)
        process_time = time.time() - start_time
        
        logger.info(f"数据处理耗时: {process_time:.4f} 秒, 结果: {'成功' if result else '失败'}")
        
        # 验证处理结果
        assert result, "数据处理失败"
        
        logger.info("数据处理功能测试通过")
        return True
    except Exception as e:
        logger.error(f"数据处理功能测试失败: {e}")
        return False

def test_batch_processing():
    """测试批处理功能"""
    logger.info("测试批处理功能...")
    
    try:
        # 创建测试数据
        test_data = [f"测试数据{i}" for i in range(100)]
        
        # 模拟批处理函数
        def process_batch(batch, memory_pool):
            processed = []
            for item in batch:
                processed.append(f"已处理: {item}")
            return processed
        
        # 创建内存池
        memory_pool = MemoryPool(max_size=1024 * 1024)  # 1MB
        
        # 执行批处理
        batch_size = 10
        batches = [test_data[i:i+batch_size] for i in range(0, len(test_data), batch_size)]
        
        start_time = time.time()
        results = []
        for batch in batches:
            with memory_pool.acquire() as mem:
                result = process_batch(batch, mem)
                results.extend(result)
        
        batch_time = time.time() - start_time
        logger.info(f"批处理耗时: {batch_time:.4f} 秒, 处理了 {len(results)} 个项目")
        
        # 验证结果
        assert len(results) == len(test_data), "批处理结果数量不匹配"
        
        logger.info("批处理功能测试通过")
        return True
    except Exception as e:
        logger.error(f"批处理功能测试失败: {e}")
        return False

def run_all_tests():
    """运行所有测试"""
    logger.info("开始运行所有处理器测试...")
    
    # 设置测试环境
    test_dir = setup_test_environment()
    
    tests = [
        ("处理器初始化", lambda: test_processor_initialization(test_dir)),
        ("内存池功能", test_memory_pool),
        ("数据处理功能", lambda: test_data_processor(test_dir)),
        ("批处理功能", test_batch_processing)
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
    logger.info("处理器测试汇总:")
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

def collect_performance_data():
    """收集性能数据"""
    logger.info("开始收集处理器性能数据...")
    
    # 设置测试环境
    test_dir = setup_test_environment()
    
    try:
        # 初始化处理器
        config_path = test_dir / "config" / "config.yaml"
        processor = ProcessorMain(config_path=str(config_path))
        
        # 修改数据目录为测试目录
        processor.dataset_dir = test_dir / "dataset"
        
        # 收集不同批次大小的性能数据
        batch_sizes = [5, 10, 20, 50]
        performance_data = []
        
        for batch_size in batch_sizes:
            # 修改批次大小
            processor.config["batch_size"] = batch_size
            
            # 测量处理时间
            start_time = time.time()
            processor.process_data(force=True)
            process_time = time.time() - start_time
            
            # 记录性能数据
            performance_data.append({
                "batch_size": batch_size,
                "process_time": process_time
            })
            
            logger.info(f"批次大小 {batch_size}: 处理时间 {process_time:.4f} 秒")
        
        # 输出性能数据
        logger.info("\n性能数据汇总:")
        for data in performance_data:
            logger.info(f"批次大小: {data['batch_size']}, 处理时间: {data['process_time']:.4f} 秒")
        
        # 找出最佳批次大小
        best_batch = min(performance_data, key=lambda x: x["process_time"])
        logger.info(f"\n最佳批次大小: {best_batch['batch_size']}, 处理时间: {best_batch['process_time']:.4f} 秒")
        
        return performance_data
    except Exception as e:
        logger.error(f"收集性能数据时出错: {e}")
        return None
    finally:
        # 清理测试环境
        cleanup_test_environment(test_dir)

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="处理器测试脚本")
    parser.add_argument("--all", action="store_true", help="运行所有测试")
    parser.add_argument("--init", action="store_true", help="测试初始化")
    parser.add_argument("--memory", action="store_true", help="测试内存池")
    parser.add_argument("--process", action="store_true", help="测试数据处理")
    parser.add_argument("--batch", action="store_true", help="测试批处理")
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
            if args.init:
                test_processor_initialization(test_dir)
            if args.memory:
                test_memory_pool()
            if args.process:
                test_data_processor(test_dir)
            if args.batch:
                test_batch_processing()
            if args.perf:
                collect_performance_data()
        finally:
            cleanup_test_environment(test_dir)

if __name__ == "__main__":
    main()