#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分词器(tokenizer.py)增强测试脚本
用于全面测试分词器的功能、性能和边界条件
"""

import os
import sys
import time
import json
import logging
import tempfile
import shutil
import random
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("分词器增强测试")

# 导入分词器组件
sys.path.append(str(Path(__file__).resolve().parent.parent))
from tokenizer import ClassicalTokenizer

def setup_test_environment():
    """设置测试环境，创建必要的目录和文件"""
    logger.info("设置测试环境...")
    
    # 创建临时测试目录
    test_dir = Path(tempfile.mkdtemp(prefix="tokenizer_test_"))
    logger.info(f"创建临时测试目录: {test_dir}")
    
    # 创建必要的子目录
    dataset_dir = test_dir / "dataset"
    dictionaries_dir = dataset_dir / "dictionaries"
    logs_dir = test_dir / "logs"
    
    for d in [dataset_dir, dictionaries_dir, logs_dir]:
        d.mkdir(exist_ok=True)
    
    # 创建测试词典
    dict_path = dictionaries_dir / "test_dict.txt"
    with open(dict_path, "w", encoding="utf-8") as f:
        f.write("# 测试词典\n")
        f.write("春风 n\n")
        f.write("杨柳 n\n")
        f.write("千里 n\n")
        f.write("江山 n\n")
        f.write("如画 v\n")
        f.write("明月 n\n")
        f.write("长天 n\n")
        f.write("秋水 n\n")
        f.write("落霞 n\n")
        f.write("孤鹜 n\n")
    
    # 创建小型测试语料
    small_corpus_path = dataset_dir / "small_corpus.txt"
    with open(small_corpus_path, "w", encoding="utf-8") as f:
        f.write("春风又绿江南岸，明月何时照我还。\n")
        f.write("千里江山如画，风景独好。\n")
        f.write("落霞与孤鹜齐飞，秋水共长天一色。\n")
    
    # 创建中型测试语料
    medium_corpus_path = dataset_dir / "medium_corpus.txt"
    with open(medium_corpus_path, "w", encoding="utf-8") as f:
        poems = [
            "春风又绿江南岸，明月何时照我还。",
            "千里江山如画，风景独好。",
            "落霞与孤鹜齐飞，秋水共长天一色。",
            "人闲桂花落，夜静春山空。",
            "欲穷千里目，更上一层楼。",
            "会当凌绝顶，一览众山小。",
            "白日依山尽，黄河入海流。",
            "两岸猿声啼不住，轻舟已过万重山。",
            "孤帆远影碧空尽，唯见长江天际流。",
            "飞流直下三千尺，疑是银河落九天。"
        ]
        for _ in range(20):  # 重复20次以创建更大的语料
            for poem in poems:
                f.write(poem + "\n")
    
    # 创建JSONL格式测试数据
    jsonl_path = dataset_dir / "test_data.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        poems = [
            "春风又绿江南岸，明月何时照我还。",
            "千里江山如画，风景独好。",
            "落霞与孤鹜齐飞，秋水共长天一色。",
            "人闲桂花落，夜静春山空。",
            "欲穷千里目，更上一层楼。"
        ]
        for i in range(100):
            sample = {
                "id": i,
                "title": f"测试诗词{i}",
                "content": random.choice(poems),
                "author": "测试"
            }
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    
    return test_dir, {
        "dict_path": dict_path,
        "small_corpus_path": small_corpus_path,
        "medium_corpus_path": medium_corpus_path,
        "jsonl_path": jsonl_path
    }

def cleanup_test_environment(test_dir):
    """清理测试环境"""
    logger.info(f"清理测试环境: {test_dir}")
    try:
        shutil.rmtree(test_dir)
    except Exception as e:
        logger.error(f"清理测试环境时出错: {e}")

def test_basic_functionality(test_files):
    """测试分词器基本功能"""
    logger.info("测试分词器基本功能...")
    
    try:
        # 初始化分词器
        start_time = time.time()
        tokenizer = ClassicalTokenizer(
            vocab_size=1000,  # 小词表用于快速测试
            special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[SEP]", "[MASK]"],
            dictionary_path=str(test_files["dict_path"])
        )
        init_time = time.time() - start_time
        logger.info(f"分词器初始化耗时: {init_time:.4f} 秒")
        
        # 训练分词器
        start_time = time.time()
        training_files = [str(test_files["small_corpus_path"])]
        success = tokenizer.train(training_files)
        train_time = time.time() - start_time
        
        if not success:
            logger.error("分词器训练失败")
            return False
        
        logger.info(f"分词器训练耗时: {train_time:.4f} 秒")
        
        # 测试分词功能
        test_texts = [
            "春风又绿江南岸",
            "千里江山如画",
            "落霞与孤鹜齐飞"
        ]
        
        for text in test_texts:
            # 编码
            start_time = time.time()
            tokens = tokenizer.encode(text)
            encode_time = time.time() - start_time
            
            # 解码
            start_time = time.time()
            decoded = tokenizer.decode(tokens)
            decode_time = time.time() - start_time
            
            logger.info(f"测试文本: '{text}'")
            logger.info(f"  编码结果: {tokens}")
            logger.info(f"  解码结果: '{decoded}'")
            logger.info(f"  编码耗时: {encode_time:.6f} 秒")
            logger.info(f"  解码耗时: {decode_time:.6f} 秒")
            
            # 验证解码结果是否与原文本相似
            similarity = len(set(text) & set(decoded)) / len(set(text) | set(decoded))
            logger.info(f"  文本相似度: {similarity:.2f}")
            assert similarity > 0.5, f"解码结果与原文本相似度过低: {similarity:.2f}"
        
        logger.info("分词器基本功能测试通过")
        return True
    except Exception as e:
        logger.error(f"分词器基本功能测试失败: {e}")
        return False

def test_performance(test_files):
    """测试分词器性能"""
    logger.info("测试分词器性能...")
    
    try:
        # 初始化分词器
        tokenizer = ClassicalTokenizer(
            vocab_size=1000,
            special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[SEP]", "[MASK]"],
            dictionary_path=str(test_files["dict_path"])
        )
        
        # 训练分词器 - 使用中型语料
        logger.info("使用中型语料训练分词器...")
        start_time = time.time()
        training_files = [str(test_files["medium_corpus_path"])]
        success = tokenizer.train(training_files)
        train_time = time.time() - start_time
        
        if not success:
            logger.error("分词器训练失败")
            return False
        
        logger.info(f"中型语料训练耗时: {train_time:.4f} 秒")
        
        # 性能测试 - 单个文本分词
        test_text = "春风又绿江南岸，明月何时照我还。千里江山如画，风景独好。" * 10
        
        # 预热
        for _ in range(5):
            tokenizer.encode(test_text)
        
        # 测量单个文本分词性能
        iterations = 100
        encode_times = []
        
        for i in range(iterations):
            start_time = time.time()
            tokens = tokenizer.encode(test_text)
            encode_time = time.time() - start_time
            encode_times.append(encode_time)
        
        avg_encode_time = sum(encode_times) / len(encode_times)
        min_encode_time = min(encode_times)
        max_encode_time = max(encode_times)
        
        logger.info(f"单个文本分词性能 ({iterations} 次迭代):")
        logger.info(f"  平均耗时: {avg_encode_time:.6f} 秒")
        logger.info(f"  最小耗时: {min_encode_time:.6f} 秒")
        logger.info(f"  最大耗时: {max_encode_time:.6f} 秒")
        
        # 性能测试 - 批量分词
        test_texts = [test_text] * 10
        
        # 预热
        if hasattr(tokenizer, 'batch_tokenize'):
            tokenizer.batch_tokenize(test_texts[:5])
        
        # 测量批量分词性能
        batch_iterations = 10
        batch_times = []
        
        for i in range(batch_iterations):
            start_time = time.time()
            if hasattr(tokenizer, 'batch_tokenize'):
                batch_tokens = tokenizer.batch_tokenize(test_texts)
            else:
                batch_tokens = [tokenizer.tokenize(text) for text in test_texts]
            batch_time = time.time() - start_time
            batch_times.append(batch_time)
        
        avg_batch_time = sum(batch_times) / len(batch_times)
        min_batch_time = min(batch_times)
        max_batch_time = max(batch_times)
        
        logger.info(f"批量分词性能 ({batch_iterations} 次迭代, 每批 {len(test_texts)} 个文本):")
        logger.info(f"  平均耗时: {avg_batch_time:.6f} 秒")
        logger.info(f"  最小耗时: {min_batch_time:.6f} 秒")
        logger.info(f"  最大耗时: {max_batch_time:.6f} 秒")
        
        # 计算加速比
        single_total_time = avg_encode_time * len(test_texts)
        speedup = single_total_time / avg_batch_time if avg_batch_time > 0 else float('inf')
        logger.info(f"批量处理加速比: {speedup:.2f}x")
        
        # 测试缓存效果
        logger.info("测试缓存效果...")
        
        # 第一次调用 - 无缓存
        start_time = time.time()
        tokenizer.encode(test_text)
        no_cache_time = time.time() - start_time
        
        # 第二次调用 - 有缓存
        start_time = time.time()
        tokenizer.encode(test_text)
        with_cache_time = time.time() - start_time
        
        cache_speedup = no_cache_time / with_cache_time if with_cache_time > 0 else float('inf')
        logger.info(f"无缓存耗时: {no_cache_time:.6f} 秒")
        logger.info(f"有缓存耗时: {with_cache_time:.6f} 秒")
        logger.info(f"缓存加速比: {cache_speedup:.2f}x")
        
        logger.info("分词器性能测试通过")
        return True, {
            "avg_encode_time": avg_encode_time,
            "avg_batch_time": avg_batch_time,
            "batch_speedup": speedup,
            "cache_speedup": cache_speedup
        }
    except Exception as e:
        logger.error(f"分词器性能测试失败: {e}")
        return False, None

def test_edge_cases(test_files):
    """测试分词器边界条件"""
    logger.info("测试分词器边界条件...")
    
    try:
        # 初始化分词器
        tokenizer = ClassicalTokenizer(
            vocab_size=1000,
            special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[SEP]", "[MASK]"],
            dictionary_path=str(test_files["dict_path"])
        )
        
        # 训练分词器
        training_files = [str(test_files["small_corpus_path"])]
        success = tokenizer.train(training_files)
        
        if not success:
            logger.error("分词器训练失败")
            return False
        
        # 测试边界条件
        edge_cases = [
            "",  # 空字符串
            "a",  # 单个字符
            "春",  # 单个中文字符
            "春风" * 1000,  # 超长文本
            "12345",  # 纯数字
            "!@#$%^",  # 特殊字符
            "\n\t\r",  # 控制字符
            "春风abc123",  # 混合字符
            "春风\n又绿\t江南",  # 包含换行和制表符
            "\u3000\u3000春风",  # 包含全角空格
        ]
        
        for i, case in enumerate(edge_cases):
            logger.info(f"测试边界条件 #{i+1}: '{case[:20]}{'...' if len(case) > 20 else ''}'")
            
            try:
                # 编码
                tokens = tokenizer.encode(case)
                
                # 解码
                decoded = tokenizer.decode(tokens)
                
                logger.info(f"  编码结果: {tokens[:10]}{'...' if len(tokens) > 10 else ''}")
                logger.info(f"  解码结果: '{decoded[:20]}{'...' if len(decoded) > 20 else ''}'")
                logger.info(f"  处理成功")
            except Exception as e:
                logger.error(f"  处理失败: {e}")
                return False
        
        # 测试错误处理
        error_cases = [
            (None, "空值"),
            (123, "非字符串类型"),
            (["春风", "又绿"], "列表类型"),
            ({"text": "春风"}, "字典类型")
        ]
        
        for case, desc in error_cases:
            logger.info(f"测试错误处理: {desc}")
            
            try:
                tokens = tokenizer.encode(case)
                logger.warning(f"  预期失败但成功处理: {tokens}")
            except Exception as e:
                logger.info(f"  预期的错误处理: {e}")
        
        logger.info("分词器边界条件测试通过")
        return True
    except Exception as e:
        logger.error(f"分词器边界条件测试失败: {e}")
        return False

def test_jsonl_processing(test_files):
    """测试JSONL文件处理"""
    logger.info("测试JSONL文件处理...")
    
    try:
        # 初始化分词器
        tokenizer = ClassicalTokenizer(
            vocab_size=1000,
            special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[SEP]", "[MASK]"],
            dictionary_path=str(test_files["dict_path"])
        )
        
        # 从JSONL文件提取文本
        start_time = time.time()
        jsonl_files = [str(test_files["jsonl_path"])]
        text_lines = tokenizer.extract_text_from_jsonl(jsonl_files)
        extract_time = time.time() - start_time
        
        logger.info(f"从JSONL文件提取文本耗时: {extract_time:.4f} 秒")
        logger.info(f"提取的文本行数: {len(text_lines)}")
        
        # 验证提取的文本
        assert len(text_lines) > 0, "未从JSONL文件提取到文本"
        
        # 使用提取的文本训练分词器
        start_time = time.time()
        success = tokenizer.train(training_files=None)  # 使用提取的文本训练
        train_time = time.time() - start_time
        
        if not success:
            logger.error("使用JSONL提取的文本训练分词器失败")
            return False
        
        logger.info(f"使用JSONL提取的文本训练分词器耗时: {train_time:.4f} 秒")
        
        # 测试分词结果
        test_text = "春风又绿江南岸"
        tokens = tokenizer.encode(test_text)
        decoded = tokenizer.decode(tokens)
        
        logger.info(f"测试文本: '{test_text}'")
        logger.info(f"  编码结果: {tokens}")
        logger.info(f"  解码结果: '{decoded}'")
        
        logger.info("JSONL文件处理测试通过")
        return True
    except Exception as e:
        logger.error(f"JSONL文件处理测试失败: {e}")
        return False

def plot_performance_results(performance_data, save_dir):
    """绘制性能测试结果图表"""
    if not performance_data:
        logger.warning("没有性能数据可供绘图")
        return
    
    try:
        # 创建图表目录
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        
        # 设置图表样式
        plt.style.use('seaborn-v0_8-darkgrid')
        plt.figure(figsize=(12, 8))
        
        # 创建子图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('分词器性能测试结果', fontsize=16)
        
        # 绘制加速比对比图
        labels = ['批处理加速比', '缓存加速比']
        values = [performance_data['batch_speedup'], performance_data['cache_speedup']]
        colors = ['#3498db', '#2ecc71']
        
        ax1.bar(labels, values, color=colors)
        ax1.set_title('加速比对比')
        ax1.set_ylabel('加速比 (x倍)')
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # 在柱状图上添加数值标签
        for i, v in enumerate(values):
            ax1.text(i, v + 0.1, f'{v:.2f}x', ha='center')
        
        # 绘制处理时间对比图
        labels = ['单个文本平均时间', '批处理平均时间']
        values = [performance_data['avg_encode_time'], performance_data['avg_batch_time'] / 10]  # 除以10获取每个文本的平均时间
        
        ax2.bar(labels, values, color=colors)
        ax2.set_title('处理时间对比 (每个文本)')
        ax2.set_ylabel('时间 (秒)')
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # 在柱状图上添加数值标签
        for i, v in enumerate(values):
            ax2.text(i, v + 0.0001, f'{v:.6f}s', ha='center')
        
        # 添加测试信息
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fig.text(0.5, 0.01, f'测试时间: {timestamp}', ha='center', fontsize=10)
        
        # 保存图表
        plot_path = save_dir / f'tokenizer_performance_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(plot_path, dpi=300)
        plt.close()
        
        logger.info(f"性能测试结果图表已保存至: {plot_path}")
        return plot_path
    except Exception as e:
        logger.error(f"绘制性能测试结果图表失败: {e}")
        return None

def run_all_tests():
    """运行所有测试"""
    logger.info("开始运行所有分词器测试...")
    
    # 设置测试环境
    test_dir, test_files = setup_test_environment()
    
    tests = [
        ("基本功能", lambda: test_basic_functionality(test_files)),
        ("性能测试", lambda: test_performance(test_files)),
        ("边界条件", lambda: test_edge_cases(test_files)),
        ("JSONL处理", lambda: test_jsonl_processing(test_files))
    ]
    
    results = {}
    all_passed = True
    performance_data = None
    
    for name, test_func in tests:
        logger.info(f"\n{'=' * 50}\n测试 {name}\n{'=' * 50}")
        start_time = time.time()
        try:
            if name == "性能测试":
                passed, perf_data = test_func()
                if passed and perf_data:
                    performance_data = perf_data
            else:
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
    
    # 绘制性能测试结果图表
    if performance_data:
        plot_performance_results(performance_data, test_dir / "logs")
    
    # 汇总测试结果
    logger.info("\n" + "=" * 50)
    logger.info("分词器测试汇总:")
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
    """收集分词器性能数据"""
    logger.info("开始收集分词器性能数据...")
    
    # 设置测试环境
    test_dir, test_files = setup_test_environment()
    
    try:
        # 初始化分词器
        tokenizer = ClassicalTokenizer(
            vocab_size=1000,
            special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[SEP]", "[MASK]"],
            dictionary_path=str(test_files["dict_path"])
        )
        
        # 训练分词器
        training_files = [str(test_files["medium_corpus_path"])]
        success = tokenizer.train(training_files)
        
        if not success:
            logger.error("分词器训练失败")
            return None
        
        # 准备测试文本 - 不同长度
        test_texts = {
            "短文本": "春风又绿江南岸",
            "中文本": "春风又绿江南岸，明月何时照我还。千里江山如画，风景独好。",
            "长文本": "春风又绿江南岸，明月何时照我还。" * 10,
            "超长文本": "春风又绿江南岸，明月何时照我还。" * 100
        }
        
        # 收集不同文本长度的性能数据
        performance_data = []
        
        for name, text in test_texts.items():
            logger.info(f"测试 {name} (长度: {len(text)})...")
            
            # 预热
            for _ in range(3):
                tokenizer.encode(text)
            
            # 测量编码时间
            iterations = 10
            encode_times = []
            
            for _ in range(iterations):
                start_time = time.time()
                tokens = tokenizer.encode(text)
                encode_time = time.time() - start_time
                encode_times.append(encode_time)
            
            avg_time = sum(encode_times) / len(encode_times)
            min_time = min(encode_times)
            max_time = max(encode_times)
            
            # 记录性能数据
            performance_data.append({
                "name": name,
                "length": len(text),
                "avg_time": avg_time,
                "min_time": min_time,
                "max_time": max_time,
                "tokens_per_second": len(text) / avg_time if avg_time > 0 else float('inf')
            })
            
            logger.info(f"  平均耗时: {avg_time:.6f} 秒")
            logger.info(f"  最小耗时: {min_time:.6f} 秒")
            logger.info(f"  最大耗时: {max_time:.6f} 秒")
            logger.info(f"  处理速度: {len(text) / avg_time:.2f} 字符/秒")
        
        # 输出性能数据
        logger.info("\n性能数据汇总:")
        for data in performance_data:
            logger.info(f"{data['name']} (长度: {data['length']}):")
            logger.info(f"  平均耗时: {data['avg_time']:.6f} 秒")
            logger.info(f"  处理速度: {data['tokens_per_second']:.2f} 字符/秒")
        
        # 绘制性能图表
        plot_path = test_dir / "logs" / f"tokenizer_performance_by_length_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        try:
            plt.figure(figsize=(12, 10))
            
            # 绘制处理时间与文本长度的关系
            plt.subplot(2, 1, 1)
            lengths = [data["length"] for data in performance_data]
            times = [data["avg_time"] for data in performance_data]
            plt.plot(lengths, times, 'o-', linewidth=2, markersize=8, color='#3498db')
            plt.title('处理时间与文本长度的关系')
            plt.xlabel('文本长度 (字符数)')
            plt.ylabel('处理时间 (秒)')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # 添加数据点标签
            for i, (length, time) in enumerate(zip(lengths, times)):
                plt.annotate(f'{time:.6f}s', (length, time), 
                            textcoords="offset points", 
                            xytext=(0, 10), 
                            ha='center')
            
            # 绘制处理速度与文本长度的关系
            plt.subplot(2, 1, 2)
            speeds = [data["tokens_per_second"] for data in performance_data]
            plt.plot(lengths, speeds, 'o-', linewidth=2, markersize=8, color='#2ecc71')
            plt.title('处理速度与文本长度的关系')
            plt.xlabel('文本长度 (字符数)')
            plt.ylabel('处理速度 (字符/秒)')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # 添加数据点标签
            for i, (length, speed) in enumerate(zip(lengths, speeds)):
                plt.annotate(f'{speed:.2f}', (length, speed), 
                            textcoords="offset points", 
                            xytext=(0, 10), 
                            ha='center')
            
            plt.tight_layout()
            plt.savefig(plot_path, dpi=300)
            plt.close()
            
            logger.info(f"性能数据图表已保存至: {plot_path}")
        except Exception as e:
            logger.error(f"绘制性能数据图表失败: {e}")
        
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
    
    parser = argparse.ArgumentParser(description="分词器增强测试脚本")
    parser.add_argument("--all", action="store_true", help="运行所有测试")
    parser.add_argument("--basic", action="store_true", help="测试基本功能")
    parser.add_argument("--perf", action="store_true", help="测试性能")
    parser.add_argument("--edge", action="store_true", help="测试边界条件")
    parser.add_argument("--jsonl", action="store_true", help="测试JSONL处理")
    parser.add_argument("--collect", action="store_true", help="收集性能数据")
    
    args = parser.parse_args()
    
    # 如果没有指定任何参数，则运行所有测试
    if not any(vars(args).values()):
        args.all = True
    
    if args.all:
        run_all_tests()
    else:
        test_dir, test_files = setup_test_environment()
        try:
            if args.basic:
                test_basic_functionality(test_files)
            if args.perf:
                test_performance(test_files)
            if args.edge:
                test_edge_cases(test_files)
            if args.jsonl:
                test_jsonl_processing(test_files)
            if args.collect:
                collect_performance_data()
        finally:
            cleanup_test_environment(test_dir)

if __name__ == "__main__":
    main()