#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
灵猫墨韵古典文学数据处理系统 - 主启动器
版本: v2.1.2
"""

import os
import sys
import argparse
import logging
import random
import shutil
import glob
import time
import queue
import threading
import gc
from datetime import datetime
from pathlib import Path
import json
import yaml
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from tqdm import tqdm
from typing import Dict, List, Any, Optional, Set, Tuple
import itertools
from contextlib import contextmanager
import psutil

# 添加项目根目录到Python路径
ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR))

from processors.data_processor import DataProcessor
from processors.text_cleaner import TextCleaner
from processors.structure_organizer import StructureOrganizer
from processors.data_synthesizer import DataSynthesizer
from processors.dictionary_generator import DictionaryGenerator


class ProcessorMain:
    """灵猫墨韵古典文学数据处理系统主类"""
    
    def __init__(self, config_path="config/config.yaml"):
        """
        初始化处理器主类
        
        Args:
            config_path: 配置文件路径
        """
        # 设置日志
        self.logger = self._setup_logger()
        self.logger.info("初始化灵猫墨韵古典文学数据处理系统")
        
        # 保存配置路径
        self.config_path = config_path
        
        # 加载配置
        self.config = self._load_config(config_path)
        
        # 初始化组件
        self.text_cleaner = TextCleaner()
        self.structure_organizer = StructureOrganizer(self.config)
        self.data_processor = DataProcessor(self.config, self.text_cleaner, self.structure_organizer)
        
        # 主要路径
        self.main_dir = Path(".")
        self.dataset_dir = self.main_dir / "dataset"
        self.logs_dir = self.main_dir / "logs"
        self.temp_dirs = [self.main_dir / "temp_data", self.main_dir / "cleanup_temp"]
        self.model_weights_dir = self.main_dir / "model_weights"
        
        # 确保关键目录存在
        self.dataset_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        self.model_weights_dir.mkdir(exist_ok=True)
    
    def _setup_logger(self):
        """
        设置日志
        
        Returns:
            日志器实例
        """
        logger = logging.getLogger("LingmaoMoyun")
        logger.setLevel(logging.INFO)
        
        # 创建日志目录（如果不存在）
        self.main_dir = Path(__file__).resolve().parent
        self.logs_dir = self.main_dir / "logs"
        self.logs_dir.mkdir(exist_ok=True, parents=True)
        
        # 为各模块创建子日志目录
        self.processor_logs_dir = self.logs_dir / "processor"
        self.train_model_logs_dir = self.logs_dir / "train_model"
        self.generate_logs_dir = self.logs_dir / "generate"
        self.data_logs_dir = self.logs_dir / "data"
        
        # 确保所有日志目录存在
        self.processor_logs_dir.mkdir(exist_ok=True, parents=True)
        self.train_model_logs_dir.mkdir(exist_ok=True, parents=True)
        self.generate_logs_dir.mkdir(exist_ok=True, parents=True)
        self.data_logs_dir.mkdir(exist_ok=True, parents=True)
        
        # 使用当前日期时间创建唯一的日志文件名
        log_dir = self.processor_logs_dir  # 处理器的日志存放在processor子目录
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f"processor_{current_time}.log"
        
        # 添加文件处理器 - 文件里保留详细日志格式
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        
        # 添加控制台处理器 - 简化控制台输出格式，仅显示消息
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_formatter)
        
        # 清除现有处理器，防止重复添加
        if logger.handlers:
            logger.handlers.clear()
            
        # 添加处理器
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def _load_config(self, config_path):
        """
        加载配置文件
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            配置字典
        """
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            self.logger.info(f"已加载配置文件: {config_path}")
            return config
        except Exception as e:
            self.logger.error(f"加载配置文件失败: {e}")
            return {}
    
    def process_data(self, force=False):
        """处理数据"""
        self.logger.info("开始处理数据...")
        
        # 初始化内存池和任务队列
        memory_pool = MemoryPool(max_size=self.config.get("memory_limit", 1024 * 1024 * 1024))
        task_queue = queue.Queue()
        
        # 配置并行处理
        num_workers = min(os.cpu_count(), self.config.get("max_workers", 4))
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # 批量预加载数据
            batch_size = self.config.get("batch_size", 1000)
            for batch in self._batch_data_loader(batch_size):
                task_queue.put(batch)
            
            # 并行处理数据
            futures = []
            while not task_queue.empty():
                batch = task_queue.get()
                future = executor.submit(self._process_batch, batch, memory_pool)
                futures.append(future)
            
            # 等待所有任务完成
            for future in tqdm(futures, desc="处理数据批次"):
                try:
                    result = future.result()
                    if not result:
                        self.logger.warning("部分数据处理失败")
                except Exception as e:
                    self.logger.error(f"数据处理出错: {e}")
                    return False
                finally:
                    memory_pool.cleanup()
        
        return True
    
    def _batch_data_loader(self, batch_size):
        """改进的批量数据加载器 - 使用生成器减少内存占用"""
        # 使用Path.glob而不是list(glob.glob())避免一次性加载所有文件名
        data_path = Path(self.dataset_dir)
        # 使用迭代器处理，避免一次性创建大列表
        file_iter = data_path.glob("**/*.txt")
        
        current_batch = []
        for file_path in file_iter:
            current_batch.append(file_path)
            if len(current_batch) >= batch_size:
                yield current_batch
                current_batch = []
            
        # 返回最后不足batch_size的批次
        if current_batch:
            yield current_batch
    
    def _process_batch(self, batch, memory_pool):
        """处理单个数据批次"""
        try:
            with memory_pool.acquire() as mem:
                data_synthesizer = DataSynthesizer(self.config, text_cleaner=self.text_cleaner)
                return data_synthesizer.process_batch(batch, mem)
        except Exception as e:
            self.logger.error(f"批次处理失败: {e}")
            return False

class MemoryPool:
    """内存池管理器 - 增强版"""
    
    def __init__(self, max_size, cleanup_threshold=0.8):
        self.max_size = max_size
        self.current_size = 0
        self.lock = threading.Lock()
        self.cleanup_threshold = cleanup_threshold
        self.last_cleanup_time = time.time()
        self.cleanup_interval = 60  # 每60秒检查一次
        
        # 监控统计
        self.cleanup_count = 0
        self.peak_memory = 0
    
    @contextmanager
    def acquire(self):
        """获取内存块，加入自动清理机制"""
        current_time = time.time()
        try:
            with self.lock:
                # 定期检查或内存使用超阈值时清理
                if (self.current_size >= self.max_size * self.cleanup_threshold or 
                    current_time - self.last_cleanup_time > self.cleanup_interval):
                    self.cleanup()
                    self.last_cleanup_time = current_time
                self.current_size += 1
                # 记录峰值内存使用
                self.peak_memory = max(self.peak_memory, self.current_size)
            yield self
        finally:
            with self.lock:
                self.current_size -= 1
    
    def cleanup(self):
        """增强的内存清理"""
        # 强制垃圾收集
        gc.collect()
        self.cleanup_count += 1
        
        # 获取内存使用统计
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        current_memory_usage = memory_info.rss / (1024 * 1024)  # MB
        
        # 记录清理信息
        logging.debug(f"内存清理执行 #{self.cleanup_count}: 当前内存使用 {current_memory_usage:.2f}MB")

    def get_stats(self):
        """获取内存池统计信息"""
        return {
            "current_size": self.current_size,
            "peak_memory": self.peak_memory,
            "cleanup_count": self.cleanup_count,
            "max_size": self.max_size
        }

    def _get_data_logger(self, operation_name):
        """
        获取数据处理专用的日志记录器
        
        Args:
            operation_name: 操作名称，如'fix'、'optimize'等
            
        Returns:
            数据处理专用的日志记录器
        """
        try:
            # 设置data专用日志
            data_logger = logging.getLogger(f"LingmaoMoyun.DataProcessor.{operation_name}")
            
            # 创建data日志文件
            current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = self.data_logs_dir / f"data_{operation_name}_{current_time}.log"
            
            # 添加文件处理器
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.INFO)
            
            # 设置格式
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            
            # 添加处理器
            data_logger.addHandler(file_handler)
            
            # 记录开始信息
            data_logger.info(f"开始数据{operation_name}操作")
            self.logger.info(f"数据{operation_name}日志将记录到: {log_file}")
            
            return data_logger
        except Exception as e:
            self.logger.error(f"创建数据处理日志时出错: {e}")
            return self.logger  # 如果出错，返回主日志记录器


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="灵猫墨韵古典文学数据处理系统")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="配置文件路径")
    parser.add_argument("--force", action="store_true", help="强制重新处理所有数据")
    parser.add_argument("--clean", action="store_true", help="清理旧文件和目录")
    parser.add_argument("--optimize", action="store_true", help="优化数据（合并、拆分数据集）")
    parser.add_argument("--validate", action="store_true", help="验证数据完整性")
    parser.add_argument("--process", action="store_true", help="处理数据")
    parser.add_argument("--all", action="store_true", help="执行所有步骤（处理、优化、验证）")
    
    args = parser.parse_args()
    
    # 初始化处理器
    processor = ProcessorMain(config_path=args.config)
    
    # 检查是否指定了任何操作，如果没有，执行完整流程
    if not any([args.clean, args.optimize, args.validate, args.process, args.all]):
        print("没有指定操作，执行默认完整处理流程...")
        try:
            processor.logger.info("执行默认完整处理流程...")
            
            # 步骤1: 处理数据
            processor.logger.info("【步骤1/3】处理数据...")
            process_success = processor.process_data(force=args.force)
            
            # 步骤2: 优化数据
            processor.logger.info("【步骤2/3】优化数据...")
            if process_success:
                optimize_success = processor.optimize_data()
            else:
                processor.logger.warning("前序步骤失败，跳过数据优化")
                optimize_success = False
            
            # 步骤3: 验证数据
            processor.logger.info("【步骤3/3】验证数据...")
            if process_success:
                validate_success = processor.validate_data_integrity()
            else:
                processor.logger.warning("前序步骤失败，跳过数据验证")
                validate_success = False
            
            # 输出总结 - 使用简洁的图形格式输出结果
            from termcolor import colored
            print("\n" + "="*50)
            print(colored("💼 处理流程完成摘要", "cyan", attrs=["bold"]))
            print("="*50)
            print(f"数据处理: {colored('✓ 成功', 'green') if process_success else colored('✗ 失败', 'red')}")
            print(f"数据优化: {colored('✓ 成功', 'green') if optimize_success else colored('✗ 失败', 'red')}")
            print(f"数据验证: {colored('✓ 成功', 'green') if validate_success else colored('✗ 失败', 'red')}")
            print("="*50)
            
            return all([process_success, optimize_success, validate_success])
        except KeyboardInterrupt:
            processor.logger.warning("用户中断了处理流程")
            return False
    
    # 执行用户指定的操作
    success = True
    
    try:
        if args.all:
            processor.logger.info("执行所有步骤")
            if args.clean:
                processor.cleanup_old_files()
            process_success = processor.process_data(force=args.force)
            if process_success:
                optimize_success = processor.optimize_data()
                validate_success = processor.validate_data_integrity()
                success = all([process_success, optimize_success, validate_success])
            else:
                success = False
        else:
            if args.clean:
                processor.logger.info("清理旧文件")
                processor.cleanup_old_files()
            
            if args.process:
                processor.logger.info("开始处理数据...")
                if not processor.process_data(force=args.force):
                    processor.logger.error("数据处理失败")
                    success = False
            
            if args.optimize:
                processor.logger.info("开始优化数据...")
                if not processor.optimize_data():
                    processor.logger.error("数据优化失败")
                    success = False
            
            if args.validate:
                processor.logger.info("开始验证数据完整性...")
                if not processor.validate_data_integrity():
                    processor.logger.error("数据验证失败")
                    success = False
    except KeyboardInterrupt:
        processor.logger.warning("处理被用户中断")
        return False
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
