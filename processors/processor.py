#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
灵猫墨韵数据处理器
用于清洗、转换和去重古籍文本数据，为大语言模型训练准备高质量数据集
支持多种输入格式，包括txt、json和jsonl
"""

import os
import sys
import json
import re
import hashlib
import argparse
import logging
import time
import chardet
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Set, Optional, Generator, Tuple, Any, Union
import yaml
from bs4 import BeautifulSoup
import pandas as pd
import regex
import nltk
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import colorlog
import inquirer
from functools import partial

# 全局变量
logger = None

# 设置日志记录
def setup_logging(log_dir: str = "logs", log_level: str = "INFO", colored: bool = True) -> logging.Logger:
    """配置日志系统
    
    Args:
        log_dir: 日志目录
        log_level: 日志级别
        colored: 是否使用彩色日志
        
    Returns:
        日志记录器
    """
    # 确保日志目录存在
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(exist_ok=True, parents=True)
    
    # 创建日志文件名，包含时间戳
    log_file = log_dir_path / f"processor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # 获取日志记录器
    logger = logging.getLogger("processor")
    logger.setLevel(getattr(logging, log_level.upper()))
    logger.handlers = []  # 清除现有处理器
    
    # 文件处理器 - 记录详细信息
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(getattr(logging, log_level.upper()))
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logger.addHandler(file_handler)
    
    # 控制台处理器 - 显示简洁信息
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    
    if colored:
        # 使用彩色日志
        color_formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(levelname)s: %(message)s",
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        )
        console_handler.setFormatter(color_formatter)
    else:
        console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        
    logger.addHandler(console_handler)
    
    return logger

def load_config(config_path: str = "config/config.yaml") -> Dict:
    """加载配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        logger.info(f"配置文件加载成功: {config_path}")
        return config
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        return {}

def detect_encoding(file_path: str) -> str:
    """检测文件编码
    
    Args:
        file_path: 文件路径
        
    Returns:
        文件编码
    """
    try:
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read(1024))
        encoding = result['encoding'] or 'utf-8'
        confidence = result['confidence']
        logger.debug(f"检测到文件编码: {encoding} (置信度: {confidence:.2f}) - {file_path}")
        return encoding
    except Exception as e:
        logger.warning(f"编码检测失败，使用默认编码 utf-8: {e}")
        return 'utf-8'

def read_data(input_paths: List[str], file_extensions: List[str], 
              batch_size: int = 1000, preview: bool = False, 
              preview_count: int = 5) -> Generator[List[str], None, None]:
    """读取指定路径下所有支持格式的文件中的文本数据
    
    Args:
        input_paths: 输入路径列表，可以是目录或文件
        file_extensions: 支持的文件扩展名列表，如 [".txt", ".json", ".jsonl"]
        batch_size: 批处理大小
        preview: 是否预览数据
        preview_count: 预览数据条数
        
    Yields:
        文本数据批次
    """
    # 展开所有文件路径
    all_files = []
    for input_path in input_paths:
        path = Path(input_path)
        if path.is_dir():
            # 如果是目录，递归查找所有匹配的文件
            for ext in file_extensions:
                all_files.extend(list(path.glob(f"**/*{ext}")))
        elif path.is_file() and path.suffix.lower() in file_extensions:
            # 如果是文件且扩展名匹配，直接添加
            all_files.append(path)
    
    logger.info(f"找到 {len(all_files)} 个文件")
    
    # 预览模式
    if preview and all_files:
        preview_files = all_files[:min(preview_count, len(all_files))]
        logger.info(f"数据预览 (来自 {len(preview_files)} 个文件):")
        for file_path in preview_files:
            encoding = detect_encoding(str(file_path))
            try:
                with open(file_path, "r", encoding=encoding, errors="replace") as f:
                    content = f.read(500)  # 读取前500个字符
                    logger.info(f"\n文件: {file_path}\n内容预览:\n{content}...")
            except Exception as e:
                logger.warning(f"预览文件失败: {file_path} - {e}")
    
    # 批量读取数据
    batch = []
    file_count = 0
    
    # 使用tqdm创建进度条
    with tqdm(total=len(all_files), desc="读取文件", unit="文件") as pbar:
        for file_path in all_files:
            file_count += 1
            file_ext = file_path.suffix.lower()
            encoding = detect_encoding(str(file_path))
            
            try:
                with open(file_path, "r", encoding=encoding, errors="replace") as f:
                    if file_ext == ".txt":
                        # 对于txt文件，按行读取
                        for line in f:
                            line = line.strip()
                            if line:  # 跳过空行
                                batch.append(line)
                                if len(batch) >= batch_size:
                                    yield batch
                                    batch = []
                    elif file_ext == ".json":
                        # 对于json文件，解析整个文件
                        data = json.load(f)
                        if isinstance(data, list):
                            for item in data:
                                if isinstance(item, dict) and "text" in item:
                                    batch.append(item["text"])
                                elif isinstance(item, str):
                                    batch.append(item)
                                if len(batch) >= batch_size:
                                    yield batch
                                    batch = []
                        elif isinstance(data, dict) and "text" in data:
                            batch.append(data["text"])
                            if len(batch) >= batch_size:
                                yield batch
                                batch = []
                    elif file_ext == ".jsonl":
                        # 对于jsonl文件，按行解析
                        for line in f:
                            try:
                                data = json.loads(line)
                                if isinstance(data, dict) and "text" in data:
                                    batch.append(data["text"])
                                    if len(batch) >= batch_size:
                                        yield batch
                                        batch = []
                            except json.JSONDecodeError:
                                logger.warning(f"解析JSONL行失败: {line[:50]}...")
                pbar.update(1)
            except Exception as e:
                logger.error(f"读取文件失败: {file_path} - {e}")
                pbar.update(1)
    
    # 返回最后一个批次
    if batch:
        yield batch

def clean_text(text: str, cleaning_rules: Dict[str, bool] = None) -> str:
    """清洗文本数据
    
    清除HTML标签、特殊字符、控制字符，规范化空白字符，
    并进行基本的文本规范化处理
    
    Args:
        text: 原始文本
        cleaning_rules: 清洗规则字典，键为规则名，值为是否启用
        
    Returns:
        清洗后的文本
    """
    if not text or not isinstance(text, str):
        return ""
    
    # 默认清洗规则
    default_rules = {
        "remove_html": True,           # 移除HTML标签
        "normalize_whitespace": True,  # 规范化空白字符
        "remove_control_chars": True,  # 移除控制字符
        "normalize_punctuation": True, # 规范化标点符号
        "remove_urls": True,          # 移除URL
        "remove_emojis": True,        # 移除表情符号
    }
    
    # 使用提供的规则覆盖默认规则
    rules = default_rules.copy()
    if cleaning_rules:
        rules.update(cleaning_rules)
    
    # 移除HTML标签
    if rules["remove_html"]:
        try:
            text = BeautifulSoup(text, "html.parser").get_text()
        except Exception as e:
            logger.warning(f"BeautifulSoup处理失败: {e}")
    
    # 移除控制字符
    if rules["remove_control_chars"]:
        text = regex.sub(r"[\p{C}&&[^\n\t]]", "", text)
    
    # 规范化空白字符
    if rules["normalize_whitespace"]:
        text = regex.sub(r"\s+", " ", text).strip()
    
    # 移除URL
    if rules["remove_urls"]:
        text = regex.sub(r"https?://\S+|www\.\S+", "", text)
    
    # 规范化标点符号
    if rules["normalize_punctuation"]:
        # 移除重复的标点符号
        text = regex.sub(r"([，。！？；：、,.!?;:]){2,}", r"\1", text)
        # 确保中文标点前后没有不必要的空格
        text = regex.sub(r"\s*([，。！？；：、])\s*", r"\1", text)
    
    # 移除表情符号
    if rules["remove_emojis"]:
        text = regex.sub(r"[\U00010000-\U0010ffff]", "", text)
    
    return text.strip()

def process_batch(batch: List[str], cleaning_rules: Dict[str, bool] = None) -> List[Tuple[str, str]]:
    """处理一批文本数据
    
    Args:
        batch: 文本数据批次
        cleaning_rules: 清洗规则
        
    Returns:
        处理后的文本及其哈希值列表
    """
    results = []
    for text in batch:
        cleaned_text = clean_text(text, cleaning_rules)
        if cleaned_text:  # 跳过清洗后为空的文本
            # 计算文本哈希值用于去重
            text_hash = hashlib.md5(cleaned_text.encode("utf-8")).hexdigest()
            results.append((cleaned_text, text_hash))
    return results

def save_results(results: List[str], output_file: str, output_format: str = "txt") -> None:
    """保存处理结果
    
    Args:
        results: 处理后的文本列表
        output_file: 输出文件路径
        output_format: 输出格式，支持txt、json、jsonl
    """
    # 确保输出目录存在
    output_path = Path(output_file)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    try:
        if output_format == "txt":
            with open(output_file, "w", encoding="utf-8") as f:
                for text in results:
                    f.write(f"{text}\n")
        elif output_format == "json":
            with open(output_file, "w", encoding="utf-8") as f:
                json_data = [{"text": text} for text in results]
                json.dump(json_data, f, ensure_ascii=False, indent=2)
        elif output_format == "jsonl":
            with open(output_file, "w", encoding="utf-8") as f:
                for text in results:
                    f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
        else:
            logger.error(f"不支持的输出格式: {output_format}")
            return
            
        logger.info(f"结果已保存到: {output_file} (格式: {output_format})")
    except Exception as e:
        logger.error(f"保存结果失败: {e}")

def interactive_config() -> Dict:
    """交互式配置
    
    Returns:
        配置字典
    """
    questions = [
        inquirer.Text('input_paths', 
                     message="输入路径 (多个路径用逗号分隔)",
                     validate=lambda _, x: len(x) > 0),
        inquirer.Text('output_file',
                     message="输出文件路径",
                     validate=lambda _, x: len(x) > 0),
        inquirer.List('output_format',
                     message="输出格式",
                     choices=['txt', 'json', 'jsonl'],
                     default='txt'),
        inquirer.Checkbox('file_extensions',
                        message="支持的文件扩展名",
                        choices=[".txt", ".json", ".jsonl"],
                        default=[".txt", ".json", ".jsonl"]),
        inquirer.Confirm('preview',
                        message="是否预览数据",
                        default=True),
        inquirer.Text('batch_size',
                     message="批处理大小",
                     default="1000",
                     validate=lambda _, x: x.isdigit() and int(x) > 0),
        inquirer.Text('max_workers',
                     message="最大工作进程数",
                     default=str(min(os.cpu_count(), 8)),
                     validate=lambda _, x: x.isdigit() and int(x) > 0),
        inquirer.Confirm('colored_log',
                        message="是否使用彩色日志",
                        default=True),
    ]
    
    answers = inquirer.prompt(questions)
    if not answers:
        logger.error("交互式配置被取消")
        sys.exit(1)
    
    # 处理输入路径
    answers['input_paths'] = [path.strip() for path in answers['input_paths'].split(',')]
    answers['batch_size'] = int(answers['batch_size'])
    answers['max_workers'] = int(answers['max_workers'])
    
    return answers

def main():
    """主函数"""
    global logger
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description="灵猫墨韵数据处理器 - 用于清洗、转换和去重古籍文本数据",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument("-i", "--input", dest="input_paths", nargs="*",
                        help="输入路径，可以是目录或文件，支持多个路径")
    parser.add_argument("-o", "--output", dest="output_file",
                        help="输出文件路径")
    parser.add_argument("-f", "--format", dest="output_format", choices=["txt", "json", "jsonl"], default="txt",
                        help="输出格式，支持txt、json、jsonl (默认: txt)")
    parser.add_argument("-e", "--extensions", dest="file_extensions", nargs="*", default=[".txt", ".json", ".jsonl"],
                        help="支持的文件扩展名 (默认: .txt .json .jsonl)")
    parser.add_argument("-c", "--config", dest="config_path", default="config/config.yaml",
                        help="配置文件路径 (默认: config/config.yaml)")
    parser.add_argument("-b", "--batch-size", dest="batch_size", type=int, default=1000,
                        help="批处理大小 (默认: 1000)")
    parser.add_argument("-w", "--workers", dest="max_workers", type=int, default=min(os.cpu_count(), 8),
                        help=f"最大工作进程数 (默认: {min(os.cpu_count(), 8)})")
    parser.add_argument("-p", "--preview", dest="preview", action="store_true",
                        help="预览数据")
    parser.add_argument("-n", "--preview-count", dest="preview_count", type=int, default=5,
                        help="预览数据条数 (默认: 5)")
    parser.add_argument("-l", "--log-level", dest="log_level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO",
                        help="日志级别 (默认: INFO)")
    parser.add_argument("--no-color", dest="no_color", action="store_true",
                        help="禁用彩色日志")
    parser.add_argument("--interactive", dest="interactive", action="store_true",
                        help="交互式配置")
    
    args = parser.parse_args()
    
    # 设置日志记录器
    logger = setup_logging(log_level=args.log_level, colored=not args.no_color)
    
    # 加载配置文件
    config = load_config(args.config_path)
    
    # 交互式配置
    if args.interactive:
        interactive_config_dict = interactive_config()
        # 更新参数
        for key, value in interactive_config_dict.items():
            setattr(args, key, value)
    
    # 检查必要参数
    if not args.input_paths:
        if "paths" in config and "input_dir" in config["paths"]:
            args.input_paths = [config["paths"]["input_dir"]]
        else:
            logger.error("未指定输入路径")
            parser.print_help()
            sys.exit(1)
    
    if not args.output_file:
        if "paths" in config and "output_dir" in config["paths"]:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.output_file = os.path.join(
                config["paths"]["output_dir"], 
                f"processed_{timestamp}.{args.output_format}"
            )
        else:
            logger.error("未指定输出文件路径")
            parser.print_help()
            sys.exit(1)
    
    # 显示处理参数
    logger.info("处理参数:")
    logger.info(f"  输入路径: {args.input_paths}")
    logger.info(f"  输出文件: {args.output_file}")
    logger.info(f"  输出格式: {args.output_format}")
    logger.info(f"  文件扩展名: {args.file_extensions}")
    logger.info(f"  批处理大小: {args.batch_size}")
    logger.info(f"  最大工作进程数: {args.max_workers}")
    
    # 开始处理
    start_time = time.time()
    
    # 用于去重的集合
    unique_hashes = set()
    processed_results = []
    
    # 创建进程池
    process_func = partial(process_batch, cleaning_rules=None)
    
    # 读取并处理数据
    total_batches = 0
    total_texts = 0
    total_unique = 0
    
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        # 获取数据批次
        data_generator = read_data(
            args.input_paths, 
            args.file_extensions, 
            args.batch_size,
            args.preview,
            args.preview_count
        )
        
        # 提交批处理任务
        futures = []
        for batch in data_generator:
            total_batches += 1
            total_texts += len(batch)
            futures.append(executor.submit(process_func, batch))
        
        # 处理结果
        for future in tqdm(as_completed(futures), total=len(futures), desc="处理批次", unit="批"):
            try:
                batch_results = future.result()
                for cleaned_text, text_hash in batch_results:
                    if text_hash not in unique_hashes:
                        unique_hashes.add(text_hash)
                        processed_results.append(cleaned_text)
                        total_unique += 1
            except Exception as e:
                logger.error(f"处理批次失败: {e}")
    
    # 保存结果
    save_results(processed_results, args.output_file, args.output_format)
    
    # 显示处理统计
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info("处理统计:")
    logger.info(f"  总批次数: {total_batches}")
    logger.info(f"  总文本数: {total_texts}")
    logger.info(f"  去重后文本数: {total_unique}")
    logger.info(f"  去重率: {(1 - total_unique / total_texts) * 100:.2f}% (如果总文本数为0则忽略)" if total_texts > 0 else "  去重率: N/A")
    logger.info(f"  处理时间: {elapsed_time:.2f} 秒")
    logger.info(f"  处理速度: {total_texts / elapsed_time:.2f} 文本/秒" if elapsed_time > 0 else "  处理速度: N/A")

if __name__ == "__main__":
    main()