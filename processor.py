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
# import nltk # nltk is not used currently, comment out to avoid confusion
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import colorlog
import inquirer
from functools import partial

# --- New Imports for Enhanced Filtering ---
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
import spacy # Required by presidio
from langdetect import detect as langdetect_detect, LangDetectException
# import fasttext # Optional: uncomment if using fasttext for language detection

# --- Global Variables ---
logger = None
# Initialize Presidio components globally or within a function context if preferred
# Lazy initialization might be better for performance if not always used
presidio_analyzer = None
presidio_anonymizer = None
# Load spacy model (consider doing this only if PII redaction is enabled)
# try:
#     nlp = spacy.load("en_core_web_lg") # Or other languages
# except OSError:
#     logger.warning("Spacy model 'en_core_web_lg' not found. PII detection might be limited. Run: python -m spacy download en_core_web_lg")
#     nlp = None

# --- Setup Logging ---
def setup_logging(config) -> logging.Logger:
    '''初始化结构化日志系统（返回专用处理器）
    Returns:
        Logger: 专用于数据处理流程的logger对象
    '''
    # 添加第三方库警告过滤
    import warnings
    from urllib3.exceptions import NotOpenSSLWarning
    from bs4 import MarkupResemblesLocatorWarning
    warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
    warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

    # 创建专用logger而非根logger
    processor_logger = logging.getLogger('data_processor')
    processor_logger.propagate = False  # 防止传播到根logger

    # 从配置获取参数
    log_dir = config.get("log_dir", "logs")
    log_level_str = config.get("log_level", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    
    # 创建日志目录
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(exist_ok=True, parents=True)

    # 清理现有处理器
    processor_logger.handlers.clear()

    # 结构化日志格式
    console_format = colorlog.ColoredFormatter(
        '%(log_color)s[%(asctime)s] %(levelname)-8s %(cyan)s%(name)s:%(lineno)d%(reset)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_format = logging.Formatter(
        '[%(asctime)s] %(levelname)-8s %(name)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # --- 控制台处理器（简洁格式）---
    console_format = "%(log_color)s%(levelname)-8s%(reset)s %(message)s"
    console_handler = colorlog.StreamHandler()
    console_handler.setFormatter(colorlog.ColoredFormatter(console_format))
    console_handler.setLevel(logging.INFO)  # 控制台只显示INFO及以上级别
    
    # --- 文件处理器（详细格式）---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir_path / f"processor_{timestamp}.log"
    file_format = "%(asctime)s - %(name)s - %(levelname)-8s - %(filename)s:%(lineno)d - %(message)s"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter(file_format, datefmt='%Y-%m-%d %H:%M:%S'))
    
    # 添加处理器
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # 抑制第三方库的冗余日志
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("chardet").setLevel(logging.WARNING)
    logging.getLogger("spacy").setLevel(logging.WARNING)

    # 添加处理分隔符
    def log_section_header(logger, title: str, char: str = '=', length: int = 60):
        logger.info(char * length)
        logger.info(f'{title.center(length-4)}')
        logger.info(char * length)

    # 记录初始化完成信息
    log_section_header(processor_logger, '日志系统初始化完成')
    processor_logger.info(f'日志文件: {log_file}')
    processor_logger.info(f'日志级别: {log_level_str}')
    log_section_header(processor_logger, '初始化完成')

    return processor_logger
    except Exception as e:
        # 在日志系统设置失败时，打印到stderr
        print(f"Error setting up logging: {e}", file=sys.stderr)
        # 返回一个基本的、未配置的logger，或者根据需要处理
        return logging.getLogger("processor_fallback")

# --- Load Config ---
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
        logger.info("应用清洗规则配置：\n%s", "\n".join([f"{k}: {v}" for k,v in config.get('cleaning_rules', {}).items()]))
        return config
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        return {}

# --- Detect Encoding ---
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

# --- Read Data ---
def read_data(input_paths: List[str], file_extensions: List[str], 
              batch_size: int = 1000, preview: bool = False, 
              preview_count: int = 5) -> Generator[dict, None, None]:
    """读取数据并返回结构化结果（包含错误统计）
    Yields:
        dict: {
            'texts': List[str],  # 文本批次
            'json_errors': int   # 本批次JSON解析错误数
        }
    """
    batch_json_errors = 0
    
    Args:
        input_paths: 输入路径列表，可以是目录或文件
        file_extensions: 支持的文件扩展名列表，如 [".txt", ".json", ".jsonl"]
        batch_size: 批处理大小
        preview: 是否预览数据
        preview_count: 预览数据条数
        
    Yields:
        包含文本数据和错误计数的字典
    """
    malformed_files_count = 0
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
    
    # 使用tqdm.write替代普通日志输出
    tqdm.write(f'[文件扫描] 发现 {len(all_files)} 个待处理文件', file=sys.stdout)
    
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
                logger.warning(f"预览文件失败 ({type(e).__name__}): {file_path} - {e}")
    
    # 批量读取数据
    batch = []
    file_count = 0
    
    with tqdm(total=len(all_files), desc="读取文件", unit="个文件", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]') as pbar:
        for file_path in all_files:
            file_count += 1
            file_ext = file_path.suffix.lower()
            encoding = detect_encoding(str(file_path))
            
            try:
                with open(file_path, "r", encoding=encoding, errors="replace") as f:
                    if file_ext == ".txt":
                        for line in f:
                            line = line.strip()
                            if line:
                                batch.append(line)
                                if len(batch) >= batch_size:
                                    yield {'texts': batch, 'json_errors': 0}
                                    batch = []
                    elif file_ext == ".json":
                        try:
                            data = json.load(f)
                            if isinstance(data, list):
                                batch.extend([item.get("text", "") if isinstance(item, dict) else str(item) for item in data])
                            elif isinstance(data, dict):
                                batch.append(data.get("text", ""))
                            else:
                                raise json.JSONDecodeError("Invalid JSON structure", "", 0)
                            if batch:
                                yield {'texts': batch, 'json_errors': 0}
                                batch = []
                        except json.JSONDecodeError as e:
                            json_decode_errors += 1
                            logger.warning(f"无效JSON文件已跳过：{file_path} - 错误位置：第{e.lineno}行，列{e.colno}，详情：{e.msg}")
                            yield {'texts': [], 'json_errors': 1}
                    elif file_ext == ".jsonl":
                        file_errors = 0
                        for line in f:
                            try:
                                data = json.loads(line.strip())
                                batch.append(data.get("text", "") if isinstance(data, dict) else str(data))
                                if len(batch) >= batch_size:
                                    yield {'texts': batch, 'json_errors': 0}
                                    batch = []
                            except json.JSONDecodeError as e:
                                file_errors += 1
                                json_decode_errors += 1
                                logger.warning(f"解析JSONL行失败: {file_path} - 错误位置:{e.pos} - 行内容:{line[:50]}...")
                        if file_errors > 0:
                            yield {'texts': batch, 'json_errors': file_errors}
                            batch = []
                pbar.update(1)
            except Exception as e:
                logger.error(f"读取文件失败 ({type(e).__name__}): {file_path} - {e}")
                pbar.update(1)
    
    # 返回最后一个批次
    if batch:
        yield {'texts': batch, 'json_errors': 0}
    return json_decode_errors

# --- Text Cleaning and Filtering ---
def clean_text(text: str, cleaning_rules: Dict[str, Any] = None) -> str:
    """清洗和过滤文本数据
    
    应用一系列规则来清理、规范化和过滤文本。
    
    Args:
        text: 原始文本
        cleaning_rules: 清洗和过滤规则字典
        
    Returns:
        清洗和过滤后的文本，如果文本被过滤则返回空字符串
    """
    if not text or not isinstance(text, str):
        return ""
    
    # 默认清洗和过滤规则 (包含新规则)
    default_rules = {
        # Basic Cleaning
        "remove_html": True,
        "normalize_whitespace": True,
        "remove_control_chars": True,
        "normalize_punctuation": True,
        "remove_urls": True,
        "remove_emojis": True,
        # PII Redaction
        "redact_pii": False,
        "pii_entities": ["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "LOCATION", "CREDIT_CARD", "CRYPTO", "DATE_TIME", "IBAN_CODE", "IP_ADDRESS", "NRP", "MEDICAL_LICENSE", "URL", "US_BANK_NUMBER", "US_DRIVER_LICENSE", "US_ITIN", "US_PASSPORT", "US_SSN"], # Presidio默认支持的部分
        "pii_redaction_method": "replace", # 'replace', 'hash', 'mask', 'remove'
        "pii_replacement_tag": "[PII]",
        "pii_spacy_model": "en_core_web_lg", # Spacy model for Presidio
        # Harmful Content Filtering

        # Quality Filtering
        "filter_quality": False,
        "min_length": 10,
        "max_length": 10000, # Added max length
        "max_symbol_ratio": 0.1,
        "filter_by_language": False,
        "allowed_languages": ["zh", "en"], # Languages to keep (ISO 639-1 codes)
        "lang_detection_method": "langdetect", # 'langdetect' or 'fasttext'
        "fasttext_model_path": None, # Path to fasttext language model if used
        "filter_repetition": False,
        "repetition_ngram_size": 5,
        "repetition_threshold": 0.3 # Max ratio of repeated ngrams
    }
    
    # 使用提供的规则覆盖默认规则
    rules = default_rules.copy()
    if cleaning_rules:
        rules.update(cleaning_rules)

    # --- 1. Basic Cleaning Steps ---
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
    
    # 移除URL (改进正则)
    if rules["remove_urls"]:
        text = regex.sub(r"https?:\/\/(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&\/=]*)", "", text)
        text = regex.sub(r"www\.[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&\/=]*)", "", text)

    # 规范化标点符号
    if rules["normalize_punctuation"]:
        # 移除重复的标点符号
        text = regex.sub(r"([，。！？；：、,.!?;:]){2,}", r"\1", text)
        # 确保中文标点前后没有不必要的空格 (保留英文标点后的空格)
        text = regex.sub(r"\s*([，。！？；：、])\s*", r"\1", text)
        text = regex.sub(r"([,.!?;:])\s+", r"\1 ", text) # Ensure space after English punct
        text = regex.sub(r"\s+([,.!?;:])", r"\1", text) # Remove space before English punct
    
    # 移除表情符号
    if rules["remove_emojis"]:
        text = regex.sub(r"[\U00010000-\U0010ffff]", "", text)

    # --- 2. PII Redaction ---
    if rules.get("redact_pii", False):
        text = pii_redactor(text, rules)
        if not text: # If redaction somehow empties the text
            return ""


        
    # --- 4. Quality Filtering ---
    if rules.get("filter_quality", False):
        # 长度过滤
        min_len = rules.get("min_length", 10)
        max_len = rules.get("max_length", 10000)
        if not (min_len <= len(text) <= max_len):
            logger.debug(f"文本因长度 ({len(text)}) 不在 [{min_len}, {max_len}] 范围内被过滤: {text[:50]}...")
            return ""
            
        # 符号比例过滤
        max_ratio = rules.get("max_symbol_ratio", 0.1)
        if len(text) > 0:
            # More robust symbol detection (non-letter, non-number, non-whitespace)
            symbol_count = len(regex.findall(r'[^\p{L}\p{N}\s]', text))
            if symbol_count / len(text) > max_ratio:
                logger.debug(f"文本因符号比例过高 ({symbol_count / len(text):.2f} > {max_ratio}) 被过滤: {text[:50]}...")
                return ""
                
        # 语言过滤
        if rules.get("filter_by_language", False):
            lang = detect_language(text, rules)
            allowed_langs = rules.get("allowed_languages", ["zh", "en"])
            if lang not in allowed_langs:
                logger.debug(f"文本因语言 ({lang}) 不在允许列表 {allowed_langs} 中被过滤: {text[:50]}...")
                return ""
                
        # 重复度过滤
        if rules.get("filter_repetition", False):
            ngram_size = rules.get("repetition_ngram_size", 5)
            threshold = rules.get("repetition_threshold", 0.3)
            if check_repetition(text, ngram_size, threshold):
                 logger.debug(f"文本因重复度过高 (ngram={ngram_size}, threshold={threshold}) 被过滤: {text[:50]}...")
                 return ""

        # TODO: Add more quality filters (stopword ratio, sentence length distribution, etc.)
        
    return text.strip()

# --- PII Redaction Implementation ---
def initialize_presidio(rules: Dict[str, Any]):
    """Initializes Presidio components based on rules."""
    global presidio_analyzer, presidio_anonymizer
    if presidio_analyzer is None:
        try:
            # Load the specific spacy model defined in rules
            spacy_model_name = rules.get("pii_spacy_model", "en_core_web_lg")
            try:
                nlp = spacy.load(spacy_model_name)
            except OSError:
                logger.error(f"Spacy model '{spacy_model_name}' not found. PII detection may fail. Run: python -m spacy download {spacy_model_name}")
                # Fallback or raise error?
                # For now, let Presidio handle potential errors if NLP engine is missing
                nlp = None 

            logger.info(f"Initializing Presidio Analyzer with Spacy model: {spacy_model_name}")
            presidio_analyzer = AnalyzerEngine(nlp_engine=nlp) # Pass NLP engine if loaded
            presidio_anonymizer = AnonymizerEngine()
            logger.info("Presidio components initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize Presidio: {e}", exc_info=True)
            # Disable PII redaction if initialization fails?
            # rules['redact_pii'] = False # Or handle differently

def pii_redactor(text: str, rules: Dict[str, Any]) -> str:
    """使用 Presidio 编辑文本中的 PII
    
    Args:
        text: 输入文本
        rules: 包含 Presidio 配置的规则字典
        
    Returns:
        编辑 PII 后的文本
    """
    global presidio_analyzer, presidio_anonymizer
    
    # Initialize Presidio if not already done
    if presidio_analyzer is None:
        initialize_presidio(rules)
        # If initialization failed, analyzer might still be None
        if presidio_analyzer is None:
            logger.warning("Presidio Analyzer not available, skipping PII redaction.")
            return text
            
    try:
        supported_entities = rules.get("pii_entities", [])
        analyzer_results = presidio_analyzer.analyze(text=text, 
                                                   entities=supported_entities, 
                                                   language='en') # TODO: Make language configurable
        
        if not analyzer_results:
            return text # No PII found
            
        # Configure anonymization based on rules
        redaction_method = rules.get("pii_redaction_method", "replace")
        replacement_tag = rules.get("pii_replacement_tag", "[PII]")
        
        operator_config = {}
        if redaction_method == "replace":
            operator_config = {"DEFAULT": OperatorConfig("replace", {"new_value": replacement_tag})}
        elif redaction_method == "mask":
             operator_config = {"DEFAULT": OperatorConfig("mask", {"masking_char": "*", "chars_to_mask": 4, "from_end": False})}
        elif redaction_method == "hash":
             operator_config = {"DEFAULT": OperatorConfig("hash", {"hash_type": "sha256"})}
        # 'remove' is handled by default if no operator is specified for an entity
        # else: # Default to replace if method is unknown
        #     operator_config = {"DEFAULT": OperatorConfig("replace", {"new_value": replacement_tag})}

        anonymized_result = presidio_anonymizer.anonymize(
            text=text, 
            analyzer_results=analyzer_results,
            operators=operator_config
        )
        
        logger.debug(f"PII Redaction applied. Original length: {len(text)}, New length: {len(anonymized_result.text)}")
        return anonymized_result.text
        
    except Exception as e:
        logger.error(f"Presidio PII redaction failed: {e}", exc_info=True)
        return text # Return original text on error

# --- Harmful Content Filtering Implementation ---


# --- Quality Filtering Helpers ---
def detect_language(text: str, rules: Dict[str, Any]) -> Optional[str]:
    """Detects the language of the text."""
    method = rules.get("lang_detection_method", "langdetect")
    
    if method == "langdetect":
        try:
            # Limit text length for langdetect performance
            lang = langdetect_detect(text[:500])
            return lang
        except LangDetectException:
            logger.debug(f"Langdetect failed for text: {text[:50]}...")
            return None
        except Exception as e:
            logger.warning(f"Error during langdetect: {e}")
            return None
    elif method == "fasttext":
        model_path = rules.get("fasttext_model_path")
        if not model_path or not Path(model_path).exists():
            logger.warning(f"FastText model path '{model_path}' not found or not specified. Cannot detect language.")
            return None
        try:
            # TODO: Implement fasttext detection
            # import fasttext
            # model = fasttext.load_model(model_path)
            # # Predict requires text with newline
            # predictions = model.predict(text.replace('\n', ' '), k=1)
            # lang = predictions[0][0].replace('__label__', '')
            # return lang
            logger.warning("FastText language detection not yet implemented.")
            return None # Placeholder
        except Exception as e:
            logger.error(f"Error during FastText language detection: {e}", exc_info=True)
            return None
    else:
        logger.warning(f"Unsupported language detection method: {method}")
        return None

def check_repetition(text: str, ngram_size: int, threshold: float) -> bool:
    """Checks for excessive repetition of n-grams."""
    if len(text) < ngram_size * 2: # Need enough text to compare
        return False
    try:
        words = regex.findall(r'\p{L}+', text.lower()) # Simple word tokenization
        if len(words) < ngram_size * 2:
            return False
            
        ngrams = [' '.join(words[i:i+ngram_size]) for i in range(len(words) - ngram_size + 1)]
        if not ngrams:
            return False
            
        ngram_counts = {} 
        for ng in ngrams:
            ngram_counts[ng] = ngram_counts.get(ng, 0) + 1
            
        # Calculate repetition ratio based on the most frequent ngram
        # More sophisticated methods exist (e.g., ratio of unique ngrams)
        max_count = 0
        if ngram_counts:
             max_count = max(ngram_counts.values())
             
        repetition_ratio = (max_count / len(ngrams)) if len(ngrams) > 0 else 0
        
        # Alternative: Ratio of duplicate ngrams
        # duplicate_ngram_count = sum(1 for count in ngram_counts.values() if count > 1)
        # repetition_ratio = duplicate_ngram_count / len(ngrams) if len(ngrams) > 0 else 0

        return repetition_ratio > threshold
    except Exception as e:
        logger.warning(f"Error during repetition check: {e}")
        return False # Don't filter if check fails

# --- Process Batch ---
def process_batch(batch: List[str], cleaning_rules: Dict[str, Any] = None) -> List[Tuple[str, str]]:
    """处理一批文本数据
    
    Args:
        batch: 文本数据批次
        cleaning_rules: 清洗和过滤规则
        
    Returns:
        处理后的文本及其哈希值列表 (过滤掉的文本不包含在内)
    """
    results = []
    for text in batch:
        processed_text = clean_text(text, cleaning_rules)
        if processed_text:  # Only include non-empty results (i.e., not filtered out)
            # 计算文本哈希值用于去重
            text_hash = hashlib.md5(processed_text.encode("utf-8")).hexdigest()
            results.append((processed_text, text_hash))
    return results

# --- Save Results ---
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

# --- Interactive Config ---
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
        description="灵猫墨韵数据处理器 - 用于清洗、转换和去重古籍文本数据，为大语言模型训练准备高质量数据集。",
    
    # 添加命令行参数
    parser.add_argument("-i", "--input", nargs="+", required=True,
                      help="输入路径（文件或目录）")
    parser.add_argument("-o", "--output", required=True,
                      help="输出文件路径（JSONL格式）")
    parser.add_argument("--log_dir", default="logs",
                      help="日志文件存储目录")
    parser.add_argument("--log_level", default="INFO",
                      choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                      help="日志级别")
    parser.add_argument("--batch_size", type=int, default=1000,
                      help="批处理大小")
    args = parser.parse_args()

    # 初始化结构化日志系统
    logger = setup_logging(vars(args))
    logger.info('\n' + '='*60)
    logger.info('启动数据处理流程'.center(60))
    logger.info('='*60)

    # 加载配置文件
    config = load_config()
    
    # 显示清洗规则配置
    tqdm.write('\n' + '='*60)
    tqdm.write('当前应用的清洗规则配置：'.center(60))
    tqdm.write('='*60)
    for rule, params in config.get('cleaning_rules', {}).items():
        tqdm.write(f'{rule.upper():<20}: {params}')
    tqdm.write('='*60 + '\n')
    
    # 输入/输出参数组
    io_group = parser.add_argument_group('输入/输出选项')
    io_group.add_argument("-i", "--input", dest="input_paths", nargs="*",
                        help="指定一个或多个输入文件或目录路径。可以是包含文本文件的目录，或直接指定文件。")
    io_group.add_argument("-o", "--output", dest="output_file",
                        help="指定处理结果的输出文件路径。如果未指定，将根据配置文件或默认规则生成。")
    io_group.add_argument("-f", "--format", dest="output_format", choices=["txt", "json", "jsonl"], default="txt",
                        help="指定输出文件的格式。可选 'txt', 'json', 'jsonl'。")
    io_group.add_argument("-e", "--extensions", dest="file_extensions", nargs="*", default=[".txt", ".json", ".jsonl"],
                        help="指定要处理的文件扩展名列表。")

    # 处理参数组
    processing_group = parser.add_argument_group('处理选项')
    processing_group.add_argument("-b", "--batch-size", type=int, default=1000,
                                help="指定批处理的大小，即一次处理多少条文本。")
    processing_group.add_argument("-w", "--workers", dest="max_workers", type=int, default=min(os.cpu_count(), 8),
                                help="指定用于并行处理的最大工作进程数。")
    processing_group.add_argument("--preview", action="store_true", default=False,
                                help="预览输入文件的前几行内容，不进行实际处理。")
    processing_group.add_argument("--preview-count", type=int, default=5,
                                help="预览模式下显示的文件数量。")

    # 清洗规则参数组
    cleaning_group = parser.add_argument_group('清洗规则选项')
    # 可以添加参数来覆盖配置文件中的清洗规则
    cleaning_group.add_argument("--no-html", action="store_false", dest="remove_html", help="禁用HTML标签移除")
    cleaning_group.add_argument("--no-whitespace-norm", action="store_false", dest="normalize_whitespace", help="禁用空白字符规范化")
    cleaning_group.add_argument("--no-control-chars", action="store_false", dest="remove_control_chars", help="禁用控制字符移除")
    cleaning_group.add_argument("--no-punct-norm", action="store_false", dest="normalize_punctuation", help="禁用标点符号规范化")
    cleaning_group.add_argument("--no-urls", action="store_false", dest="remove_urls", help="禁用URL移除")
    cleaning_group.add_argument("--no-emojis", action="store_false", dest="remove_emojis", help="禁用表情符号移除")
    # 新增规则开关
    cleaning_group.add_argument("--redact-pii", action="store_true", default=False, help="启用PII编辑 (当前为占位符)")
    cleaning_group.add_argument("--filter-quality", action="store_true", default=False, help="启用基本质量过滤 (当前为占位符)")
    cleaning_group.add_argument("--min-length", type=int, default=10, help="质量过滤：最小文本长度")
    cleaning_group.add_argument("--max-symbol-ratio", type=float, default=0.1, help="质量过滤：最大符号比例")
    # ... other keys ...

    # 其他参数组
    other_group = parser.add_argument_group('其他选项')
    other_group.add_argument("-c", "--config", dest="config_path", default="config/config.yaml",
                         help="指定配置文件的路径。")
    other_group.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                         help="设置日志记录级别。")
    other_group.add_argument("--no-color", action="store_true", default=False,
                         help="禁用控制台彩色日志输出。")
    other_group.add_argument("--interactive", action="store_true", default=False,
                         help="使用交互式命令行界面配置参数。")

    args = parser.parse_args()

    # 将清洗规则相关的 args 传递给 cleaning_rules_config (在 main 函数中处理)
    # 注意: argparse 的 action='store_false' 需要特殊处理，如果提供了flag，值为False
    # 我们需要在 main 函数中构建最终的 cleaning_rules 字典
    
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
    
    # 获取清洗规则 (合并配置和命令行参数，命令行优先)
    cleaning_rules_config = config.get("cleaning_rules", {}).copy() # 使用副本以防修改原始配置
    
    # 从 args 更新清洗规则配置，命令行参数优先
    # 使用 getattr 安全地获取属性，避免 AttributeError
    # 对于 action='store_false'，如果命令行提供了flag，args对应属性为False
    # 对于 action='store_true'，如果命令行提供了flag，args对应属性为True
    rule_args = {
        "remove_html": getattr(args, 'remove_html', None),
        "normalize_whitespace": getattr(args, 'normalize_whitespace', None),
        "remove_control_chars": getattr(args, 'remove_control_chars', None),
        "normalize_punctuation": getattr(args, 'normalize_punctuation', None),
        "remove_urls": getattr(args, 'remove_urls', None),
        "remove_emojis": getattr(args, 'remove_emojis', None),
        "redact_pii": getattr(args, 'redact_pii', None),
        "filter_quality": getattr(args, 'filter_quality', None),
        "min_length": getattr(args, 'min_length', None),
        "max_symbol_ratio": getattr(args, 'max_symbol_ratio', None),
        "filter_harmful": getattr(args, 'filter_harmful', None)
    }
    
    for key, value in rule_args.items():
        # 只有当命令行参数被明确设置时才覆盖 (对于bool类型，检查是否非None；对于其他类型，也检查非None)
        # 注意：args的默认值可能与config不同，这里逻辑是命令行指定了就覆盖
        # 获取 argparse 定义的默认值，用于判断命令行参数是否被用户显式设置
        arg_defaults = {action.dest: action.default for action in parser._actions if hasattr(action, 'dest')}

        # 定义命令行参数与配置键的映射 (在此例中它们相同)
        rule_arg_keys = [
            "remove_html", "normalize_whitespace", "remove_control_chars",
            "normalize_punctuation", "remove_urls", "remove_emojis",
            "redact_pii", "filter_quality", "min_length", "max_symbol_ratio",
            "filter_harmful"
            # 注意: 如果为 config.yaml 中的其他 cleaning_rules 添加了命令行参数,
            # 例如 --pii-spacy-model, 需要将对应的 dest 名称添加到此列表
        ]

        # 遍历相关参数，如果命令行值与 argparse 默认值不同，则覆盖配置
        for key in rule_arg_keys:
            if hasattr(args, key):
                arg_value = getattr(args, key)
                # 检查命令行参数值是否与其 argparse 默认值不同
                if key in arg_defaults and arg_value != arg_defaults[key]:
                    logger.debug(f"使用命令行参数覆盖配置 '{key}': {arg_value} (默认值: {arg_defaults[key]})")
                    cleaning_rules_config[key] = arg_value
                # 如果参数在 args 中但不在 arg_defaults 中 (理论上不应发生), 也应用命令行值
                elif key not in arg_defaults:
                    logger.debug(f"应用命令行参数值 '{key}': {arg_value} (未找到 argparse 默认值)")
                    cleaning_rules_config[key] = arg_value

        # 确保数值类型的规则存在且有效
        cleaning_rules_config['min_length'] = int(cleaning_rules_config.get('min_length', 10))
        cleaning_rules_config['max_symbol_ratio'] = float(cleaning_rules_config.get('max_symbol_ratio', 0.1))
        # 对其他需要特定类型的配置进行类似的检查和转换 (例如 max_length, harmful_threshold 等)
        if 'max_length' in cleaning_rules_config:
            cleaning_rules_config['max_length'] = int(cleaning_rules_config['max_length'])
        if 'harmful_threshold' in cleaning_rules_config:
            cleaning_rules_config['harmful_threshold'] = float(cleaning_rules_config['harmful_threshold'])
        if 'repetition_ngram_size' in cleaning_rules_config:
            cleaning_rules_config['repetition_ngram_size'] = int(cleaning_rules_config['repetition_ngram_size'])
        if 'repetition_threshold' in cleaning_rules_config:
            cleaning_rules_config['repetition_threshold'] = float(cleaning_rules_config['repetition_threshold'])

        logger.info("=" * 50)
        logger.info("Starting Data Processing")
        logger.info("-" * 50)
        logger.info("Effective Cleaning Rules:")
        for key, value in cleaning_rules_config.items():
            logger.info(f"  - {key}: {value}")
        logger.info("-" * 50)
 
      # 创建进程池
    process_func = partial(process_batch, cleaning_rules=cleaning_rules_config)
    
    # 读取并处理数据
    total_batches = 0
    total_texts = 0
    total_unique = 0

    # 初始化统计指标
    total_texts_processed = 0
    total_json_errors = 0
    unique_texts = set()
    start_time = time.time()

    try:
        with ProcessPoolExecutor() as executor:
            futures = []
            with open(output_path, 'w', encoding='utf-8') as f_out:
                # 使用增强型进度条
                with tqdm(total=total_files, desc="处理进度", unit="file", 
                         bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [剩余:{remaining}]') as pbar:
                    
                    for data_batch in read_data(args.input, [".txt", ".json", ".jsonl"], 
                                              args.batch_size, args.preview, args.preview_count):
                        # 累加批次级错误
                        total_json_errors += data_batch.get('json_errors', 0)
                        
                        # 提交处理任务
                        futures.append(executor.submit(
                            process_batch, 
                            data_batch['texts'], 
                            config
                        ))
                        
                        # 更新进度条并处理完成的任务
                        pbar.update(1)
                        for future in as_completed(futures):
                            processed_results = future.result()
                            if processed_results:
                                # 保存处理结果
                                processed_results_to_save = [
                                    {'text': text, 'hash': hashlib.md5(text.encode()).hexdigest()}
                                    for text in processed_results
                                ]
                                for result in processed_results_to_save:
                                    f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
                                    unique_texts.add(result['text'])
                                    total_texts_processed += 1
                            futures.remove(future)
    
    # 保存结果
    save_results(processed_results, args.output_file, args.output_format)
    
    # 显示处理统计
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info("处理统计:")
    logger.info(f"  总批次数: {total_batches}")
    logger.info(f"  总文本数: {total_texts}")
    logger.info(f"  格式错误文件数: {total_malformed}")
    logger.info(f"  去重后文本数: {total_unique}")
    logger.info(f"  去重率: {(1 - total_unique / total_texts) * 100:.2f}% (如果总文本数为0则忽略)" if total_texts > 0 else "  去重率: N/A")
    logger.info(f"  处理时间: {elapsed_time:.2f} 秒")
    logger.info(f"  处理速度: {total_texts / elapsed_time:.2f} 文本/秒" if elapsed_time > 0 else "  处理速度: N/A")

if __name__ == "__main__":
    main()
    
    # 在主循环中添加错误计数器
    json_decode_errors = 0
    
    try:
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            data_generator = read_data(
                args.input_paths,
                args.file_extensions,
                args.batch_size,
                args.preview,
                args.preview_count
            )
            
            # 处理数据批次
            for batch_data in tqdm(data_generator, desc="处理数据批次", unit="batch"):
                json_decode_errors += batch_data['json_errors']
                # ... existing processing logic ...
    
    except StopIteration as e:
        json_decode_errors += e.value
    
    # 在最终总结中显示错误计数
    if json_decode_errors > 0:
        logger.warning(f"JSON解析错误总数: {json_decode_errors}")
    
    # ... rest of existing code ...