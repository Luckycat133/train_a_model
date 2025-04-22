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

# --- Define Project Root ---
# Assuming the script is in the project root or a subdirectory
PROJECT_ROOT = Path(__file__).resolve().parent

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
    try:
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
        # Use PROJECT_ROOT to ensure logs are always relative to the project base
        log_dir_relative = config.get("log_dir", "logs")
        log_dir_path = PROJECT_ROOT / log_dir_relative
        log_level_str = config.get("log_level", "INFO").upper()
        log_level = getattr(logging, log_level_str, logging.INFO)
    
    # 创建日志目录
    # log_dir_path = Path(log_dir) # Removed as log_dir_path is now defined above
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
    # Explicitly set stream to stdout
    console_handler = colorlog.StreamHandler(sys.stdout)
    console_handler.setFormatter(colorlog.ColoredFormatter(console_format))
    console_handler.setLevel(logging.INFO)  # 控制台只显示INFO及以上级别
    
    # --- 文件处理器（详细格式）---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir_path / f"processor_{timestamp}.log"
    file_format = "%(asctime)s - %(name)s - %(levelname)-8s - %(filename)s:%(lineno)d - %(message)s"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter(file_format, datefmt='%Y-%m-%d %H:%M:%S'))
    
    # 添加处理器
    # Add handlers to the specific logger, not the root logger
    processor_logger.addHandler(console_handler)
    processor_logger.addHandler(file_handler)

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
        # Resolve config path relative to project root if it's not absolute
        config_file = Path(config_path)
        if not config_file.is_absolute():
            config_file = PROJECT_ROOT / config_path

        if not config_file.exists():
            # Fallback or error if config doesn't exist
            if logger: # Check if logger is initialized
                 logger.error(f"配置文件未找到: {config_file}")
            else:
                 print(f"Error: Config file not found: {config_file}", file=sys.stderr)
            return {}

        with open(config_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
       if logger:
            logger.info(f"配置文件加载成功: {config_file}")
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
        # Use tqdm.write for progress-related info if called within loop context
        # Or keep as debug log if called outside loops
        # logger.debug(f"检测到文件编码: {encoding} (置信度: {confidence:.2f}) - {file_path}")
        return encoding
    except Exception as e:
        # Use logger.warning for issues outside the main processing loop
        if logger:
             logger.warning(f"编码检测失败，使用默认编码 utf-8: {e} - {file_path}")
        return 'utf-8'

# --- Read Data ---
def read_data(input_paths: List[str], file_extensions: List[str],
              batch_size: int = 1000, preview: bool = False,
              preview_count: int = 5) -> Generator[dict, None, Tuple[int, int]]: # Return tuple (json_errors, malformed_files)
    """读取数据并返回结构化结果（包含错误统计）
    Yields:
        dict: {
            'texts': List[str],  # 文本批次
            'json_errors': int   # 本批次JSON解析错误数
        }
    Returns:
        Tuple[int, int]: (total_json_errors, total_malformed_files)
    """
    total_json_errors = 0
    malformed_files_count = 0

    # 展开所有文件路径
    all_files = []
    for input_path_str in input_paths:
        input_path = Path(input_path_str)
        # Resolve relative paths against project root if they don't exist as absolute
        if not input_path.is_absolute() and not input_path.exists():
             input_path = PROJECT_ROOT / input_path_str

        if input_path.is_dir():
            # 如果是目录，递归查找所有匹配的文件
            for ext in file_extensions:
                try:
                    all_files.extend(list(input_path.glob(f"**/*{ext}")))
                except Exception as e:
                     if logger:
                         logger.warning(f"扫描目录时出错 {input_path} for {ext}: {e}")
        elif input_path.is_file() and input_path.suffix.lower() in file_extensions:
            # 如果是文件且扩展名匹配，直接添加
            all_files.append(input_path)
        elif not input_path.exists():
             if logger:
                 logger.warning(f"输入路径不存在，已跳过: {input_path_str} (Resolved: {input_path})")

    # 使用tqdm.write替代普通日志输出 (确保在 tqdm context 外或之前打印)
    if logger:
        logger.info(f'[文件扫描] 发现 {len(all_files)} 个待处理文件')

    # 预览模式 (在 tqdm 循环外执行)
    if preview and all_files:
        preview_files = all_files[:min(preview_count, len(all_files))]
        if logger:
             logger.info(f"数据预览 (来自 {len(preview_files)} 个文件):")
        for file_path in preview_files:
            encoding = detect_encoding(str(file_path))
            try:
                with open(file_path, "r", encoding=encoding, errors="replace") as f:
                    content = f.read(500)  # 读取前500个字符
                    if logger:
                         logger.info(f"\n文件: {file_path}\n内容预览:\n{content}...")
            except Exception as e:
                if logger:
                     logger.warning(f"预览文件失败 ({type(e).__name__}): {file_path} - {e}")

    # 批量读取数据
    batch = []
    file_count = 0
    json_decode_errors = 0 # Errors within the current file being processed

    # Explicitly set tqdm file to stderr
    with tqdm(total=len(all_files), desc="读取文件", unit="个文件", 
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]',
              file=sys.stderr) as pbar:
        for file_path in all_files:
            file_count += 1
            file_ext = file_path.suffix.lower()
            encoding = detect_encoding(str(file_path))
            file_json_errors_in_batch = 0 # Errors for the current file

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
                            # else: # Allow other valid JSON structures, just don't extract text
                            #     pass # Or log a debug message
                            # Yield immediately if batch is full after processing the JSON file
                            while len(batch) >= batch_size:
                                yield {'texts': batch[:batch_size], 'json_errors': 0}
                                batch = batch[batch_size:]
                        except json.JSONDecodeError as e:
                            # Use tqdm.write for non-critical errors inside the loop
                            tqdm.write(f"[WARN] 无效JSON文件已跳过：{file_path} - {e.msg} (line {e.lineno} col {e.colno})", file=sys.stderr)
                            json_decode_errors += 1 # Increment local counter for the generator's return
                            total_json_errors += 1 # Increment global counter (if needed elsewhere)
                            malformed_files_count += 1 # Count malformed files
                    elif file_ext == ".jsonl":
                        file_errors = 0
                        for line_num, line in enumerate(f, 1):
                            try:
                                data = json.loads(line.strip())
                                batch.append(data.get("text", "") if isinstance(data, dict) else str(data))
                                if len(batch) >= batch_size:
                                    yield {'texts': batch, 'json_errors': 0}
                                    batch = []
                            except json.JSONDecodeError as e:
                                file_errors += 1
                                json_decode_errors += 1 # Increment local counter
                                total_json_errors += 1 # Increment global counter
                                # Use tqdm.write for non-critical errors inside the loop
                                tqdm.write(f"[WARN] 解析JSONL行失败: {file_path} - Line:{line_num} Pos:{e.pos} - Line:'{line.strip()[:50]}...'", file=sys.stderr)
                        if file_errors > 0:
                            malformed_files_count += 1 # Count malformed files
                            # Yield remaining batch along with errors from this file
                            # Only yield if there's content in the batch
                            if batch:
                                 yield {'texts': batch, 'json_errors': file_errors}
                                 batch = [] # Reset batch after yielding
                            # If batch is empty but there were errors, just count the malformed file

                pbar.update(1)
                pbar.set_postfix(malformed=malformed_files_count, refresh=True)
            except Exception as e:
                # Keep logger.error for significant file reading errors
                if logger:
                     logger.error(f"读取文件失败 ({type(e).__name__}): {file_path} - {e}")
                malformed_files_count += 1 # Count files that failed to open/read
                pbar.update(1)
                pbar.set_postfix(malformed=malformed_files_count, refresh=True)

    # 返回最后一个批次
    if batch:
        yield {'texts': batch, 'json_errors': 0}

    # Return the total count of JSON decoding errors encountered by this generator
    # And the count of malformed/skipped files
    # Using StopIteration to return values is a common pattern for generators
    # The actual return happens implicitly when the generator finishes
    # We store the values to be returned
    final_return_values = (json_decode_errors, malformed_files_count)

    # Log summary before finishing the generator
    if logger:
        logger.info(f"文件读取完成. 总JSON解析错误: {json_decode_errors}, 无法处理/格式错误文件数: {malformed_files_count}")

    # This return statement is for type hinting and clarity;
    # the actual values are passed via StopIteration when the generator exhausts.
    return final_return_values

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
        "filter_harmful": False, # Add toggle for harmful content
        "harmful_categories": ["hate", "sexual", "violence"], # Example categories
        "harmful_threshold": 0.7, # Confidence threshold
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
    if rules.get("remove_html", False):
        try:
            text = BeautifulSoup(text, "html.parser").get_text()
        except Exception as e:
            # Log warning if cleaning fails, but don't stop processing
            # Use logger here as it's less frequent than per-line errors
            if logger:
                 logger.warning(f"BeautifulSoup处理失败: {e} for text starting with: {text[:50]}...")

    # 移除控制字符
    if rules.get("remove_control_chars", False):
        text = regex.sub(r"[\p{C}&&[^\n\t]]", "", text)

    # 规范化空白字符
    if rules.get("normalize_whitespace", False):
        text = regex.sub(r"\s+", " ", text).strip()

    # 移除URL (改进正则)
    if rules.get("remove_urls", False):
        text = regex.sub(r"https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&/=]*)", "", text)
        text = regex.sub(r"www\.[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&/=]*)", "", text)

    # 规范化标点符号
    if rules.get("normalize_punctuation", False):
        # 移除重复的标点符号
        text = regex.sub(r"([，。！？；：、,.!?;:]){2,}", r"\1", text)
        # 确保中文标点前后没有不必要的空格 (保留英文标点后的空格)
        text = regex.sub(r"\s*([，。！？；：、])\s*", r"\1", text)
        text = regex.sub(r"([,.!?;:])\s+", r"\1 ", text) # Ensure space after English punct
        text = regex.sub(r"\s+([,.!?;:])", r"\1", text) # Remove space before English punct

    # 移除表情符号
    if rules.get("remove_emojis", False):
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
    """处理一批文本数据，应用清洗规则并计算哈希值。
    
    Args:
        batch: 文本数据批次。
        cleaning_rules: 清洗和过滤规则字典。
        
    Returns:
        处理后的文本及其MD5哈希值列表 (过滤掉的文本不包含在内)。
    """
    results = []
    # Apply cleaning rules within the batch processing
    # This part seems correct, just ensuring cleaning_rules are passed
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
    global presidio_analyzer, presidio_anonymizer # Allow modification
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(
         description="灵猫墨韵数据处理器 - 清洗、转换和去重文本数据",
         formatter_class=argparse.ArgumentDefaultsHelpFormatter # Show defaults
     )
     
     # 添加命令行参数
     parser.add_argument("-i", "--input", nargs='+', 
                         default=["collection/"], # Default to collection directory
                         help="输入文件或目录路径列表")
     parser.add_argument("-o", "--output-file", 
                         default="dataset/preprocessed_data.txt", # Default output file
                         help="输出文件路径")
     parser.add_argument("-f", "--output-format", 
                         choices=["txt", "json", "jsonl"], 
                         help="指定输出文件的格式。可选 'txt', 'json', 'jsonl'。")
     parser.add_argument("-e", "--extensions", dest="file_extensions", nargs="*", default=[".txt", ".json", ".jsonl"],
                         help="指定要处理的文件扩展名列表。")

    # 处理参数组
    processing_group = parser.add_argument_group('处理选项')
    processing_group.add_argument("--batch-size", type=int, default=1000,
                                help="批处理大小")
    processing_group.add_argument("--max-workers", type=int, default=min(os.cpu_count(), 8),
                                help="最大工作进程数")
    processing_group.add_argument("--preview", action="store_true", default=False,
                                help="预览输入文件的前几行")
    processing_group.add_argument("--preview-count", type=int, default=5,
                                help="预览的文件数量")

    # 日志参数组
    log_group = parser.add_argument_group('日志选项')
    log_group.add_argument("--log-dir", default="logs",
                         help="日志文件存储目录")
    log_group.add_argument("--log-level", default="INFO",
                         choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                         help="日志级别")
    # log_group.add_argument("--colored-log", action="store_true", default=True, # Colorlog handles this
    #                      help="是否使用彩色日志")

    # 清洗规则参数组 (允许命令行覆盖config.yaml)
    cleaning_group = parser.add_argument_group('清洗与过滤规则 (覆盖配置文件)')
    cleaning_group.add_argument("--remove-html", action=argparse.BooleanOptionalAction, help="移除HTML标签")
    cleaning_group.add_argument("--normalize-whitespace", action=argparse.BooleanOptionalAction, help="规范化空白字符")
    cleaning_group.add_argument("--remove-control-chars", action=argparse.BooleanOptionalAction, help="移除控制字符")
    cleaning_group.add_argument("--normalize-punctuation", action=argparse.BooleanOptionalAction, help="规范化标点符号")
    cleaning_group.add_argument("--remove-urls", action=argparse.BooleanOptionalAction, help="移除URL")
    cleaning_group.add_argument("--remove-emojis", action=argparse.BooleanOptionalAction, help="移除表情符号")
    cleaning_group.add_argument("--redact-pii", action=argparse.BooleanOptionalAction, help="启用PII（个人身份信息）编辑")
    cleaning_group.add_argument("--filter-quality", action=argparse.BooleanOptionalAction, help="启用质量过滤")
    cleaning_group.add_argument("--min-length", type=int, help="质量过滤：最小文本长度")
    cleaning_group.add_argument("--max-symbol-ratio", type=float, help="质量过滤：最大符号比例")
    cleaning_group.add_argument("--filter-harmful", action=argparse.BooleanOptionalAction, help="启用有害内容过滤")
    # Add more specific args if needed, e.g., --pii-spacy-model

    args = parser.parse_args()

    # --- 初始化日志系统 --- (使用命令行参数和默认值)
    log_config = {
        "log_dir": args.log_dir,
        "log_level": args.log_level
    }
    logger = setup_logging(log_config)
    logger.info('\n' + '='*60)
    logger.info('启动数据处理流程'.center(60))
    logger.info('='*60)

    # --- 加载配置文件 --- (config.yaml 优先，命令行可覆盖部分规则)
    config = load_config() # Loads from default path "config/config.yaml"
    if not config:
        logger.error("无法加载配置，使用默认值和命令行参数。")
        config = {} # Ensure config is a dict

    # --- 清理之前的输出文件 --- #
    output_file_path = PROJECT_ROOT / args.output_file
    if output_file_path.exists():
        logger.warning(f"警告：输出文件 '{args.output_file}' 已存在，将被覆盖。")
        try:
            output_file_path.unlink()
            logger.info(f"已删除旧的输出文件: {args.output_file}")
        except OSError as e:
            logger.error(f"无法删除旧的输出文件 '{args.output_file}': {e}", exc_info=True)
            # 根据需要决定是否退出
            # sys.exit(1)

    # --- 合并配置与命令行参数 --- #
    # 命令行参数优先覆盖 config.yaml 中的 cleaning_rules
    cleaning_rules_config = config.get('cleaning_rules', {}).copy() # Start with config rules

    # 获取 argparse 定义的默认值，用于判断命令行参数是否被用户显式设置
    arg_defaults = {action.dest: action.default for action in parser._actions if hasattr(action, 'dest')}

    # 定义命令行参数与配置键的映射 (在此例中它们大部分相同)
    rule_arg_keys = [
        "remove_html", "normalize_whitespace", "remove_control_chars",
        "normalize_punctuation", "remove_urls", "remove_emojis",
        "redact_pii", "filter_quality", "min_length", "max_symbol_ratio",
        "filter_harmful"
        # Add other cleaning rule keys here if they have corresponding cmd args
    ]

    # 遍历相关参数，如果命令行值与 argparse 默认值不同，则覆盖配置
    for key in rule_arg_keys:
        if hasattr(args, key):
            arg_value = getattr(args, key)
            # Check if the argument was explicitly set (not None for optional args, different from default for others)
            # For BooleanOptionalAction, None means not set, True/False means set.
            if arg_value is not None:
                logger.debug(f"使用命令行参数覆盖配置 '{key}': {arg_value}")
                cleaning_rules_config[key] = arg_value
            # If the arg wasn't set (is None), but it exists in config, keep the config value.
            # If it wasn't set and not in config, it will use the default defined in clean_text.

    # 确保数值类型的规则存在且有效 (从合并后的配置中获取)
    cleaning_rules_config['min_length'] = int(cleaning_rules_config.get('min_length', 10))
    cleaning_rules_config['max_symbol_ratio'] = float(cleaning_rules_config.get('max_symbol_ratio', 0.1))
    # 对其他需要特定类型的配置进行类似的检查和转换
    if 'max_length' in cleaning_rules_config:
        cleaning_rules_config['max_length'] = int(cleaning_rules_config['max_length'])
    if 'harmful_threshold' in cleaning_rules_config:
        cleaning_rules_config['harmful_threshold'] = float(cleaning_rules_config['harmful_threshold'])
    if 'repetition_ngram_size' in cleaning_rules_config:
        cleaning_rules_config['repetition_ngram_size'] = int(cleaning_rules_config['repetition_ngram_size'])
    if 'repetition_threshold' in cleaning_rules_config:
        cleaning_rules_config['repetition_threshold'] = float(cleaning_rules_config['repetition_threshold'])

    # --- 初始化 PII Redaction (如果启用) ---
    if cleaning_rules_config.get('redact_pii'):
        try:
            # Lazy load spacy model only when needed
            spacy_model = cleaning_rules_config.get("pii_spacy_model", "en_core_web_lg")
            try:
                nlp = spacy.load(spacy_model)
            except OSError:
                logger.warning(f"Spacy model '{spacy_model}' not found. PII detection might be limited. Run: python -m spacy download {spacy_model}")
                nlp = None # Continue without Spacy if model not found

            if nlp:
                presidio_analyzer = AnalyzerEngine(nlp_engine=nlp, supported_languages=["en"]) # Adjust languages if needed
                presidio_anonymizer = AnonymizerEngine()
                logger.info("Presidio PII Redaction Engine 初始化成功")
            else:
                 logger.warning("无法加载 Spacy 模型，PII 编辑功能受限。")
                 cleaning_rules_config['redact_pii'] = False # Disable if spacy failed

        except ImportError:
            logger.warning("Presidio 或 Spacy 未安装。PII 编辑功能已禁用。请运行 'pip install presidio-analyzer presidio-anonymizer spacy'")
            cleaning_rules_config['redact_pii'] = False # Disable if import fails
        except Exception as e:
            logger.error(f"初始化 Presidio 时出错: {e}", exc_info=True)
            cleaning_rules_config['redact_pii'] = False # Disable on other errors

    logger.info("=" * 50)
    logger.info("开始数据处理")
    logger.info("-" * 50)
    logger.info("生效的清洗规则:")
    for key, value in cleaning_rules_config.items():
        logger.info(f"  - {key}: {value}")
    logger.info("-" * 50)
 
    # --- 数据处理 --- #
    process_func = partial(process_batch, cleaning_rules=cleaning_rules_config)
    
    # 初始化统计指标
    total_texts_processed = 0
    total_json_errors_returned = 0 # Errors returned by the generator
    total_malformed_files_returned = 0 # Malformed files returned by the generator
    unique_hashes = set()
    output_path = PROJECT_ROOT / args.output_file # Ensure output path is relative to project root
    output_path.parent.mkdir(exist_ok=True, parents=True) # Create output dir
    start_time = time.time()
    total_files_found = 0 # We need to know the total files for tqdm

    # --- First pass to count files (for accurate tqdm) --- #
    # This is necessary because the generator yields batches, not files
    all_files_for_count = []
    for input_path_str in args.input:
        input_path = Path(input_path_str)
        if not input_path.is_absolute() and not input_path.exists():
            input_path = PROJECT_ROOT / input_path_str
        if input_path.is_dir():
            for ext in args.file_extensions:
                try:
                    all_files_for_count.extend(list(input_path.glob(f"**/*{ext}")))
                except Exception as e:
                    logger.warning(f"扫描目录时出错 {input_path} for {ext}: {e}")
        elif input_path.is_file() and input_path.suffix.lower() in args.file_extensions:
            all_files_for_count.append(input_path)
    total_files_found = len(all_files_for_count)
    logger.info(f"发现 {total_files_found} 个待处理文件。")

    try:
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            futures = []
            processed_count_in_loop = 0
            # Use a single output file handle
            with open(output_path, 'w', encoding='utf-8') as f_out:
                # Initialize the data generator
                data_generator = read_data(
                    args.input, 
                    args.file_extensions, 
                    args.batch_size, 
                    args.preview, 
                    args.preview_count
                )
                
                # Use tqdm for the generator itself, showing batches processed
                # Note: tqdm(total=?) based on batches is tricky if batch size varies or files are empty
                # Option 1: Leave total unknown
                # Option 2: Estimate total batches (total_files / avg_files_per_batch)
                # Using total=total_files_found provides progress based on files read by the generator
                with tqdm(total=total_files_found, desc="处理文件", unit="文件", 
                          bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]',
                          file=sys.stderr) as pbar: # Ensure tqdm output goes to stderr
                    
                    generator_finished = False
                    while not generator_finished:
                        # Submit new tasks if capacity allows
                        while len(futures) < args.max_workers * 2: # Keep the queue reasonably full
                            try:
                                data_batch = next(data_generator)
                                if data_batch and data_batch['texts']:
                                    futures.append(executor.submit(
                                        process_func, 
                                        data_batch['texts']
                                    ))
                                    # Update progress based on files processed by generator (implicitly tracked by generator)
                                    # pbar.update(??) # Difficult to update accurately here
                            except StopIteration as e:
                                # Generator finished, capture return values
                                if e.value and isinstance(e.value, tuple) and len(e.value) == 2:
                                    total_json_errors_returned, total_malformed_files_returned = e.value
                                else:
                                    # Handle case where generator didn't return expected tuple
                                    logger.warning("read_data generator did not return expected error counts.")
                                generator_finished = True
                                break # Exit inner loop
                            except Exception as e:
                                logger.error(f"从 read_data 获取批次时出错: {e}", exc_info=True)
                                # Decide whether to break or continue
                                generator_finished = True # Assume fatal error in generator
                                break
                        
                        # Process completed futures
                        if not futures:
                            if generator_finished:
                                break # Exit outer loop if generator is done and no pending tasks
                            else:
                                time.sleep(0.1) # Wait if generator is still running but no futures ready
                                continue

                        # Use as_completed to process results as they finish
                        for future in as_completed(futures): 
                            try:
                                processed_results = future.result() # List[Tuple[str, str]]
                                if processed_results:
                                    for text, text_hash in processed_results:
                                        if text_hash not in unique_hashes:
                                            unique_hashes.add(text_hash)
                                            # Write based on output format
                                            if args.output_format == "txt":
                                                f_out.write(f"{text}\n")
                                            elif args.output_format == "jsonl":
                                                f_out.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                                            elif args.output_format == "json":
                                                # JSON requires collecting all results first, less ideal for large datasets
                                                # Consider warning or disallowing this for large files
                                                # For now, we handle it but it's inefficient
                                                pass # Handled after loop for json
                                            total_texts_processed += 1
                                # Update pbar manually based on completed futures if needed, though file-based is better
                                # pbar.update(1) # This would track completed batches/futures, not files
                            except Exception as e:
                                logger.error(f"处理批次时出错: {e}", exc_info=True)
                            futures.remove(future) # Remove completed future
                            # Update pbar based on the generator's internal progress
                            # This relies on the generator yielding batches corresponding somewhat to files
                            pbar.update(1) # Increment pbar for each file processed by the generator (approximated)
                            pbar.set_postfix(malformed=total_malformed_files_returned, json_err=total_json_errors_returned, unique=len(unique_hashes), refresh=True)

                        # Break outer loop if generator is done and all futures processed
                        if generator_finished and not futures:
                            break

            # Handle JSON output (requires collecting all unique texts first)
            if args.output_format == "json":
                logger.warning("JSON output format requires loading all unique results into memory.")
                # Re-open in write mode, overwriting previous writes if any
                with open(output_path, 'w', encoding='utf-8') as f_out_json:
                    # Need to store unique texts if using JSON output
                    unique_text_list = []
                    # This requires re-reading the temp output or storing in memory - inefficient
                    # A better approach for JSON would be needed for large scale.
                    # For now, assuming unique_hashes holds hashes, not text. We need the text.
                    # Let's re-read the temp file if it was txt/jsonl or store in memory.
                    # Simplification: We'll assume unique_hashes contained the text for demo, which is wrong.
                    # Correct implementation would need storing unique texts in memory or a temp DB.
                    logger.error("JSON output format is not efficiently implemented for streaming. Please use txt or jsonl.")
                    # json_data = [{"text": text} for text in unique_texts_placeholder]
                    # json.dump(json_data, f_out_json, ensure_ascii=False, indent=2)

    except Exception as e:
        logger.error(f"处理过程中发生未捕获的异常: {e}", exc_info=True)
    finally:
        # 显示处理统计
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info("=" * 50)
        logger.info("处理完成")
        logger.info("-" * 50)
        logger.info("统计信息:")
        logger.info(f"  总处理文件数 (预估): {total_files_found}")
        logger.info(f"  无法处理/格式错误文件数: {total_malformed_files_returned}")
        logger.info(f"  JSON 解析错误总数 (行/文件): {total_json_errors_returned}")
        logger.info(f"  处理后唯一文本数: {len(unique_hashes)}")
        # Calculate unique rate based on processed texts if possible, otherwise total found
        # total_texts_input = ? # Hard to get exact input count without more tracking
        # logger.info(f"  去重率: {(1 - len(unique_hashes) / total_texts_input) * 100:.2f}%" if total_texts_input > 0 else "N/A")
        logger.info(f"  结果保存至: {output_path}")
        logger.info(f"  处理时间: {elapsed_time:.2f} 秒")
        logger.info(f"  处理速度 (文件/秒): {total_files_found / elapsed_time:.2f} 文件/秒" if elapsed_time > 0 else "N/A")
        logger.info(f"  处理速度 (文本/秒): {total_texts_processed / elapsed_time:.2f} 文本/秒" if elapsed_time > 0 else "N/A")
        logger.info("=" * 50)

if __name__ == "__main__":
    main()
    
    # Removed redundant try-except block and error counting logic here,
    # as it's now integrated into the main processing loop.