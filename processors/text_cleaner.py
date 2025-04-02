#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
灵猫墨韵古典文学数据处理系统 - 文本清洗工具
"""

import re
import unicodedata
import logging
from typing import Dict, List, Any, Optional
from opencc import OpenCC


class TextCleaner:
    """古典文学文本清洗工具"""
    
    def __init__(self):
        """初始化文本清洗工具"""
        self.logger = logging.getLogger("LingmaoMoyun.TextCleaner")
        
        # 延迟初始化OpenCC
        self.cc_t2s = None
        self.cc_s2t = None
        self.opencc_available = False
        
        # 常见错误模式
        self.error_patterns = {
            r'\s+': ' ',                  # 多余空格
            r'([，。！？；：、])\1+': r'\1', # 重复标点
            r'\.{2,}': '…',               # 将多个点替换为省略号
        }
        
        # 编译正则表达式
        self.compiled_patterns = {
            re.compile(pattern): replacement
            for pattern, replacement in self.error_patterns.items()
        }
        
        self.logger.info("文本清洗工具初始化完成")
    
    def _init_opencc(self):
        """懒加载初始化OpenCC"""
        if self.cc_t2s is None:
            try:
                self.cc_t2s = OpenCC('t2s')  # 繁体转简体
                self.cc_s2t = OpenCC('s2t')  # 简体转繁体
                self.opencc_available = True
            except Exception as e:
                self.logger.warning(f"OpenCC初始化失败，繁简转换功能不可用: {e}")
                self.opencc_available = False
    
    def clean_text(self, text: str) -> str:
        """
        清洗文本
        
        Args:
            text: 原始文本
            
        Returns:
            清洗后的文本
        """
        if not text:
            return ""
        
        # 标准化Unicode
        text = unicodedata.normalize('NFKC', text)
        
        # 应用错误模式替换
        for pattern, replacement in self.compiled_patterns.items():
            text = pattern.sub(replacement, text)
        
        # 去除首尾空白
        text = text.strip()
        
        return text
    
    def traditional_to_simplified(self, text: str) -> str:
        """
        繁体转简体
        
        Args:
            text: 繁体文本
            
        Returns:
            简体文本
        """
        if not text:
            return text
        
        self._init_opencc()  # 延迟初始化
        if not self.opencc_available:
            return text
        
        return self.cc_t2s.convert(text)
    
    def simplified_to_traditional(self, text: str) -> str:
        """
        简体转繁体
        
        Args:
            text: 简体文本
            
        Returns:
            繁体文本
        """
        if not text:
            return text
        
        self._init_opencc()  # 延迟初始化
        if not self.opencc_available:
            return text
        
        return self.cc_s2t.convert(text)
    
    def normalize_punctuation(self, text: str) -> str:
        """
        标准化标点符号
        
        Args:
            text: 原始文本
            
        Returns:
            标准化后的文本
        """
        if not text:
            return ""
        
        # 全角标点映射到半角标点
        punct_map = {
            '，': ',',
            '。': '.',
            '！': '!',
            '？': '?',
            '；': ';',
            '：': ':',
            '（': '(',
            '）': ')',
            '【': '[',
            '】': ']',
            '「': '"',
            '」': '"',
            '"': '"',
            '"': '"',
            ''': "'",
            ''': "'",
            '《': '<',
            '》': '>',
            '—': '-',
            '～': '~',
            '·': '.'
        }
        
        # 替换标点
        for old, new in punct_map.items():
            text = text.replace(old, new)
        
        return text
    
    def clean_item(self, item: Any) -> Any:
        """
        清洗数据项
        
        Args:
            item: 原始数据项
            
        Returns:
            清洗后的数据项
        """
        # 如果不是字典，直接返回
        if not isinstance(item, dict):
            return item
            
        result = item.copy()
        
        # 清洗标题
        if "title" in result and isinstance(result["title"], str):
            result["title"] = self.clean_text(result["title"])
        
        # 清洗内容
        if "content" in result and isinstance(result["content"], str):
            result["content"] = self.clean_text(result["content"])
        
        # 清洗段落
        if "paragraphs" in result and isinstance(result["paragraphs"], list):
            result["paragraphs"] = [
                self.clean_text(p) for p in result["paragraphs"] if p
            ]
        
        # 清洗翻译
        if "translation" in result and isinstance(result["translation"], str):
            result["translation"] = self.clean_text(result["translation"])
        
        # 清洗注释
        if "notes" in result and isinstance(result["notes"], str):
            result["notes"] = self.clean_text(result["notes"])
        
        return result
    
    def clean_batch(self, items: List[Dict]) -> List[Dict]:
        """
        批量清洗数据
        
        Args:
            items: 原始数据列表
            
        Returns:
            清洗后的数据列表
        """
        return [self.clean_item(item) for item in items]


# 简单测试
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    cleaner = TextCleaner()
    
    # 测试文本清洗
    test_text = "床前明月光，，  疑是地上霜。。。"
    cleaned_text = cleaner.clean_text(test_text)
    print(f"原文: '{test_text}'")
    print(f"清洗后: '{cleaned_text}'")
    
    # 测试繁简转换
    if cleaner.opencc_available:
        trad_text = "風花雪月"
        simp_text = cleaner.traditional_to_simplified(trad_text)
        print(f"繁体: '{trad_text}'")
        print(f"简体: '{simp_text}'")
