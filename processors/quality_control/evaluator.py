#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
灵猫墨韵古典文学数据处理系统 - 质量评估器
"""

import logging
from typing import Dict, List, Any, Union, Optional
import re
import json


class QualityEvaluator:
    """古典文学数据质量评估器"""
    
    def __init__(self, config):
        """
        初始化质量评估器
        
        Args:
            config: 配置字典
        """
        self.logger = logging.getLogger("LingmaoMoyun.QualityEvaluator")
        self.config = config
        
        # 从配置中获取质量控制参数
        self.quality_config = config.get("quality_control", {})
        self.min_score = self.quality_config.get("min_score", 0.7)
        
        self.logger.info(f"质量评估器初始化完成，最低接受分数: {self.min_score}")
    
    def evaluate_item(self, item: Dict) -> Dict:
        """
        评估单个数据项的质量
        
        Args:
            item: 待评估的数据项
            
        Returns:
            添加了质量评分的数据项
        """
        # 初始化评分
        score = 1.0
        issues = []
        
        # 根据数据类型选择不同的评估策略
        if "content" in item:
            content_score, content_issues = self._evaluate_content(item["content"])
            score *= content_score
            issues.extend(content_issues)
        
        if "title" in item:
            title_score, title_issues = self._evaluate_title(item["title"])
            score *= title_score
            issues.extend(title_issues)
            
        if "paragraphs" in item and isinstance(item["paragraphs"], list):
            paragraphs_score, paragraphs_issues = self._evaluate_paragraphs(item["paragraphs"])
            score *= paragraphs_score
            issues.extend(paragraphs_issues)
        
        # 添加评分到数据项
        item["_quality_score"] = round(score, 3)
        item["_quality_issues"] = issues
        
        return item
    
    def evaluate_batch(self, items: List[Dict]) -> List[Dict]:
        """
        批量评估数据项的质量
        
        Args:
            items: 待评估的数据项列表
            
        Returns:
            添加了质量评分的数据项列表
        """
        self.logger.info(f"开始批量评估 {len(items)} 个数据项")
        
        evaluated_items = []
        for item in items:
            evaluated_item = self.evaluate_item(item)
            evaluated_items.append(evaluated_item)
        
        # 统计评分情况
        passed = sum(1 for item in evaluated_items if item["_quality_score"] >= self.min_score)
        self.logger.info(f"评估完成，通过率: {passed}/{len(items)} ({passed/len(items)*100:.2f}%)")
        
        return evaluated_items
    
    def filter_by_quality(self, items: List[Dict]) -> List[Dict]:
        """
        根据质量分数过滤数据项
        
        Args:
            items: 已评估的数据项列表
            
        Returns:
            质量合格的数据项列表
        """
        filtered_items = [item for item in items if item.get("_quality_score", 0) >= self.min_score]
        self.logger.info(f"质量过滤: {len(filtered_items)}/{len(items)} 个数据项通过")
        return filtered_items
    
    def _evaluate_content(self, content: str) -> tuple:
        """
        评估内容质量
        
        Args:
            content: 内容文本
            
        Returns:
            (分数, 问题列表)
        """
        score = 1.0
        issues = []
        
        # 检查内容长度
        if not content or len(content) < 10:
            score *= 0.5
            issues.append("内容过短")
        
        # 检查重复内容
        if self._has_excessive_repetition(content):
            score *= 0.7
            issues.append("内容存在过多重复")
        
        # 检查特殊字符
        if self._has_excessive_special_chars(content):
            score *= 0.8
            issues.append("内容包含过多特殊字符")
        
        return score, issues
    
    def _evaluate_title(self, title: str) -> tuple:
        """
        评估标题质量
        
        Args:
            title: 标题文本
            
        Returns:
            (分数, 问题列表)
        """
        score = 1.0
        issues = []
        
        # 检查标题长度
        if not title:
            score *= 0.3
            issues.append("标题为空")
        elif len(title) < 2:
            score *= 0.7
            issues.append("标题过短")
        elif len(title) > 50:
            score *= 0.8
            issues.append("标题过长")
        
        # 检查特殊字符
        if self._has_excessive_special_chars(title):
            score *= 0.7
            issues.append("标题包含特殊字符")
        
        return score, issues
    
    def _evaluate_paragraphs(self, paragraphs: List[str]) -> tuple:
        """
        评估段落质量
        
        Args:
            paragraphs: 段落列表
            
        Returns:
            (分数, 问题列表)
        """
        score = 1.0
        issues = []
        
        # 检查段落数量
        if not paragraphs:
            score *= 0.3
            issues.append("段落为空")
            return score, issues
        
        # 检查段落内容
        empty_paragraphs = sum(1 for p in paragraphs if not p.strip())
        if empty_paragraphs > 0:
            score *= (1 - 0.1 * empty_paragraphs / len(paragraphs))
            issues.append(f"包含 {empty_paragraphs} 个空段落")
        
        # 检查段落重复
        unique_paragraphs = set(paragraphs)
        if len(unique_paragraphs) < len(paragraphs) * 0.8:
            score *= 0.7
            issues.append("段落存在大量重复")
        
        return score, issues
    
    def _has_excessive_repetition(self, text: str) -> bool:
        """
        检查文本是否存在过多重复
        
        Args:
            text: 待检查文本
            
        Returns:
            是否存在过多重复
        """
        # 简单实现：检查连续重复的字符
        for char in set(text):
            if char * 5 in text:
                return True
        
        # 检查重复的短语
        words = re.findall(r'\w{2,}', text)
        word_count = {}
        for word in words:
            word_count[word] = word_count.get(word, 0) + 1
        
        # 如果某个词出现次数超过总词数的20%，认为存在过多重复
        for count in word_count.values():
            if count > len(words) * 0.2 and count > 3:
                return True
        
        return False
    
    def _has_excessive_special_chars(self, text: str) -> bool:
        """
        检查文本是否包含过多特殊字符
        
        Args:
            text: 待检查文本
            
        Returns:
            是否包含过多特殊字符
        """
        # 定义允许的标点符号
        allowed_punctuation = set('，。！？；："''（）【】《》、')