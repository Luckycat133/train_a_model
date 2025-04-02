#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
灵猫墨韵古典文学数据处理系统 - 加权抽样工具
"""

import logging
import random
from typing import List, Dict, Any, Union, Optional, TypeVar, Generic, Callable

T = TypeVar('T')

class WeightedSampler:
    """加权抽样工具，用于从数据集中按照权重进行抽样"""
    
    def __init__(self, config):
        """
        初始化加权抽样工具
        
        Args:
            config: 配置字典
        """
        self.logger = logging.getLogger("LingmaoMoyun.WeightedSampler")
        self.config = config
        
        # 从配置中获取抽样参数
        self.quality_config = config.get("quality_control", {})
        self.sample_rate = self.quality_config.get("sample_rate", 0.1)
        
        self.logger.info(f"加权抽样工具初始化完成，默认抽样率: {self.sample_rate}")
    
    def sample(self, items: List[T], weights: Optional[List[float]] = None, k: Optional[int] = None) -> List[T]:
        """
        从列表中按权重抽样
        
        Args:
            items: 待抽样的项目列表
            weights: 权重列表，如果为None则等权重抽样
            k: 抽样数量，如果为None则使用配置的抽样率
            
        Returns:
            抽样结果列表
        """
        if not items:
            return []
        
        # 确定抽样数量
        if k is None:
            k = max(1, int(len(items) * self.sample_rate))
        k = min(k, len(items))  # 确保k不超过列表长度
        
        # 如果没有提供权重，则使用等权重
        if weights is None:
            weights = [1.0] * len(items)
        
        # 确保权重列表长度与项目列表长度一致
        if len(weights) != len(items):
            self.logger.warning(f"权重列表长度({len(weights)})与项目列表长度({len(items)})不一致，使用等权重")
            weights = [1.0] * len(items)
        
        # 处理权重为0或负数的情况
        for i, w in enumerate(weights):
            if w <= 0:
                weights[i] = 0.0001  # 设置一个很小的正数
        
        # 使用random.choices进行加权抽样
        try:
            sampled = random.choices(items, weights=weights, k=k)
            self.logger.info(f"成功抽样 {len(sampled)}/{len(items)} 个项目")
            return sampled
        except Exception as e:
            self.logger.error(f"抽样过程中出错: {e}")
            # 出错时使用无放回等权重抽样
            return random.sample(items, k)
    
    def stratified_sample(self, items: List[Dict], strata_key: str, sample_rates: Dict[str, float] = None) -> List[Dict]:
        """
        分层抽样
        
        Args:
            items: 待抽样的项目列表
            strata_key: 分层的键名
            sample_rates: 各层的抽样率字典，如果为None则使用配置的抽样率
            
        Returns:
            抽样结果列表
        """
        if not items:
            return []
        
        # 按strata_key分组
        strata = {}
        for item in items:
            key = item.get(strata_key, "unknown")
            if key not in strata:
                strata[key] = []
            strata[key].append(item)
        
        # 如果没有提供各层抽样率，则使用默认抽样率
        if sample_rates is None:
            sample_rates = {key: self.sample_rate for key in strata.keys()}
        
        # 对每一层进行抽样
        sampled_items = []
        for key, group in strata.items():
            rate = sample_rates.get(key, self.sample_rate)
            k = max(1, int(len(group) * rate))
            sampled = self.sample(group, k=k)
            sampled_items.extend(sampled)
        
        self.logger.info(f"分层抽样完成，共抽取 {len(sampled_items)}/{len(items)} 个项目")
        return sampled_items
    
    def weighted_sample_by_score(self, items: List[Dict], score_key: str = "_quality_score", 
                               higher_weight: bool = True) -> List[Dict]:
        """
        根据评分进行加权抽样
        
        Args:
            items: 待抽样的项目列表
            score_key: 评分的键名
            higher_weight: 是否高分高权重，如果为False则低分高权重
            
        Returns:
            抽样结果列表
        """
        if not items:
            return []
        
        # 提取评分
        scores = []
        for item in items:
            score = item.get(score_key, 0.5)  # 默认评分0.5
            if not isinstance(score, (int, float)):
                score = 0.5
            scores.append(score)
        
        # 计算权重
        if higher_weight:
            # 高分高权重
            weights = [max(0.0001, s) for s in scores]  # 确保权重为正
        else:
            # 低分高权重
            weights = [max(0.0001, 1.0 - s) for s in scores]  # 确保权重为正
        
        # 进行加权抽样
        k = max(1, int(len(items) * self.sample_rate))
        sampled = self.sample(items, weights=weights, k=k)
        
        self.logger.info(f"根据评分加权抽样完成，共抽取 {len(sampled)}/{len(items)} 个项目")
        return sampled