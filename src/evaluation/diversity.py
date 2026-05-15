"""多样性评估模块 - 灵猫墨韵评估框架

本模块提供多种多样性评估指标，用于衡量语言模型生成内容的多样性特征。
包括词级别、序列级别、语义级别等多维度的多样性评估功能。
"""

from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from collections import Counter, defaultdict
import math
import random

try:
    import torch
    import numpy as np
except Exception:
    torch = None
    np = None

from src.evaluation.base_evaluator import (
    BaseEvaluator,
    EvaluationResult,
    EvaluatorType,
)


@dataclass
class DiversityMetrics:
    """多样性指标容器类
    
    属性:
        token_entropy: 词元级别的熵值
        sequence_diversity: 序列级别多样性
        vocabulary_richness: 词汇丰富度
        ngram_diversity: N-gram多样性
        semantic_spread: 语义分散度
    """
    token_entropy: float = 0.0
    sequence_diversity: float = 0.0
    vocabulary_richness: float = 0.0
    ngram_diversity: Dict[int, float] = field(default_factory=dict)
    semantic_spread: float = 0.0


class DIVERSITYMETRICS:
    """多样性指标计算工具集
    
    该类提供静态方法用于计算各种多样性评估指标，
    包括基于频率、基于熵、基于覆盖率等多种计算方式。
    """
    
    @staticmethod
    def type_token_ratio(tokens: List[int]) -> float:
        """计算类型-词元比率 (Type-Token Ratio, TTR)
        
        TTR是不同词元数量与总词元数量的比率，用于衡量词汇多样性。
        较高的TTR表示更大的词汇多样性。
        
        参数:
            tokens: 词元ID列表
            
        返回:
            TTR值，范围[0, 1]
        """
        if not tokens:
            return 0.0
        
        unique_tokens = len(set(tokens))
        total_tokens = len(tokens)
        
        return unique_tokens / total_tokens
    
    @staticmethod
    def hapax_legomena_ratio(tokens: List[int]) -> float:
        """计算仅出现一次的词元比率 (Hapax Legomena Ratio)
        
        Hapax legomena是指在整个语料中只出现一次的词。
        这个比率可以反映词汇使用的创新性和多样性。
        
        参数:
            tokens: 词元ID列表
            
        返回:
            Hapax比率，范围[0, 1]
        """
        if not tokens:
            return 0.0
        
        freq_counter = Counter(tokens)
        hapax_count = sum(1 for count in freq_counter.values() if count == 1)
        
        return hapax_count / len(tokens)
    
    @staticmethod
    def yule_k_statistic(tokens: List[int]) -> float:
        """计算Yule's K统计量 - 词汇丰富度度量
        
        Yule's K测量词汇分布的均匀性，值越小表示词汇分布越均匀。
        该指标对高频词敏感，适用于评估词汇多样性。
        
        参数:
            tokens: 词元ID列表
            
        返回:
            Yule's K值
        """
        if not tokens:
            return 0.0
        
        freq_counter = Counter(tokens)
        freq_counts = Counter(freq_counter.values())
        
        m1 = len(tokens)
        m2 = sum(f * f * c for f, c in freq_counts.items())
        
        k = 10000 * (m2 - m1) / (m1 * m1)
        
        return k
    
    @staticmethod
    def shannon_entropy(tokens: List[int], base: float = 2) -> float:
        """计算香农熵 (Shannon Entropy)
        
        香农熵衡量信息的不确定性或随机性。
        较高的熵值表示词汇分布更加均匀和多样。
        
        参数:
            tokens: 词元ID列表
            base: 对数底数，默认2
            
        返回:
            熵值
        """
        if not tokens:
            return 0.0
        
        freq_counter = Counter(tokens)
        total = len(tokens)
        
        entropy = 0.0
        for freq in freq_counter.values():
            p = freq / total
            if p > 0:
                entropy -= p * math.log(p, base)
        
        return entropy
    
    @staticmethod
    def renyi_entropy(tokens: List[int], alpha: float = 2) -> float:
        """计算Renyi熵 - 香农熵的广义形式
        
        Renyi熵是香农熵的推广，通过参数alpha控制对频率分布的敏感度。
        - alpha < 1: 对低频词更敏感
        - alpha > 1: 对高频词更敏感
        - alpha = 1: 退化为香农熵
        
        参数:
            tokens: 词元ID列表
            alpha: Rényi参数
            
        返回:
            Renyi熵值
        """
        if not tokens:
            return 0.0
        
        freq_counter = Counter(tokens)
        total = len(tokens)
        
        if alpha == 1:
            return DIVERSITYMETRICS.shannon_entropy(tokens)
        
        renyi_sum = 0.0
        for freq in freq_counter.values():
            p = freq / total
            if p > 0:
                renyi_sum += p ** alpha
        
        return (1 / (1 - alpha)) * math.log(renyi_sum) if alpha != 1 else 0.0
    
    @staticmethod
    def simpson_diversity_index(tokens: List[int]) -> float:
        """计算Simpson多样性指数
        
        Simpson指数测量随机抽取两个词元来自不同类型的概率。
        值越高表示多样性越好。
        
        参数:
            tokens: 词元ID列表
            
        返回:
            Simpson指数，范围[0, 1]
        """
        if not tokens:
            return 0.0
        
        freq_counter = Counter(tokens)
        total = len(tokens)
        
        simpson_sum = 0.0
        for freq in freq_counter.values():
            simpson_sum += (freq / total) ** 2
        
        return 1 - simpson_sum
    
    @staticmethod
    def inverse_simpson_index(tokens: List[int]) -> float:
        """计算逆Simpson指数
        
        逆Simpson指数是Simpson指数的倒数，更容易直观理解。
        值越高表示多样性越好。
        
        参数:
            tokens: 词元ID列表
            
        返回:
            逆Simpson指数
        """
        if not tokens:
            return 0.0
        
        freq_counter = Counter(tokens)
        total = len(tokens)
        
        simpson_sum = 0.0
        for freq in freq_counter.values():
            simpson_sum += (freq / total) ** 2
        
        return 1 / simpson_sum if simpson_sum > 0 else 0.0
    
    @staticmethod
    def ngram_diversity(
        tokens: List[int],
        n: int = 2,
    ) -> Dict[str, float]:
        """计算N-gram多样性
        
        测量不同N-gram的使用多样性。
        
        参数:
            tokens: 词元ID列表
            n: N-gram的大小
            
        返回:
            包含多种多样性度量的字典
        """
        if not tokens or len(tokens) < n:
            return {
                "unique_ratio": 0.0,
                "ttr": 0.0,
                "entropy": 0.0,
            }
        
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            ngrams.append(ngram)
        
        if not ngrams:
            return {
                "unique_ratio": 0.0,
                "ttr": 0.0,
                "entropy": 0.0,
            }
        
        unique_ngrams = set(ngrams)
        
        return {
            "unique_ratio": len(unique_ngrams) / len(ngrams),
            "ttr": len(unique_ngrams) / len(ngrams),
            "entropy": DIVERSITYMETRICS.shannon_entropy(ngrams),
            "total_ngrams": len(ngrams),
            "unique_count": len(unique_ngrams),
        }
    
    @staticmethod
    def sequence_level_diversity(
        sequences: List[List[int]],
    ) -> Dict[str, float]:
        """计算序列级别的多样性
        
        评估多个序列之间的多样性，而不是单个序列内部的多样性。
        
        参数:
            sequences: 多个序列的列表
            
        返回:
            序列级别多样性指标字典
        """
        if not sequences or len(sequences) < 2:
            return {
                "pairwise_similarity": 0.0,
                "unique_sequences": 0.0,
                "sequence_entropy": 0.0,
            }
        
        unique_sequences = len(set(tuple(seq) for seq in sequences))
        unique_ratio = unique_sequences / len(sequences)
        
        def jaccard_similarity(seq1: List[int], seq2: List[int]) -> float:
            set1, set2 = set(seq1), set(seq2)
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            return intersection / union if union > 0 else 0.0
        
        total_similarity = 0.0
        pair_count = 0
        for i in range(len(sequences)):
            for j in range(i + 1, len(sequences)):
                total_similarity += jaccard_similarity(sequences[i], sequences[j])
                pair_count += 1
        
        avg_similarity = total_similarity / pair_count if pair_count > 0 else 0.0
        
        seq_repr = [tuple(seq) for seq in sequences]
        seq_entropy = DIVERSITYMETRICS.shannon_entropy(seq_repr)
        
        return {
            "pairwise_similarity": 1 - avg_similarity,
            "unique_sequences": unique_ratio,
            "sequence_entropy": seq_entropy,
        }
    
    @staticmethod
    def positional_diversity(tokens: List[int], num_positions: int = 10) -> float:
        """计算位置多样性
        
        测量不同位置上的词元分布多样性。
        用于评估模型是否能够在不同位置产生多样化的输出。
        
        参数:
            tokens: 词元ID列表
            num_positions: 分成多少个位置段
            
        返回:
            位置多样性分数，范围[0, 1]
        """
        if not tokens or len(tokens) < num_positions:
            return 0.0
        
        segment_size = len(tokens) // num_positions
        segments = []
        
        for i in range(num_positions):
            start = i * segment_size
            end = start + segment_size if i < num_positions - 1 else len(tokens)
            segment = tokens[start:end]
            segments.append(set(segment))
        
        total_unique = set()
        for segment in segments:
            total_unique.update(segment)
        
        overlap_count = 0
        for i in range(len(segments)):
            for j in range(i + 1, len(segments)):
                overlap_count += len(segments[i] & segments[j])
        
        max_overlap = num_positions * (num_positions - 1) // 2
        normalized_overlap = overlap_count / max_overlap if max_overlap > 0 else 0.0
        
        return 1.0 - normalized_overlap
    
    @staticmethod
    def repetition_free_ratio(text: str, n: int = 3) -> float:
        """计算无重复N-gram的比例
        
        测量文本中没有出现重复的N-gram的比例。
        
        参数:
            text: 输入文本（空格分隔的词）
            n: N-gram大小
            
        返回:
            无重复比例，范围[0, 1]
        """
        if not text:
            return 0.0
        
        words = text.split() if isinstance(text, str) else text
        
        if len(words) < n:
            return 1.0 if len(words) > 0 else 0.0
        
        seen = set()
        total_ngrams = len(words) - n + 1
        unique_count = 0
        
        for i in range(total_ngrams):
            ngram = tuple(words[i:i+n])
            if ngram not in seen:
                unique_count += 1
                seen.add(ngram)
        
        return unique_count / total_ngrams if total_ngrams > 0 else 0.0


class DiversityEvaluator(BaseEvaluator):
    """多样性评估器
    
    该评估器用于衡量语言模型生成内容的多样性特征。
    支持多种多样性指标的计算和聚合。
    
    属性:
        metrics_config: 各指标的配置参数
        compute_all: 是否计算所有可用指标
    """
    
    def __init__(
        self,
        name: str = "diversity_evaluator",
        compute_ttr: bool = True,
        compute_entropy: bool = True,
        compute_ngram: bool = True,
        compute_simpson: bool = True,
        ngram_range: Tuple[int, int] = (2, 4),
        config: Optional[Dict[str, Any]] = None,
    ):
        """初始化多样性评估器
        
        参数:
            name: 评估器名称
            compute_ttr: 是否计算TTR
            compute_entropy: 是否计算熵值
            compute_ngram: 是否计算N-gram多样性
            compute_simpson: 是否计算Simpson指数
            ngram_range: N-gram的范围
            config: 额外配置
        """
        super().__init__(
            name=name,
            evaluator_type=EvaluatorType.DIVERSITY,
            config=config,
        )
        
        self.compute_ttr = compute_ttr
        self.compute_entropy = compute_entropy
        self.compute_ngram = compute_ngram
        self.compute_simpson = compute_simpson
        self.ngram_range = ngram_range
        
        self._register_diversity_metrics()
    
    def _register_diversity_metrics(self) -> None:
        """注册多样性指标"""
        self.register_metric("type_token_ratio", self._compute_ttr)
        self.register_metric("shannon_entropy", self._compute_entropy)
        self.register_metric("simpson_diversity", self._compute_simpson)
        self.register_metric("hapax_ratio", self._compute_hapax)
        self.register_metric("yule_k", self._compute_yule_k)
    
    def _collect_tokens(
        self,
        model: Any,
        dataset: Any,
    ) -> List[int]:
        """从数据集收集词元
        
        参数:
            model: 语言模型
            dataset: 数据集
            
        返回:
            收集的词元列表
        """
        tokens = []
        
        if torch is None:
            return tokens
        
        model.eval()
        
        with torch.no_grad():
            for batch in dataset:
                if isinstance(batch, dict):
                    input_ids = batch.get("input_ids")
                    if input_ids is None:
                        continue
                elif isinstance(batch, torch.Tensor):
                    input_ids = batch
                else:
                    continue
                
                if isinstance(input_ids, torch.Tensor):
                    tokens.extend(input_ids.flatten().tolist())
        
        return tokens
    
    def _compute_ttr(
        self,
        model: Any,
        dataset: Any,
        **kwargs,
    ) -> float:
        """计算类型-词元比率
        
        参数:
            model: 语言模型
            dataset: 数据集
            **kwargs: 额外参数
            
        返回:
            TTR值
        """
        tokens = self._collect_tokens(model, dataset)
        return DIVERSITYMETRICS.type_token_ratio(tokens)
    
    def _compute_entropy(
        self,
        model: Any,
        dataset: Any,
        **kwargs,
    ) -> float:
        """计算香农熵
        
        参数:
            model: 语言模型
            dataset: 数据集
            **kwargs: 额外参数
            
        返回:
            熵值
        """
        tokens = self._collect_tokens(model, dataset)
        return DIVERSITYMETRICS.shannon_entropy(tokens)
    
    def _compute_simpson(
        self,
        model: Any,
        dataset: Any,
        **kwargs,
    ) -> float:
        """计算Simpson多样性指数
        
        参数:
            model: 语言模型
            dataset: 数据集
            **kwargs: 额外参数
            
        返回:
            Simpson指数
        """
        tokens = self._collect_tokens(model, dataset)
        return DIVERSITYMETRICS.simpson_diversity_index(tokens)
    
    def _compute_hapax(
        self,
        model: Any,
        dataset: Any,
        **kwargs,
    ) -> float:
        """计算Hapax比率
        
        参数:
            model: 语言模型
            dataset: 数据集
            **kwargs: 额外参数
            
        返回:
            Hapax比率
        """
        tokens = self._collect_tokens(model, dataset)
        return DIVERSITYMETRICS.hapax_legomena_ratio(tokens)
    
    def _compute_yule_k(
        self,
        model: Any,
        dataset: Any,
        **kwargs,
    ) -> float:
        """计算Yule's K统计量
        
        参数:
            model: 语言模型
            dataset: 数据集
            **kwargs: 额外参数
            
        返回:
            Yule's K值
        """
        tokens = self._collect_tokens(model, dataset)
        return DIVERSITYMETRICS.yule_k_statistic(tokens)
    
    def _evaluate(
        self,
        model: Any,
        dataset: Any,
        **kwargs,
    ) -> List[EvaluationResult]:
        """执行多样性评估
        
        参数:
            model: 语言模型
            dataset: 数据集
            **kwargs: 额外参数
            
        返回:
            评估结果列表
        """
        results = []
        
        if self.compute_ttr:
            ttr = self.compute_metric("type_token_ratio", model, dataset)
            if ttr is not None:
                results.append(
                    EvaluationResult(
                        metric_name="type_token_ratio",
                        value=ttr,
                        metadata={"higher_is_better": True},
                    )
                )
        
        if self.compute_entropy:
            entropy = self.compute_metric("shannon_entropy", model, dataset)
            if entropy is not None:
                results.append(
                    EvaluationResult(
                        metric_name="shannon_entropy",
                        value=entropy,
                        metadata={"higher_is_better": True},
                    )
                )
        
        if self.compute_simpson:
            simpson = self.compute_metric("simpson_diversity", model, dataset)
            if simpson is not None:
                results.append(
                    EvaluationResult(
                        metric_name="simpson_diversity",
                        value=simpson,
                        metadata={"higher_is_better": True},
                    )
                )
        
        results.append(
            EvaluationResult(
                metric_name="hapax_ratio",
                value=self.compute_metric("hapax_ratio", model, dataset) or 0.0,
                metadata={"higher_is_better": True},
            )
        )
        
        results.append(
            EvaluationResult(
                metric_name="yule_k",
                value=self.compute_metric("yule_k", model, dataset) or 0.0,
                metadata={"lower_is_better": True},
            )
        )
        
        if self.compute_ngram:
            tokens = self._collect_tokens(model, dataset)
            for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
                ngram_metrics = DIVERSITYMETRICS.ngram_diversity(tokens, n)
                results.append(
                    EvaluationResult(
                        metric_name=f"ngram_{n}_diversity",
                        value=ngram_metrics.get("unique_ratio", 0.0),
                        metadata={
                            "higher_is_better": True,
                            "n": n,
                            "total_ngrams": ngram_metrics.get("total_ngrams", 0),
                            "unique_count": ngram_metrics.get("unique_count", 0),
                        },
                    )
                )
        
        return results


class ComparativeDiversityEvaluator(DiversityEvaluator):
    """比较多样性评估器
    
    用于比较不同模型或不同配置下的生成多样性。
    提供配对比较和统计分析功能。
    """
    
    def __init__(
        self,
        name: str = "comparative_diversity_evaluator",
        **kwargs,
    ):
        """初始化比较多样性评估器
        
        参数:
            name: 评估器名称
            **kwargs: 传递给父类的参数
        """
        super().__init__(name=name, **kwargs)
        
        self._baseline_results: Optional[List[EvaluationResult]] = None
    
    def set_baseline(
        self,
        baseline_results: List[EvaluationResult],
    ) -> None:
        """设置基准结果用于比较
        
        参数:
            baseline_results: 基准评估结果
        """
        self._baseline_results = baseline_results
    
    def compare_with_baseline(
        self,
        current_results: List[EvaluationResult],
    ) -> Dict[str, Dict[str, float]]:
        """与基准结果进行比较
        
        参数:
            current_results: 当前评估结果
            
        返回:
            比较结果字典
        """
        if self._baseline_results is None:
            return {}
        
        comparisons = {}
        
        current_dict = {r.metric_name: r.value for r in current_results}
        baseline_dict = {r.metric_name: r.value for r in self._baseline_results}
        
        for metric_name in current_dict:
            if metric_name in baseline_dict:
                baseline_val = baseline_dict[metric_name]
                current_val = current_dict[metric_name]
                
                if baseline_val != 0:
                    relative_change = (current_val - baseline_val) / baseline_val
                else:
                    relative_change = current_val
                
                comparisons[metric_name] = {
                    "current": current_val,
                    "baseline": baseline_val,
                    "absolute_change": current_val - baseline_val,
                    "relative_change": relative_change,
                }
        
        return comparisons
    
    def rank_by_diversity(
        self,
        model_results: Dict[str, List[EvaluationResult]],
    ) -> List[Tuple[str, float]]:
        """根据多样性排名模型
        
        参数:
            model_results: 模型名称到评估结果的字典
            
        返回:
            排名列表 (模型名, 平均多样性分数)
        """
        rankings = []
        
        for model_name, results in model_results.items():
            avg_diversity = sum(r.value for r in results) / len(results)
            rankings.append((model_name, avg_diversity))
        
        rankings.sort(key=lambda x: x[1], reverse=True)
        
        return rankings


class StreamingDiversityEvaluator(DiversityEvaluator):
    """流式多样性评估器
    
    用于处理无法一次性加载到内存的大规模数据集。
    采用流式处理方式逐步累积统计信息。
    
    属性:
        streaming_stats: 流式统计信息
    """
    
    def __init__(
        self,
        name: str = "streaming_diversity_evaluator",
        **kwargs,
    ):
        """初始化流式多样性评估器
        
        参数:
            name: 评估器名称
            **kwargs: 传递给父类的参数
        """
        super().__init__(name=name, **kwargs)
        
        self._streaming_stats = {
            "total_tokens": 0,
            "unique_tokens": set(),
            "token_frequencies": Counter(),
            "ngram_frequencies": defaultdict(Counter),
        }
    
    def reset_stats(self) -> None:
        """重置流式统计信息"""
        self._streaming_stats = {
            "total_tokens": 0,
            "unique_tokens": set(),
            "token_frequencies": Counter(),
            "ngram_frequencies": defaultdict(Counter),
        }
    
    def update_stats(self, tokens: List[int]) -> None:
        """更新统计信息
        
        参数:
            tokens: 新增的词元列表
        """
        stats = self._streaming_stats
        
        stats["total_tokens"] += len(tokens)
        stats["unique_tokens"].update(tokens)
        
        for token in tokens:
            stats["token_frequencies"][token] += 1
        
        for n in range(self.ngram_range[0], min(self.ngram_range[1] + 1, 5)):
            if len(tokens) >= n:
                for i in range(len(tokens) - n + 1):
                    ngram = tuple(tokens[i:i+n])
                    stats["ngram_frequencies"][n][ngram] += 1
    
    def get_metrics(self) -> Dict[str, float]:
        """获取计算的多样性指标
        
        返回:
            多样性指标字典
        """
        stats = self._streaming_stats
        
        if stats["total_tokens"] == 0:
            return {
                "type_token_ratio": 0.0,
                "shannon_entropy": 0.0,
                "simpson_diversity": 0.0,
                "hapax_ratio": 0.0,
            }
        
        freq_counter = stats["token_frequencies"]
        total = stats["total_tokens"]
        
        entropy = 0.0
        simpson_sum = 0.0
        hapax_count = 0
        
        for freq in freq_counter.values():
            p = freq / total
            entropy -= p * math.log(p, 2)
            simpson_sum += p ** 2
            if freq == 1:
                hapax_count += 1
        
        ttr = len(stats["unique_tokens"]) / total
        simpson = 1 - simpson_sum
        hapax_ratio = hapax_count / total
        
        return {
            "type_token_ratio": ttr,
            "shannon_entropy": entropy,
            "simpson_diversity": simpson,
            "hapax_ratio": hapax_ratio,
        }
    
    def _evaluate(
        self,
        model: Any,
        dataset: Any,
        **kwargs,
    ) -> List[EvaluationResult]:
        """执行流式多样性评估
        
        参数:
            model: 语言模型
            dataset: 数据集
            **kwargs: 额外参数
            
        返回:
            评估结果列表
        """
        self.reset_stats()
        
        if torch is None:
            return []
        
        model.eval()
        
        with torch.no_grad():
            for batch in dataset:
                if isinstance(batch, dict):
                    input_ids = batch.get("input_ids")
                    if input_ids is None:
                        continue
                elif isinstance(batch, torch.Tensor):
                    input_ids = batch
                else:
                    continue
                
                if isinstance(input_ids, torch.Tensor):
                    tokens = input_ids.flatten().tolist()
                    self.update_stats(tokens)
        
        metrics = self.get_metrics()
        
        results = []
        for metric_name, value in metrics.items():
            results.append(
                EvaluationResult(
                    metric_name=metric_name,
                    value=value,
                    metadata={"streaming": True},
                )
            )
        
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            ngram_freq = self._streaming_stats["ngram_frequencies"].get(n, Counter())
            total_ngrams = sum(ngram_freq.values())
            unique_ngrams = len(ngram_freq)
            
            if total_ngrams > 0:
                ngram_diversity = unique_ngrams / total_ngrams
            else:
                ngram_diversity = 0.0
            
            results.append(
                EvaluationResult(
                    metric_name=f"ngram_{n}_diversity",
                    value=ngram_diversity,
                    metadata={
                        "streaming": True,
                        "n": n,
                        "unique_count": unique_ngrams,
                        "total_count": total_ngrams,
                    },
                )
            )
        
        return results
