"""数据过滤模块。

提供数据质量过滤功能，包括：
- 基于长度的过滤（最小/最大长度）
- 质量评分过滤
- 重复率检测和过滤
- 语言检测过滤
- 噪声数据过滤
- 自定义规则过滤

示例：
    >>> data_filter = DataFilter(min_length=10, max_length=5000)
    >>> data_filter.min_quality_score = 0.5
    >>> if data_filter.is_valid(sample):
    ...     # 保留数据
    >>> filtered = data_filter.filter_batch(samples)
"""

import re
import hashlib
from typing import Dict, List, Optional, Set, Callable, Any, Tuple
from dataclasses import dataclass, field
from collections import Counter

from src.logger import get_logger

logger = get_logger("LingmaoMoyun.Data.Filter")


@dataclass
class FilterConfig:
    """数据过滤配置类。"""

    min_length: int = 10
    max_length: int = 100000
    min_char_ratio: float = 0.3
    max_repetition_ratio: float = 0.5
    min_chinese_ratio: float = 0.1
    max_url_ratio: float = 0.1
    max_special_char_ratio: float = 0.3
    check_duplicates: bool = True
    check_length: bool = True
    check_quality: bool = True
    check_language: bool = False


class DataFilter:
    """数据过滤器。

    基于多种规则过滤低质量数据。

    Attributes:
        config: 过滤配置对象。
        seen_hashes: 已见文本哈希集合（用于去重）。

    Example:
        >>> data_filter = DataFilter(min_length=10, max_length=5000)
        >>> sample = {'text': '这是一个测试文本。', 'id': '1'}
        >>> if data_filter.is_valid(sample):
        ...     print("Valid sample")
        >>> filtered = data_filter.filter_batch(samples)
    """

    URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
    SPECIAL_CHAR_PATTERN = re.compile(r'[!@#$%^&*()_+\-=\[\]{}|;:\'",.<>?/\\`~·「」『』【】〔〕〖〗〘〙〚〛〈〉《》]+')
    REPETITION_PATTERN = re.compile(r'(.{3,}?)\1{2,}')

    def __init__(self, config: Optional[FilterConfig] = None):
        """初始化数据过滤器。

        Args:
            config: 过滤配置对象。如果为None，使用默认配置。
        """
        self.config = config or FilterConfig()
        self.seen_hashes: Set[str] = set()
        self._stats = {
            'total': 0,
            'passed': 0,
            'filtered': 0,
            'length_filtered': 0,
            'quality_filtered': 0,
            'duplicate_filtered': 0,
            'repetition_filtered': 0,
        }

    def is_valid(self, sample: Dict[str, Any]) -> bool:
        """检查样本是否通过所有过滤规则。

        Args:
            sample: 待检查的样本字典，必须包含'text'或'content'字段。

        Returns:
            如果样本通过所有过滤规则返回True，否则返回False。

        Example:
            >>> data_filter = DataFilter(min_length=10)
            >>> sample = {'text': '这是一个有效的测试文本。'}
            >>> data_filter.is_valid(sample)
            True
        """
        self._stats['total'] += 1

        if self.config.check_length and not self._check_length(sample):
            self._stats['length_filtered'] += 1
            self._stats['filtered'] += 1
            return False

        text = self._extract_text(sample)
        if not text:
            self._stats['filtered'] += 1
            return False

        if self.config.check_duplicates and not self._check_duplicate(text):
            self._stats['duplicate_filtered'] += 1
            self._stats['filtered'] += 1
            return False

        if self.config.check_quality and not self._check_quality(text):
            self._stats['quality_filtered'] += 1
            self._stats['filtered'] += 1
            return False

        self._stats['passed'] += 1
        return True

    def _extract_text(self, sample: Dict[str, Any]) -> Optional[str]:
        """从样本中提取文本内容。

        Args:
            sample: 样本字典。

        Returns:
            提取的文本，如果不存在返回None。
        """
        for key in ['text', 'content', 'body', 'input', 'output']:
            if key in sample and isinstance(sample[key], str):
                return sample[key]
        return None

    def _check_length(self, sample: Dict[str, Any]) -> bool:
        """检查文本长度是否在有效范围内。

        Args:
            sample: 样本字典。

        Returns:
            如果长度有效返回True。
        """
        text = self._extract_text(sample)
        if text is None:
            return False

        length = len(text.strip())
        return self.config.min_length <= length <= self.config.max_length

    def _check_duplicate(self, text: str) -> bool:
        """检查文本是否重复。

        Args:
            text: 文本内容。

        Returns:
            如果文本不重复返回True。
        """
        text_hash = self._compute_hash(text)
        if text_hash in self.seen_hashes:
            return False
        self.seen_hashes.add(text_hash)
        return True

    def _compute_hash(self, text: str) -> str:
        """计算文本的哈希值。

        Args:
            text: 文本内容。

        Returns:
            文本的SHA256哈希值。
        """
        normalized = text.lower().strip()
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()

    def _check_quality(self, text: str) -> bool:
        """检查文本质量。

        Args:
            text: 文本内容。

        Returns:
            如果文本质量合格返回True。
        """
        if not text or len(text.strip()) == 0:
            return False

        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        total_chars = len(text)
        chinese_ratio = chinese_chars / total_chars if total_chars > 0 else 0

        if chinese_ratio < self.config.min_chinese_ratio:
            logger.debug(f"Chinese ratio {chinese_ratio:.2f} below threshold")
            return False

        url_matches = self.URL_PATTERN.findall(text)
        url_ratio = len(' '.join(url_matches)) / len(text) if len(text) > 0 else 0
        if url_ratio > self.config.max_url_ratio:
            logger.debug(f"URL ratio {url_ratio:.2f} above threshold")
            return False

        special_matches = self.SPECIAL_CHAR_PATTERN.findall(text)
        special_ratio = len(' '.join(special_matches)) / len(text) if len(text) > 0 else 0
        if special_ratio > self.config.max_special_char_ratio:
            logger.debug(f"Special char ratio {special_ratio:.2f} above threshold")
            return False

        if self._check_repetition(text):
            logger.debug("Text has high repetition")
            return False

        return True

    def _check_repetition(self, text: str) -> bool:
        """检查文本重复率。

        Args:
            text: 文本内容。

        Returns:
            如果重复率过高返回True。
        """
        if len(text) < 10:
            return False

        n = 3
        ngrams = [text[i:i+n] for i in range(len(text) - n + 1)]
        if not ngrams:
            return False

        ngram_counts = Counter(ngrams)
        most_common_count = ngram_counts.most_common(1)[0][1] if ngram_counts else 0
        repetition_ratio = most_common_count / len(ngrams) if len(ngrams) > 0 else 0

        return repetition_ratio > self.config.max_repetition_ratio

    def filter_batch(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """批量过滤数据。

        Args:
            samples: 待过滤的样本列表。

        Returns:
            过滤后的样本列表。

        Example:
            >>> data_filter = DataFilter(min_length=10)
            >>> samples = [
            ...     {'text': '短文本', 'id': '1'},
            ...     {'text': '这是一个足够长的有效测试文本。', 'id': '2'},
            ... ]
            >>> filtered = data_filter.filter_batch(samples)
            >>> print(len(filtered))
            1
        """
        return [sample for sample in samples if self.is_valid(sample)]

    def reset_seen_hashes(self) -> None:
        """重置已见哈希集合（用于处理新数据集）。"""
        self.seen_hashes.clear()
        logger.info("Reset seen hashes")

    def get_stats(self) -> Dict[str, int]:
        """获取过滤统计信息。

        Returns:
            包含过滤统计信息的字典。

        Example:
            >>> data_filter = DataFilter()
            >>> # ... process samples ...
            >>> stats = data_filter.get_stats()
            >>> print(f"Pass rate: {stats['passed'] / stats['total']:.2%}")
        """
        return self._stats.copy()

    def reset_stats(self) -> None:
        """重置统计信息。"""
        self._stats = {
            'total': 0,
            'passed': 0,
            'filtered': 0,
            'length_filtered': 0,
            'quality_filtered': 0,
            'duplicate_filtered': 0,
            'repetition_filtered': 0,
        }


class QualityScorer:
    """数据质量评分器。

    为文本样本计算质量分数。

    Example:
        >>> scorer = QualityScorer()
        >>> score = scorer.score(sample)
        >>> if score >= 0.7:
        ...     # 高质量数据
    """

    def __init__(self):
        """初始化质量评分器。"""
        self.weights = {
            'length': 0.2,
            'chinese_ratio': 0.2,
            'diversity': 0.3,
            'coherence': 0.3,
        }

    def score(self, sample: Dict[str, Any]) -> float:
        """计算样本的质量分数。

        Args:
            sample: 待评分的样本字典。

        Returns:
            质量分数，范围0-1。
        """
        text = self._extract_text(sample)
        if not text:
            return 0.0

        length_score = self._score_length(text)
        chinese_score = self._score_chinese_ratio(text)
        diversity_score = self._score_diversity(text)
        coherence_score = self._score_coherence(text)

        total_score = (
            self.weights['length'] * length_score +
            self.weights['chinese_ratio'] * chinese_score +
            self.weights['diversity'] * diversity_score +
            self.weights['coherence'] * coherence_score
        )

        return min(1.0, max(0.0, total_score))

    def _extract_text(self, sample: Dict[str, Any]) -> Optional[str]:
        """从样本中提取文本。"""
        for key in ['text', 'content', 'body']:
            if key in sample and isinstance(sample[key], str):
                return sample[key]
        return None

    def _score_length(self, text: str) -> float:
        """长度评分。"""
        length = len(text)
        if length < 10:
            return length / 10 * 0.5
        if length < 100:
            return 0.5 + (length - 10) / 90 * 0.3
        if length < 1000:
            return 0.8 + (length - 100) / 900 * 0.2
        return 1.0

    def _score_chinese_ratio(self, text: str) -> float:
        """中文比例评分。"""
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        total_chars = len(text)
        if total_chars == 0:
            return 0.0
        ratio = chinese_chars / total_chars
        if ratio < 0.1:
            return ratio / 0.1 * 0.3
        if ratio < 0.3:
            return 0.3 + (ratio - 0.1) / 0.2 * 0.3
        if ratio < 0.7:
            return 0.6 + (ratio - 0.3) / 0.4 * 0.4
        return 1.0

    def _score_diversity(self, text: str) -> float:
        """多样性评分。"""
        if len(text) < 2:
            return 0.0
        unique_chars = len(set(text))
        diversity = unique_chars / len(text)
        return min(1.0, diversity * 2)

    def _score_coherence(self, text: str) -> float:
        """连贯性评分（简化版本）。"""
        if len(text) < 10:
            return 0.5
        avg_word_length = sum(len(w) for w in text.split()) / max(1, len(text.split()))
        return min(1.0, avg_word_length / 2.0)
