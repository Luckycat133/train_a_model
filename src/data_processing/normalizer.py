"""文本标准化模块。

提供文本标准化功能，包括：
- 繁体中文到简体中文转换
- 标点符号统一（繁简、中英文）
- 特殊字符处理
- Unicode规范化
- 大小写标准化
- 数字格式标准化

示例：
    >>> normalizer = DataNormalizer()
    >>> normalizer.config.simplify_chinese = True
    >>> normalizer.config.normalize_punctuation = True
    >>> normalized = normalizer.normalize(text)
"""

import re
import unicodedata
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass

from src.logger import get_logger

logger = get_logger("LingmaoMoyun.Data.Normalizer")


TRADITIONAL_TO_SIMPLIFIED: Dict[str, str] = {
    '\u5c0d': '\u5bf9', '\u5011': '\u4ed6', '\u6642': '\u65f6', '\u570b': '\u56fd', '\u6703': '\u4f1a',
    '\u958b': '\u5f00', '\u73fe': '\u73b0', '\u5834': '\u573a', '\u9577': '\u9577', '\u696d': '\u4e1a',
    '\u767c': '\u53d1', '\u99ac': '\u9a6c', '\u6771': '\u4e1c', '\u8eca': '\u8f66', '\u96fb': '\u7535',
    '\u7db2': '\u7f51', '\u6a5f': '\u673a', '\u52d9': '\u52a1', '\u8655': '\u5904', '\u66f8': '\u4e66',
    '\u5b78': '\u5b66', '\u54e1': '\u5458', '\u7d93': '\u7ecf', '\u6a02': '\u4e50', '\u8072': '\u58f0',
    '\u8207': '\u5174', '\u8aaa': '\u8bf4', '\u8a9e': '\u8bed', '\u8a8d': '\u8ba4', '\u8a18': '\u8bb0',
    '\u8a31': '\u8bb8', '\u8ad6': '\u8bba', '\u8a2d': '\u8bbe', '\u9ede': '\u70b9', '\u8b8a': '\u53d8',
    '\u9593': '\u95f4', '\u984c': '\u9898', '\u61c9': '\u5e94', '\u6a23': '\u6837', '\u95dc': '\u5173',
    '\u6c23': '\u6c14', '\u7a2e': '\u79cd', '\u9084': '\u8fd8', '\u9019': '\u8fd9', '\u9032': '\u8fdb',
    '\u9060': '\u8fdc', '\u9023': '\u8fde', '\u904b': '\u8fd0', '\u904e': '\u8fc7', '\u842c': '\u4e07',
    '\u88cf': '\u91cc', '\u90f5': '\u90ae', '\u9673': '\u9648', '\u96e2': '\u79bb', '\u96f2': '\u4e91',
    '\u8a69': '\u8bd7', '\u8a5b': '\u8bcd', '\u8acb': '\u8bf7', '\u7e3d': '\u603b', '\u7d50': '\u7ed3',
    '\u92b7': '\u94b1', '\u9280': '\u94f6', '\u9435': '\u94c1', '\u6a4b': '\u6865', '\u76e3': '\u76d1',
    '\u8996': '\u89c6', '\u9580': '\u95e8', '\u6a13': '\u697c', '\u5eda': '\u53a8', '\u5340': '\u533a',
    '\u91ab': '\u533b', '\u85e5': '\u836f', '\u53a0': '\u5382', '\u9921': '\u9986', '\u5712': '\u56ed',
    '\u9ad4': '\u4f53', '\u5104': '\u4ebf', '\u53c3': '\u53c2', '\u89c0': '\u89c2', '\u8b80': '\u8bfb',
    '\u5beb': '\u5199', '\u807d': '\u542c', '\u50b3': '\u4f30', '\u50b5': '\u4ef7', '\u766b': '\u5f02',
    '\u96e8': '\u96e8', '\u9700': '\u9700', '\u9748': '\u7075', '\u8c93': '\u732b', '\u58a8': '\u58a8',
    '\u97fb': '\u97fb', '\u66f8': '\u4e66', '\u5b78': '\u5b66', '\u53e4': '\u53e4', '\u5178': '\u5178',
    '\u4f5c': '\u4f5c', '\u5bb6': '\u5bb6', '\u8cde': '\u8d4f', '\u6790': '\u6790', '\u8a9e': '\u8bed',
    '\u6559': '\u6559', '\u6750': '\u6750', '\u8ab2': '\u8bfe', '\u672c': '\u672c', '\u5167': '\u5185',
    '\u5bb9': '\u5bb9', '\u8207': '\u4e0e', '\u65b9': '\u65b9', '\u6cd5': '\u6cd5', '\u7e3d': '\u603b',
    '\u7d50': '\u7ed3', '\u958b': '\u5f00', '\u767c': '\u53d1', '\u74b0': '\u73af', '\u5883': '\u5883',
    '\u932e': '\u9519', '\u8aa4': '\u8bef', '\u7e79': '\u4e49', '\u52d9': '\u52a1', '\u9109': '\u4e61',
    '\u93ae': '\u9547', '\u92b7': '\u94b1', '\u9280': '\u94f6', '\u9435': '\u94c1', '\u6a4b': '\u6865',
    '\u76e3': '\u76d1', '\u8996': '\u89c6', '\u9580': '\u95e8', '\u6a13': '\u697c', '\u5eda': '\u53a8',
    '\u5340': '\u533a', '\u91ab': '\u533b', '\u85e5': '\u836f', '\u53a0': '\u5382', '\u9921': '\u9986',
    '\u5712': '\u56ed', '\u9ad4': '\u4f53', '\u5104': '\u4ebf', '\u53c3': '\u53c2', '\u89c0': '\u89c2',
    '\u8b80': '\u8bfb', '\u5beb': '\u5199', '\u807d': '\u542c', '\u50b3': '\u4f30', '\u50b5': '\u4ef7',
    '\u766b': '\u5f02', '\u97fb': '\u8f7d', '\u8207': '\u4e0e', '\u65bc': '\u4e8e', '\u8457': '\u7740',
    '\u4e86': '\u4e86', '\u904e': '\u8fc7', '\u5fae': '\u5fae', '\u5fae': '\u5fae', '\u6bcf': '\u6bcf',
    '\u5374': '\u5374',
}


@dataclass
class NormalizerConfig:
    """文本标准化配置类。"""

    simplify_chinese: bool = True
    normalize_punctuation: bool = True
    normalize_whitespace: bool = True
    normalize_quotes: bool = True
    normalize_brackets: bool = True
    unicode_normalize: str = "NFKC"
    lowercase_english: bool = False
    normalize_numbers: bool = False


class DataNormalizer:
    """文本标准化器。

    对文本进行标准化处理，包括繁简转换、标点规范化等。

    Attributes:
        config: 标准化配置对象。

    Example:
        >>> normalizer = DataNormalizer()
        >>> normalizer.config.simplify_chinese = True
        >>> result = normalizer.normalize("\u7e41\u9ad4\u4e2d\u6587")
        >>> print(result)
        \u7e41\u4f53\u4e2d\u6587
    """

    FULLWIDTH_PAIRS = {
        '\uff01': '!', '\uff1f': '?', '\uff0c': ',', '\u3002': '.',
        '\uff1a': ':', '\uff1b': ';', '\u201c': '"', '\u201d': '"',
        '\uff08': '(', '\uff09': ')', '\u3010': '[', '\u3011': ']',
        '\u300a': '<', '\u300b': '>', '\u3001': ',', '\uff5e': '~',
        '\u300c': '"', '\u300d': '"', '\u300e': "'", '\u300f': "'",
    }

    PUNCTUATION_NORMALIZE = {
        '\uff0c': ',', '\u3002': '.', '\uff01': '!', '\uff1f': '?',
        '\uff1a': ':', '\uff1b': ';', '\u3001': ',', '\uff5e': '~',
    }

    def __init__(self, config: Optional[NormalizerConfig] = None):
        """初始化文本标准化器。

        Args:
            config: 标准化配置对象。如果为None，使用默认配置。
        """
        self.config = config or NormalizerConfig()

    def normalize(self, text: str) -> str:
        """执行完整的文本标准化流程。

        Args:
            text: 待标准化的原始文本。

        Returns:
            标准化后的文本。

        Example:
            >>> normalizer = DataNormalizer()
            >>> result = normalizer.normalize("\u7e41\u9ad4\u4e2d\u6587\u3000\u6e2c\u8a66")
            >>> print(result)
            \u7e41\u4f53\u4e2d\u6587 \u6e2c\u8a66
        """
        if not isinstance(text, str):
            text = str(text) if text is not None else ""

        if not text:
            return text

        if self.config.simplify_chinese:
            text = self._simplify_chinese(text)

        if self.config.unicode_normalize:
            text = unicodedata.normalize(self.config.unicode_normalize, text)

        if self.config.normalize_punctuation:
            text = self._normalize_punctuation(text)

        if self.config.normalize_whitespace:
            text = self._normalize_whitespace(text)

        if self.config.normalize_quotes:
            text = self._normalize_quotes(text)

        if self.config.normalize_brackets:
            text = self._normalize_brackets(text)

        if self.config.lowercase_english:
            text = self._lowercase_english(text)

        if self.config.normalize_numbers:
            text = self._normalize_numbers(text)

        return text

    def _simplify_chinese(self, text: str) -> str:
        """将繁体中文转换为简体中文。"""
        result = []
        for char in text:
            if char in TRADITIONAL_TO_SIMPLIFIED:
                result.append(TRADITIONAL_TO_SIMPLIFIED[char])
            else:
                result.append(char)
        return ''.join(result)

    def _normalize_punctuation(self, text: str) -> str:
        """规范化标点符号。"""
        for old, new in self.PUNCTUATION_NORMALIZE.items():
            text = text.replace(old, new)
        return text

    def _normalize_whitespace(self, text: str) -> str:
        """规范化空白字符。"""
        text = re.sub(r'\s+', ' ', text)
        text = text.replace('\u3000', ' ')
        return text.strip()

    def _normalize_quotes(self, text: str) -> str:
        """规范化引号。"""
        replacements = {
            '\u201c': '"', '\u201d': '"', '\u2018': "'", '\u2019': "'",
            '\u00ab': '"', '\u00bb': '"', '\u2039': "'", '\u203a': "'",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    def _normalize_brackets(self, text: str) -> str:
        """规范化括号。"""
        replacements = {
            '\u3010': '[', '\u3011': ']', '\u3016': '[', '\u3017': ']',
            '\u3018': '[', '\u3019': ']', '\u301a': '[', '\u301b': ']',
            '\u3008': '<', '\u3009': '>', '\u300a': '<', '\u300b': '>',
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    def _lowercase_english(self, text: str) -> str:
        """将英文字母转为小写。"""
        return text.lower()

    def _normalize_numbers(self, text: str) -> str:
        """规范化数字格式。"""
        text = re.sub(r'\u96f6+', '\u96f6', text)
        return text

    def normalize_batch(self, texts: List[str]) -> List[str]:
        """批量标准化文本。

        Args:
            texts: 待标准化的文本列表。

        Returns:
            标准化后的文本列表。

        Example:
            >>> normalizer = DataNormalizer()
            >>> results = normalizer.normalize_batch(['\u7e41\u9ad4', '\u7c21\u9ad4'])
            >>> print(results)
            ['\u7e41\u4f53', '\u7c21\u4f53']
        """
        return [self.normalize(text) for text in texts]

    def get_normalization_stats(self) -> Dict[str, int]:
        """获取标准化统计信息。

        Returns:
            包含标准化统计信息的字典。
        """
        return {
            'total_normalized': 0,
            'chinese_simplified': 0,
            'punctuation_normalized': 0,
        }
