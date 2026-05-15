"""Lingmao Moyun 数据预处理管道。

提供完整的数据预处理流程，包括清洗、过滤、标准化、格式转换和验证等功能。
支持预训练数据、SFT对话数据等多种数据格式的处理。

模块结构：
    - cleaner: 数据清洗（去除噪声、修正编码）
    - filter: 数据过滤（长度、质量、重复率）
    - normalizer: 文本标准化（繁简转换、标点统一）
    - formatter: 格式转换（JSONL、JSON、CSV、TXT）
    - validator: 数据验证（完整性、格式检查）

示例：
    >>> from src.data_processing import DataCleaner, DataFilter, DataNormalizer
    >>> from src.data_processing import format_converter, DataValidator
    >>>
    >>> # 数据清洗
    >>> cleaner = DataCleaner()
    >>> cleaned = cleaner.clean(text)
    >>>
    >>> # 数据过滤
    >>> filter = DataFilter(min_length=10, max_length=5000)
    >>> if filter.is_valid(sample):
    ...     # 保留数据
    >>>
    >>> # 文本标准化
    >>> normalizer = DataNormalizer(simplify_chinese=True)
    >>> normalized = normalizer.normalize(text)
    >>>
    >>> # 格式转换
    >>> formatter = DataFormatter(input_format='json', output_format='jsonl')
    >>> formatter.convert('input.json', 'output.jsonl')
    >>>
    >>> # 数据验证
    >>> validator = DataValidator()
    >>> if validator.validate(sample):
    ...     # 数据有效
"""

from src.data_processing.cleaner import DataCleaner
from src.data_processing.filter import DataFilter
from src.data_processing.normalizer import DataNormalizer
from src.data_processing.formatter import DataFormatter
from src.data_processing.validator import DataValidator

__version__ = "1.0.0"

__all__ = [
    "DataCleaner",
    "DataFilter",
    "DataNormalizer",
    "DataFormatter",
    "DataValidator",
]
