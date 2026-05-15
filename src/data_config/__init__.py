"""Lingmao Moyun 数据配置管理系统。

本模块提供统一的数据集配置、注册和加载接口，支持多种数据格式和流式加载。
主要用于管理古诗词、古文典籍等中文古典文献数据集的加载和预处理。

功能特性：
- 数据集注册表：管理所有可用数据集的元信息
- 数据加载器：统一的数据加载接口，支持多种格式
- 配置Schema：标准化的数据集配置验证

支持的数据格式：
- JSONL：JSON Lines格式，每行一个JSON对象
- JSON：JSON数组或单个JSON对象
- TXT：纯文本文件，每行一个样本

示例：
    >>> from src.data_config import DatasetRegistry, DatasetLoader, DatasetConfig
    >>>
    >>> # 注册新数据集
    >>> registry = DatasetRegistry()
    >>> registry.register_dataset(
    ...     name="poetry",
    ...     source="https://example.com/poetry.jsonl",
    ...     path="data/poetry.jsonl",
    ...     size=10000,
    ...     description="古诗词数据集"
    ... )
    >>>
    >>> # 加载数据集
    >>> loader = DatasetLoader()
    >>> dataset = loader.load("poetry")
"""

from src.data_config.schema import (
    DatasetConfig,
    DataFormat,
    ValidationRule,
    PreprocessingStep,
    DataSource,
    FieldMapping,
    MixedDatasetConfig,
    DatasetStats,
    DatasetProfile,
)
from src.data_config.registry import DatasetRegistry, DatasetInfo
from src.data_config.loader import DatasetLoader, StreamingDataLoader, DataMixer

__version__ = "0.1.0"

__all__ = [
    "DatasetConfig",
    "DataFormat",
    "ValidationRule",
    "PreprocessingStep",
    "DataSource",
    "FieldMapping",
    "MixedDatasetConfig",
    "DatasetStats",
    "DatasetProfile",
    "DatasetRegistry",
    "DatasetInfo",
    "DatasetLoader",
    "StreamingDataLoader",
    "DataMixer",
]
