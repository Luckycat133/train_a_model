"""Lingmao Moyun 数据管道模块。

本模块提供统一的数据加载和处理接口，支持预训练、SFT对话和长序列打包等多种数据格式。
包含流式处理、混合采样和动态打包等高级功能。

示例：
    >>> from src.data import PretrainDataset, SFTDataset, PackedDataset
    >>>
    >>> # 预训练数据加载
    >>> pretrain_ds = PretrainDataset(
    ...     data_paths="data/pretrain.jsonl",
    ...     context_length=512
    ... )
    >>>
    >>> # SFT对话数据加载
    >>> sft_ds = SFTDataset(
    ...     data_paths="data/sft.jsonl",
    ...     dialogue_format="sharegpt"
    ... )
    >>>
    >>> # 长序列打包
    >>> packed_ds = PackedDataset(
    ...     data_paths="data/corpus/",
    ...     context_length=2048
    ... )
"""

from src.data.base_dataset import (
    BaseDataset,
    StreamingDataset,
    WeightedMixingDataset,
)

from src.data.pretrain_dataset import (
    PretrainDataset,
    ConcatPretrainDataset,
    PretrainDatasetFactory,
)

from src.data.sft_dataset import (
    SFTDataset,
    SFTTrainingCollator,
    SFTDatasetFactory,
    ChatTemplate,
    DialogueFormat,
)

from src.data.packed_dataset import (
    PackedDataset,
    DynamicPackingDataset,
)

from src.data_config import (
    DatasetConfig,
    DataFormat,
    ValidationRule,
    PreprocessingStep,
    DatasetRegistry,
    DatasetInfo,
    DatasetLoader,
    StreamingDataLoader,
    DataMixer,
)

__all__ = [
    # 基础接口
    "BaseDataset",
    "StreamingDataset",
    "WeightedMixingDataset",
    # 预训练数据
    "PretrainDataset",
    "ConcatPretrainDataset",
    "PretrainDatasetFactory",
    # SFT数据
    "SFTDataset",
    "SFTTrainingCollator",
    "SFTDatasetFactory",
    "ChatTemplate",
    "DialogueFormat",
    # 打包数据
    "PackedDataset",
    "DynamicPackingDataset",
    # 数据配置系统
    "DatasetConfig",
    "DataFormat",
    "ValidationRule",
    "PreprocessingStep",
    "DatasetRegistry",
    "DatasetInfo",
    "DatasetLoader",
    "StreamingDataLoader",
    "DataMixer",
]
