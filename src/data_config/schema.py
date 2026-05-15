"""数据集配置Schema定义。

本模块定义了数据集配置的Pydantic模型，用于验证和结构化数据集配置信息。
支持的数据类型包括：数据源URL、预处理步骤、数据格式、验证规则等。
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator


class DataFormat(str, Enum):
    """支持的数据格式枚举。"""
    JSONL = "jsonl"
    JSON = "json"
    TXT = "txt"
    CSV = "csv"
    PARQUET = "parquet"


class ValidationType(str, Enum):
    """数据验证类型枚举。"""
    REQUIRED = "required"
    SCHEMA = "schema"
    RANGE = "range"
    REGEX = "regex"
    CUSTOM = "custom"


class PreprocessingType(str, Enum):
    """预处理类型枚举。"""
    FILTER = "filter"
    TRANSFORM = "transform"
    AUGMENT = "augment"
    NORMALIZE = "normalize"


class ValidationRule(BaseModel):
    """数据验证规则配置。
    
    定义单个字段的验证规则，包括必填检查、范围验证、正则匹配等。
    
    属性：
        field: 要验证的字段名
        rule_type: 验证类型（required, schema, range, regex, custom）
        value: 验证规则的值（可以是默认值、范围、模式等）
        message: 验证失败时的错误消息
    """
    field: str = Field(..., description="要验证的字段名")
    rule_type: ValidationType = Field(default=ValidationType.REQUIRED, description="验证类型")
    value: Optional[Any] = Field(default=None, description="验证规则的值")
    message: Optional[str] = Field(default=None, description="验证失败消息")
    
    @field_validator('field')
    @classmethod
    def validate_field_name(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("字段名不能为空")
        return v.strip()


class PreprocessingStep(BaseModel):
    """数据预处理步骤配置。
    
    定义单个预处理步骤，包括过滤、转换、增强、标准化等操作。
    
    属性：
        step_type: 预处理类型（filter, transform, augment, normalize）
        field: 要处理的字段名
        params: 处理参数
        condition: 过滤条件（仅filter类型需要）
    """
    step_type: PreprocessingType = Field(..., description="预处理类型")
    field: str = Field(default="text", description="处理的字段名")
    params: Dict[str, Any] = Field(default_factory=dict, description="处理参数")
    condition: Optional[str] = Field(default=None, description="过滤条件表达式")


class DataSource(BaseModel):
    """数据源配置。
    
    定义数据集的来源信息，包括URL、本地路径、下载方式等。
    
    属性：
        url: 数据源的URL地址
        path: 本地存储路径（下载后或已存在的路径）
        download_method: 下载方式（http, git, local, huggingface）
        verify_ssl: 是否验证SSL证书
        timeout: 下载超时时间（秒）
    """
    url: Optional[str] = Field(default=None, description="数据源URL")
    path: str = Field(..., description="本地路径或下载目标路径")
    download_method: str = Field(default="http", description="下载方式: http, git, local, huggingface")
    verify_ssl: bool = Field(default=True, description="是否验证SSL")
    timeout: int = Field(default=300, ge=10, le=3600, description="超时时间（秒）")
    
    def is_local(self) -> bool:
        """检查是否为本地数据源。"""
        from pathlib import Path
        p = Path(self.path)
        return p.exists() or self.download_method == "local"


class FieldMapping(BaseModel):
    """字段映射配置。
    
    定义输入数据字段到标准字段的映射关系。
    
    属性：
        source_field: 源数据中的字段名
        target_field: 映射到的目标字段名
        transform: 字段值转换函数（可选）
    """
    source_field: str = Field(..., description="源字段名")
    target_field: str = Field(..., description="目标字段名")
    transform: Optional[str] = Field(default=None, description="转换函数名")


class DatasetConfig(BaseModel):
    """完整的数据集配置模型。
    
    这是数据集配置的主Schema，定义了数据集的所有配置信息，包括：
    - 基本信息（名称、版本、描述）
    - 数据源（URL、本地路径）
    - 数据格式和处理方式
    - 预处理和验证规则
    - 字段映射
    - 统计信息
    
    属性：
        name: 数据集名称（唯一标识符）
        version: 数据集版本
        description: 数据集描述
        source: 数据源配置
        format: 数据格式（jsonl, json, txt, csv, parquet）
        preprocessing: 预处理步骤列表
        validation: 验证规则列表
        field_mapping: 字段映射配置
        metadata: 额外的元数据信息
        tags: 标签列表，用于分类和搜索
        size: 数据集大小（样本数）
        license: 数据许可证
    """
    name: str = Field(..., description="数据集名称（唯一标识）", min_length=1, max_length=100)
    version: str = Field(default="1.0.0", description="数据集版本")
    description: Optional[str] = Field(default=None, description="数据集描述")
    
    source: DataSource = Field(..., description="数据源配置")
    format: DataFormat = Field(default=DataFormat.JSONL, description="数据格式")
    
    preprocessing: List[PreprocessingStep] = Field(default_factory=list, description="预处理步骤")
    validation: List[ValidationRule] = Field(default_factory=list, description="验证规则")
    
    field_mapping: Optional[List[FieldMapping]] = Field(default=None, description="字段映射")
    
    metadata: Dict[str, Any] = Field(default_factory=dict, description="额外元数据")
    tags: List[str] = Field(default_factory=list, description="标签")
    
    size: Optional[int] = Field(default=None, description="数据集大小（样本数）")
    license: Optional[str] = Field(default=None, description="许可证")
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("数据集名称不能为空")
        name = v.strip().lower().replace(' ', '_').replace('-', '_')
        if not name.replace('_', '').isalnum():
            raise ValueError("数据集名称只能包含字母、数字、下划线")
        return name
    
    @field_validator('tags')
    @classmethod
    def validate_tags(cls, v: List[str]) -> List[str]:
        return [tag.strip().lower() for tag in v if tag.strip()]
    
    def get_required_fields(self) -> List[str]:
        """获取所有必填字段。"""
        required = []
        for rule in self.validation:
            if rule.rule_type == ValidationType.REQUIRED:
                required.append(rule.field)
        return required
    
    def get_filter_steps(self) -> List[PreprocessingStep]:
        """获取所有过滤步骤。"""
        return [step for step in self.preprocessing if step.step_type == PreprocessingType.FILTER]
    
    def get_transform_steps(self) -> List[PreprocessingStep]:
        """获取所有转换步骤。"""
        return [step for step in self.preprocessing if step.step_type == PreprocessingType.TRANSFORM]


class MixedDatasetConfig(BaseModel):
    """混合数据集配置。
    
    用于配置多个数据集的混合训练，支持设置各数据集的采样权重。
    
    属性：
        name: 混合数据集名称
        datasets: 子数据集配置列表
        weights: 各数据集的采样权重（与datasets对应）
        shuffle: 是否打乱混合顺序
        seed: 随机种子（用于可重现性）
    """
    name: str = Field(..., description="混合数据集名称")
    datasets: List[str] = Field(..., description="子数据集名称列表")
    weights: List[float] = Field(..., description="采样权重列表")
    shuffle: bool = Field(default=True, description="是否打乱")
    seed: int = Field(default=42, description="随机种子")
    
    @field_validator('weights')
    @classmethod
    def validate_weights(cls, v: List[float]) -> List[float]:
        if not v:
            raise ValueError("权重列表不能为空")
        if any(w < 0 for w in v):
            raise ValueError("权重不能为负数")
        total = sum(v)
        if total <= 0:
            raise ValueError("权重总和必须大于0")
        return v
    
    @field_validator('datasets', 'weights')
    @classmethod
    def validate_lengths(cls, v: List) -> List:
        if len(v) == 0:
            raise ValueError("列表不能为空")
        return v


class DatasetStats(BaseModel):
    """数据集统计信息。
    
    记录数据集的各种统计指标，用于监控和分析。
    
    属性：
        name: 数据集名称
        total_samples: 总样本数
        total_size_bytes: 总大小（字节）
        avg_sample_size: 平均样本大小
        field_stats: 各字段的统计信息
        quality_score: 质量评分（0-100）
        last_updated: 最后更新时间
    """
    name: str = Field(..., description="数据集名称")
    total_samples: int = Field(default=0, ge=0, description="总样本数")
    total_size_bytes: int = Field(default=0, ge=0, description="总大小（字节）")
    avg_sample_size: float = Field(default=0.0, ge=0, description="平均样本大小")
    field_stats: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="字段统计")
    quality_score: float = Field(default=0.0, ge=0.0, le=100.0, description="质量评分")
    last_updated: Optional[str] = Field(default=None, description="最后更新时间")


class DatasetProfile(BaseModel):
    """数据集配置文件。
    
    包含数据集配置和统计信息的完整配置。
    
    属性：
        config: 数据集配置
        stats: 数据集统计信息
    """
    config: DatasetConfig = Field(..., description="数据集配置")
    stats: DatasetStats = Field(..., description="数据集统计")
