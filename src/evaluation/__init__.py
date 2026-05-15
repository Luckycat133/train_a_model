"""灵猫墨韵评估框架

本模块提供完整的语言模型评估功能，包括：
- 标准LM评估（困惑度、BPC等）
- 开放式生成评估
- 能力涌现检测
- 多样性评估
- 自定义评估器接口
"""

from src.evaluation.base_evaluator import (
    BaseEvaluator,
    CustomEvaluator,
    EvaluatorRegistry,
    EvaluationResult,
    EvaluationReport,
    EvaluatorType,
    EvaluationStage,
)

from src.evaluation.lm_metrics import (
    LMEvaluator,
    StreamingLMEvaluator,
    LMMETRICS,
    LMMetricResult,
)

from src.evaluation.open_evaluator import (
    OpenEndedEvaluator,
    ReferenceFreeEvaluator,
    EmergenceDetector,
    OPENENDEDMETRICS,
    EmergenceMetric,
    EmergenceDetectionResult,
    GenerationMetrics,
)

from src.evaluation.diversity import (
    DiversityEvaluator,
    ComparativeDiversityEvaluator,
    StreamingDiversityEvaluator,
    DIVERSITYMETRICS,
    DiversityMetrics,
)


__all__ = [
    "BaseEvaluator",
    "CustomEvaluator",
    "EvaluatorRegistry",
    "EvaluationResult",
    "EvaluationReport",
    "EvaluatorType",
    "EvaluationStage",
    "LMEvaluator",
    "StreamingLMEvaluator",
    "LMMETRICS",
    "LMMetricResult",
    "OpenEndedEvaluator",
    "ReferenceFreeEvaluator",
    "EmergenceDetector",
    "OPENENDEDMETRICS",
    "EmergenceMetric",
    "EmergenceDetectionResult",
    "GenerationMetrics",
    "DiversityEvaluator",
    "ComparativeDiversityEvaluator",
    "StreamingDiversityEvaluator",
    "DIVERSITYMETRICS",
    "DiversityMetrics",
]


def create_standard_evaluator(
    name: str = "standard_evaluator",
    compute_perplexity: bool = True,
    compute_bpc: bool = True,
    compute_accuracy: bool = True,
    vocab_size: int = 256,
) -> LMEvaluator:
    """创建标准LM评估器
    
    参数:
        name: 评估器名称
        compute_perplexity: 是否计算困惑度
        compute_bpc: 是否计算BPC
        compute_accuracy: 是否计算准确率
        vocab_size: 词汇表大小
        
    返回:
        配置好的LMEvaluator实例
    """
    return LMEvaluator(
        name=name,
        compute_perplexity=compute_perplexity,
        compute_bpc=compute_bpc,
        compute_accuracy=compute_accuracy,
        vocab_size=vocab_size,
    )


def create_open_ended_evaluator(
    name: str = "open_ended_evaluator",
    emergence_threshold: float = 0.2,
) -> OpenEndedEvaluator:
    """创建开放式评估器
    
    参数:
        name: 评估器名称
        emergence_threshold: 涌现检测阈值
        
    返回:
        配置好的OpenEndedEvaluator实例
    """
    return OpenEndedEvaluator(
        name=name,
        emergence_threshold=emergence_threshold,
    )


def create_diversity_evaluator(
    name: str = "diversity_evaluator",
    compute_ttr: bool = True,
    compute_entropy: bool = True,
    compute_ngram: bool = True,
    ngram_range: tuple = (2, 4),
) -> DiversityEvaluator:
    """创建多样性评估器
    
    参数:
        name: 评估器名称
        compute_ttr: 是否计算TTR
        compute_entropy: 是否计算熵值
        compute_ngram: 是否计算N-gram多样性
        ngram_range: N-gram范围
        
    返回:
        配置好的DiversityEvaluator实例
    """
    return DiversityEvaluator(
        name=name,
        compute_ttr=compute_ttr,
        compute_entropy=compute_entropy,
        compute_ngram=compute_ngram,
        ngram_range=ngram_range,
    )


def get_default_registry() -> EvaluatorRegistry:
    """获取默认的评估器注册表实例（单例模式）
    
    返回:
        EvaluatorRegistry单例实例
    """
    return EvaluatorRegistry()
