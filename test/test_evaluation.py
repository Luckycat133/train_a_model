"""评估框架测试模块

测试灵猫墨韵评估框架的各个组件：
- 基础评估器
- 语言模型指标
- 开放式评估
- 多样性评估
"""

import pytest
import json
import tempfile
from pathlib import Path
from collections import Counter
from typing import Any, Dict, List

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

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
)
from src.evaluation.open_evaluator import (
    OpenEndedEvaluator,
    ReferenceFreeEvaluator,
    EmergenceDetector,
    OPENENDEDMETRICS,
)
from src.evaluation.diversity import (
    DiversityEvaluator,
    DIVERSITYMETRICS,
)


class TestEvaluationResult:
    """测试EvaluationResult类"""

    def test_creation(self):
        """测试基本创建"""
        result = EvaluationResult(
            metric_name="test_metric",
            value=0.95,
            confidence=0.02,
        )
        assert result.metric_name == "test_metric"
        assert result.value == 0.95
        assert result.confidence == 0.02
        assert result.timestamp > 0

    def test_to_dict(self):
        """测试转换为字典"""
        result = EvaluationResult(
            metric_name="accuracy",
            value=0.85,
            metadata={"batch_size": 32},
        )
        result_dict = result.to_dict()
        
        assert result_dict["metric_name"] == "accuracy"
        assert result_dict["value"] == 0.85
        assert result_dict["metadata"]["batch_size"] == 32
        assert "timestamp" in result_dict

    def test_str_representation(self):
        """测试字符串表示"""
        result = EvaluationResult(metric_name="perplexity", value=15.5)
        assert "perplexity" in str(result)
        assert "15.5" in str(result)

    def test_str_with_confidence(self):
        """测试带置信度的字符串表示"""
        result = EvaluationResult(
            metric_name="accuracy",
            value=0.92,
            confidence=0.01,
        )
        result_str = str(result)
        assert "accuracy" in result_str
        assert "0.920" in result_str
        assert "±" in result_str


class TestEvaluationReport:
    """测试EvaluationReport类"""

    def test_creation(self):
        """测试报告创建"""
        report = EvaluationReport(evaluator_name="test_evaluator")
        assert report.evaluator_name == "test_evaluator"
        assert len(report.results) == 0

    def test_add_single_result(self):
        """测试添加单个结果"""
        report = EvaluationReport(evaluator_name="test")
        result = EvaluationResult(metric_name="metric1", value=1.0)
        report.add_result(result)
        
        assert len(report.results) == 1
        assert report.results[0].metric_name == "metric1"

    def test_add_multiple_results(self):
        """测试批量添加结果"""
        report = EvaluationReport(evaluator_name="test")
        results = [
            EvaluationResult(metric_name="m1", value=1.0),
            EvaluationResult(metric_name="m2", value=2.0),
            EvaluationResult(metric_name="m3", value=3.0),
        ]
        report.add_results(results)
        
        assert len(report.results) == 3

    def test_compute_summary(self):
        """测试统计汇总计算"""
        report = EvaluationReport(evaluator_name="test")
        report.add_results([
            EvaluationResult(metric_name="m1", value=1.0),
            EvaluationResult(metric_name="m2", value=2.0),
            EvaluationResult(metric_name="m3", value=3.0),
        ])
        
        summary = report.compute_summary()
        
        assert "mean" in summary
        assert "min" in summary
        assert "max" in summary
        assert summary["count"] == 3
        assert summary["mean"] == 2.0

    def test_save_and_load(self):
        """测试报告保存"""
        with tempfile.TemporaryDirectory() as tmpdir:
            report = EvaluationReport(evaluator_name="test")
            report.add_result(EvaluationResult(metric_name="metric", value=0.5))
            report.compute_summary()
            
            save_path = Path(tmpdir) / "report.json"
            report.save(save_path)
            
            assert save_path.exists()
            
            with open(save_path, "r") as f:
                loaded = json.load(f)
            
            assert loaded["evaluator_name"] == "test"
            assert len(loaded["results"]) == 1


class TestBaseEvaluator:
    """测试BaseEvaluator类"""

    def test_creation(self):
        """测试评估器创建"""
        evaluator = BaseEvaluator(
            name="test_evaluator",
            evaluator_type=EvaluatorType.STANDARD,
        )
        
        assert evaluator.name == "test_evaluator"
        assert evaluator.evaluator_type == EvaluatorType.STANDARD
        assert len(evaluator.metrics) == 0

    def test_register_metric(self):
        """测试指标注册"""
        evaluator = BaseEvaluator(name="test")
        
        def mock_metric(x):
            return x * 2
        
        evaluator.register_metric("double", mock_metric)
        
        assert "double" in evaluator.metrics

    def test_compute_metric(self):
        """测试指标计算"""
        evaluator = BaseEvaluator(name="test")
        
        def mock_metric(x):
            return x * 2
        
        evaluator.register_metric("double", mock_metric)
        result = evaluator.compute_metric("double", 5)
        
        assert result == 10

    def test_compute_metric_not_found(self):
        """测试不存在的指标"""
        evaluator = BaseEvaluator(name="test")
        result = evaluator.compute_metric("nonexistent", 5)
        
        assert result is None

    def test_stage_property(self):
        """测试阶段属性"""
        evaluator = BaseEvaluator(name="test")
        assert evaluator.stage == EvaluationStage.INITIALIZATION


class TestCustomEvaluator:
    """测试CustomEvaluator类"""

    def test_creation_with_custom_metrics(self):
        """测试带自定义指标的创建"""
        def custom_metric(model, dataset):
            return 42.0
        
        evaluator = CustomEvaluator(
            name="custom_test",
            custom_metrics={"custom_score": custom_metric},
        )
        
        assert "custom_score" in evaluator.metrics
        assert evaluator.evaluator_type == EvaluatorType.CUSTOM

    def test_evaluate_with_custom_metric(self):
        """测试自定义指标评估"""
        def custom_metric(model, dataset):
            return 42.0
        
        evaluator = CustomEvaluator(
            name="custom_test",
            custom_metrics={"custom_score": custom_metric},
        )
        
        results = evaluator._evaluate(None, None)
        
        assert len(results) == 1
        assert results[0].metric_name == "custom_score"
        assert results[0].value == 42.0


class TestEvaluatorRegistry:
    """测试EvaluatorRegistry类"""

    def test_singleton_pattern(self):
        """测试单例模式"""
        registry1 = EvaluatorRegistry()
        registry2 = EvaluatorRegistry()
        
        assert registry1 is registry2

    def test_register_and_get(self):
        """测试注册和获取"""
        registry = EvaluatorRegistry()
        evaluator = BaseEvaluator(name="test")
        
        registry.register("test_eval", evaluator)
        retrieved = registry.get("test_eval")
        
        assert retrieved is evaluator

    def test_get_nonexistent(self):
        """测试获取不存在的评估器"""
        registry = EvaluatorRegistry()
        result = registry.get("nonexistent")
        
        assert result is None

    def test_list_evaluators(self):
        """测试列出评估器"""
        registry = EvaluatorRegistry()
        
        registry.register("eval1", BaseEvaluator(name="e1"))
        registry.register("eval2", BaseEvaluator(name="e2"))
        
        names = registry.list_evaluators()
        
        assert "eval1" in names
        assert "eval2" in names

    def test_evaluate_all(self):
        """测试批量评估"""
        registry = EvaluatorRegistry()
        
        evaluator = BaseEvaluator(name="test")
        evaluator.register_metric("score", lambda m, d: 1.0)
        registry.register("test", evaluator)
        
        reports = registry.evaluate_all(None, None)
        
        assert "test" in reports


class TestLMMETRICS:
    """测试LMMETRICS类"""

    def test_perplexity(self):
        """测试困惑度计算"""
        log_likelihoods = [-0.5, -0.5, -0.5, -0.5]
        ppl = LMMETRICS.perplexity(log_likelihoods)
        
        expected = 1.64872
        assert abs(ppl - expected) < 0.001

    def test_perplexity_empty(self):
        """测试空列表的困惑度"""
        ppl = LMMETRICS.perplexity([])
        assert ppl == float("inf")

    def test_bits_per_character(self):
        """测试BPC计算"""
        log_probs = [-0.69, -0.69, -0.69]
        bpc = LMMETRICS.bits_per_character(log_probs, vocab_size=2)
        
        assert bpc > 0
        assert bpc < 2

    def test_token_accuracy(self):
        """测试token准确率"""
        if not HAS_TORCH:
            pytest.skip("PyTorch not available")
        
        predictions = torch.tensor([1, 2, 3, 4])
        targets = torch.tensor([1, 2, 3, 5])
        
        acc = LMMETRICS.token_accuracy(predictions, targets)
        assert abs(acc - 0.75) < 0.001

    def test_entropy(self):
        """测试熵计算"""
        probs = [0.5, 0.5]
        entropy = LMMETRICS.entropy(probs)
        
        expected = 1.0
        assert abs(entropy - expected) < 0.001


class TestLMEvaluator:
    """测试LMEvaluator类"""

    def test_creation(self):
        """测试创建"""
        evaluator = LMEvaluator(
            name="test_lm_eval",
            compute_perplexity=True,
            compute_bpc=True,
        )
        
        assert evaluator.name == "test_lm_eval"
        assert evaluator.compute_perplexity
        assert evaluator.compute_bpc

    def test_evaluate_single_sequence(self):
        """测试单序列评估"""
        if not HAS_TORCH:
            pytest.skip("PyTorch not available")
        
        evaluator = LMEvaluator(name="test")
        
        class SimpleModel:
            def __init__(self):
                self.vocab_size = 10
                self.embed = nn.Embedding(10, 8)
                self.lm_head = nn.Linear(8, 10)
            
            def __call__(self, x):
                emb = self.embed(x)
                logits = self.lm_head(emb)
                return type('obj', (object,), {'logits': logits})()
            
            def eval(self):
                pass
        
        model = SimpleModel()
        tokens = [1, 2, 3, 4, 5]
        
        metrics = evaluator.evaluate_single_sequence(model, tokens)
        
        assert "perplexity" in metrics
        assert "bpc" in metrics


class TestOPENENDEDMETRICS:
    """测试OPENENDEDMETRICS类"""

    def test_n_gram_diversity(self):
        """测试N-gram多样性"""
        generations = [
            "the cat sat on the mat",
            "a dog ran in the park",
            "birds fly in the sky",
        ]
        
        diversity = OPENENDEDMETRICS.n_gram_diversity(generations, n=2)
        
        assert diversity > 0
        assert diversity <= 1

    def test_unique_ratio(self):
        """测试唯一性比率"""
        generations = ["a", "b", "a", "c"]
        
        ratio = OPENENDEDMETRICS.unique_ratio(generations)
        assert ratio == 0.75

    def test_repetition_rate(self):
        """测试重复率"""
        text = "the cat the cat the cat"
        
        rate = OPENENDEDMETRICS.repetition_rate(text, n=2)
        
        assert rate > 0

    def test_coherence_score(self):
        """测试连贯性分数"""
        text = "The weather is nice today. The sun is shining brightly."
        
        score = OPENENDEDMETRICS.coherence_score(text)
        
        assert 0 <= score <= 1

    def test_fluency_score(self):
        """测试流畅性分数"""
        text = "The quick brown fox jumps over the lazy dog."
        
        score = OPENENDEDMETRICS.fluency_score(text)
        
        assert 0 <= score <= 1

    def test_text_length_stats(self):
        """测试文本长度统计"""
        generations = [
            "short text",
            "medium length text here",
            "a",
        ]
        
        stats = OPENENDEDMETRICS.text_length_stats(generations)
        
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats


class TestEmergenceDetector:
    """测试EmergenceDetector类"""

    def test_creation(self):
        """测试创建"""
        detector = EmergenceDetector(threshold=0.3)
        
        assert detector.threshold == 0.3
        assert len(detector.history) == 0

    def test_record_capability(self):
        """测试记录能力"""
        detector = EmergenceDetector()
        
        detector.record_capability("reasoning", scale=1.0, performance=0.5)
        detector.record_capability("reasoning", scale=2.0, performance=0.7)
        
        assert "reasoning" in detector.history
        assert len(detector.history["reasoning"]) == 2

    def test_detect_emergence(self):
        """测试涌现检测"""
        detector = EmergenceDetector(threshold=0.1)
        
        for scale in [1.0, 2.0, 3.0]:
            performance = 0.1 + (scale * 0.3)
            detector.record_capability("math", scale, performance)
        
        result = detector.detect_emergence("math")
        
        if result:
            assert result.capability_name == "math"
            assert result.emergence_scale is not None

    def test_detect_all_emergences(self):
        """测试检测所有涌现"""
        detector = EmergenceDetector()
        
        detector.record_capability("ability1", 1.0, 0.5)
        detector.record_capability("ability1", 2.0, 0.9)
        detector.record_capability("ability2", 1.0, 0.3)
        
        results = detector.detect_all_emergences()
        
        assert isinstance(results, list)


class TestOpenEndedEvaluator:
    """测试OpenEndedEvaluator类"""

    def test_creation(self):
        """测试创建"""
        evaluator = OpenEndedEvaluator(name="test_open")
        
        assert evaluator.name == "test_open"
        assert evaluator.evaluator_type == EvaluatorType.OPEN_ENDED

    def test_register_metrics(self):
        """测试指标注册"""
        evaluator = OpenEndedEvaluator()
        
        assert "ngram_diversity" in evaluator.metrics
        assert "unique_ratio" in evaluator.metrics


class TestReferenceFreeEvaluator:
    """测试ReferenceFreeEvaluator类"""

    def test_creation(self):
        """测试创建"""
        evaluator = ReferenceFreeEvaluator(name="ref_free")
        
        assert evaluator.name == "ref_free"


class TestDIVERSITYMETRICS:
    """测试DIVERSITYMETRICS类"""

    def test_type_token_ratio(self):
        """测试TTR计算"""
        tokens = [1, 2, 3, 1, 2]
        
        ttr = DIVERSITYMETRICS.type_token_ratio(tokens)
        
        assert ttr == 0.6

    def test_type_token_ratio_empty(self):
        """测试空列表的TTR"""
        ttr = DIVERSITYMETRICS.type_token_ratio([])
        assert ttr == 0.0

    def test_hapax_legomena_ratio(self):
        """测试Hapax比率"""
        tokens = [1, 2, 3, 1]
        
        ratio = DIVERSITYMETRICS.hapax_legomena_ratio(tokens)
        
        assert ratio == 0.5

    def test_shannon_entropy(self):
        """测试香农熵"""
        tokens = [1, 1, 2, 2]
        
        entropy = DIVERSITYMETRICS.shannon_entropy(tokens)
        
        assert entropy > 0

    def test_simpson_diversity_index(self):
        """测试Simpson多样性指数"""
        tokens = [1, 1, 2, 3]
        
        index = DIVERSITYMETRICS.simpson_diversity_index(tokens)
        
        assert 0 <= index <= 1

    def test_ngram_diversity(self):
        """测试N-gram多样性"""
        tokens = [1, 2, 3, 4, 5]
        
        result = DIVERSITYMETRICS.ngram_diversity(tokens, n=2)
        
        assert "unique_ratio" in result
        assert "entropy" in result
        assert result["unique_ratio"] == 1.0

    def test_sequence_level_diversity(self):
        """测试序列级多样性"""
        sequences = [
            [1, 2, 3],
            [4, 5, 6],
            [1, 2, 3],
        ]
        
        result = DIVERSITYMETRICS.sequence_level_diversity(sequences)
        
        assert "pairwise_similarity" in result
        assert "unique_sequences" in result
        assert result["unique_sequences"] == 2/3

    def test_repetition_free_ratio(self):
        """测试无重复比例"""
        text = "a b c d e"
        
        ratio = DIVERSITYMETRICS.repetition_free_ratio(text, n=2)
        
        assert ratio == 1.0

    def test_positional_diversity(self):
        """测试位置多样性"""
        tokens = list(range(100))
        
        diversity = DIVERSITYMETRICS.positional_diversity(tokens, num_positions=5)
        
        assert 0 <= diversity <= 1


class TestDiversityEvaluator:
    """测试DiversityEvaluator类"""

    def test_creation(self):
        """测试创建"""
        evaluator = DiversityEvaluator(
            name="test_diversity",
            compute_ttr=True,
            compute_entropy=True,
        )
        
        assert evaluator.name == "test_diversity"
        assert evaluator.compute_ttr
        assert evaluator.compute_entropy

    def test_metrics_registration(self):
        """测试指标注册"""
        evaluator = DiversityEvaluator()
        
        assert "type_token_ratio" in evaluator.metrics
        assert "shannon_entropy" in evaluator.metrics
        assert "simpson_diversity" in evaluator.metrics


class TestIntegration:
    """集成测试"""

    def test_full_evaluation_workflow(self):
        """测试完整评估流程"""
        report = EvaluationReport(evaluator_name="integration_test")
        
        results = [
            EvaluationResult(metric_name="perplexity", value=25.0),
            EvaluationResult(metric_name="accuracy", value=0.85),
            EvaluationResult(metric_name="diversity", value=0.72),
        ]
        
        report.add_results(results)
        report.compute_summary()
        
        assert len(report.results) == 3
        assert "mean" in report.summary

    def test_evaluator_composition(self):
        """测试评估器组合"""
        registry = EvaluatorRegistry()
        
        lm_eval = LMEvaluator(name="lm")
        diversity_eval = DiversityEvaluator(name="diversity")
        
        registry.register("lm", lm_eval)
        registry.register("diversity", diversity_eval)
        
        evaluators = registry.list_evaluators()
        
        assert "lm" in evaluators
        assert "diversity" in evaluators

    def test_custom_metric_integration(self):
        """测试自定义指标集成"""
        def domain_specific_metric(model, dataset):
            return 0.95
        
        evaluator = CustomEvaluator(
            name="domain_eval",
            custom_metrics={"domain_score": domain_specific_metric},
        )
        
        results = evaluator._evaluate(None, None)
        
        assert len(results) == 1
        assert results[0].value == 0.95


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
