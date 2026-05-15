"""Open-ended generation evaluation for the Lingmao Moyun framework.

This module provides evaluation capabilities for open-ended generation tasks
where there is no single correct answer. It includes metrics for assessing
generation quality, coherence, and the detection of emergent capabilities.
"""

from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re
import json
from collections import Counter

try:
    import torch
    import torch.nn.functional as F
except Exception:
    torch = None

from src.evaluation.base_evaluator import (
    BaseEvaluator,
    EvaluationResult,
    EvaluatorType,
)


class EmergenceMetric(Enum):
    """Metrics for detecting emergent capabilities."""
    SUDDEN_IMPROVEMENT = "sudden_improvement"
    PHASE_TRANSITION = "phase_transition"
    CAPABILITY_THRESHOLD = "capability_threshold"
    NOVEL_BEHAVIOR = "novel_behavior"


@dataclass
class EmergenceDetectionResult:
    """Result of emergent capability detection.
    
    Attributes:
        capability_name: Name of the detected capability
        emergence_scale: Scale at which emergence was detected
        confidence: Confidence level of the detection
        evidence: Supporting evidence for the emergence claim
        transition_type: Type of transition observed
    """
    capability_name: str
    emergence_scale: Optional[float] = None
    confidence: float = 0.0
    evidence: List[str] = field(default_factory=list)
    transition_type: str = "unknown"


@dataclass
class GenerationMetrics:
    """Metrics for open-ended generation evaluation.
    
    Attributes:
        coherence: Text coherence score
        diversity: Generation diversity score
        fluency: Language fluency score
        relevance: Relevance to prompt score
        creativity: Creative generation score
    """
    coherence: float = 0.0
    diversity: float = 0.0
    fluency: float = 0.0
    relevance: float = 0.0
    creativity: float = 0.0


class OPENENDEDMETRICS:
    """Collection of open-ended generation metrics.
    
    This class provides static methods for computing various
    open-ended generation evaluation metrics without reference answers.
    """
    
    @staticmethod
    def n_gram_diversity(generations: List[str], n: int = 2) -> float:
        """Calculate n-gram diversity of generations.
        
        Args:
            generations: List of generated texts
            n: Size of n-grams
            
        Returns:
            N-gram diversity score (0-1)
        """
        if not generations:
            return 0.0
        
        total_ngrams = 0
        unique_ngrams = set()
        
        for text in generations:
            tokens = text.split()
            for i in range(len(tokens) - n + 1):
                ngram = tuple(tokens[i:i+n])
                unique_ngrams.add(ngram)
                total_ngrams += 1
        
        if total_ngrams == 0:
            return 0.0
        
        return len(unique_ngrams) / total_ngrams
    
    @staticmethod
    def unique_ratio(generations: List[str]) -> float:
        """Calculate ratio of unique generations.
        
        Args:
            generations: List of generated texts
            
        Returns:
            Ratio of unique texts (0-1)
        """
        if not generations:
            return 0.0
        
        unique_count = len(set(generations))
        return unique_count / len(generations)
    
    @staticmethod
    def repetition_rate(text: str, n: int = 4) -> float:
        """Calculate rate of n-gram repetition.
        
        Args:
            text: Input text
            n: Size of n-grams
            
        Returns:
            Repetition rate (0-1, lower is better)
        """
        if not text or len(text) < n:
            return 0.0
        
        tokens = text.split()
        if len(tokens) < n:
            return 0.0
        
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            ngrams.append(ngram)
        
        if not ngrams:
            return 0.0
        
        unique_count = len(set(ngrams))
        return 1.0 - (unique_count / len(ngrams))
    
    @staticmethod
    def text_length_stats(generations: List[str]) -> Dict[str, float]:
        """Calculate statistics of generation lengths.
        
        Args:
            generations: List of generated texts
            
        Returns:
            Dictionary of length statistics
        """
        if not generations:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        
        lengths = [len(text.split()) for text in generations]
        
        mean_len = sum(lengths) / len(lengths)
        variance = sum((l - mean_len) ** 2 for l in lengths) / len(lengths)
        std_len = variance ** 0.5
        
        return {
            "mean": mean_len,
            "std": std_len,
            "min": min(lengths),
            "max": max(lengths),
        }
    
    @staticmethod
    def coherence_score(text: str) -> float:
        """Estimate text coherence without reference.
        
        Uses simple heuristics based on sentence structure
        and topic consistency.
        
        Args:
            text: Input text
            
        Returns:
            Coherence score (0-1)
        """
        if not text or len(text.strip()) == 0:
            return 0.0
        
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 2:
            return 1.0 if len(text.split()) > 3 else 0.0
        
        coherence = 0.5
        
        words = [w.lower() for w in text.split()]
        if len(words) > 1:
            unique_ratio = len(set(words)) / len(words)
            coherence = 0.3 + 0.4 * unique_ratio
        
        if len(sentences) > 2:
            coherence += 0.1
        
        return min(1.0, max(0.0, coherence))
    
    @staticmethod
    def fluency_score(text: str) -> float:
        """Estimate text fluency without reference.
        
        Uses heuristics based on word structure and sentence formation.
        
        Args:
            text: Input text
            
        Returns:
            Fluency score (0-1)
        """
        if not text:
            return 0.0
        
        score = 0.5
        
        words = text.split()
        if words:
            valid_words = sum(1 for w in words if len(w) > 0 and w.isalpha())
            if len(words) > 0:
                word_validity = valid_words / len(words)
                score = 0.3 + 0.5 * word_validity
        
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if sentences:
            avg_sentence_len = sum(len(s.split()) for s in sentences) / len(sentences)
            if 5 <= avg_sentence_len <= 50:
                score = min(1.0, score + 0.2)
        
        return min(1.0, max(0.0, score))
    
    @staticmethod
    def semantic_diversity(embeddings: List[List[float]]) -> float:
        """Calculate semantic diversity using embeddings.
        
        Args:
            embeddings: List of embedding vectors
            
        Returns:
            Semantic diversity score (0-1)
        """
        if not embeddings or len(embeddings) < 2:
            return 0.0
        
        import math
        
        total_distance = 0.0
        count = 0
        
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                dist = sum(
                    (a - b) ** 2 for a, b in zip(embeddings[i], embeddings[j])
                ) ** 0.5
                total_distance += dist
                count += 1
        
        if count == 0:
            return 0.0
        
        avg_distance = total_distance / count
        normalized = avg_distance / (math.sqrt(len(embeddings[0])) * 10)
        
        return min(1.0, normalized)


class EmergenceDetector:
    """Detector for emergent capabilities in language models.
    
    This class analyzes model behavior across different scales
    (parameters, training steps, etc.) to detect when new
    capabilities emerge.
    """
    
    def __init__(self, threshold: float = 0.2):
        """Initialize emergence detector.
        
        Args:
            threshold: Threshold for detecting significant changes
        """
        self.threshold = threshold
        self.history: Dict[str, List[Tuple[float, float]]] = {}
    
    def record_capability(
        self,
        capability: str,
        scale: float,
        performance: float,
    ) -> None:
        """Record a capability measurement.
        
        Args:
            capability: Name of the capability
            scale: Scale value (e.g., model size, training steps)
            performance: Performance metric value
        """
        if capability not in self.history:
            self.history[capability] = []
        
        self.history[capability].append((scale, performance))
    
    def detect_emergence(
        self,
        capability: str,
        window_size: int = 3,
    ) -> Optional[EmergenceDetectionResult]:
        """Detect emergence of a capability.
        
        Args:
            capability: Name of the capability to check
            window_size: Size of sliding window for analysis
            
        Returns:
            EmergenceDetectionResult if emergence detected, None otherwise
        """
        if capability not in self.history:
            return None
        
        history = self.history[capability]
        if len(history) < window_size + 1:
            return None
        
        improvements = []
        for i in range(1, len(history)):
            prev_scale, prev_perf = history[i - 1]
            curr_scale, curr_perf = history[i]
            
            if prev_scale > 0:
                improvement = (curr_perf - prev_perf) / prev_scale
                improvements.append((curr_scale, improvement))
        
        if not improvements:
            return None
        
        avg_improvement = sum(imp for _, imp in improvements) / len(improvements)
        
        if avg_improvement > self.threshold:
            latest_scale = history[-1][0]
            
            phase_changes = 0
            for i in range(1, len(history)):
                if (history[i][1] - history[i-1][1]) > self.threshold * 2:
                    phase_changes += 1
            
            return EmergenceDetectionResult(
                capability_name=capability,
                emergence_scale=latest_scale,
                confidence=min(1.0, avg_improvement),
                evidence=[
                    f"Average improvement rate: {avg_improvement:.4f}",
                    f"Phase transitions detected: {phase_changes}",
                ],
                transition_type="sudden" if phase_changes > 0 else "gradual",
            )
        
        return None
    
    def detect_all_emergences(self) -> List[EmergenceDetectionResult]:
        """Detect emergence across all recorded capabilities.
        
        Returns:
            List of emergence detection results
        """
        results = []
        for capability in self.history.keys():
            result = self.detect_emergence(capability)
            if result:
                results.append(result)
        return results
    
    def clear_history(self, capability: Optional[str] = None) -> None:
        """Clear recorded history.
        
        Args:
            capability: Specific capability to clear, or None for all
        """
        if capability is None:
            self.history.clear()
        elif capability in self.history:
            del self.history[capability]


class OpenEndedEvaluator(BaseEvaluator):
    """Evaluator for open-ended generation tasks.
    
    This evaluator assesses model performance on tasks without
    predetermined correct answers, such as creative writing,
    question answering, and instruction following.
    
    Attributes:
        metrics_functions: Dictionary of metric computation functions
        emergence_detector: Detector for emergent capabilities
    """
    
    def __init__(
        self,
        name: str = "open_ended_evaluator",
        use_custom_metrics: bool = True,
        emergence_threshold: float = 0.2,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize open-ended evaluator.
        
        Args:
            name: Name of the evaluator
            use_custom_metrics: Whether to use custom metrics
            emergence_threshold: Threshold for emergence detection
            config: Additional configuration
        """
        super().__init__(
            name=name,
            evaluator_type=EvaluatorType.OPEN_ENDED,
            config=config,
        )
        
        self.emergence_detector = EmergenceDetector(threshold=emergence_threshold)
        self._use_custom_metrics = use_custom_metrics
        
        self._register_open_ended_metrics()
    
    def _register_open_ended_metrics(self) -> None:
        """Register open-ended generation metrics."""
        self.register_metric("ngram_diversity", self._compute_ngram_diversity)
        self.register_metric("unique_ratio", self._compute_unique_ratio)
        self.register_metric("repetition_rate", self._compute_repetition_rate)
        self.register_metric("coherence", self._compute_coherence)
        self.register_metric("fluency", self._compute_fluency)
    
    def _compute_ngram_diversity(
        self,
        model: Any,
        dataset: Any,
        n: int = 2,
        **kwargs,
    ) -> float:
        """Compute n-gram diversity.
        
        Args:
            model: The model
            dataset: Dataset with generation prompts
            n: N-gram size
            **kwargs: Additional arguments
            
        Returns:
            N-gram diversity score
        """
        generations = self._generate_samples(model, dataset)
        return OPENENDEDMETRICS.n_gram_diversity(generations, n)
    
    def _compute_unique_ratio(
        self,
        model: Any,
        dataset: Any,
        **kwargs,
    ) -> float:
        """Compute ratio of unique generations.
        
        Args:
            model: The model
            dataset: Dataset with generation prompts
            **kwargs: Additional arguments
            
        Returns:
            Unique ratio score
        """
        generations = self._generate_samples(model, dataset)
        return OPENENDEDMETRICS.unique_ratio(generations)
    
    def _compute_repetition_rate(
        self,
        model: Any,
        dataset: Any,
        n: int = 4,
        **kwargs,
    ) -> float:
        """Compute n-gram repetition rate.
        
        Args:
            model: The model
            dataset: Dataset with generation prompts
            n: N-gram size
            **kwargs: Additional arguments
            
        Returns:
            Repetition rate
        """
        generations = self._generate_samples(model, dataset)
        
        total_repetition = 0.0
        for text in generations:
            total_repetition += OPENENDEDMETRICS.repetition_rate(text, n)
        
        return total_repetition / len(generations) if generations else 0.0
    
    def _compute_coherence(
        self,
        model: Any,
        dataset: Any,
        **kwargs,
    ) -> float:
        """Compute coherence score.
        
        Args:
            model: The model
            dataset: Dataset with generation prompts
            **kwargs: Additional arguments
            
        Returns:
            Coherence score
        """
        generations = self._generate_samples(model, dataset)
        
        total_coherence = 0.0
        for text in generations:
            total_coherence += OPENENDEDMETRICS.coherence_score(text)
        
        return total_coherence / len(generations) if generations else 0.0
    
    def _compute_fluency(
        self,
        model: Any,
        dataset: Any,
        **kwargs,
    ) -> float:
        """Compute fluency score.
        
        Args:
            model: The model
            dataset: Dataset with generation prompts
            **kwargs: Additional arguments
            
        Returns:
            Fluency score
        """
        generations = self._generate_samples(model, dataset)
        
        total_fluency = 0.0
        for text in generations:
            total_fluency += OPENENDEDMETRICS.fluency_score(text)
        
        return total_fluency / len(generations) if generations else 0.0
    
    def _generate_samples(
        self,
        model: Any,
        dataset: Any,
        max_length: int = 100,
        num_samples: int = 10,
    ) -> List[str]:
        """Generate samples for evaluation.
        
        Args:
            model: The model to generate with
            dataset: Dataset containing prompts
            max_length: Maximum generation length
            num_samples: Number of samples per prompt
            
        Returns:
            List of generated texts
        """
        if torch is None:
            return []
        
        generations = []
        
        model.eval()
        
        with torch.no_grad():
            for i, batch in enumerate(dataset):
                if i >= num_samples:
                    break
                
                if isinstance(batch, dict):
                    prompt = batch.get("input_ids")
                    if prompt is None:
                        continue
                elif isinstance(batch, torch.Tensor):
                    prompt = batch
                else:
                    continue
                
                if hasattr(model, "generate"):
                    generated = model.generate(
                        prompt,
                        max_length=max_length,
                        num_return_sequences=1,
                    )
                else:
                    generated = self._simple_generate(
                        model, prompt, max_length
                    )
                
                if isinstance(generated, torch.Tensor):
                    generated = generated.cpu().tolist()
                
                if isinstance(generated, list) and len(generated) > 0:
                    if isinstance(generated[0], list):
                        generated = generated[0]
                
                generations.append(str(generated))
        
        return generations
    
    def _simple_generate(
        self,
        model: Any,
        input_ids: torch.Tensor,
        max_length: int,
    ) -> torch.Tensor:
        """Simple greedy generation as fallback.
        
        Args:
            model: The model
            input_ids: Input tensor
            max_length: Maximum length
            
        Returns:
            Generated tensor
        """
        if torch is None:
            return input_ids
        
        input_ids = input_ids.to(next(model.parameters()).device)
        
        for _ in range(max_length):
            outputs = model(input_ids)
            
            if hasattr(outputs, "logits"):
                logits = outputs.logits
            else:
                logits = outputs
            
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
        
        return input_ids
    
    def _evaluate(
        self,
        model: Any,
        dataset: Any,
        **kwargs,
    ) -> List[EvaluationResult]:
        """Perform open-ended evaluation.
        
        Args:
            model: The model to evaluate
            dataset: The evaluation dataset
            **kwargs: Additional arguments
            
        Returns:
            List of evaluation results
        """
        results = []
        
        ngram_div = self.compute_metric("ngram_diversity", model, dataset)
        if ngram_div is not None:
            results.append(
                EvaluationResult(
                    metric_name="ngram_diversity",
                    value=ngram_div,
                    metadata={"higher_is_better": True, "n": 2},
                )
            )
        
        unique_ratio = self.compute_metric("unique_ratio", model, dataset)
        if unique_ratio is not None:
            results.append(
                EvaluationResult(
                    metric_name="unique_ratio",
                    value=unique_ratio,
                    metadata={"higher_is_better": True},
                )
            )
        
        repetition = self.compute_metric("repetition_rate", model, dataset)
        if repetition is not None:
            results.append(
                EvaluationResult(
                    metric_name="repetition_rate",
                    value=repetition,
                    metadata={"lower_is_better": True, "n": 4},
                )
            )
        
        coherence = self.compute_metric("coherence", model, dataset)
        if coherence is not None:
            results.append(
                EvaluationResult(
                    metric_name="coherence",
                    value=coherence,
                    metadata={"higher_is_better": True},
                )
            )
        
        fluency = self.compute_metric("fluency", model, dataset)
        if fluency is not None:
            results.append(
                EvaluationResult(
                    metric_name="fluency",
                    value=fluency,
                    metadata={"higher_is_better": True},
                )
            )
        
        return results
    
    def evaluate_capability_emergence(
        self,
        model: Any,
        scales: List[float],
        capability_name: str,
        evaluation_fn: Callable[[Any, float], float],
    ) -> EmergenceDetectionResult:
        """Evaluate and detect capability emergence across scales.
        
        Args:
            model: The model to evaluate
            scales: List of scale values to evaluate
            capability_name: Name of the capability
            evaluation_fn: Function to evaluate capability at each scale
            
        Returns:
            EmergenceDetectionResult if emergence detected
        """
        for scale in scales:
            performance = evaluation_fn(model, scale)
            self.emergence_detector.record_capability(
                capability_name, scale, performance
            )
        
        return self.emergence_detector.detect_emergence(capability_name)
    
    def add_custom_generation_metric(
        self,
        name: str,
        metric_fn: Callable[[List[str]], float],
    ) -> None:
        """Add a custom metric for generation evaluation.
        
        Args:
            name: Name of the metric
            metric_fn: Function that takes list of generations and returns score
        """
        def wrapped_metric(model: Any, dataset: Any, **kwargs) -> float:
            generations = self._generate_samples(model, dataset)
            return metric_fn(generations)
        
        self.register_metric(name, wrapped_metric)


class ReferenceFreeEvaluator(OpenEndedEvaluator):
    """Reference-free evaluator for open-ended generation.
    
    This evaluator focuses on assessing generation quality without
    requiring reference answers, making it suitable for evaluating
    models on tasks like creative writing or open-domain QA.
    """
    
    def __init__(
        self,
        name: str = "reference_free_evaluator",
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize reference-free evaluator.
        
        Args:
            name: Name of the evaluator
            config: Additional configuration
        """
        super().__init__(name=name, config=config)
        
        self._register_reference_free_metrics()
    
    def _register_reference_free_metrics(self) -> None:
        """Register reference-free assessment metrics."""
        self.register_metric("length_stats", self._compute_length_stats)
        self.register_metric("semantic_coverage", self._compute_semantic_coverage)
    
    def _compute_length_stats(
        self,
        model: Any,
        dataset: Any,
        **kwargs,
    ) -> float:
        """Compute normalized length statistics score.
        
        Args:
            model: The model
            dataset: Dataset with prompts
            **kwargs: Additional arguments
            
        Returns:
            Length statistics score
        """
        generations = self._generate_samples(model, dataset)
        stats = OPENENDEDMETRICS.text_length_stats(generations)
        
        cv = stats["std"] / stats["mean"] if stats["mean"] > 0 else 0
        normalized = min(1.0, cv)
        
        return normalized
    
    def _compute_semantic_coverage(
        self,
        model: Any,
        dataset: Any,
        **kwargs,
    ) -> float:
        """Compute semantic coverage score.
        
        Args:
            model: The model
            dataset: Dataset with prompts
            **kwargs: Additional arguments
            
        Returns:
            Semantic coverage score
        """
        generations = self._generate_samples(model, dataset)
        
        if not generations:
            return 0.0
        
        all_words = []
        for text in generations:
            all_words.extend(text.split())
        
        word_freq = Counter(all_words)
        unique_words = len(word_freq)
        total_words = len(all_words)
        
        if total_words == 0:
            return 0.0
        
        return unique_words / total_words
    
    def _evaluate(
        self,
        model: Any,
        dataset: Any,
        **kwargs,
    ) -> List[EvaluationResult]:
        """Perform reference-free evaluation.
        
        Args:
            model: The model to evaluate
            dataset: The evaluation dataset
            **kwargs: Additional arguments
            
        Returns:
            List of evaluation results
        """
        results = []
        
        length_stats = self.compute_metric("length_stats", model, dataset)
        if length_stats is not None:
            results.append(
                EvaluationResult(
                    metric_name="length_variation",
                    value=length_stats,
                    metadata={"higher_variation_is_better": True},
                )
            )
        
        semantic_coverage = self.compute_metric("semantic_coverage", model, dataset)
        if semantic_coverage is not None:
            results.append(
                EvaluationResult(
                    metric_name="semantic_coverage",
                    value=semantic_coverage,
                    metadata={"higher_is_better": True},
                )
            )
        
        return results
