"""Base evaluator framework for the Lingmao Moyun language model.

This module provides the foundational infrastructure for evaluating language models,
including standard evaluation interfaces, metric tracking, and custom evaluator support.
"""

from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import time
import json
from pathlib import Path


class EvaluatorType(Enum):
    """Types of evaluators supported by the framework."""
    STANDARD = "standard"
    OPEN_ENDED = "open_ended"
    DIVERSITY = "diversity"
    CUSTOM = "custom"


class EvaluationStage(Enum):
    """Stages of the evaluation process."""
    INITIALIZATION = "initialization"
    PREPARATION = "preparation"
    EVALUATION = "evaluation"
    AGGREGATION = "aggregation"
    FINALIZATION = "finalization"


@dataclass
class EvaluationResult:
    """Container for evaluation results.
    
    Attributes:
        metric_name: Name of the metric being evaluated
        value: The computed metric value
        confidence: Optional confidence interval or standard deviation
        metadata: Additional information about the evaluation
        timestamp: When the evaluation was performed
    """
    metric_name: str
    value: float
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        return {
            "metric_name": self.metric_name,
            "value": self.value,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }
    
    def __str__(self) -> str:
        """String representation of the evaluation result."""
        base_str = f"{self.metric_name}: {self.value:.6f}"
        if self.confidence is not None:
            base_str += f" (±{self.confidence:.6f})"
        return base_str


@dataclass
class EvaluationReport:
    """Complete evaluation report containing multiple results.
    
    Attributes:
        evaluator_name: Name of the evaluator that generated this report
        results: List of individual evaluation results
        summary: Aggregated summary statistics
        config: Configuration used for evaluation
        duration: Total evaluation time in seconds
    """
    evaluator_name: str
    results: List[EvaluationResult] = field(default_factory=list)
    summary: Dict[str, float] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    duration: float = 0.0
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    
    def add_result(self, result: EvaluationResult) -> None:
        """Add a single evaluation result to the report."""
        self.results.append(result)
    
    def add_results(self, results: List[EvaluationResult]) -> None:
        """Add multiple evaluation results to the report."""
        self.results.extend(results)
    
    def compute_summary(self) -> Dict[str, float]:
        """Compute summary statistics from all results."""
        if not self.results:
            return {}
        
        values = [r.value for r in self.results]
        self.summary = {
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "count": len(values),
        }
        
        if self.duration > 0:
            self.summary["duration"] = self.duration
        if self.end_time is not None:
            self.summary["end_time"] = self.end_time
        
        return self.summary
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary format."""
        return {
            "evaluator_name": self.evaluator_name,
            "results": [r.to_dict() for r in self.results],
            "summary": self.summary,
            "config": self.config,
            "duration": self.duration,
            "start_time": self.start_time,
            "end_time": self.end_time,
        }
    
    def save(self, path: Union[str, Path]) -> None:
        """Save the evaluation report to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    def __str__(self) -> str:
        """String representation of the evaluation report."""
        lines = [f"Evaluation Report: {self.evaluator_name}"]
        lines.append(f"Duration: {self.duration:.2f}s")
        lines.append(f"Number of metrics: {len(self.results)}")
        
        if self.summary:
            lines.append("Summary:")
            for key, value in self.summary.items():
                if isinstance(value, float):
                    lines.append(f"  {key}: {value:.6f}")
                else:
                    lines.append(f"  {key}: {value}")
        
        return "\n".join(lines)


class BaseEvaluator:
    """Abstract base class for all evaluators.
    
    This class defines the standard interface that all evaluators must implement.
    It provides common functionality for metric tracking, result aggregation,
    and evaluation lifecycle management.
    
    Attributes:
        name: Unique identifier for this evaluator
        evaluator_type: Type of evaluation this evaluator performs
        metrics: Dictionary of registered metrics
        results: List of evaluation results
    """
    
    def __init__(
        self,
        name: str,
        evaluator_type: EvaluatorType = EvaluatorType.STANDARD,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the base evaluator.
        
        Args:
            name: Unique identifier for this evaluator
            evaluator_type: Type of evaluation this evaluator performs
            config: Optional configuration dictionary
        """
        self.name = name
        self.evaluator_type = evaluator_type
        self.config = config or {}
        self.metrics: Dict[str, Callable] = {}
        self.results: List[EvaluationResult] = []
        self._stage = EvaluationStage.INITIALIZATION
        
    def register_metric(
        self,
        name: str,
        metric_fn: Callable[[Any], float],
    ) -> None:
        """Register a new metric with the evaluator.
        
        Args:
            name: Name of the metric
            metric_fn: Function that computes the metric value
        """
        self.metrics[name] = metric_fn
    
    def evaluate(
        self,
        model: Any,
        dataset: Any,
        **kwargs,
    ) -> EvaluationReport:
        """Execute the evaluation process.
        
        This method orchestrates the complete evaluation workflow:
        1. Preparation stage
        2. Evaluation stage (performs actual metric computation)
        3. Aggregation stage
        4. Finalization stage
        
        Args:
            model: The model to evaluate
            dataset: The dataset to evaluate on
            **kwargs: Additional arguments for specific evaluators
            
        Returns:
            EvaluationReport containing all results and summary statistics
        """
        report = EvaluationReport(
            evaluator_name=self.name,
            config=self.config,
        )
        
        self._stage = EvaluationStage.PREPARATION
        self._prepare(model, dataset)
        
        self._stage = EvaluationStage.EVALUATION
        results = self._evaluate(model, dataset, **kwargs)
        
        self._stage = EvaluationStage.AGGREGATION
        report.add_results(results)
        report.compute_summary()
        
        self._stage = EvaluationStage.FINALIZATION
        report.end_time = time.time()
        report.duration = report.end_time - report.start_time
        
        self._finalize(report)
        
        return report
    
    def _prepare(self, model: Any, dataset: Any) -> None:
        """Prepare for evaluation (hook method).
        
        Override this method to implement custom preparation logic,
        such as loading additional resources or preprocessing data.
        
        Args:
            model: The model being evaluated
            dataset: The dataset being used
        """
        pass
    
    def _evaluate(
        self,
        model: Any,
        dataset: Any,
        **kwargs,
    ) -> List[EvaluationResult]:
        """Perform the actual evaluation (must be implemented by subclasses).
        
        Args:
            model: The model to evaluate
            dataset: The dataset to evaluate on
            **kwargs: Additional arguments
            
        Returns:
            List of EvaluationResult objects
        """
        raise NotImplementedError("Subclasses must implement _evaluate method")
    
    def _finalize(self, report: EvaluationReport) -> None:
        """Finalize evaluation (hook method).
        
        Override this method to implement custom finalization logic,
        such as cleanup or additional result processing.
        
        Args:
            report: The evaluation report to finalize
        """
        pass
    
    def compute_metric(
        self,
        name: str,
        *args,
        **kwargs,
    ) -> Optional[float]:
        """Compute a registered metric.
        
        Args:
            name: Name of the metric to compute
            *args: Positional arguments for the metric function
            **kwargs: Keyword arguments for the metric function
            
        Returns:
            The computed metric value, or None if metric not found
        """
        if name not in self.metrics:
            return None
        return self.metrics[name](*args, **kwargs)
    
    @property
    def stage(self) -> EvaluationStage:
        """Get the current evaluation stage."""
        return self._stage


class CustomEvaluator(BaseEvaluator):
    """Evaluator for user-defined custom metrics.
    
    This class allows users to define their own evaluation metrics
    without modifying the core evaluator framework. It provides
    a flexible interface for implementing domain-specific evaluations.
    """
    
    def __init__(
        self,
        name: str,
        custom_metrics: Optional[Dict[str, Callable]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the custom evaluator.
        
        Args:
            name: Unique identifier for this evaluator
            custom_metrics: Dictionary of custom metric functions
            config: Optional configuration dictionary
        """
        super().__init__(
            name=name,
            evaluator_type=EvaluatorType.CUSTOM,
            config=config,
        )
        
        if custom_metrics:
            for metric_name, metric_fn in custom_metrics.items():
                self.register_metric(metric_name, metric_fn)
    
    def _evaluate(
        self,
        model: Any,
        dataset: Any,
        **kwargs,
    ) -> List[EvaluationResult]:
        """Evaluate using registered custom metrics.
        
        Args:
            model: The model to evaluate
            dataset: The dataset to evaluate on
            **kwargs: Additional arguments passed to metric functions
            
        Returns:
            List of EvaluationResult objects
        """
        results = []
        
        for metric_name, metric_fn in self.metrics.items():
            try:
                value = metric_fn(model, dataset, **kwargs)
                result = EvaluationResult(
                    metric_name=metric_name,
                    value=value,
                    metadata={"evaluator": self.name},
                )
                results.append(result)
            except Exception as e:
                result = EvaluationResult(
                    metric_name=metric_name,
                    value=float("nan"),
                    metadata={
                        "evaluator": self.name,
                        "error": str(e),
                    },
                )
                results.append(result)
        
        return results


class EvaluatorRegistry:
    """Registry for managing multiple evaluators.
    
    This class provides a centralized registry for managing different
    types of evaluators, allowing for easy registration, retrieval,
    and batch execution of evaluations.
    """
    
    _instance: Optional["EvaluatorRegistry"] = None
    
    def __new__(cls):
        """Singleton pattern implementation."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._evaluators = {}
            cls._instance._metrics = {}
        return cls._instance
    
    def register(
        self,
        name: str,
        evaluator: BaseEvaluator,
    ) -> None:
        """Register an evaluator with the registry.
        
        Args:
            name: Unique identifier for the evaluator
            evaluator: The evaluator instance to register
        """
        self._evaluators[name] = evaluator
    
    def get(self, name: str) -> Optional[BaseEvaluator]:
        """Retrieve a registered evaluator.
        
        Args:
            name: Name of the evaluator to retrieve
            
        Returns:
            The evaluator instance, or None if not found
        """
        return self._evaluators.get(name)
    
    def list_evaluators(self) -> List[str]:
        """List all registered evaluator names.
        
        Returns:
            List of evaluator names
        """
        return list(self._evaluators.keys())
    
    def evaluate_all(
        self,
        model: Any,
        dataset: Any,
        **kwargs,
    ) -> Dict[str, EvaluationReport]:
        """Run all registered evaluators.
        
        Args:
            model: The model to evaluate
            dataset: The dataset to evaluate on
            **kwargs: Additional arguments for evaluators
            
        Returns:
            Dictionary mapping evaluator names to their reports
        """
        reports = {}
        for name, evaluator in self._evaluators.items():
            try:
                report = evaluator.evaluate(model, dataset, **kwargs)
                reports[name] = report
            except Exception as e:
                report = EvaluationReport(evaluator_name=name)
                report.add_result(
                    EvaluationResult(
                        metric_name="evaluation_error",
                        value=float("nan"),
                        metadata={"error": str(e)},
                    )
                )
                reports[name] = report
        return reports
    
    def register_metric(
        self,
        name: str,
        metric_fn: Callable[[Any, Any], float],
    ) -> None:
        """Register a global metric.
        
        Args:
            name: Name of the metric
            metric_fn: Function that computes the metric value
        """
        self._metrics[name] = metric_fn
    
    def get_metric(self, name: str) -> Optional[Callable]:
        """Retrieve a global metric.
        
        Args:
            name: Name of the metric
            
        Returns:
            The metric function, or None if not found
        """
        return self._metrics.get(name)
