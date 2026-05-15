"""Language model metrics for the Lingmao Moyun evaluation framework.

This module implements standard language model evaluation metrics including
perplexity, bits per character (BPC), cross-entropy, and other standard metrics
for measuring language model performance.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import math
from dataclasses import dataclass, field
from collections import Counter

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:
    torch = None
    F = None
    nn = None

from src.evaluation.base_evaluator import (
    BaseEvaluator,
    EvaluationResult,
    EvaluatorType,
)


@dataclass
class LMMetricResult:
    """Container for language model metric results.
    
    Attributes:
        perplexity: Perplexity score
        bits_per_character: Bits per character score
        cross_entropy: Cross-entropy loss
        token_accuracy: Token-level accuracy
        sequence_lengths: Distribution of sequence lengths
    """
    perplexity: float = 0.0
    bits_per_character: float = 0.0
    cross_entropy: float = 0.0
    token_accuracy: float = 0.0
    sequence_lengths: List[int] = field(default_factory=list)
    token_counts: Dict[str, int] = field(default_factory=dict)


class LMMETRICS:
    """Collection of language model metric functions.
    
    This class provides static methods for computing various
    language model evaluation metrics.
    """
    
    @staticmethod
    def perplexity(log_likelihoods: List[float]) -> float:
        """Calculate perplexity from log likelihoods.
        
        Perplexity measures how well the model predicts a sample.
        Lower perplexity indicates better performance.
        
        Args:
            log_likelihoods: List of log likelihood values
            
        Returns:
            Perplexity score
        """
        if not log_likelihoods:
            return float("inf")
        
        avg_log_likelihood = sum(log_likelihoods) / len(log_likelihoods)
        perplexity = math.exp(-avg_log_likelihood)
        return perplexity
    
    @staticmethod
    def bits_per_character(log_probs: List[float], vocab_size: int = 256) -> float:
        """Calculate bits per character (BPC).
        
        BPC measures the average number of bits needed to encode each character.
        Lower BPC indicates better compression and better model performance.
        
        Args:
            log_probs: List of log probabilities
            vocab_size: Size of the vocabulary
            
        Returns:
            BPC score
        """
        if not log_probs:
            return float("inf")
        
        avg_log_prob = sum(log_probs) / len(log_probs)
        log2_vocab = math.log2(vocab_size)
        bpc = -avg_log_prob / log2_vocab
        return bpc
    
    @staticmethod
    def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> float:
        """Calculate cross-entropy loss.
        
        Args:
            logits: Model predictions (logits)
            targets: Ground truth targets
            
        Returns:
            Cross-entropy loss value
        """
        if torch is None:
            return float("inf")
        
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits, targets)
        return loss.item()
    
    @staticmethod
    def token_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Calculate token-level accuracy.
        
        Args:
            predictions: Predicted token IDs
            targets: Ground truth token IDs
            
        Returns:
            Token accuracy as a float between 0 and 1
        """
        if torch is None:
            return 0.0
        
        correct = (predictions == targets).sum().item()
        total = targets.numel()
        return correct / total if total > 0 else 0.0
    
    @staticmethod
    def sequence_perplexity(
        token_ids: List[int],
        log_probs: List[float],
    ) -> float:
        """Calculate perplexity for a single sequence.
        
        Args:
            token_ids: List of token IDs
            log_probs: List of log probabilities for each token
            
        Returns:
            Per-sequence perplexity
        """
        if not log_probs or len(log_probs) == 0:
            return float("inf")
        
        n = len(log_probs)
        avg_log_prob = sum(log_probs) / n
        perplexity = math.exp(-avg_log_prob * n) ** (1.0 / n)
        return perplexity
    
    @staticmethod
    def entropy(probabilities: List[float]) -> float:
        """Calculate entropy of a probability distribution.
        
        Args:
            probabilities: List of probabilities
            
        Returns:
            Entropy value
        """
        entropy = 0.0
        for p in probabilities:
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy
    
    @staticmethod
    def conditional_entropy(
        joint_probs: Dict[Tuple[Any, Any], float],
        marginal_probs: Dict[Any, float],
    ) -> float:
        """Calculate conditional entropy H(Y|X).
        
        Args:
            joint_probs: Joint probability distribution P(X, Y)
            marginal_probs: Marginal probability distribution P(X)
            
        Returns:
            Conditional entropy value
        """
        cond_entropy = 0.0
        for (x, y), joint_prob in joint_probs.items():
            if joint_prob > 0 and marginal_probs.get(x, 0) > 0:
                cond_prob = joint_prob / marginal_probs[x]
                cond_entropy -= joint_prob * math.log2(cond_prob)
        return cond_entropy
    
    @staticmethod
    def mutual_information(
        joint_probs: Dict[Tuple[Any, Any], float],
        marginal_x: Dict[Any, float],
        marginal_y: Dict[Any, float],
    ) -> float:
        """Calculate mutual information I(X;Y).
        
        Args:
            joint_probs: Joint probability distribution P(X, Y)
            marginal_x: Marginal distribution P(X)
            marginal_y: Marginal distribution P(Y)
            
        Returns:
            Mutual information value
        """
        mi = 0.0
        for (x, y), joint_prob in joint_probs.items():
            if joint_prob > 0:
                expected = marginal_x.get(x, 0) * marginal_y.get(y, 0)
                if expected > 0:
                    mi += joint_prob * math.log2(joint_prob / expected)
        return mi


class LMEvaluator(BaseEvaluator):
    """Evaluator for standard language model metrics.
    
    This evaluator computes comprehensive language model metrics
    including perplexity, bits per character, cross-entropy, and accuracy.
    
    Attributes:
        compute_perplexity: Whether to compute perplexity
        compute_bpc: Whether to compute BPC
        compute_accuracy: Whether to compute token accuracy
        vocab_size: Size of the vocabulary for BPC calculation
    """
    
    def __init__(
        self,
        name: str = "lm_evaluator",
        compute_perplexity: bool = True,
        compute_bpc: bool = True,
        compute_accuracy: bool = True,
        vocab_size: int = 256,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the LM evaluator.
        
        Args:
            name: Name of the evaluator
            compute_perplexity: Whether to compute perplexity
            compute_bpc: Whether to compute BPC
            compute_accuracy: Whether to compute token accuracy
            vocab_size: Vocabulary size for BPC calculation
            config: Additional configuration
        """
        super().__init__(
            name=name,
            evaluator_type=EvaluatorType.STANDARD,
            config=config,
        )
        
        self.compute_perplexity = compute_perplexity
        self.compute_bpc = compute_bpc
        self.compute_accuracy = compute_accuracy
        self.vocab_size = vocab_size
        
        self._register_default_metrics()
    
    def _register_default_metrics(self) -> None:
        """Register default metric functions."""
        self.register_metric("perplexity", self._compute_perplexity)
        self.register_metric("bpc", self._compute_bpc)
        self.register_metric("accuracy", self._compute_accuracy)
    
    def _compute_perplexity(
        self,
        model: Any,
        dataset: Any,
        **kwargs,
    ) -> float:
        """Compute perplexity for the model on the dataset.
        
        Args:
            model: The language model
            dataset: The evaluation dataset
            **kwargs: Additional arguments
            
        Returns:
            Perplexity score
        """
        if torch is None:
            return float("inf")
        
        model.eval()
        total_log_likelihood = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in dataset:
                if isinstance(batch, dict):
                    input_ids = batch.get("input_ids")
                    if input_ids is None:
                        continue
                elif isinstance(batch, torch.Tensor):
                    input_ids = batch
                else:
                    continue
                
                outputs = model(input_ids)
                
                if hasattr(outputs, "logits"):
                    logits = outputs.logits
                else:
                    logits = outputs
                
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction="sum",
                )
                
                total_log_likelihood -= loss.item()
                total_tokens += shift_labels.numel()
        
        if total_tokens == 0:
            return float("inf")
        
        avg_log_likelihood = total_log_likelihood / total_tokens
        perplexity = math.exp(-avg_log_likelihood)
        
        return perplexity
    
    def _compute_bpc(
        self,
        model: Any,
        dataset: Any,
        **kwargs,
    ) -> float:
        """Compute bits per character for the model.
        
        Args:
            model: The language model
            dataset: The evaluation dataset
            **kwargs: Additional arguments
            
        Returns:
            BPC score
        """
        if torch is None:
            return float("inf")
        
        model.eval()
        total_log_prob = 0.0
        total_chars = 0
        
        with torch.no_grad():
            for batch in dataset:
                if isinstance(batch, dict):
                    input_ids = batch.get("input_ids")
                    if input_ids is None:
                        continue
                elif isinstance(batch, torch.Tensor):
                    input_ids = batch
                else:
                    continue
                
                outputs = model(input_ids)
                
                if hasattr(outputs, "logits"):
                    logits = outputs.logits
                else:
                    logits = outputs
                
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                
                log_probs = F.log_softmax(shift_logits, dim=-1)
                
                nll = F.nll_loss(
                    log_probs.view(-1, log_probs.size(-1)),
                    shift_labels.view(-1),
                    reduction="sum",
                )
                
                total_log_prob -= nll.item()
                total_chars += shift_labels.numel()
        
        if total_chars == 0:
            return float("inf")
        
        avg_log_prob = total_log_prob / total_chars
        log2_vocab = math.log2(self.vocab_size)
        bpc = -avg_log_prob / log2_vocab
        
        return bpc
    
    def _compute_accuracy(
        self,
        model: Any,
        dataset: Any,
        **kwargs,
    ) -> float:
        """Compute token-level accuracy.
        
        Args:
            model: The language model
            dataset: The evaluation dataset
            **kwargs: Additional arguments
            
        Returns:
            Token accuracy
        """
        if torch is None:
            return 0.0
        
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in dataset:
                if isinstance(batch, dict):
                    input_ids = batch.get("input_ids")
                    if input_ids is None:
                        continue
                elif isinstance(batch, torch.Tensor):
                    input_ids = batch
                else:
                    continue
                
                outputs = model(input_ids)
                
                if hasattr(outputs, "logits"):
                    logits = outputs.logits
                else:
                    logits = outputs
                
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                
                predictions = shift_logits.argmax(dim=-1)
                
                correct += (predictions == shift_labels).sum().item()
                total += shift_labels.numel()
        
        if total == 0:
            return 0.0
        
        return correct / total
    
    def _evaluate(
        self,
        model: Any,
        dataset: Any,
        **kwargs,
    ) -> List[EvaluationResult]:
        """Perform LM evaluation.
        
        Args:
            model: The language model
            dataset: The evaluation dataset
            **kwargs: Additional arguments
            
        Returns:
            List of evaluation results
        """
        results = []
        
        if self.compute_perplexity:
            ppl = self._compute_perplexity(model, dataset, **kwargs)
            results.append(
                EvaluationResult(
                    metric_name="perplexity",
                    value=ppl,
                    metadata={"lower_is_better": True},
                )
            )
        
        if self.compute_bpc:
            bpc = self._compute_bpc(model, dataset, **kwargs)
            results.append(
                EvaluationResult(
                    metric_name="bits_per_character",
                    value=bpc,
                    metadata={"lower_is_better": True, "vocab_size": self.vocab_size},
                )
            )
        
        if self.compute_accuracy:
            acc = self._compute_accuracy(model, dataset, **kwargs)
            results.append(
                EvaluationResult(
                    metric_name="token_accuracy",
                    value=acc,
                    metadata={"higher_is_better": True},
                )
            )
        
        return results
    
    def evaluate_single_sequence(
        self,
        model: Any,
        token_ids: List[int],
    ) -> Dict[str, float]:
        """Evaluate a single sequence.
        
        Args:
            model: The language model
            token_ids: List of token IDs
            
        Returns:
            Dictionary of metrics for the sequence
        """
        if torch is None:
            return {"perplexity": float("inf"), "bpc": float("inf"), "accuracy": 0.0}
        
        model.eval()
        
        if not isinstance(token_ids, torch.Tensor):
            input_ids = torch.tensor(token_ids).unsqueeze(0)
        else:
            input_ids = token_ids.unsqueeze(0) if token_ids.dim() == 1 else token_ids
        
        with torch.no_grad():
            outputs = model(input_ids)
            
            if hasattr(outputs, "logits"):
                logits = outputs.logits
            else:
                logits = outputs
            
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            
            log_probs = F.log_softmax(shift_logits, dim=-1)
            
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="mean",
            )
            
            predictions = shift_logits.argmax(dim=-1)
            accuracy = (predictions == shift_labels).float().mean().item()
        
        perplexity = math.exp(loss.item())
        log2_vocab = math.log2(self.vocab_size)
        bpc = loss.item() / log2_vocab
        
        return {
            "perplexity": perplexity,
            "bpc": bpc,
            "accuracy": accuracy,
            "loss": loss.item(),
        }


class StreamingLMEvaluator(LMEvaluator):
    """Streaming version of LM evaluator for large datasets.
    
    This evaluator processes data in a streaming fashion to handle
    datasets that don't fit in memory.
    
    Attributes:
        batch_size: Number of sequences to process at once
        accumulation_steps: Gradient accumulation steps
    """
    
    def __init__(
        self,
        name: str = "streaming_lm_evaluator",
        batch_size: int = 32,
        accumulation_steps: int = 1,
        **kwargs,
    ):
        """Initialize streaming LM evaluator.
        
        Args:
            name: Name of the evaluator
            batch_size: Batch size for processing
            accumulation_steps: Number of batches to accumulate
            **kwargs: Additional arguments for LMEvaluator
        """
        super().__init__(name=name, **kwargs)
        
        self.batch_size = batch_size
        self.accumulation_steps = accumulation_steps
        self._streaming_stats = {
            "total_tokens": 0,
            "total_correct": 0,
            "total_log_likelihood": 0.0,
        }
    
    def reset_stats(self) -> None:
        """Reset streaming statistics."""
        self._streaming_stats = {
            "total_tokens": 0,
            "total_correct": 0,
            "total_log_likelihood": 0.0,
        }
    
    def update_stats(
        self,
        tokens: int,
        correct: int,
        log_likelihood: float,
    ) -> None:
        """Update streaming statistics.
        
        Args:
            tokens: Number of tokens in the batch
            correct: Number of correct predictions
            log_likelihood: Log likelihood of the batch
        """
        self._streaming_stats["total_tokens"] += tokens
        self._streaming_stats["total_correct"] += correct
        self._streaming_stats["total_log_likelihood"] += log_likelihood
    
    def get_metrics(self) -> Dict[str, float]:
        """Get computed metrics from streaming statistics.
        
        Returns:
            Dictionary of metrics
        """
        stats = self._streaming_stats
        
        if stats["total_tokens"] == 0:
            return {
                "perplexity": float("inf"),
                "bpc": float("inf"),
                "accuracy": 0.0,
            }
        
        avg_log_likelihood = stats["total_log_likelihood"] / stats["total_tokens"]
        perplexity = math.exp(-avg_log_likelihood)
        
        log2_vocab = math.log2(self.vocab_size)
        bpc = -avg_log_likelihood / log2_vocab
        
        accuracy = stats["total_correct"] / stats["total_tokens"]
        
        return {
            "perplexity": perplexity,
            "bpc": bpc,
            "accuracy": accuracy,
        }
    
    def _evaluate(
        self,
        model: Any,
        dataset: Any,
        **kwargs,
    ) -> List[EvaluationResult]:
        """Perform streaming LM evaluation.
        
        Args:
            model: The language model
            dataset: The evaluation dataset
            **kwargs: Additional arguments
            
        Returns:
            List of evaluation results
        """
        self.reset_stats()
        
        if torch is None:
            return [
                EvaluationResult(metric_name="perplexity", value=float("inf")),
                EvaluationResult(metric_name="bits_per_character", value=float("inf")),
                EvaluationResult(metric_name="token_accuracy", value=0.0),
            ]
        
        model.eval()
        
        with torch.no_grad():
            for batch in dataset:
                if isinstance(batch, dict):
                    input_ids = batch.get("input_ids")
                    if input_ids is None:
                        continue
                elif isinstance(batch, torch.Tensor):
                    input_ids = batch
                else:
                    continue
                
                outputs = model(input_ids)
                
                if hasattr(outputs, "logits"):
                    logits = outputs.logits
                else:
                    logits = outputs
                
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                
                log_probs = F.log_softmax(shift_logits, dim=-1)
                
                nll = F.nll_loss(
                    log_probs.view(-1, log_probs.size(-1)),
                    shift_labels.view(-1),
                    reduction="sum",
                )
                
                predictions = shift_logits.argmax(dim=-1)
                correct = (predictions == shift_labels).sum().item()
                tokens = shift_labels.numel()
                
                self.update_stats(
                    tokens=tokens,
                    correct=correct,
                    log_likelihood=-nll.item(),
                )
        
        metrics = self.get_metrics()
        
        results = []
        if self.compute_perplexity:
            results.append(
                EvaluationResult(
                    metric_name="perplexity",
                    value=metrics["perplexity"],
                    metadata={"streaming": True},
                )
            )
        
        if self.compute_bpc:
            results.append(
                EvaluationResult(
                    metric_name="bits_per_character",
                    value=metrics["bpc"],
                    metadata={"streaming": True, "vocab_size": self.vocab_size},
                )
            )
        
        if self.compute_accuracy:
            results.append(
                EvaluationResult(
                    metric_name="token_accuracy",
                    value=metrics["accuracy"],
                    metadata={"streaming": True},
                )
            )
        
        return results
