"""Evaluation Example Script

This script demonstrates how to evaluate trained models using the
Lingmao Moyun evaluation framework.
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.model import SimpleTransformer
from src.dataset import LMDataset
from src.logger import get_logger
from src.config.schema import ModelConfig, ExperimentConfig

try:
    from torch.utils.data import DataLoader
except ImportError:
    DataLoader = None

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **k): return x

logger = get_logger("LingmaoMoyun.Evaluation")


class EvaluationMetrics:
    """Container for evaluation metrics."""
    
    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
    
    def add_metric(self, name: str, value: float):
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)
    
    def compute_summary(self) -> Dict[str, Any]:
        summary = {}
        for name, values in self.metrics.items():
            if values:
                summary[name] = {
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "count": len(values),
                }
        
        if self.start_time and self.end_time:
            summary["elapsed_time"] = self.end_time - self.start_time
        
        return summary
    
    def __str__(self) -> str:
        summary = self.compute_summary()
        lines = ["Evaluation Metrics Summary:", "-" * 40]
        
        for name, stats in summary.items():
            if isinstance(stats, dict) and "mean" in stats:
                lines.append(
                    f"  {name}: {stats['mean']:.4f} "
                    f"(min={stats['min']:.4f}, max={stats['max']:.4f}, n={stats['count']})"
                )
            else:
                lines.append(f"  {name}: {stats}")
        
        return "\n".join(lines)


def load_model_for_eval(
    model_path: str,
    config: Optional[ExperimentConfig] = None,
    device: Optional[torch.device] = None
) -> SimpleTransformer:
    """Load a model from checkpoint for evaluation.
    
    Args:
        model_path: Path to model checkpoint.
        config: Optional experiment configuration.
        device: Device to load model on.
    
    Returns:
        Loaded model instance.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if config is None:
        model_config = ModelConfig()
    else:
        model_config = config.model
    
    model_kwargs = config.to_model_kwargs() if config else {}
    
    model = SimpleTransformer(**model_kwargs)
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    logger.info(f"Model loaded from {model_path}")
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


def evaluate_model(
    model: SimpleTransformer,
    test_file: str,
    tokenizer_path: str = "tokenizer.json",
    context_length: int = 512,
    batch_size: int = 16,
    device: Optional[torch.device] = None,
    max_batches: Optional[int] = None,
) -> EvaluationMetrics:
    """Evaluate model on test dataset.
    
    Args:
        model: Model to evaluate.
        test_file: Path to test data file.
        tokenizer_path: Path to tokenizer.
        context_length: Context length for evaluation.
        batch_size: Evaluation batch size.
        device: Device to use.
        max_batches: Maximum number of batches to evaluate.
    
    Returns:
        Evaluation metrics.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    metrics = EvaluationMetrics()
    metrics.start_time = time.time()
    
    logger.info(f"Loading test data from {test_file}")
    
    tok = None
    if os.path.exists(tokenizer_path):
        try:
            from tokenizer import ClassicalTokenizer
            tok = ClassicalTokenizer()
            tok.load(tokenizer_path)
            logger.info(f"Tokenizer loaded: {len(tok.token_to_id)} tokens")
        except Exception as e:
            logger.warning(f"Could not load tokenizer: {e}")
    
    test_dataset = LMDataset(
        test_file,
        context_length=context_length,
        tokenizer=tok,
        stride=context_length,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(device.type != "cpu"),
    )
    
    logger.info(f"Evaluating {len(test_loader)} batches...")
    
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            if max_batches and batch_idx >= max_batches:
                break
            
            try:
                if isinstance(batch, dict):
                    input_ids = batch["input_ids"].to(device)
                    target_ids = batch["target_ids"].to(device)
                else:
                    input_ids, target_ids = batch
                    input_ids = input_ids.to(device)
                    target_ids = target_ids.to(device)
                
                outputs = model(input_ids)
                loss = criterion(
                    outputs.view(-1, outputs.size(-1)),
                    target_ids.view(-1)
                )
                
                metrics.add_metric("loss", loss.item())
                total_loss += loss.item()
                num_batches += 1
                
            except Exception as e:
                logger.warning(f"Error processing batch {batch_idx}: {e}")
                continue
    
    metrics.end_time = time.time()
    
    avg_loss = total_loss / max(num_batches, 1)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    metrics.add_metric("perplexity", perplexity)
    
    return metrics


def evaluate_generation(
    model: SimpleTransformer,
    prompts: List[str],
    tokenizer_path: str = "tokenizer.json",
    max_length: int = 100,
    temperature: float = 1.0,
    top_k: int = 50,
    device: Optional[torch.device] = None,
) -> List[Dict[str, Any]]:
    """Evaluate model generation capabilities.
    
    Args:
        model: Model to evaluate.
        prompts: List of prompt texts.
        tokenizer_path: Path to tokenizer.
        max_length: Maximum generation length.
        temperature: Sampling temperature.
        top_k: Top-k sampling parameter.
        device: Device to use.
    
    Returns:
        List of generation results.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tok = None
    if os.path.exists(tokenizer_path):
        try:
            from tokenizer import ClassicalTokenizer
            tok = ClassicalTokenizer()
            tok.load(tokenizer_path)
        except Exception as e:
            logger.warning(f"Could not load tokenizer: {e}")
    
    results = []
    
    for i, prompt in enumerate(tqdm(prompts, desc="Generating")):
        try:
            if tok:
                input_ids = torch.tensor([tok.encode(prompt)]).to(device)
            else:
                input_ids = torch.tensor([[ord(c) for c in prompt]]).to(device)
            
            with torch.no_grad():
                generated = model.generate(
                    input_ids,
                    max_length=max_length,
                    temperature=temperature,
                    top_k=top_k,
                )
            
            if tok:
                output_text = tok.decode(generated[0].tolist())
            else:
                output_text = "".join(chr(c) for c in generated[0].tolist())
            
            results.append({
                "prompt": prompt,
                "generated": output_text,
                "length": len(generated[0]),
            })
            
        except Exception as e:
            logger.warning(f"Error generating for prompt {i}: {e}")
            results.append({
                "prompt": prompt,
                "generated": "",
                "error": str(e),
            })
    
    return results


def save_evaluation_report(
    metrics: EvaluationMetrics,
    output_path: str,
    generation_results: Optional[List[Dict]] = None,
):
    """Save evaluation report to file.
    
    Args:
        metrics: Evaluation metrics.
        output_path: Path to save report.
        generation_results: Optional generation results to include.
    """
    report = {
        "metrics": metrics.compute_summary(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    if generation_results:
        report["generation_samples"] = generation_results[:10]
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Evaluation report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run model evaluation")
    parser.add_argument(
        "--model", "-m",
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--config", "-c",
        default=None,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--test-file", "-t",
        default="dataset/test.txt",
        help="Path to test data file"
    )
    parser.add_argument(
        "--tokenizer", "-tok",
        default="tokenizer.json",
        help="Path to tokenizer"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=16,
        help="Evaluation batch size"
    )
    parser.add_argument(
        "--context-length", "-l",
        type=int,
        default=512,
        help="Context length"
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Maximum batches to evaluate"
    )
    parser.add_argument(
        "--output", "-o",
        default="eval_results.json",
        help="Output report path"
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Run generation evaluation"
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=["床前明月光，疑是地上霜。", "春风得意马蹄疾，"],
        help="Prompts for generation evaluation"
    )
    parser.add_argument(
        "--max-gen-length",
        type=int,
        default=100,
        help="Maximum generation length"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Generation temperature"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device (auto/cpu/cuda)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Lingmao Moyun Model Evaluation")
    print("=" * 60)
    
    config = None
    if args.config:
        from src.config.validator import load_and_validate_config
        _, config, _ = load_and_validate_config(args.config)
    
    device = torch.device("cpu")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device != "auto":
        device = torch.device(args.device)
    
    print(f"\nLoading model from {args.model}...")
    model = load_model_for_eval(args.model, config, device)
    
    metrics = evaluate_model(
        model,
        args.test_file,
        tokenizer_path=args.tokenizer,
        context_length=args.context_length,
        batch_size=args.batch_size,
        device=device,
        max_batches=args.max_batches,
    )
    
    print(f"\n{metrics}")
    
    generation_results = None
    if args.generate:
        print("\nRunning generation evaluation...")
        generation_results = evaluate_generation(
            model,
            args.prompts,
            tokenizer_path=args.tokenizer,
            max_length=args.max_gen_length,
            temperature=args.temperature,
            device=device,
        )
        
        print("\nGeneration samples:")
        for i, result in enumerate(generation_results):
            print(f"\n[{i+1}] Prompt: {result['prompt']}")
            print(f"    Generated: {result['generated'][:200]}...")
    
    save_evaluation_report(metrics, args.output, generation_results)
    
    print("\n" + "=" * 60)
    print("Evaluation completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
