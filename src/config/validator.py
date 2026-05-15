"""Configuration validation utilities for Lingmao Moyun training system.

This module provides validation functions for configuration files and parameters,
supporting multiple training modes: pretraining, SFT, and RL.
"""

import os
import sys
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import logging

try:
    from pydantic import ValidationError
    from src.config.schema import (
        ExperimentConfig,
        PretrainConfig,
        SFTConfig,
        RLExperimentConfig,
        ModelConfig,
        TrainingConfig,
        DatasetConfig,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.config.schema import (
        ExperimentConfig,
        PretrainConfig,
        SFTConfig,
        RLExperimentConfig,
        ModelConfig,
        TrainingConfig,
        DatasetConfig,
    )

logger = logging.getLogger("LingmaoMoyun.Validator")


class ConfigValidationError(Exception):
    """Configuration validation error."""
    pass


class ConfigValidator:
    """Validator for configuration files and parameters."""

    def __init__(self, strict: bool = True):
        """Initialize validator.
        
        Args:
            strict: Whether to use strict validation mode.
        """
        self.strict = strict

    def validate_file(self, config_path: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Validate a configuration file.
        
        Args:
            config_path: Path to the YAML configuration file.
            
        Returns:
            Tuple of (is_valid, config_dict or None).
        """
        if not os.path.exists(config_path):
            logger.error(f"Configuration file not found: {config_path}")
            return False, None

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)

            if config_dict is None:
                logger.error("Configuration file is empty")
                return False, None

            return True, config_dict

        except yaml.YAMLError as e:
            logger.error(f"YAML parsing error: {e}")
            return False, None
        except Exception as e:
            logger.error(f"Error reading configuration file: {e}")
            return False, None

    def validate_with_schema(
        self, 
        config_dict: Dict[str, Any], 
        config_class: type = ExperimentConfig
    ) -> Tuple[bool, Optional[ExperimentConfig], List[str]]:
        """Validate configuration dictionary with Pydantic schema.
        
        Args:
            config_dict: Configuration dictionary.
            config_class: Configuration class to validate against.
            
        Returns:
            Tuple of (is_valid, validated_config, error_messages).
        """
        errors = []
        
        try:
            config = config_class(**config_dict)
            return True, config, errors
        except ValidationError as e:
            for error in e.errors():
                field = ".".join(str(loc) for loc in error["loc"])
                msg = error["msg"]
                errors.append(f"{field}: {msg}")
                logger.error(f"Validation error in {field}: {msg}")
            
            if self.strict:
                return False, None, errors
            else:
                try:
                    config = config_class(**config_dict, strict=False)
                    return True, config, errors
                except:
                    return False, None, errors
        
        except Exception as e:
            errors.append(f"Unexpected error: {str(e)}")
            logger.error(f"Unexpected validation error: {e}")
            return False, None, errors

    def validate_experiment_type(
        self, 
        config_dict: Dict[str, Any]
    ) -> Tuple[str, Optional[ExperimentConfig]]:
        """Detect and validate experiment type from configuration.
        
        Args:
            config_dict: Configuration dictionary.
            
        Returns:
            Tuple of (experiment_type, validated_config or None).
        """
        exp_type = config_dict.get("experiment_type", "pretrain").lower()
        
        type_mapping = {
            "pretrain": PretrainConfig,
            "pre-train": PretrainConfig,
            "sft": SFTConfig,
            "rl": RLExperimentConfig,
            "reinforcement": RLExperimentConfig,
        }
        
        config_class = type_mapping.get(exp_type, PretrainConfig)
        
        is_valid, config, errors = self.validate_with_schema(config_dict, config_class)
        
        if not is_valid:
            logger.warning(f"Configuration validation has errors: {errors}")
            if not self.strict:
                logger.warning("Proceeding in non-strict mode")
            else:
                return exp_type, None
        
        return exp_type, config

    def validate_model_params(self, model_config: ModelConfig) -> List[str]:
        """Validate model parameters compatibility.
        
        Args:
            model_config: Model configuration object.
            
        Returns:
            List of validation warnings.
        """
        warnings = []
        
        if model_config.num_kv_heads is not None:
            if model_config.num_kv_heads > model_config.nhead:
                warnings.append(
                    f"num_kv_heads ({model_config.num_kv_heads}) should not exceed "
                    f"nhead ({model_config.nhead}). Setting num_kv_heads to nhead."
                )
            elif model_config.num_kv_heads < model_config.nhead:
                logger.info(f"Using GQA with {model_config.num_kv_heads} KV heads")
        
        if model_config.use_mla and model_config.use_sliding_window:
            warnings.append(
                "Using both MLA and sliding window attention together. "
                "This may impact performance."
            )
        
        if model_config.use_moe:
            if model_config.num_experts < 2:
                warnings.append("MoE requires at least 2 experts")
            if model_config.top_k >= model_config.num_experts:
                warnings.append(
                    f"top_k ({model_config.top_k}) should be less than "
                    f"num_experts ({model_config.num_experts})"
                )
        
        if model_config.d_model % model_config.nhead != 0:
            warnings.append(
                f"d_model ({model_config.d_model}) should be divisible by "
                f"nhead ({model_config.nhead}) for optimal attention computation"
            )
        
        if model_config.d_model % model_config.head_dim != 0:
            warnings.append(
                f"d_model ({model_config.d_model}) should be divisible by "
                f"head_dim ({model_config.head_dim})"
            )
        
        return warnings

    def validate_training_params(
        self, 
        training_config: TrainingConfig,
        model_config: ModelConfig
    ) -> List[str]:
        """Validate training parameters.
        
        Args:
            training_config: Training configuration object.
            model_config: Model configuration object.
            
        Returns:
            List of validation warnings.
        """
        warnings = []
        
        if training_config.context_length > model_config.max_len:
            warnings.append(
                f"context_length ({training_config.context_length}) exceeds "
                f"model.max_len ({model_config.max_len})"
            )
        
        if training_config.batch_size * training_config.accumulation_steps < 8:
            warnings.append(
                "Small effective batch size (batch_size * accumulation_steps). "
                "Consider increasing for stable training."
            )
        
        if training_config.warmup_steps > training_config.total_training_steps * 0.2:
            warnings.append(
                f"warmup_steps ({training_config.warmup_steps}) is more than 20% of "
                f"total_training_steps ({training_config.total_training_steps})"
            )
        
        if training_config.min_lr_ratio < 0.05:
            warnings.append(
                f"Very low min_lr_ratio ({training_config.min_lr_ratio}). "
                "This may cause training instability."
            )
        
        return warnings

    def validate_dataset_paths(
        self, 
        dataset_config: DatasetConfig
    ) -> List[str]:
        """Validate dataset paths and accessibility.
        
        Args:
            dataset_config: Dataset configuration object.
            
        Returns:
            List of validation warnings.
        """
        warnings = []
        
        if dataset_config.train_file and not os.path.exists(dataset_config.train_file):
            warnings.append(f"Training file not found: {dataset_config.train_file}")
        
        if dataset_config.test_file and not os.path.exists(dataset_config.test_file):
            warnings.append(f"Test file not found: {dataset_config.test_file}")
        
        if dataset_config.val_file and not os.path.exists(dataset_config.val_file):
            warnings.append(f"Validation file not found: {dataset_config.val_file}")
        
        if not dataset_config.train_file:
            warnings.append("No training file specified")
        
        return warnings

    def validate_system_resources(
        self,
        training_config: TrainingConfig,
        model_config: ModelConfig,
        gpu_available: bool = False
    ) -> List[str]:
        """Validate system resource requirements.
        
        Args:
            training_config: Training configuration object.
            model_config: Model configuration object.
            gpu_available: Whether GPU is available.
            
        Returns:
            List of validation warnings.
        """
        warnings = []
        
        estimated_memory = self._estimate_model_memory(model_config)
        
        if gpu_available:
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory
                    if estimated_memory > gpu_memory * 0.9:
                        warnings.append(
                            f"Model size (~{estimated_memory / 1e9:.1f} GB) is very large "
                            "for available GPU memory. Consider enabling gradient checkpointing "
                            "or reducing model size."
                        )
            except ImportError:
                pass
        else:
            if estimated_memory > 10 * 1024**3:
                warnings.append(
                    f"Estimated model size (~{estimated_memory / 1e9:.1f} GB) may cause "
                    "CPU training to be very slow. Consider using a smaller model or "
                    "reducing batch size."
                )
        
        return warnings

    def _estimate_model_memory(self, model_config: ModelConfig) -> int:
        """Estimate model memory requirements in bytes.
        
        Args:
            model_config: Model configuration object.
            
        Returns:
            Estimated memory in bytes.
        """
        params = (
            model_config.vocab_size * model_config.d_model +  # Embedding
            model_config.d_model * model_config.d_model * 4 * model_config.num_layers +  # Transformer layers
            model_config.d_model * model_config.vocab_size  # Output projection
        )
        
        if model_config.use_moe:
            params *= (1 + (model_config.num_experts - 1) * 0.1)
        
        bytes_per_param = 2
        return int(params * bytes_per_param * 4)

    def validate_complete(
        self, 
        config_dict: Dict[str, Any],
        experiment_type: Optional[str] = None
    ) -> Tuple[bool, Optional[ExperimentConfig], Dict[str, List[str]]]:
        """Perform complete configuration validation.
        
        Args:
            config_dict: Configuration dictionary.
            experiment_type: Optional experiment type override.
            
        Returns:
            Tuple of (is_valid, validated_config, warnings_dict).
        """
        warnings = {
            "model": [],
            "training": [],
            "dataset": [],
            "system": [],
            "general": [],
        }
        
        if experiment_type is None:
            exp_type, config = self.validate_experiment_type(config_dict)
        else:
            type_mapping = {
                "pretrain": PretrainConfig,
                "sft": SFTConfig,
                "rl": RLExperimentConfig,
            }
            config_class = type_mapping.get(experiment_type.lower(), PretrainConfig)
            is_valid, config, errors = self.validate_with_schema(config_dict, config_class)
            
            if not is_valid:
                warnings["general"].extend(errors)
                return False, None, warnings
        
        if config is None:
            return False, None, warnings
        
        warnings["model"].extend(self.validate_model_params(config.model))
        warnings["training"].extend(
            self.validate_training_params(config.training, config.model)
        )
        warnings["dataset"].extend(self.validate_dataset_paths(config.dataset))
        warnings["system"].extend(
            self.validate_system_resources(
                config.training, 
                config.model,
                gpu_available=True
            )
        )
        
        has_warnings = any(w for w in warnings.values())
        
        return not has_warnings, config, warnings


def load_and_validate_config(
    config_path: str,
    experiment_type: Optional[str] = None,
    strict: bool = True
) -> Tuple[bool, Optional[ExperimentConfig], Dict[str, List[str]]]:
    """Load and validate a configuration file.
    
    Args:
        config_path: Path to the configuration file.
        experiment_type: Optional experiment type override.
        strict: Whether to use strict validation mode.
        
    Returns:
        Tuple of (is_valid, validated_config, warnings_dict).
    """
    validator = ConfigValidator(strict=strict)
    
    is_valid, config_dict = validator.validate_file(config_path)
    
    if not is_valid:
        return False, None, {"general": [f"Failed to load config file: {config_path}"]}
    
    return validator.validate_complete(config_dict, experiment_type)


def create_config_template(
    output_path: str,
    experiment_type: str = "pretrain"
) -> bool:
    """Create a configuration template file.
    
    Args:
        output_path: Path where to save the template.
        experiment_type: Type of experiment template to create.
        
    Returns:
        True if successful, False otherwise.
    """
    templates = {
        "pretrain": {
            "experiment_type": "pretrain",
            "experiment_name": "pretrain_baseline",
            "seed": 42,
            "model": {
                "d_model": 768,
                "nhead": 12,
                "num_layers": 12,
                "dim_feedforward": 3072,
                "dropout": 0.1,
                "max_len": 1024,
                "vocab_size": 50000,
                "mode": "modern",
            },
            "training": {
                "context_length": 512,
                "batch_size": 8,
                "learning_rate": 5e-5,
                "epochs": 10,
                "accumulation_steps": 4,
                "max_grad_norm": 1.0,
                "weight_decay": 0.01,
                "warmup_steps": 1000,
                "total_training_steps": 10000,
            },
            "dataset": {
                "stride": 256,
                "train_file": "dataset/train.txt",
                "test_file": "dataset/test.txt",
            },
            "tokenizer": {
                "path": "tokenizer.json",
                "vocab_size": 16000,
            },
            "paths": {
                "model_save_dir": "model_weights",
                "log_dir": "logs",
            },
        },
        "sft": {
            "experiment_type": "sft",
            "experiment_name": "sft_baseline",
            "seed": 42,
            "model": {
                "d_model": 512,
                "nhead": 8,
                "num_layers": 6,
                "dim_feedforward": 2048,
                "dropout": 0.05,
                "max_len": 512,
                "vocab_size": 50000,
                "mode": "modern",
            },
            "training": {
                "context_length": 256,
                "batch_size": 16,
                "learning_rate": 2e-5,
                "epochs": 3,
                "accumulation_steps": 2,
                "max_grad_norm": 1.0,
                "weight_decay": 0.01,
                "warmup_steps": 100,
                "total_training_steps": 1000,
                "label_smoothing": 0.05,
            },
            "dataset": {
                "stride": 256,
                "train_file": "dataset/sft_train.jsonl",
                "val_file": "dataset/sft_val.jsonl",
            },
            "tokenizer": {
                "path": "tokenizer.json",
                "vocab_size": 16000,
            },
            "paths": {
                "model_save_dir": "model_weights/sft",
                "log_dir": "logs/sft",
            },
        },
        "rl": {
            "experiment_type": "rl",
            "experiment_name": "rl_baseline",
            "seed": 42,
            "model": {
                "d_model": 512,
                "nhead": 8,
                "num_layers": 6,
                "dim_feedforward": 2048,
                "dropout": 0.0,
                "max_len": 512,
                "vocab_size": 50000,
                "mode": "modern",
            },
            "training": {
                "context_length": 256,
                "batch_size": 4,
                "learning_rate": 1e-5,
                "epochs": 10,
                "accumulation_steps": 8,
                "max_grad_norm": 1.0,
                "weight_decay": 0.01,
            },
            "dataset": {
                "train_file": "dataset/rl_train.jsonl",
            },
            "rl": {
                "reward_scale": 1.0,
                "kl_coef": 0.1,
                "gamma": 0.99,
                "lam": 0.95,
                "clip_range": 0.2,
                "entropy_coef": 0.01,
                "ppo_epochs": 4,
                "mini_batch_size": 4,
            },
            "tokenizer": {
                "path": "tokenizer.json",
                "vocab_size": 16000,
            },
            "paths": {
                "model_save_dir": "model_weights/rl",
                "log_dir": "logs/rl",
            },
        },
    }
    
    template = templates.get(experiment_type.lower())
    
    if template is None:
        logger.error(f"Unknown experiment type: {experiment_type}")
        return False
    
    try:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(template, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"Template saved to {output_path}")
        return True
    
    except Exception as e:
        logger.error(f"Failed to save template: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Configuration validation utility")
    parser.add_argument("config", nargs="?", help="Configuration file to validate")
    parser.add_argument("--type", "-t", choices=["pretrain", "sft", "rl"],
                        help="Experiment type")
    parser.add_argument("--create-template", "-c", choices=["pretrain", "sft", "rl"],
                        help="Create configuration template")
    parser.add_argument("--output", "-o", default="config_template.yaml",
                        help="Output path for template")
    parser.add_argument("--strict", action="store_true", default=True,
                        help="Use strict validation mode")
    
    args = parser.parse_args()
    
    if args.create_template:
        success = create_config_template(args.output, args.create_template)
        sys.exit(0 if success else 1)
    
    if args.config:
        is_valid, config, warnings = load_and_validate_config(
            args.config, args.type, args.strict
        )
        
        print(f"\nValidation result: {'PASSED' if is_valid else 'FAILED'}")
        
        if warnings:
            print("\nWarnings:")
            for category, msgs in warnings.items():
                if msgs:
                    print(f"  [{category.upper()}]")
                    for msg in msgs:
                        print(f"    - {msg}")
        
        if is_valid and config:
            print(f"\nExperiment type: {config.experiment_name}")
            print(f"Model: {config.model.d_model}d, {config.model.num_layers}L, {config.model.nhead}head")
            print(f"Training: {config.training.batch_size} batch, {config.training.epochs} epochs")
        
        sys.exit(0 if is_valid else 1)
    
    parser.print_help()
