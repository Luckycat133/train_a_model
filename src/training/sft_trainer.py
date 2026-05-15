"""Supervised Fine-Tuning (SFT) trainer for instruction-following models.

This module implements SFT training where the model learns to generate
appropriate responses given instruction-input pairs. SFT is typically performed
after pretraining to adapt the model for specific tasks or following instructions.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
except Exception:
    torch = None
    nn = None

try:
    from src.logger import get_logger
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    def get_logger(name):
        return logging.getLogger(name)

try:
    from .base_trainer import BaseTrainer, TrainingConfig
except ImportError:
    from src.training.base_trainer import BaseTrainer, TrainingConfig

logger = get_logger("LingmaoMoyun.SFT")


@dataclass
class SFTConfig(TrainingConfig):
    """Configuration for supervised fine-tuning.

    Attributes:
        response_loss_only: If True, only compute loss on response tokens,
            ignoring the instruction/prompt tokens.
        mask_prompt_tokens: If True, mask out loss on prompt tokens.
        max_prompt_length: Maximum length for prompt tokens.
        max_response_length: Maximum length for response tokens.
        truncate_response: Truncate responses exceeding max_response_length.
        train_file: Path to training data file.
        test_file: Optional path to test/validation data file.
        tokenizer_path: Path to tokenizer file.
        ignore_token_id: Token ID to ignore in loss computation (e.g., padding).
        sample_weights: Optional dict of sample weights for weighted loss.
    """

    response_loss_only: bool = True
    mask_prompt_tokens: bool = True
    max_prompt_length: int = 512
    max_response_length: int = 512
    truncate_response: bool = True
    train_file: Optional[str] = None
    test_file: Optional[str] = None
    tokenizer_path: str = "tokenizer.json"
    ignore_token_id: int = -100
    sample_weights: Optional[Dict[str, float]] = None
    response_template: str = "\n\n### Response:\n"
    prompt_template: str = "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n"


class InstructionResponseDataset(Dataset):
    """Dataset for instruction-response pairs used in SFT.

    Supports various data formats:
    - JSONL with 'instruction', 'input', 'output' fields
    - JSONL with 'prompt', 'completion' fields
    - JSONL with 'messages' field (chat format)
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: Any,
        max_length: int = 1024,
        max_prompt_length: int = 512,
        max_response_length: int = 512,
        truncate_response: bool = True,
        response_template: str = "\n\n### Response:\n",
        prompt_template: str = "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n",
        ignore_token_id: int = -100,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        self.truncate_response = truncate_response
        self.response_template = response_template
        self.prompt_template = prompt_template
        self.ignore_token_id = ignore_token_id

        self.samples = self._load_data(data_path)
        logger.info(f"Loaded {len(self.samples)} instruction-response samples")

    def _load_data(self, data_path: str) -> List[Dict[str, Any]]:
        """Load and parse the dataset from file."""
        samples = []

        with open(data_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(f"Skipping malformed JSON at line {line_num + 1}")
                    continue

                sample = self._parse_item(item)
                if sample:
                    samples.append(sample)

        return samples

    def _parse_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse a single data item into instruction-response format."""
        if "messages" in item:
            return self._parse_chat_format(item)
        elif "instruction" in item and "output" in item:
            return self._parse_instruction_format(item)
        elif "prompt" in item and "completion" in item:
            return self._parse_completion_format(item)
        else:
            logger.warning(f"Unknown data format: {list(item.keys())}")
            return None

    def _parse_instruction_format(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Parse instruction-input-output format."""
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        output = item.get("output", "")

        prompt = self.prompt_template.format(
            instruction=instruction,
            input=input_text if input_text else "N/A"
        )
        response = output

        return {
            "prompt": prompt,
            "response": response,
            "prompt_ids": None,
            "response_ids": None,
        }

    def _parse_completion_format(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Parse prompt-completion format."""
        prompt = item.get("prompt", "")
        completion = item.get("completion", "")

        return {
            "prompt": prompt,
            "response": completion,
            "prompt_ids": None,
            "response_ids": None,
        }

    def _parse_chat_format(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Parse chat/messages format."""
        messages = item.get("messages", [])
        prompt_parts = []
        response = ""

        for i, msg in enumerate(messages):
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "user":
                prompt_parts.append(f"### Human: {content}\n\n")
            elif role == "assistant":
                if not response:
                    response = content
                prompt_parts.append(f"### Assistant: {content}\n\n")

        prompt = "".join(prompt_parts)

        return {
            "prompt": prompt,
            "response": response,
            "prompt_ids": None,
            "response_ids": None,
        }

    def _tokenize_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Tokenize a single sample."""
        if sample.get("prompt_ids") is not None:
            return sample

        prompt_ids = self.tokenizer.encode(
            sample["prompt"],
            max_length=self.max_prompt_length,
            truncation=True,
        )

        response_ids = self.tokenizer.encode(
            sample["response"],
            max_length=self.max_response_length,
            truncation=self.truncate_response,
        )

        sample["prompt_ids"] = prompt_ids
        sample["response_ids"] = response_ids

        return sample

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self._tokenize_sample(self.samples[idx])

        prompt_ids = sample["prompt_ids"]
        response_ids = sample["response_ids"]

        input_ids = prompt_ids + response_ids
        labels = [self.ignore_token_id] * len(prompt_ids) + response_ids

        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor([1] * len(input_ids), dtype=torch.long),
            "prompt_length": len(prompt_ids),
        }


class SFTTrainer(BaseTrainer):
    """Trainer for supervised fine-tuning on instruction-response pairs.

    SFT trains the model to generate appropriate responses given instructions.
    The model learns via standard next-token prediction on curated data.

    Features:
    - Instruction-following training
    - Optional loss masking for prompt tokens (only train on response)
    - Packed sequences for memory efficiency
    - Mixed precision training
    - Gradient accumulation
    """

    def __init__(
        self,
        config: Optional[SFTConfig] = None,
        model: Optional[nn.Module] = None,
        train_loader: Optional[DataLoader] = None,
        eval_loader: Optional[DataLoader] = None,
        device: Optional[torch.device] = None,
        tokenizer: Optional[Any] = None,
    ):
        if config is None:
            config = SFTConfig()

        super().__init__(
            config=config,
            model=model,
            train_loader=train_loader,
            eval_loader=eval_loader,
            device=device,
        )

        self.tokenizer = tokenizer
        self._setup_label_ids()

    def _setup_label_ids(self) -> None:
        """Setup the ignore token ID for loss computation."""
        self.ignore_token_id = self.config.ignore_token_id

    def build_model(self) -> nn.Module:
        """Build the model for SFT.

        SFT typically uses the same model architecture as pretraining,
        but may use a smaller learning rate and fewer epochs.
        """
        from src.model import SimpleTransformer

        if self.model is not None:
            return self.model

        cfg = self.config
        vocab_size = cfg.vocab_size

        if os.path.exists(cfg.tokenizer_path):
            try:
                from tokenizer import ClassicalTokenizer
                self.tokenizer = ClassicalTokenizer()
                self.tokenizer.load(cfg.tokenizer_path)
                vocab_size = len(self.tokenizer.token_to_id)
            except Exception:
                pass

        model = SimpleTransformer(
            vocab_size=vocab_size,
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            num_layers=cfg.num_layers,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            max_len=cfg.max_len,
            mode=cfg.mode,
        )

        logger.info(f"SFT model built with {sum(p.numel() for p in model.parameters()):,} parameters")
        return model

    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor],
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute the SFT loss.

        Computes cross-entropy loss on response tokens only (prompt tokens masked).
        This is the standard approach for SFT where we only train on responses.

        Args:
            batch: Dictionary containing 'input_ids', 'labels', 'attention_mask'.

        Returns:
            Tuple of (loss tensor, metrics dictionary).
        """
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        attention_mask = batch.get("attention_mask")

        outputs, _ = self.model(input_ids, attention_mask=attention_mask)

        if self.criterion is None:
            self.criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_token_id)

        logits_flat = outputs.view(-1, outputs.size(-1))
        labels_flat = labels.view(-1)

        loss = self.criterion(logits_flat, labels_flat)

        with torch.no_grad():
            valid_mask = labels_flat != self.ignore_token_id
            if valid_mask.any():
                preds = logits_flat.argmax(dim=-1)
                accuracy = (preds[valid_mask] == labels_flat[valid_mask]).float().mean().item()
                num_tokens = valid_mask.sum().item()
            else:
                accuracy = 0.0
                num_tokens = 0

        metrics = {
            "loss": loss.item(),
            "accuracy": accuracy,
            "num_tokens": num_tokens,
        }

        return loss, metrics

    def _prepare_batch(self, batch: Union[Dict[str, torch.Tensor], tuple]) -> Dict[str, torch.Tensor]:
        """Prepare batch for SFT training."""
        if isinstance(batch, dict):
            result = {k: v.to(self.device) for k, v in batch.items()}
            return result
        elif isinstance(batch, (tuple, list)) and len(batch) >= 2:
            input_ids, labels = batch[0], batch[1]
            return {
                "input_ids": input_ids.to(self.device),
                "labels": labels.to(self.device),
            }
        else:
            raise ValueError(f"Unknown batch type: {type(batch)}")

    def _get_extra_checkpoint_state(self) -> Dict[str, Any]:
        """Return extra state for SFT checkpointing."""
        return {
            "trainer_type": "sft",
            "ignore_token_id": self.ignore_token_id,
        }

    def create_dataloaders(
        self,
        train_file: str,
        test_file: Optional[str] = None,
        batch_size: Optional[int] = None,
        num_workers: int = 4,
    ) -> Tuple[DataLoader, Optional[DataLoader]]:
        """Create SFT training and evaluation dataloaders.

        Args:
            train_file: Path to training JSONL file with instruction-response pairs.
            test_file: Optional path to test/validation data.
            batch_size: Batch size (defaults to config value).
            num_workers: Number of data loading workers.

        Returns:
            Tuple of (train_loader, eval_loader).
        """
        if batch_size is None:
            batch_size = self.config.batch_size

        if self.tokenizer is None:
            if os.path.exists(self.config.tokenizer_path):
                try:
                    from tokenizer import ClassicalTokenizer
                    self.tokenizer = ClassicalTokenizer()
                    self.tokenizer.load(self.config.tokenizer_path)
                except Exception as e:
                    logger.warning(f"Failed to load tokenizer: {e}")
                    raise RuntimeError("Tokenizer required for SFT")
            else:
                raise RuntimeError("Tokenizer required for SFT")

        train_dataset = InstructionResponseDataset(
            data_path=train_file,
            tokenizer=self.tokenizer,
            max_length=self.config.context_length,
            max_prompt_length=self.config.max_prompt_length,
            max_response_length=self.config.max_response_length,
            truncate_response=self.config.truncate_response,
            response_template=self.config.response_template,
            prompt_template=self.config.prompt_template,
            ignore_token_id=self.config.ignore_token_id,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(self.device.type == "cuda"),
            drop_last=True,
        )

        eval_loader = None
        if test_file and os.path.exists(test_file):
            eval_dataset = InstructionResponseDataset(
                data_path=test_file,
                tokenizer=self.tokenizer,
                max_length=self.config.context_length,
                max_prompt_length=self.config.max_prompt_length,
                max_response_length=self.config.max_response_length,
                truncate_response=self.config.truncate_response,
                response_template=self.config.response_template,
                prompt_template=self.config.prompt_template,
                ignore_token_id=self.config.ignore_token_id,
            )
            eval_loader = DataLoader(
                eval_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=(self.device.type == "cuda"),
            )
            logger.info(f"Created eval dataloader with {len(eval_dataset)} samples")

        logger.info(f"Created train dataloader with {len(train_dataset)} samples")
        return train_loader, eval_loader

    @classmethod
    def from_pretrained(
        cls,
        pretrained_path: str,
        config: Optional[SFTConfig] = None,
        device: Optional[torch.device] = None,
    ) -> "SFTTrainer":
        """Create an SFT trainer from a pretrained checkpoint.

        Args:
            pretrained_path: Path to pretrained model checkpoint.
            config: SFT configuration.
            device: Target device.

        Returns:
            SFTTrainer with loaded pretrained weights.
        """
        if config is None:
            config = SFTConfig()

        trainer = cls(config=config, device=device)

        model = trainer.build_model()

        checkpoint = torch.load(pretrained_path, map_location=device or torch.device("cpu"))
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        trainer.model = model
        logger.info(f"Loaded pretrained model from {pretrained_path}")

        return trainer

    def generate_response(
        self,
        instruction: str,
        input_text: Optional[str] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """Generate a response for a given instruction (for evaluation).

        Args:
            instruction: The instruction/prompt.
            input_text: Optional input context.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.

        Returns:
            Generated response string.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model and tokenizer must be initialized")

        self.model.eval()

        prompt = self.config.prompt_template.format(
            instruction=instruction,
            input=input_text if input_text else "N/A"
        )

        input_ids = self.tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor)
            logits = outputs[0]

            logits = logits[-1] / temperature
            probs = torch.softmax(logits, dim=-1)

            if top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
                probs[indices_to_remove] = 0
                probs = probs / probs.sum()

            next_token = torch.multinomial(probs, num_samples=1)

            generated = input_tensor.tolist()[0]
            generated.append(next_token.item())

            for _ in range(max_new_tokens - 1):
                output_tensor = torch.tensor([generated[-self.config.context_length:]], dtype=torch.long, device=self.device)
                outputs = self.model(output_tensor)
                logits = outputs[0][-1] / temperature

                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated.append(next_token.item())

                if next_token == self.tokenizer.eos_id:
                    break

        response_ids = generated[len(input_ids):]
        response = self.tokenizer.decode(response_ids)

        self.model.train()
        return response


def create_sft_trainer(
    train_file: str,
    pretrained_path: Optional[str] = None,
    test_file: Optional[str] = None,
    model_save_dir: str = "model_weights/sft",
    tokenizer_path: str = "tokenizer.json",
    context_length: int = 1024,
    batch_size: int = 4,
    learning_rate: float = 1e-5,
    epochs: int = 3,
    accumulation_steps: int = 4,
    max_grad_norm: float = 1.0,
    weight_decay: float = 0.01,
    checkpoint_every: int = 1,
    response_loss_only: bool = True,
    device: Optional[torch.device] = None,
    **kwargs
) -> SFTTrainer:
    """Factory function to create a preconfigured SFT trainer.

    Args:
        train_file: Path to training JSONL file.
        pretrained_path: Optional path to pretrained model checkpoint.
        test_file: Optional path to test/validation data.
        model_save_dir: Directory to save checkpoints.
        tokenizer_path: Path to tokenizer file.
        context_length: Maximum sequence length.
        batch_size: Training batch size.
        learning_rate: Learning rate (typically lower than pretraining).
        epochs: Number of training epochs.
        accumulation_steps: Gradient accumulation steps.
        max_grad_norm: Gradient clipping norm.
        weight_decay: Weight decay strength.
        checkpoint_every: Save checkpoint every N epochs.
        response_loss_only: Only compute loss on response tokens.
        device: Target device.
        **kwargs: Additional configuration options.

    Returns:
        Configured SFTTrainer instance.
    """
    config = SFTConfig(
        train_file=train_file,
        test_file=test_file,
        model_save_dir=model_save_dir,
        tokenizer_path=tokenizer_path,
        context_length=context_length,
        batch_size=batch_size,
        learning_rate=learning_rate,
        epochs=epochs,
        accumulation_steps=accumulation_steps,
        max_grad_norm=max_grad_norm,
        weight_decay=weight_decay,
        checkpoint_every=checkpoint_every,
        response_loss_only=response_loss_only,
    )

    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    trainer = SFTTrainer(config=config, device=device)

    if pretrained_path:
        trainer.model = trainer.build_model()
        checkpoint = torch.load(pretrained_path, map_location=device or torch.device("cpu"))
        if "model_state_dict" in checkpoint:
            trainer.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            trainer.model.load_state_dict(checkpoint)
        logger.info(f"Loaded pretrained weights from {pretrained_path}")

    if train_file:
        train_loader, eval_loader = trainer.create_dataloaders(
            train_file=train_file,
            test_file=test_file,
        )
        trainer.train_loader = train_loader
        trainer.eval_loader = eval_loader

    return trainer
