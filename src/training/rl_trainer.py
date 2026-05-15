"""Reinforcement Learning trainer for GRPO/DPO alignment training.

This module implements reinforcement learning-based training methods for aligning
language models with human preferences:

1. GRPO (Group Relative Policy Optimization): A simplified PPO variant that uses
   group-relative advantage estimation, similar to the approach used by DeepSeek.

2. DPO (Direct Preference Optimization): A preference learning method that
   optimizes the policy directly using preference pairs without requiring a
   separate reward model.

These methods are typically applied after SFT to further improve model quality
and alignment with human preferences.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except Exception:
    torch = None
    nn = None
    F = None

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

logger = get_logger("LingmaoMoyun.RL")


@dataclass
class RLConfig(TrainingConfig):
    """Configuration for reinforcement learning training.

    Attributes:
        method: RL training method ('grpo' or 'dpo').
        beta: DPO/GRPO temperature parameter (controls KL penalty strength).
        gamma: GRPO discount factor for future rewards.
        epsilon: PPO/GRPO clipping parameter.
        epsilon_old: Reference policy clipping (for PPO-style methods).
        ref_model: Reference model for KL divergence computation.
        reward_model_path: Optional path to reward model checkpoint.
        preference_data_file: Path to preference data for DPO training.
        grpo_group_size: Number of samples per prompt for GRPO advantage estimation.
        max_response_length: Maximum length for generated responses.
        reward_weights: Weights for multi-objective reward (if applicable).
        kl_coef: KL divergence coefficient for regularization.
        clip_reward: Clip rewards to this range for stability.
        normalize_rewards: Whether to normalize rewards.
        use_exponential Moving Average: Use EMA for reference model updates.
        ema_beta: EMA decay factor.
    """

    method: str = "dpo"
    beta: float = 0.1
    gamma: float = 1.0
    epsilon: float = 0.2
    epsilon_old: float = 0.2
    ref_model: Optional[nn.Module] = None
    reward_model_path: Optional[str] = None
    preference_data_file: Optional[str] = None
    grpo_group_size: int = 4
    max_response_length: int = 512
    reward_weights: Optional[Dict[str, float]] = None
    kl_coef: float = 0.1
    clip_reward: float = 5.0
    normalize_rewards: bool = True
    use_ema: bool = False
    ema_beta: float = 0.99

    dpo_alpha: float = 1.0
    grpo_advantage_norm: bool = True


class PreferenceDataset(Dataset):
    """Dataset for DPO/GRPO preference training.

    Supports standard preference data formats:
    - {'prompt': str, 'chosen': str, 'rejected': str}
    - {'instruction': str, 'chosen': str, 'rejected': str}
    - {'input': str, 'chosen_response': str, 'rejected_response': str}
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: Any,
        max_length: int = 1024,
        max_prompt_length: int = 512,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length

        self.samples = self._load_data(data_path)
        logger.info(f"Loaded {len(self.samples)} preference pairs for RL training")

    def _load_data(self, data_path: str) -> List[Dict[str, Any]]:
        """Load preference data from file."""
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
        """Parse preference data item."""
        prompt = None
        chosen = None
        rejected = None

        if "prompt" in item:
            prompt = item["prompt"]
        elif "instruction" in item:
            prompt = item["instruction"]

        for chosen_key in ["chosen", "chosen_response", "output"]:
            if chosen_key in item:
                chosen = item[chosen_key]
                break

        for rejected_key in ["rejected", "rejected_response"]:
            if rejected_key in item:
                rejected = item[rejected_key]
                break

        if prompt and chosen and rejected:
            return {"prompt": prompt, "chosen": chosen, "rejected": rejected}

        return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        prompt_ids = self.tokenizer.encode(
            sample["prompt"],
            max_length=self.max_prompt_length,
            truncation=True,
        )

        chosen_ids = self.tokenizer.encode(
            sample["chosen"],
            max_length=self.max_length - len(prompt_ids),
            truncation=True,
        )

        rejected_ids = self.tokenizer.encode(
            sample["rejected"],
            max_length=self.max_length - len(prompt_ids),
            truncation=True,
        )

        return {
            "prompt_ids": torch.tensor(prompt_ids, dtype=torch.long),
            "chosen_ids": torch.tensor(chosen_ids, dtype=torch.long),
            "rejected_ids": torch.tensor(rejected_ids, dtype=torch.long),
            "prompt": sample["prompt"],
            "chosen": sample["chosen"],
            "rejected": sample["rejected"],
        }


class RLTrainer(BaseTrainer):
    """Base trainer for reinforcement learning alignment methods.

    Provides common functionality for GRPO and DPO training.
    """

    def __init__(
        self,
        config: Optional[RLConfig] = None,
        model: Optional[nn.Module] = None,
        ref_model: Optional[nn.Module] = None,
        train_loader: Optional[DataLoader] = None,
        eval_loader: Optional[DataLoader] = None,
        device: Optional[torch.device] = None,
        tokenizer: Optional[Any] = None,
    ):
        if config is None:
            config = RLConfig()

        super().__init__(
            config=config,
            model=model,
            train_loader=train_loader,
            eval_loader=eval_loader,
            device=device,
        )

        self.tokenizer = tokenizer
        self.ref_model = ref_model
        self.ema_model: Optional[nn.Module] = None

        if config.use_ema:
            self._setup_ema()

    def _setup_ema(self) -> None:
        """Setup exponential moving average for reference model."""
        if self.model is None:
            return

        self.ema_model = type(self.model)(**self.model.state_dict())
        for param in self.ema_model.parameters():
            param.requires_grad = False
        self.ema_model.to(self.device)

    def _update_ema(self) -> None:
        """Update EMA model parameters."""
        if self.ema_model is None or self.model is None:
            return

        with torch.no_grad():
            for ema_p, p in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_p.data.mul_(self.config.ema_beta).add_(p.data, alpha=1 - self.config.ema_beta)


class DPOTrainer(RLTrainer):
    """Direct Preference Optimization trainer.

    DPO optimizes a language model directly on preference data without
    requiring a separate reward model or extensive hyperparameter tuning.

    The DPO loss is:
    L = - E_{(x,y_w,y_l) ~ D}[log σ(r(x, y_w) - r(x, y_l))]

    Where r is the implicit reward from the policy and reference models.

    Reference: "Direct Preference Optimization: Your Language Model is
    Secretly a Reward Model" (Rafailov et al., 2023)
    """

    def __init__(
        self,
        config: Optional[RLConfig] = None,
        model: Optional[nn.Module] = None,
        ref_model: Optional[nn.Module] = None,
        train_loader: Optional[DataLoader] = None,
        eval_loader: Optional[DataLoader] = None,
        device: Optional[torch.device] = None,
        tokenizer: Optional[Any] = None,
    ):
        if config is None:
            config = RLConfig(method="dpo")
        config.method = "dpo"

        super().__init__(
            config=config,
            model=model,
            ref_model=ref_model,
            train_loader=train_loader,
            eval_loader=eval_loader,
            device=device,
            tokenizer=tokenizer,
        )

    def build_model(self) -> nn.Module:
        """Build the policy model for DPO training."""
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

        logger.info(f"DPO model built with {sum(p.numel() for p in model.parameters()):,} parameters")
        return model

    def _build_ref_model(self) -> nn.Module:
        """Build a reference model for KL divergence computation."""
        from src.model import SimpleTransformer

        ref_model = SimpleTransformer(
            vocab_size=self.config.vocab_size,
            d_model=self.config.d_model,
            nhead=self.config.nhead,
            num_layers=self.config.num_layers,
            dim_feedforward=self.config.dim_feedforward,
            dropout=self.config.dropout,
            max_len=self.config.max_len,
            mode=self.config.mode,
        )

        if self.model is not None:
            ref_model.load_state_dict(self.model.state_dict())

        for param in ref_model.parameters():
            param.requires_grad = False

        ref_model.to(self.device)
        ref_model.eval()

        return ref_model

    def compute_log_probs(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute log probabilities for each token.

        Args:
            model: The model to use for computing log probs.
            input_ids: Input token IDs.

        Returns:
            Tuple of (log_probs, logits).
        """
        outputs, _ = model(input_ids, attention_mask=attention_mask)
        logits = outputs[:, :-1]

        log_probs = F.log_softmax(logits, dim=-1)

        labels = input_ids[:, 1:]
        token_log_probs = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

        return token_log_probs, logits

    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor],
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute the DPO loss.

        The DPO loss compares the policy's preference for chosen vs rejected
        responses with the reference model's preferences.

        Args:
            batch: Dictionary containing prompt_ids, chosen_ids, rejected_ids.

        Returns:
            Tuple of (loss, metrics).
        """
        if self.ref_model is None:
            self.ref_model = self._build_ref_model()

        prompt_ids = batch["prompt_ids"].to(self.device)
        chosen_ids = batch["chosen_ids"].to(self.device)
        rejected_ids = batch["rejected_ids"].to(self.device)

        chosen_full = torch.cat([prompt_ids, chosen_ids], dim=-1)
        rejected_full = torch.cat([prompt_ids, rejected_ids], dim=-1)

        chosen_log_probs, _ = self.compute_log_probs(self.model, chosen_full)
        rejected_log_probs, _ = self.compute_log_probs(self.model, rejected_full)

        with torch.no_grad():
            ref_chosen_log_probs, _ = self.compute_log_probs(self.ref_model, chosen_full)
            ref_rejected_log_probs, _ = self.compute_log_probs(self.ref_model, rejected_full)

        chosen_log_probs_sum = chosen_log_probs.sum(dim=-1)
        rejected_log_probs_sum = rejected_log_probs.sum(dim=-1)

        ref_chosen_log_probs_sum = ref_chosen_log_probs.sum(dim=-1)
        ref_rejected_log_probs_sum = ref_rejected_log_probs.sum(dim=-1)

        chosen_rewards = self.config.beta * (chosen_log_probs_sum - ref_chosen_log_probs_sum)
        rejected_rewards = self.config.beta * (rejected_log_probs_sum - ref_rejected_log_probs_sum)

        reward_margin = chosen_rewards - rejected_rewards

        loss = -F.logsigmoid(reward_margin).mean()

        with torch.no_grad():
            chosen_acc = (reward_margin > 0).float().mean().item()
            reward_diff = reward_margin.mean().item()

        metrics = {
            "loss": loss.item(),
            "chosen_acc": chosen_acc,
            "reward_margin": reward_diff,
            "chosen_reward": chosen_rewards.mean().item(),
            "rejected_reward": rejected_rewards.mean().item(),
        }

        return loss, metrics

    def _prepare_batch(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Prepare batch for DPO training."""
        if isinstance(batch.get("prompt_ids"), torch.Tensor):
            return {
                "prompt_ids": batch["prompt_ids"].to(self.device),
                "chosen_ids": batch["chosen_ids"].to(self.device),
                "rejected_ids": batch["rejected_ids"].to(self.device),
            }
        return batch

    def _get_extra_checkpoint_state(self) -> Dict[str, Any]:
        """Return extra state for DPO checkpointing."""
        return {
            "trainer_type": "dpo",
            "method": "dpo",
        }


class GRPOTrainer(RLTrainer):
    """Group Relative Policy Optimization trainer.

    GRPO is a simplified PPO variant that uses group-relative advantage
    estimation. For each prompt, multiple responses are sampled and the
    relative reward among them is used to estimate advantages.

    Key features:
    - No value network required (unlike PPO)
    - Group-relative advantage estimation
    - KL divergence penalty against reference model
    - Supports multiple reward signals

    Reference: Inspired by DeepSeek's GRPO and similar approaches.
    """

    def __init__(
        self,
        config: Optional[RLConfig] = None,
        model: Optional[nn.Module] = None,
        ref_model: Optional[nn.Module] = None,
        reward_fn: Optional[Callable] = None,
        train_loader: Optional[DataLoader] = None,
        eval_loader: Optional[DataLoader] = None,
        device: Optional[torch.device] = None,
        tokenizer: Optional[Any] = None,
    ):
        if config is None:
            config = RLConfig(method="grpo")
        config.method = "grpo"

        super().__init__(
            config=config,
            model=model,
            ref_model=ref_model,
            train_loader=train_loader,
            eval_loader=eval_loader,
            device=device,
            tokenizer=tokenizer,
        )

        self.reward_fn = reward_fn

    def build_model(self) -> nn.Module:
        """Build the policy model for GRPO training."""
        from src.model import SimpleTransformer

        if self.model is not None:
            return self.model

        cfg = self.config
        vocab_size = cfg.vocab_size

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

        logger.info(f"GRPO model built with {sum(p.numel() for p in model.parameters()):,} parameters")
        return model

    def _sample_responses(
        self,
        prompts: List[str],
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Tuple[List[List[int]], List[float]]:
        """Sample multiple responses for each prompt.

        Args:
            prompts: List of input prompts.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.

        Returns:
            Tuple of (list of response token IDs, list of rewards).
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model and tokenizer required for GRPO")

        self.model.eval()

        all_responses = []
        all_rewards = []

        for prompt in prompts:
            prompt_ids = self.tokenizer.encode(prompt)
            prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)

            group_responses = []
            group_rewards = []

            for _ in range(self.config.grpo_group_size):
                response_ids = self._generate_response(
                    prompt_tensor,
                    temperature=temperature,
                    top_p=top_p,
                )
                group_responses.append(response_ids)

                if self.reward_fn:
                    response_text = self.tokenizer.decode(response_ids)
                    reward = self.reward_fn(prompt, response_text)
                else:
                    reward = 0.0

                group_rewards.append(reward)

            all_responses.append(group_responses)
            all_rewards.append(group_rewards)

        self.model.train()

        return all_responses, all_rewards

    def _generate_response(
        self,
        prompt_tensor: torch.Tensor,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> List[int]:
        """Generate a single response using the policy model."""
        max_new_tokens = self.config.max_response_length

        generated = prompt_tensor.tolist()[0]

        for _ in range(max_new_tokens):
            output_tensor = torch.tensor(
                [generated[-self.config.context_length:]],
                dtype=torch.long,
                device=self.device
            )
            outputs, _ = self.model(output_tensor)
            logits = outputs[0, -1] / temperature

            probs = F.softmax(logits, dim=-1)

            if top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
                probs[indices_to_remove] = 0
                probs = probs / probs.sum()

            next_token = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_token)

            if next_token == getattr(self.tokenizer, 'eos_id', None) or next_token == 0:
                break

        return generated[len(prompt_tensor.tolist()[0]):]

    def compute_advantages(
        self,
        rewards: List[List[float]],
    ) -> List[List[float]]:
        """Compute group-relative advantages.

        For GRPO, we use the mean-normalized reward within each group as
        the advantage estimate.

        Args:
            rewards: List of rewards for each group.

        Returns:
            List of advantages for each sample.
        """
        advantages = []

        for group_rewards in rewards:
            group_rewards_tensor = torch.tensor(group_rewards, device=self.device)
            mean_reward = group_rewards_tensor.mean()
            std_reward = group_rewards_tensor.std() + 1e-8

            if self.config.grpo_advantage_norm:
                group_advantages = (group_rewards_tensor - mean_reward) / std_reward
            else:
                group_advantages = group_rewards_tensor - mean_reward

            advantages.append(group_advantages.tolist())

        return advantages

    def compute_loss(
        self,
        batch: Dict[str, Any],
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute the GRPO loss.

        The GRPO loss combines policy gradient with KL divergence penalty
        against the reference model.

        Args:
            batch: Dictionary containing prompts.

        Returns:
            Tuple of (loss, metrics).
        """
        prompts = batch.get("prompts", [])

        if not prompts:
            return torch.tensor(0.0, device=self.device), {"loss": 0.0, "reward": 0.0}

        if self.ref_model is None:
            self.ref_model = self._build_ref_model()

        responses, rewards = self._sample_responses(prompts)
        advantages = self.compute_advantages(rewards)

        total_loss = torch.tensor(0.0, device=self.device)
        num_samples = 0

        for prompt, group_responses, group_advantages in zip(prompts, responses, advantages):
            prompt_ids = self.tokenizer.encode(prompt)
            prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)

            for response_ids, advantage in zip(group_responses, group_advantages):
                response_tensor = torch.tensor([response_ids], dtype=torch.long, device=self.device)
                full_tensor = torch.cat([prompt_tensor, response_tensor], dim=-1)

                logits, _ = self.model(full_tensor)
                log_probs = F.log_softmax(logits[:-1], dim=-1)

                response_log_probs = log_probs.gather(
                    dim=-1,
                    index=response_tensor[:, 1:].unsqueeze(-1)
                ).squeeze(-1)

                policy_loss = -(response_log_probs.mean() * advantage)

                if self.config.kl_coef > 0:
                    with torch.no_grad():
                        ref_logits, _ = self.ref_model(full_tensor)
                        ref_log_probs = F.log_softmax(ref_logits[:-1], dim=-1)
                        ref_log_probs = ref_log_probs.detach()

                    kl_div = F.kl_div(
                        log_probs,
                        ref_log_probs,
                        reduction="batchmean",
                        log_target=True,
                    )
                    kl_penalty = self.config.kl_coef * kl_div

                    total_loss += policy_loss + kl_penalty
                else:
                    total_loss += policy_loss

                num_samples += 1

        if num_samples > 0:
            total_loss = total_loss / num_samples

        with torch.no_grad():
            flat_rewards = [r for group in rewards for r in group]
            avg_reward = sum(flat_rewards) / len(flat_rewards) if flat_rewards else 0.0

        metrics = {
            "loss": total_loss.item(),
            "reward": avg_reward,
            "num_samples": num_samples,
        }

        return total_loss, metrics

    def _build_ref_model(self) -> nn.Module:
        """Build a reference model for KL divergence computation."""
        from src.model import SimpleTransformer

        ref_model = SimpleTransformer(
            vocab_size=self.config.vocab_size,
            d_model=self.config.d_model,
            nhead=self.config.nhead,
            num_layers=self.config.num_layers,
            dim_feedforward=self.config.dim_feedforward,
            dropout=self.config.dropout,
            max_len=self.config.max_len,
            mode=self.config.mode,
        )

        if self.model is not None:
            ref_model.load_state_dict(self.model.state_dict())

        for param in ref_model.parameters():
            param.requires_grad = False

        ref_model.to(self.device)
        ref_model.eval()

        return ref_model

    def _prepare_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare batch for GRPO training."""
        return batch

    def _get_extra_checkpoint_state(self) -> Dict[str, Any]:
        """Return extra state for GRPO checkpointing."""
        return {
            "trainer_type": "grpo",
            "method": "grpo",
        }


def create_rl_trainer(
    method: str = "dpo",
    preference_data_file: Optional[str] = None,
    model: Optional[nn.Module] = None,
    pretrained_path: Optional[str] = None,
    model_save_dir: str = "model_weights/rl",
    tokenizer_path: str = "tokenizer.json",
    batch_size: int = 4,
    learning_rate: float = 1e-6,
    epochs: int = 3,
    beta: float = 0.1,
    kl_coef: float = 0.1,
    device: Optional[torch.device] = None,
    **kwargs
) -> Union[DPOTrainer, GRPOTrainer]:
    """Factory function to create an RL trainer.

    Args:
        method: RL training method ('dpo' or 'grpo').
        preference_data_file: Path to preference data file.
        model: Optional pre-built model.
        pretrained_path: Optional path to pretrained checkpoint.
        model_save_dir: Directory to save checkpoints.
        tokenizer_path: Path to tokenizer file.
        batch_size: Training batch size.
        learning_rate: Learning rate.
        epochs: Number of training epochs.
        beta: DPO/GRPO temperature parameter.
        kl_coef: KL divergence coefficient.
        device: Target device.
        **kwargs: Additional configuration.

    Returns:
        Configured DPOTrainer or GRPOTrainer instance.
    """
    config = RLConfig(
        method=method,
        preference_data_file=preference_data_file,
        model_save_dir=model_save_dir,
        tokenizer_path=tokenizer_path,
        batch_size=batch_size,
        learning_rate=learning_rate,
        epochs=epochs,
        beta=beta,
        kl_coef=kl_coef,
        **kwargs
    )

    if method.lower() == "dpo":
        trainer = DPOTrainer(config=config, model=model, device=device)
    elif method.lower() == "grpo":
        trainer = GRPOTrainer(config=config, model=model, device=device)
    else:
        raise ValueError(f"Unknown RL method: {method}. Use 'dpo' or 'grpo'.")

    if pretrained_path:
        trainer.model = trainer.build_model()
        checkpoint = torch.load(pretrained_path, map_location=device or torch.device("cpu"))
        if "model_state_dict" in checkpoint:
            trainer.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            trainer.model.load_state_dict(checkpoint)
        logger.info(f"Loaded pretrained weights from {pretrained_path}")

    if preference_data_file and os.path.exists(preference_data_file):
        dataset = PreferenceDataset(
            data_path=preference_data_file,
            tokenizer=trainer.tokenizer,
            max_length=config.context_length,
        )
        trainer.train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=(device.type == "cuda") if device else False,
        )
        logger.info(f"Created RL dataloader with {len(dataset)} samples")

    return trainer
