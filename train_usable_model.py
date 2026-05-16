#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train a usable classical Chinese language model."""

import sys
import json
import os
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.model import SimpleTransformer
from src.dataset import LMDataset
from src.logger import get_logger
from src.config import (
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_D_MODEL,
    DEFAULT_DIM_FEEDFORWARD,
    DEFAULT_DROPOUT,
    DEFAULT_NHEAD,
    DEFAULT_NUM_LAYERS,
    DEFAULT_VOCAB_SIZE,
    DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_EPOCHS,
    DEFAULT_MAX_GRAD_NORM,
    DEFAULT_WEIGHT_DECAY,
)

logger = get_logger("LingmaoMoyun.TrainUsable")


class CharTokenizer:
    """Simple character-level tokenizer for quick training."""
    
    def __init__(self):
        self.token_to_id = {}
        self.id_to_token = {}
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.sep_token = "<sep>"
        self.cls_token = "<cls>"
        
        # Initialize special tokens
        special_tokens = [
            self.pad_token,
            self.unk_token,
            self.bos_token,
            self.eos_token,
            self.sep_token,
            self.cls_token,
        ]
        for idx, token in enumerate(special_tokens):
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
        
        self.pad_token_id = self.token_to_id[self.pad_token]
        self.unk_token_id = self.token_to_id[self.unk_token]
        self.bos_token_id = self.token_to_id[self.bos_token]
        self.eos_token_id = self.token_to_id[self.eos_token]
    
    def fit(self, texts):
        """Learn vocabulary from texts."""
        chars = set()
        for text in texts:
            chars.update(text)
        
        idx = len(self.token_to_id)
        for char in sorted(chars):
            if char not in self.token_to_id:
                self.token_to_id[char] = idx
                self.id_to_token[idx] = char
                idx += 1
        
        logger.info(f"Vocab size: {len(self.token_to_id)}")
    
    def encode(self, text):
        """Encode text to list of IDs."""
        tokens = []
        for char in text:
            tokens.append(self.token_to_id.get(char, self.unk_token_id))
        return tokens
    
    def decode(self, ids):
        """Decode IDs to text."""
        chars = []
        for token_id in ids:
            chars.append(self.id_to_token.get(token_id, "<?>"))
        return "".join(chars)
    
    def save(self, path):
        """Save tokenizer to JSON."""
        data = {
            "token_to_id": self.token_to_id,
            "id_to_token": {int(k): v for k, v in self.id_to_token.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path):
        """Load tokenizer from JSON."""
        tok = cls()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok.token_to_id = data.get("token_to_id", {})
        tok.id_to_token = {int(k): v for k, v in data.get("id_to_token", {}).items()}
        return tok


def prepare_texts_from_jsonl(jsonl_path):
    """Prepare texts from JSONL file."""
    texts = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                text = item.get("text", "") or item.get("content", "")
                if text:
                    texts.append(text)
            except:
                pass
    return texts


def train_model(
    train_file: str,
    model_save_dir: str = "model_weights/usable_model",
    context_length: int = 64,
    batch_size: int = 8,
    learning_rate: float = 3e-4,
    epochs: int = 20,
    d_model: int = 192,
    nhead: int = 4,
    num_layers: int = 6,
    dim_feedforward: int = 768,
    dropout: float = 0.1,
    max_grad_norm: float = 1.0,
    weight_decay: float = 0.01,
    device=None,
):
    """Train a usable model."""
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"Training device: {device}")
    
    # 1. Prepare tokenizer
    logger.info("Preparing tokenizer...")
    texts = prepare_texts_from_jsonl(train_file)
    logger.info(f"Loaded {len(texts)} training texts")
    
    tokenizer = CharTokenizer()
    tokenizer.fit(texts)
    tokenizer.save(os.path.join(model_save_dir, "tokenizer.json"))
    logger.info("Tokenizer saved")
    
    # 2. Prepare dataset
    logger.info("Preparing dataset...")
    # Create a simple tokenized dataset
    all_tokens = []
    for text in texts:
        tokens = tokenizer.encode(text)
        all_tokens.extend(tokens)
        all_tokens.append(tokenizer.eos_token_id)  # Separate samples
    
    logger.info(f"Total tokens: {len(all_tokens)}")
    
    # Create sliding window samples
    samples = []
    stride = context_length // 2
    for i in range(0, len(all_tokens) - context_length, stride):
        input_seq = all_tokens[i:i + context_length]
        target_seq = all_tokens[i + 1:i + context_length + 1]
        if len(target_seq) == context_length:
            samples.append({
                "input_ids": torch.tensor(input_seq, dtype=torch.long),
                "target_ids": torch.tensor(target_seq, dtype=torch.long),
            })
    
    logger.info(f"Created {len(samples)} training samples")
    
    dataloader = DataLoader(samples, batch_size=batch_size, shuffle=True)
    
    # 3. Create model
    logger.info("Creating model...")
    vocab_size = len(tokenizer.token_to_id)
    model = SimpleTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        max_len=context_length * 2,
        mode="modern",
        use_weight_tying=True,
    ).to(device)
    
    # 4. Training setup
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs * len(dataloader)
    )
    
    # 5. Training loop
    logger.info("Starting training...")
    best_loss = float("inf")
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            optimizer.zero_grad()
            
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            
            logits, _ = model(input_ids)
            loss = criterion(
                logits.view(-1, vocab_size),
                target_ids.view(-1)
            )
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if num_batches % 10 == 0:
                avg_loss = total_loss / num_batches
                logger.info(f"Epoch {epoch + 1}, Batch {num_batches}, Loss: {avg_loss:.4f}")
        
        avg_loss = total_loss / num_batches
        logger.info(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(model_save_dir, f"checkpoint_epoch_{epoch + 1}.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
            "model_config": {
                "vocab_size": vocab_size,
                "d_model": d_model,
                "nhead": nhead,
                "num_layers": num_layers,
                "dim_feedforward": dim_feedforward,
                "dropout": dropout,
                "max_len": context_length * 2,
                "mode": "modern",
            },
        }, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(model_save_dir, "best_model.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "loss": avg_loss,
                "model_config": {
                    "vocab_size": vocab_size,
                    "d_model": d_model,
                    "nhead": nhead,
                    "num_layers": num_layers,
                    "dim_feedforward": dim_feedforward,
                    "dropout": dropout,
                    "max_len": context_length * 2,
                    "mode": "modern",
                },
            }, best_path)
            logger.info(f"New best model saved: {avg_loss:.4f}")
    
    elapsed = time.time() - start_time
    logger.info(f"Training complete in {elapsed:.2f}s")
    
    # Save final model
    final_path = os.path.join(model_save_dir, "final_model.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": {
            "vocab_size": vocab_size,
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": num_layers,
            "dim_feedforward": dim_feedforward,
            "dropout": dropout,
            "max_len": context_length * 2,
            "mode": "modern",
        },
    }, final_path)
    logger.info(f"Final model saved to {final_path}")
    
    return model, tokenizer


def load_model(model_path, tokenizer_path=None):
    """Load trained model."""
    checkpoint = torch.load(model_path, map_location="cpu")
    config = checkpoint["model_config"]
    
    model = SimpleTransformer(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_layers=config["num_layers"],
        dim_feedforward=config["dim_feedforward"],
        dropout=config.get("dropout", 0.1),
        max_len=config["max_len"],
        mode=config["mode"],
        use_weight_tying=True,
    )
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    if tokenizer_path and os.path.exists(tokenizer_path):
        tokenizer = CharTokenizer.load(tokenizer_path)
    else:
        tokenizer = None
    
    return model, tokenizer


def generate_text(model, tokenizer, prompt, max_length=100, temperature=0.8, device=None):
    """Generate text from prompt."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    model.eval()
    
    # Encode prompt
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    
    generated = list(prompt)
    with torch.no_grad():
        for i in range(max_length):
            logits, _ = model(input_tensor)
            next_logits = logits[0, -1, :] / temperature
            
            # Sample
            probs = torch.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()
            
            if next_id == tokenizer.eos_token_id:
                break
                
            generated.append(tokenizer.id_to_token.get(next_id, "<?>"))
            input_tensor = torch.cat([
                input_tensor,
                torch.tensor([[next_id]], dtype=torch.long).to(device)
            ], dim=1)
    
    return "".join(generated)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train a usable classical Chinese model")
    parser.add_argument("--train-file", type=str, default="dataset/sample_train.jsonl")
    parser.add_argument("--model-dir", type=str, default="model_weights/usable_model")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--dim-feedforward", type=int, default=768)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--train", action="store_true", help="Train model")
    parser.add_argument("--generate", type=str, help="Generate text with prompt")
    parser.add_argument("--model-path", type=str, help="Path to model for generation")
    parser.add_argument("--tokenizer-path", type=str, help="Path to tokenizer for generation")
    
    args = parser.parse_args()
    
    if args.train:
        logger.info("=" * 80)
        logger.info("Training usable model")
        logger.info("=" * 80)
        model, tokenizer = train_model(
            train_file=args.train_file,
            model_save_dir=args.model_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_feedforward=args.dim_feedforward,
            context_length=args.context_length,
            learning_rate=args.lr,
        )
        
        # Test generation
        print("\nTesting generation...")
        test_prompts = [
            "床前明月光，",
            "白日依山尽，",
            "子曰：学而时习之，",
        ]
        for prompt in test_prompts:
            try:
                output = generate_text(model, tokenizer, prompt, max_length=60)
                print(f"\nPrompt: {prompt}")
                print(f"Output: {output}")
            except Exception as e:
                print(f"Generation failed for '{prompt}': {e}")
    
    elif args.generate:
        if not args.model_path:
            print("Error: --model-path required for generation")
            sys.exit(1)
        
        model, tokenizer = load_model(
            args.model_path,
            args.tokenizer_path or os.path.join(os.path.dirname(args.model_path), "tokenizer.json")
        )
        output = generate_text(model, tokenizer, args.generate, max_length=100)
        print(f"\nPrompt: {args.generate}")
        print(f"Output: {output}")
    
    else:
        print("Please specify --train or --generate")
