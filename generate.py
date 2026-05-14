import torch
import torch.nn as nn
import argparse
import os
import json
import logging
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
import time
from functools import lru_cache
from typing import List, Optional, Tuple, Union, Generator
from dataclasses import dataclass

from src.model import SimpleTransformer as LingmaoLM

logger = None

@dataclass
class GenerationConfig:
    """配置生成参数的数据类"""
    max_length: int = 100
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.0
    do_sample: bool = True
    use_cache: bool = True
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    max_new_tokens: Optional[int] = None
    stopping_criteria: Optional[List[int]] = None


class TrieNode:
    """前缀树节点"""
    __slots__ = ['children', 'token']
    def __init__(self):
        self.children = {}
        self.token = None


class Trie:
    """前缀树用于高效的最长匹配查找"""
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, token):
        """向前缀树中插入一个token"""
        node = self.root
        for char in token:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.token = token
    
    def find_longest_match(self, text, start):
        """从文本的start位置开始，查找最长匹配的token"""
        node = self.root
        longest_match = None
        longest_length = 0
        for i in range(start, len(text)):
            char = text[i]
            if char not in node.children:
                break
            node = node.children[char]
            if node.token is not None:
                longest_match = node.token
                longest_length = i - start + 1
        return longest_match, longest_length


class SimpleTokenizer:
    """简单的分词器类"""
    def __init__(self, tokenizer_data):
        self.token_to_id = tokenizer_data.get("vocab", {})
        if not self.token_to_id and "model" in tokenizer_data and "vocab" in tokenizer_data["model"]:
            self.token_to_id = tokenizer_data["model"]["vocab"]
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        self.unk_token = "<unk>"
        self.eos_token = "<eos>"
        if self.unk_token not in self.token_to_id:
            self.unk_token = "[UNK]"
        if self.eos_token not in self.token_to_id:
            self.eos_token = "[SEP]"
        self.unk_token_id = self.token_to_id.get(self.unk_token, 0)
        self.eos_token_id = self.token_to_id.get(self.eos_token, 1)
        self.special_tokens = {self.unk_token, self.eos_token, "<pad>", "<bos>", "<cls>", "<sep>"}
        
        self.trie = Trie()
        for token in self.token_to_id.keys():
            self.trie.insert(token)
        
        self.encode_cache = {}
        self.encode_time = 0
        self.decode_time = 0
        self.tokens_processed = 0
    
    def _encode_cached(self, text):
        """使用前缀树的编码实现，带内部缓存"""
        if text in self.encode_cache:
            return self.encode_cache[text]
        
        tokens = []
        i = 0
        while i < len(text):
            matched_token, matched_length = self.trie.find_longest_match(text, i)
            if matched_token is not None:
                tokens.append(self.token_to_id[matched_token])
                i += matched_length
            else:
                tokens.append(self.token_to_id.get(text[i], self.unk_token_id))
                i += 1
        
        result = tuple(tokens)
        self.encode_cache[text] = result
        return result
    
    def encode(self, text):
        """将文本编码为token ID序列"""
        start_time = time.time()
        result = list(self._encode_cached(text))
        end_time = time.time()
        self.encode_time += (end_time - start_time)
        self.tokens_processed += len(result)
        return result
    
    def decode(self, ids):
        """将token ID解码为文本"""
        start_time = time.time()
        tokens = [self.id_to_token.get(id, self.unk_token) for id in ids]
        result = "".join(tokens)
        end_time = time.time()
        self.decode_time += (end_time - start_time)
        return result
    
    def get_stats(self):
        """返回性能统计信息"""
        return {
            "encode_time": self.encode_time,
            "decode_time": self.decode_time,
            "tokens_processed": self.tokens_processed,
            "tokens_per_second": self.tokens_processed / max(self.encode_time, 0.001),
            "vocab_size": len(self.token_to_id)
        }


def setup_logger():
    """设置日志系统"""
    global logger
    log_dir = Path("logs") / "generate"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger("LingmaoMoyun.Generate")
    logger.setLevel(logging.INFO)
    
    if logger.handlers:
        return logger
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    log_file = log_dir / f"generate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    
    console_formatter = logging.Formatter("%(message)s")
    file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    console_handler.setFormatter(console_formatter)
    file_handler.setFormatter(file_formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


def load_tokenizer(tokenizer_path):
    """加载分词器"""
    try:
        with open(tokenizer_path, 'r', encoding='utf-8') as f:
            tokenizer_data = json.load(f)
        return SimpleTokenizer(tokenizer_data)
    except Exception as e:
        if logger:
            logger.error(f"加载分词器失败: {str(e)}")
        return None


def load_model(model_path, device="cpu"):
    """加载模型"""
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        state_dict = checkpoint['model_state_dict']
        
        embedding_weight = None
        for key, value in state_dict.items():
            if 'embedding.weight' in key:
                embedding_weight = value
                break
        
        if embedding_weight is None:
            raise ValueError("无法从模型中推断词表大小")
        
        vocab_size, d_model = embedding_weight.shape
        
        nhead = checkpoint.get('nhead', 12)
        num_layers = checkpoint.get('num_layers', 12)
        dim_feedforward = checkpoint.get('dim_feedforward', 3072)
        dropout = checkpoint.get('dropout', 0.1)
        max_len = checkpoint.get('max_len', 1024)
        
        model = LingmaoLM(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_len=max_len,
            mode="modern"
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        model = model.to(device)
        
        if logger:
            logger.info(f"模型已加载: {model_path}")
        return model
    except Exception as e:
        if logger:
            logger.error(f"加载模型失败: {str(e)}")
        return None


def apply_sampling(logits: torch.Tensor, temperature: float = 1.0, 
                   top_k: int = 0, top_p: float = 1.0,
                   repetition_penalty: float = 1.0,
                   input_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
    """应用多种采样策略
    
    Args:
        logits: 模型输出的 logits [batch_size, vocab_size]
        temperature: 温度参数
        top_k: Top-K 采样
        top_p: Top-P (nucleus) 采样
        repetition_penalty: 重复惩罚
        input_ids: 输入的 token ID（用于重复惩罚）
    
    Returns:
        采样后的 token ID [batch_size]
    """
    if temperature <= 0:
        return torch.argmax(logits, dim=-1)
    
    logits = logits / temperature
    
    if repetition_penalty != 1.0 and input_ids is not None:
        for i in range(input_ids.shape[0]):
            for token_id in input_ids[i]:
                logits[i, token_id] = logits[i, token_id] / repetition_penalty
    
    if top_k > 0:
        vocab_size = logits.size(-1)
        effective_top_k = min(top_k, vocab_size)
        indices_to_remove = logits < torch.topk(logits, effective_top_k)[0][..., -1, None]
        logits = logits.masked_fill(indices_to_remove, float('-inf'))
    
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        
        indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, float('-inf'))
    
    probs = torch.softmax(logits, dim=-1)
    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
    return next_tokens


def generate_text(model, tokenizer, prompt: Union[str, List[str]], 
                  config: Optional[GenerationConfig] = None,
                  device: str = "cpu") -> Union[str, List[str]]:
    """生成文本 - 支持单条和批量生成
    
    Args:
        model: 模型实例
        tokenizer: 分词器实例
        prompt: 提示文本或提示文本列表
        config: 生成配置
        device: 设备
    
    Returns:
        生成的文本或文本列表
    """
    if config is None:
        config = GenerationConfig()
    
    if isinstance(prompt, str):
        prompt = [prompt]
        is_single = True
    else:
        is_single = False
    
    batch_size = len(prompt)
    
    input_ids_list = []
    max_prompt_len = 0
    for p in prompt:
        ids = tokenizer.encode(p)
        input_ids_list.append(ids)
        if len(ids) > max_prompt_len:
            max_prompt_len = len(ids)
    
    pad_token_id = config.pad_token_id if config.pad_token_id is not None else tokenizer.unk_token_id
    
    input_ids = []
    attention_mask = []
    for ids in input_ids_list:
        pad_len = max_prompt_len - len(ids)
        padded_ids = [pad_token_id] * pad_len + ids
        input_ids.append(padded_ids)
        attention_mask.append([0] * pad_len + [1] * len(ids))
    
    input_ids = torch.tensor(input_ids, dtype=torch.long).to(device)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long).to(device)
    
    if config.max_new_tokens is None:
        max_len = config.max_length
    else:
        max_len = input_ids.shape[1] + config.max_new_tokens
    
    generated = input_ids.clone()
    past_key_values = None
    
    eos_token_id = config.eos_token_id if config.eos_token_id is not None else tokenizer.eos_token_id
    if config.stopping_criteria:
        stopping_criteria = set(config.stopping_criteria + [eos_token_id])
    else:
        stopping_criteria = {eos_token_id}
    
    finished = torch.zeros(batch_size, dtype=torch.bool).to(device)
    
    with torch.no_grad():
        for _ in tqdm(range(max_len - input_ids.shape[1]), desc="生成中"):
            if finished.all():
                break
            
            if past_key_values is not None and config.use_cache:
                current_input = generated[:, -1:]
                current_attention_mask = attention_mask[:, -1:]
            else:
                current_input = generated
                current_attention_mask = attention_mask
            
            outputs, present = model(
                current_input,
                attention_mask=current_attention_mask,
                use_cache=config.use_cache,
                past_key_values=past_key_values
            )
            
            if config.use_cache:
                past_key_values = present
            
            next_token_logits = outputs[:, -1, :]
            
            next_tokens = apply_sampling(
                next_token_logits,
                temperature=config.temperature,
                top_k=config.top_k,
                top_p=config.top_p,
                repetition_penalty=config.repetition_penalty,
                input_ids=generated
            )
            
            next_tokens = next_tokens.to(device)
            
            next_tokens = torch.where(finished, torch.tensor(pad_token_id, device=device), next_tokens)
            
            generated = torch.cat([generated, next_tokens.unsqueeze(1)], dim=1)
            attention_mask = torch.cat([
                attention_mask, 
                torch.ones((batch_size, 1), dtype=torch.long, device=device)
            ], dim=1)
            
            finished = finished | torch.isin(next_tokens, torch.tensor(list(stopping_criteria), device=device))
    
    generated_texts = []
    for i in range(batch_size):
        seq = generated[i].tolist()
        try:
            eos_pos = seq.index(eos_token_id)
            seq = seq[:eos_pos]
        except ValueError:
            pass
        text = tokenizer.decode(seq)
        generated_texts.append(text)
    
    return generated_texts[0] if is_single else generated_texts


def generate_stream(model, tokenizer, prompt: str, 
                    config: Optional[GenerationConfig] = None,
                    device: str = "cpu") -> Generator[str, None, None]:
    """流式生成文本
    
    Args:
        model: 模型实例
        tokenizer: 分词器实例
        prompt: 提示文本
        config: 生成配置
        device: 设备
    
    Yields:
        逐步生成的文本片段
    """
    if config is None:
        config = GenerationConfig()
    
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
    
    if config.max_new_tokens is None:
        max_len = config.max_length
    else:
        max_len = input_ids.shape[1] + config.max_new_tokens
    
    generated = input_ids.clone()
    past_key_values = None
    
    eos_token_id = config.eos_token_id if config.eos_token_id is not None else tokenizer.eos_token_id
    if config.stopping_criteria:
        stopping_criteria = set(config.stopping_criteria + [eos_token_id])
    else:
        stopping_criteria = {eos_token_id}
    
    generated_text_so_far = ""
    last_yielded_pos = len(prompt)
    
    with torch.no_grad():
        for step in range(max_len - input_ids.shape[1]):
            if past_key_values is not None and config.use_cache:
                current_input = generated[:, -1:]
            else:
                current_input = generated
            
            outputs, present = model(
                current_input,
                use_cache=config.use_cache,
                past_key_values=past_key_values
            )
            
            if config.use_cache:
                past_key_values = present
            
            next_token_logits = outputs[:, -1, :]
            
            next_tokens = apply_sampling(
                next_token_logits,
                temperature=config.temperature,
                top_k=config.top_k,
                top_p=config.top_p,
                repetition_penalty=config.repetition_penalty,
                input_ids=generated
            )
            
            next_token = next_tokens.item()
            
            if next_token in stopping_criteria:
                break
            
            generated = torch.cat([generated, torch.tensor([[next_token]], device=device)], dim=1)
            
            current_text = tokenizer.decode(generated[0].tolist())
            if len(current_text) > last_yielded_pos:
                new_text = current_text[last_yielded_pos:]
                yield new_text
                generated_text_so_far = current_text
                last_yielded_pos = len(current_text)


def main():
    setup_logger()
    
    parser = argparse.ArgumentParser(description="灵猫语言模型文本生成")
    parser.add_argument("--model", type=str, default="model_weights/best_model_v0.4.pt", help="模型路径")
    parser.add_argument("--tokenizer", type=str, default="tokenizer.json", help="分词器路径")
    parser.add_argument("--prompt", type=str, required=True, help="生成提示")
    parser.add_argument("--max_length", type=int, default=100, help="最大生成长度")
    parser.add_argument("--max_new_tokens", type=int, default=None, help="最大新生成 token 数")
    parser.add_argument("--temperature", type=float, default=0.7, help="温度参数，控制随机性")
    parser.add_argument("--top_k", type=int, default=50, help="Top-K采样参数")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p采样参数")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="重复惩罚参数")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="设备")
    parser.add_argument("--stream", action="store_true", help="流式输出")
    
    args = parser.parse_args()
    
    tokenizer = load_tokenizer(args.tokenizer)
    if tokenizer is None:
        logger.error("分词器加载失败，程序退出")
        return
    
    model = load_model(args.model, device=args.device)
    if model is None:
        logger.error("模型加载失败，程序退出")
        return
    
    config = GenerationConfig(
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        eos_token_id=tokenizer.eos_token_id
    )
    
    if args.stream:
        logger.info(f"开始流式生成，提示: '{args.prompt}'")
        print("\n生成结果:")
        print("-" * 50)
        print(args.prompt, end="", flush=True)
        for text in generate_stream(model, tokenizer, args.prompt, config, args.device):
            print(text, end="", flush=True)
        print()
        print("-" * 50)
    else:
        logger.info(f"开始生成文本，提示: '{args.prompt}'")
        generated_text = generate_text(model, tokenizer, args.prompt, config, args.device)
        
        print("\n生成结果:")
        print("-" * 50)
        print(generated_text)
        print("-" * 50)
    
    logger.info("文本生成完成")


if __name__ == "__main__":
    main()
