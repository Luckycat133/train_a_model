import torch
import torch.nn as nn
import argparse
import os
import json
import logging
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
from functools import lru_cache
import time

# 导入模型定义
from train_model import SimpleTransformer as LingmaoLM

# 设置日志
def setup_logger():
    """设置日志系统"""
    log_dir = Path("logs") / "generate"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger("LingmaoMoyun.Generate")
    logger.setLevel(logging.INFO)
    
    # 控制台处理器 - 使用简化格式，不显示日期时间
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 文件处理器 - 保留完整日志格式
    log_file = log_dir / f"generate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    
    # 设置格式
    console_formatter = logging.Formatter("%(message)s")
    file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    console_handler.setFormatter(console_formatter)
    file_handler.setFormatter(file_formatter)
    
    # 添加处理器
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

logger = setup_logger()

def load_tokenizer(tokenizer_path):
    """加载分词器"""
    try:
        with open(tokenizer_path, 'r', encoding='utf-8') as f:
            tokenizer_data = json.load(f)
        
        # 实现简单的编码和解码功能
        class SimpleTokenizer:
            def __init__(self, tokenizer_data):
                self.token_to_id = tokenizer_data.get("vocab", {})
                if not self.token_to_id and "model" in tokenizer_data and "vocab" in tokenizer_data["model"]:
                    self.token_to_id = tokenizer_data["model"]["vocab"]
                self.id_to_token = {v: k for k, v in self.token_to_id.items()}
                self.unk_token = "<unk>"
                self.eos_token = "<eos>"
                if self.unk_token not in self.token_to_id:
                    self.unk_token = "[UNK]"  # 尝试另一种常见格式
                if self.eos_token not in self.token_to_id:
                    self.eos_token = "[SEP]"  # 尝试另一种常见格式
                self.unk_token_id = self.token_to_id.get(self.unk_token, 0)
                self.eos_token_id = self.token_to_id.get(self.eos_token, 1)
                self.special_tokens = {self.unk_token, self.eos_token, "<pad>", "<cls>", "<sep>"}
                
                # 性能优化：预计算长度排序的关键词
                self.sorted_vocab = sorted(self.token_to_id.keys(), key=len, reverse=True)
                
                # 性能统计
                self.encode_time = 0
                self.decode_time = 0
                self.tokens_processed = 0
                
                logger.info(f"加载的词表大小: {len(self.token_to_id)}")
                if len(self.token_to_id) == 0:
                    logger.warning("警告：词表为空，可能导致分词错误")
                
                for token in self.special_tokens:
                    if token not in self.token_to_id:
                        logger.warning(f"特殊token '{token}' 不在词表中")
            
            @lru_cache(maxsize=16384)
            def _encode_cached(self, text):
                """带缓存的编码实现"""
                tokens = []
                i = 0
                while i < len(text):
                    # 优化：使用预排序的词表按最长匹配查找
                    matched = False
                    for word in self.sorted_vocab:
                        if text[i:].startswith(word):
                            tokens.append(self.token_to_id[word])
                            i += len(word)
                            matched = True
                            break
                    
                    # 如果没有匹配到，则使用单个字符并标记为未知token
                    if not matched:
                        tokens.append(self.token_to_id.get(text[i], self.unk_token_id))
                        i += 1
                
                return tuple(tokens)  # 返回元组以支持LRU缓存
            
            def encode(self, text):
                """将文本编码为token ID序列，采用贪婪匹配方式尽可能匹配词表中的多字符token"""
                start_time = time.time()
                
                # 使用缓存版本
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
        
        return SimpleTokenizer(tokenizer_data)
    except Exception as e:
        logger.error(f"加载分词器失败: {str(e)}")
        return None

def load_model(model_path, device="cpu"):
    """加载模型"""
    try:
        # 加载检查点
        checkpoint = torch.load(model_path, map_location=device)
        
        # 推断模型参数
        state_dict = checkpoint['model_state_dict']
        
        # 找到嵌入层大小以确定词表大小
        embedding_weight = None
        for key, value in state_dict.items():
            if 'embedding.weight' in key:
                embedding_weight = value
                break
        
        if embedding_weight is None:
            raise ValueError("无法从模型中推断词表大小")
        
        vocab_size, d_model = embedding_weight.shape
        
        # 创建模型实例
        model = LingmaoLM(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=12,
            num_layers=12,
            dim_feedforward=3072,
            dropout=0.1,
            max_len=1024
        )
        
        # 加载模型权重
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()  # 设置为评估模式
        model = model.to(device)
        
        logger.info(f"模型已加载: {model_path}")
        return model
    except Exception as e:
        logger.error(f"加载模型失败: {str(e)}")
        return None

def generate_text(model, tokenizer, prompt, max_length=100, temperature=0.7, top_k=50, top_p=0.9, device="cpu"):
    """生成文本"""
    # 将提示转换为token ID
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
    
    # 设置生成参数
    generated = input_ids.clone()
    
    # 生成文本
    with torch.no_grad():
        for _ in tqdm(range(max_length), desc="生成中"):
            # 取最后的context_length个tokens进行预测
            inputs = generated[:, -1024:] if generated.size(1) > 1024 else generated
            
            # 预测下一个token
            outputs = model(inputs)
            next_token_logits = outputs[:, -1, :] / temperature
            
            # Top-K采样
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Top-p (nucleus) 采样
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # 移除概率累计超过阈值的token
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[0, indices_to_remove] = float('-inf')
            
            # 采样下一个token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # 添加到生成序列
            generated = torch.cat((generated, next_token), dim=1)
            
            # 检查是否生成了结束标记
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    # 解码生成的文本
    generated_text = tokenizer.decode(generated[0].tolist())
    return generated_text

def main():
    parser = argparse.ArgumentParser(description="灵猫语言模型文本生成")
    parser.add_argument("--model", type=str, default="model_weights/best_model_v0.4.pt", help="模型路径")
    parser.add_argument("--tokenizer", type=str, default="tokenizer.json", help="分词器路径")
    parser.add_argument("--prompt", type=str, required=True, help="生成提示")
    parser.add_argument("--max_length", type=int, default=100, help="最大生成长度")
    parser.add_argument("--temperature", type=float, default=0.7, help="温度参数，控制随机性")
    parser.add_argument("--top_k", type=int, default=50, help="Top-K采样参数")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p采样参数")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="设备")
    
    args = parser.parse_args()
    
    # 加载分词器
    tokenizer = load_tokenizer(args.tokenizer)
    if tokenizer is None:
        logger.error("分词器加载失败，程序退出")
        return
    
    # 加载模型
    model = load_model(args.model, device=args.device)
    if model is None:
        logger.error("模型加载失败，程序退出")
        return
    
    # 生成文本
    logger.info(f"开始生成文本，提示: '{args.prompt}'")
    generated_text = generate_text(
        model, 
        tokenizer, 
        args.prompt, 
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=args.device
    )
    
    # 打印生成的文本
    print("\n生成结果:")
    print("-" * 50)
    print(generated_text)
    print("-" * 50)
    
    logger.info("文本生成完成")

if __name__ == "__main__":
    main()
