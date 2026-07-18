#!/usr/bin/env python
"""
灵猫墨韵模型量化工具
使用 AWQ (Activation-aware Weight Quantization) 进行 4-bit 或 8-bit 量化
"""

import argparse
import sys
import os
import torch
import torch.nn as nn
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.quantization import (
    QuantizationConfig,
    quantize_model,
    collect_activations,
    AWQLinear,
)
from src.model import SimpleTransformer
from src.config import (
    DEFAULT_D_MODEL,
    DEFAULT_NHEAD,
    DEFAULT_NUM_LAYERS,
    DEFAULT_DIM_FEEDFORWARD,
    DEFAULT_MAX_LEN,
    DEFAULT_DROPOUT,
)


def parse_args():
    parser = argparse.ArgumentParser(description='AWQ Model Quantization Tool')
    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to the pretrained model checkpoint'
    )
    parser.add_argument(
        '--output-path',
        type=str,
        required=True,
        help='Path to save the quantized model'
    )
    parser.add_argument(
        '--bits',
        type=int,
        default=4,
        choices=[4, 8],
        help='Quantization bits (4 or 8, default: 4)'
    )
    parser.add_argument(
        '--group-size',
        type=int,
        default=128,
        help='Quantization group size (default: 128)'
    )
    parser.add_argument(
        '--symmetric',
        action='store_true',
        default=True,
        help='Use symmetric quantization (default: True)'
    )
    parser.add_argument(
        '--has-zero-point',
        action='store_true',
        default=False,
        help='Use zero point for quantization (default: False)'
    )
    parser.add_argument(
        '--activation-aware',
        action='store_true',
        default=False,
        help='Use activation-aware quantization (default: False)'
    )
    parser.add_argument(
        '--dataset-path',
        type=str,
        default=None,
        help='Path to dataset for activation collection (required if --activation-aware is set)'
    )
    parser.add_argument(
        '--vocab-size',
        type=int,
        default=50000,
        help='Vocabulary size (default: 50000)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use (default: auto)'
    )
    
    return parser.parse_args()


def load_model(model_path, vocab_size, device):
    """加载预训练模型"""
    print(f"Loading model from {model_path}...")
    
    # 创建模型架构
    model = SimpleTransformer(
        vocab_size=vocab_size,
        d_model=DEFAULT_D_MODEL,
        nhead=DEFAULT_NHEAD,
        num_layers=DEFAULT_NUM_LAYERS,
        dim_feedforward=DEFAULT_DIM_FEEDFORWARD,
        dropout=DEFAULT_DROPOUT,
        max_len=DEFAULT_MAX_LEN,
        mode='modern',
    )
    
    # 加载权重
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    print("Model loaded successfully!")
    return model


def create_dummy_dataloader(vocab_size, seq_len=64, batch_size=4, num_batches=8):
    """创建一个简单的 dummy 数据加载器用于测试"""
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, vocab_size, seq_len, num_samples):
            self.vocab_size = vocab_size
            self.seq_len = seq_len
            self.num_samples = num_samples
        
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            return torch.randint(0, self.vocab_size, (self.seq_len,))
    
    dataset = DummyDataset(vocab_size, seq_len, batch_size * num_batches)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader


def main():
    args = parse_args()
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # 加载模型
    model = load_model(args.model_path, args.vocab_size, device)
    
    # 创建量化配置
    config = QuantizationConfig(
        bits=args.bits,
        group_size=args.group_size,
        symmetric=args.symmetric,
        has_zero_point=args.has_zero_point,
    )
    
    print(f"\nQuantization config:")
    print(f"  Bits: {config.bits}")
    print(f"  Group size: {config.group_size}")
    print(f"  Symmetric: {config.symmetric}")
    print(f"  Has zero point: {config.has_zero_point}")
    print(f"  Activation-aware: {args.activation_aware}")
    
    # 收集激活值（如果需要）
    activations = None
    if args.activation_aware:
        print("\nCollecting activations...")
        if args.dataset_path:
            # 这里应该加载真实数据集，暂时使用 dummy 数据
            print(f"Warning: Using dummy dataset. Real dataset at {args.dataset_path} not implemented yet.")
        dataloader = create_dummy_dataloader(args.vocab_size)
        activations = collect_activations(model, dataloader, num_samples=8, device=device)
        print(f"Collected activations from {len(activations)} layers")
    
    # 量化模型
    print("\nQuantizing model...")
    quantized_model = quantize_model(
        model,
        config=config,
        input_activations=activations,
    )
    
    # 保存量化模型
    print(f"\nSaving quantized model to {args.output_path}...")
    
    # 创建输出目录
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    
    # 保存
    torch.save({
        'model_state_dict': quantized_model.state_dict(),
        'quantization_config': {
            'bits': config.bits,
            'group_size': config.group_size,
            'symmetric': config.symmetric,
            'has_zero_point': config.has_zero_point,
        },
        'model_config': {
            'vocab_size': args.vocab_size,
            'd_model': DEFAULT_D_MODEL,
            'nhead': DEFAULT_NHEAD,
            'num_layers': DEFAULT_NUM_LAYERS,
            'dim_feedforward': DEFAULT_DIM_FEEDFORWARD,
            'max_len': DEFAULT_MAX_LEN,
        },
    }, args.output_path)
    
    print("\nQuantization complete!")
    
    # 打印模型大小比较
    print("\nModel size comparison:")
    original_size = sum(p.numel() * p.element_size() for p in model.parameters())
    quantized_size = sum(p.numel() * p.element_size() for p in quantized_model.parameters())
    
    # 加上量化层的 buffer
    for name, buf in quantized_model.named_buffers():
        quantized_size += buf.numel() * buf.element_size()
    
    print(f"  Original model: {original_size / (1024 * 1024):.2f} MB")
    print(f"  Quantized model: {quantized_size / (1024 * 1024):.2f} MB")
    print(f"  Compression ratio: {original_size / quantized_size:.2f}x")


if __name__ == '__main__':
    main()
