#!/usr/bin/env python
"""
AWQ 量化演示脚本
展示如何使用 AWQ 量化灵猫墨韵模型
"""

import torch
import sys
from pathlib import Path

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent))

from src.model import SimpleTransformer
from src.quantization import (
    QuantizationConfig,
    quantize_model,
    collect_activations
)


def main():
    # 创建一个小型演示模型
    vocab_size = 1000
    d_model = 128
    nhead = 4
    num_layers = 3
    dim_feedforward = 512
    
    print("=" * 60)
    print("灵猫墨韵 AWQ 量化演示")
    print("=" * 60)
    
    # 1. 创建原始模型
    print("\n[1/5] 创建原始模型...")
    model = SimpleTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=0.0,
        max_len=128,
        mode='modern'
    )
    model.eval()
    print(f"   模型参数: {sum(p.numel() for p in model.parameters()):,}")
    
    # 2. 创建 dummy 数据加载器
    print("\n[2/5] 创建 dummy 数据...")
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, num_samples=16, seq_len=32):
            self.data = [torch.randint(0, vocab_size, (seq_len,)) for _ in range(num_samples)]
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            return self.data[idx]
    
    dataloader = torch.utils.data.DataLoader(DummyDataset(), batch_size=4)
    
    # 3. 收集激活值
    print("\n[3/5] 收集激活值 (用于激活感知量化)...")
    activations = collect_activations(model, dataloader, num_samples=8)
    print(f"   从 {len(activations)} 层收集了激活值")
    
    # 4. 测试不同的量化配置
    print("\n[4/5] 测试不同量化配置...")
    
    test_configs = [
        ("4位, 分组64", QuantizationConfig(bits=4, group_size=64)),
        ("8位, 分组64", QuantizationConfig(bits=8, group_size=64)),
        ("4位, 分组128", QuantizationConfig(bits=4, group_size=128)),
    ]
    
    for config_name, config in test_configs:
        print(f"\n  配置: {config_name}")
        
        # 量化模型
        quantized_model = quantize_model(
            model,
            config=config,
            input_activations=activations
        )
        quantized_model.eval()
        
        # 测试前向传播
        input_ids = torch.randint(0, vocab_size, (2, 16))
        
        with torch.no_grad():
            original_logits, _ = model(input_ids)
            quantized_logits, _ = quantized_model(input_ids)
        
        # 计算误差
        error = torch.abs(quantized_logits - original_logits).mean().item()
        print(f"    平均误差: {error:.6f}")
        
        # 计算模型大小
        def get_model_size(m):
            size = 0
            for p in m.parameters():
                size += p.numel() * p.element_size()
            for b in m.buffers():
                size += b.numel() * b.element_size()
            return size
        
        original_size = get_model_size(model)
        quantized_size = get_model_size(quantized_model)
        print(f"    原始大小: {original_size / 1024:.2f} KB")
        print(f"    量化大小: {quantized_size / 1024:.2f} KB")
        print(f"    压缩比: {original_size / quantized_size:.2f}x")
    
    # 5. 保存量化模型示例
    print("\n[5/5] 保存量化模型示例 (演示)...")
    
    # 选择一个配置进行保存
    save_config = QuantizationConfig(bits=4, group_size=128)
    quantized_model = quantize_model(model, config=save_config, input_activations=activations)
    
    # 保存
    save_path = Path("demo_quantized_model.pt")
    torch.save({
        'model_state_dict': quantized_model.state_dict(),
        'quantization_config': {
            'bits': save_config.bits,
            'group_size': save_config.group_size,
            'symmetric': save_config.symmetric,
            'has_zero_point': save_config.has_zero_point,
        },
        'model_config': {
            'vocab_size': vocab_size,
            'd_model': d_model,
            'nhead': nhead,
            'num_layers': num_layers,
            'dim_feedforward': dim_feedforward,
        }
    }, save_path)
    print(f"   量化模型已保存到: {save_path}")
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
