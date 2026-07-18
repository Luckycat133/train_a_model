"""
AWQ 量化测试
"""

import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.quantization import (
    QuantizationConfig,
    AWQLinear,
    quantize_model,
    collect_activations,
)
from src.model import SimpleTransformer
from src.config import (
    DEFAULT_D_MODEL,
    DEFAULT_NHEAD,
    DEFAULT_NUM_LAYERS,
    DEFAULT_DIM_FEEDFORWARD,
    DEFAULT_MAX_LEN,
)


class TestAWQLinear:
    """测试 AWQ 量化线性层"""
    
    @pytest.mark.parametrize("bits", [4, 8])
    @pytest.mark.parametrize("group_size", [32, 64, 128])
    def test_quantization_basics(self, bits, group_size):
        """测试基础量化功能"""
        in_features = 256
        out_features = 128
        
        # 创建原始线性层
        linear = nn.Linear(in_features, out_features)
        linear.weight.data.normal_(0, 0.1)
        if linear.bias is not None:
            linear.bias.data.normal_(0, 0.01)
        
        # 创建量化配置
        config = QuantizationConfig(bits=bits, group_size=group_size)
        
        # 创建并量化 AWQ 线性层
        awq_linear = AWQLinear(in_features, out_features, config=config, bias=linear.bias is not None)
        awq_linear.quantize(linear.weight)
        if linear.bias is not None:
            awq_linear.bias.copy_(linear.bias)
        
        # 测试前向传播
        x = torch.randn(4, in_features)
        
        # 原始输出
        with torch.no_grad():
            original_output = linear(x)
        
        # 量化输出
        with torch.no_grad():
            quantized_output = awq_linear(x)
        
        # 检查输出形状
        assert quantized_output.shape == original_output.shape
        
        # 检查误差在合理范围内
        error = torch.abs(quantized_output - original_output).mean()
        print(f"Bits: {bits}, Group size: {group_size}, Mean error: {error.item():.6f}")
        assert error.item() < 0.5, f"Error too high: {error.item()}"
    
    def test_from_linear(self):
        """测试从现有线性层创建"""
        in_features = 128
        out_features = 64
        
        linear = nn.Linear(in_features, out_features)
        config = QuantizationConfig(bits=4, group_size=64)
        
        awq_linear = AWQLinear.from_linear(linear, config)
        
        # 检查是否已量化
        assert awq_linear._quantized
        
        # 测试前向传播
        x = torch.randn(2, in_features)
        with torch.no_grad():
            output = awq_linear(x)
        
        assert output.shape == (2, out_features)
    
    def test_activation_aware(self):
        """测试激活感知量化"""
        in_features = 128
        out_features = 64
        
        linear = nn.Linear(in_features, out_features)
        linear.weight.data.normal_(0, 0.1)
        
        # 创建一些激活样本
        activations = [
            torch.randn(4, in_features) * torch.tensor([1.0 + i * 0.1 for i in range(in_features)])
            for _ in range(4)
        ]
        
        config = QuantizationConfig(bits=4, group_size=64)
        awq_linear = AWQLinear.from_linear(linear, config, activations)
        
        # 测试前向传播
        x = torch.randn(2, in_features)
        with torch.no_grad():
            output = awq_linear(x)
        
        assert output.shape == (2, out_features)


class TestModelQuantization:
    """测试完整模型量化"""
    
    def create_small_model(self):
        """创建一个小的测试模型"""
        vocab_size = 1000
        model = SimpleTransformer(
            vocab_size=vocab_size,
            d_model=64,
            nhead=2,
            num_layers=2,
            dim_feedforward=128,
            dropout=0.0,
            max_len=128,
            mode='modern',
        )
        return model
    
    def test_quantize_model(self):
        """测试量化整个模型"""
        model = self.create_small_model()
        model.eval()
        
        config = QuantizationConfig(bits=4, group_size=64)
        
        # 量化模型
        quantized_model = quantize_model(model, config=config)
        
        # 测试前向传播
        input_ids = torch.randint(0, 1000, (2, 16))
        
        with torch.no_grad():
            original_output, _ = model(input_ids)
            quantized_output, _ = quantized_model(input_ids)
        
        # 检查输出形状
        assert quantized_output.shape == original_output.shape
        
        # 检查输出误差
        error = torch.abs(quantized_output - original_output).mean()
        print(f"Model quantization mean error: {error.item():.6f}")
        assert error.item() < 1.0, f"Model quantization error too high: {error.item()}"
    
    def test_collect_activations(self):
        """测试激活值收集"""
        model = self.create_small_model()
        model.eval()
        
        # 创建 dummy 数据加载器
        class DummyDataset(torch.utils.data.Dataset):
            def __init__(self):
                self.data = [torch.randint(0, 1000, (16,)) for _ in range(8)]
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx]
        
        dataloader = torch.utils.data.DataLoader(DummyDataset(), batch_size=2)
        
        # 收集激活
        activations = collect_activations(model, dataloader, num_samples=4)
        
        # 检查是否收集到激活
        assert len(activations) > 0
        print(f"Collected activations from {len(activations)} layers")
        
        # 检查每个层的激活数量
        for layer_name, acts in activations.items():
            assert len(acts) <= 4
            assert acts[0].dim() >= 2
    
    def test_activation_aware_quantization(self):
        """测试激活感知的模型量化"""
        model = self.create_small_model()
        model.eval()
        
        # 创建 dummy 数据加载器
        class DummyDataset(torch.utils.data.Dataset):
            def __init__(self):
                self.data = [torch.randint(0, 1000, (16,)) for _ in range(8)]
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx]
        
        dataloader = torch.utils.data.DataLoader(DummyDataset(), batch_size=2)
        
        # 收集激活
        activations = collect_activations(model, dataloader, num_samples=4)
        
        # 量化模型（使用激活感知）
        config = QuantizationConfig(bits=4, group_size=64)
        quantized_model = quantize_model(model, config=config, input_activations=activations)
        
        # 测试前向传播
        input_ids = torch.randint(0, 1000, (2, 16))
        
        with torch.no_grad():
            original_output, _ = model(input_ids)
            quantized_output, _ = quantized_model(input_ids)
        
        # 检查输出形状
        assert quantized_output.shape == original_output.shape
        
        # 检查输出误差
        error = torch.abs(quantized_output - original_output).mean()
        print(f"Activation-aware quantization mean error: {error.item():.6f}")
        assert error.item() < 1.0


class TestQuantizationConfig:
    """测试量化配置"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = QuantizationConfig()
        assert config.bits == 4
        assert config.group_size == 128
        assert config.symmetric is True
        assert config.has_zero_point is False
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = QuantizationConfig(
            bits=8,
            group_size=64,
            symmetric=False,
            has_zero_point=True,
        )
        assert config.bits == 8
        assert config.group_size == 64
        assert config.symmetric is False
        assert config.has_zero_point is True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
