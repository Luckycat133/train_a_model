"""
灵猫墨韵 AWQ 量化实现
激活感知权重量化 (Activation-aware Weight Quantization)

主要功能：
- 支持 4位 和 8位 量化
- 激活感知权重量化
- 权重分组量化
- 完整的量化/反量化流程
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class QuantizationConfig:
    """量化配置"""
    bits: int = 4  # 量化位数，支持 4 或 8
    group_size: int = 128  # 量化分组大小
    has_zero_point: bool = False  # 是否使用零点
    symmetric: bool = True  # 是否对称量化


class AWQLinear(nn.Module):
    """
    AWQ 量化线性层
    
    这是一个轻量级实现，专注于：
    1. 基础 AWQ 算法实现
    2. 激活感知权重量化
    3. 支持 4位/8位 量化
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: Optional[QuantizationConfig] = None,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config or QuantizationConfig()
        
        # 验证配置
        assert self.config.bits in [4, 8], "Only 4-bit and 8-bit quantization supported"
        assert self.config.group_size > 0, "Group size must be positive"
        
        # 计算量化参数
        self.group_size = self.config.group_size
        self.bits = self.config.bits
        self.num_groups = (in_features + self.group_size - 1) // self.group_size
        
        # 注册量化权重和缩放因子的 buffer
        # 注意：在实际部署中，这些会被进一步压缩存储
        self.register_buffer('qweight', torch.empty((out_features, in_features), dtype=torch.int8))
        self.register_buffer('scales', torch.empty((out_features, self.num_groups), dtype=torch.float32))
        
        if self.config.has_zero_point:
            self.register_buffer('zero_points', torch.empty((out_features, self.num_groups), dtype=torch.int8))
        else:
            self.zero_points = None
        
        if bias:
            self.register_buffer('bias', torch.empty(out_features, dtype=torch.float32))
        else:
            self.bias = None
        
        # 标记是否已量化
        self._quantized = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播 - 在线反量化并计算"""
        if not self._quantized:
            raise RuntimeError("AWQLinear layer not quantized yet. Call quantize() first.")
        
        # 反量化权重
        weight = self._dequantize_weight()
        
        # 确保数据类型匹配
        weight = weight.to(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        
        return F.linear(x, weight, bias)
    
    def _dequantize_weight(self) -> torch.Tensor:
        """反量化权重"""
        out_features, in_features = self.qweight.shape
        
        # 将权重重塑为分组格式
        qweight_reshaped = self.qweight.view(out_features, self.num_groups, self.group_size)
        scales_reshaped = self.scales.unsqueeze(-1)
        
        if self.config.has_zero_point:
            zero_points_reshaped = self.zero_points.unsqueeze(-1)
            weight = (qweight_reshaped.to(torch.float32) - zero_points_reshaped.to(torch.float32)) * scales_reshaped
        else:
            weight = qweight_reshaped.to(torch.float32) * scales_reshaped
        
        # 重塑回原始形状
        weight = weight.view(out_features, in_features)
        return weight
    
    @torch.no_grad()
    def quantize(
        self,
        weight: torch.Tensor,
        input_activations: Optional[List[torch.Tensor]] = None,
    ):
        """
        执行 AWQ 量化
        
        Args:
            weight: 原始全精度权重 [out_features, in_features]
            input_activations: 激活值样本，用于激活感知量化
        """
        out_features, in_features = weight.shape
        
        # 计算缩放因子
        if input_activations is not None and len(input_activations) > 0:
            scales = self._compute_activation_aware_scales(weight, input_activations)
        else:
            scales = self._compute_basic_scales(weight)
        
        # 量化权重
        qweight, zero_points = self._quantize_weight_core(weight, scales)
        
        # 保存量化结果
        self.qweight.copy_(qweight.to(torch.int8))
        self.scales.copy_(scales.to(torch.float32))
        
        if self.config.has_zero_point and zero_points is not None:
            self.zero_points.copy_(zero_points.to(torch.int8))
        
        self._quantized = True
    
    def _compute_basic_scales(self, weight: torch.Tensor) -> torch.Tensor:
        """基础缩放因子计算 - 仅使用权重"""
        out_features, in_features = weight.shape
        
        # 按组重塑权重
        weight_reshaped = weight.view(out_features, self.num_groups, self.group_size)
        
        # 计算量化边界
        qmax = 2 ** (self.bits - 1) - 1
        qmin = -2 ** (self.bits - 1)
        
        # 计算每组的缩放因子
        if self.config.symmetric:
            # 对称量化：scale = max(abs(w)) / qmax
            max_val = weight_reshaped.abs().max(dim=-1)[0]
            scales = max_val / qmax
        else:
            # 非对称量化：scale = (max - min) / (qmax - qmin)
            min_val = weight_reshaped.min(dim=-1)[0]
            max_val = weight_reshaped.max(dim=-1)[0]
            scales = (max_val - min_val) / (qmax - qmin)
        
        # 避免除以零
        scales = torch.clamp(scales, min=1e-8)
        
        return scales
    
    def _compute_activation_aware_scales(
        self,
        weight: torch.Tensor,
        input_activations: List[torch.Tensor],
    ) -> torch.Tensor:
        """激活感知缩放因子计算 - AWQ 的核心"""
        out_features, in_features = weight.shape
        
        # 计算激活的统计信息
        # AWQ 的核心思想：对激活值大的维度分配更多量化精度
        acts_list = []
        for act in input_activations:
            # 展平批次和序列维度
            act_flat = act.view(-1, in_features)
            acts_list.append(act_flat)
        
        all_acts = torch.cat(acts_list, dim=0)
        
        # 计算每个输入维度的重要性（平均激活幅度）
        act_importance = all_acts.abs().mean(dim=0)
        
        # 计算带权重的缩放因子
        # 对激活大的维度，缩放因子更小（保留更多精度）
        weighted_weight = weight * act_importance.unsqueeze(0)
        
        # 使用权重后的权重计算缩放因子
        weight_reshaped = weighted_weight.view(out_features, self.num_groups, self.group_size)
        
        qmax = 2 ** (self.bits - 1) - 1
        qmin = -2 ** (self.bits - 1)
        
        if self.config.symmetric:
            max_val = weight_reshaped.abs().max(dim=-1)[0]
            scales = max_val / qmax
        else:
            min_val = weight_reshaped.min(dim=-1)[0]
            max_val = weight_reshaped.max(dim=-1)[0]
            scales = (max_val - min_val) / (qmax - qmin)
        
        scales = torch.clamp(scales, min=1e-8)
        
        return scales
    
    def _quantize_weight_core(
        self,
        weight: torch.Tensor,
        scales: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """核心量化逻辑"""
        out_features, in_features = weight.shape
        
        # 重塑权重和缩放因子
        weight_reshaped = weight.view(out_features, self.num_groups, self.group_size)
        scales_reshaped = scales.unsqueeze(-1)
        
        qmax = 2 ** (self.bits - 1) - 1
        qmin = -2 ** (self.bits - 1)
        
        if self.config.symmetric:
            # 对称量化
            qweight = torch.round(weight_reshaped / scales_reshaped)
            zero_points = None
        else:
            # 非对称量化 - 计算零点
            min_val = weight_reshaped.min(dim=-1, keepdim=True)[0]
            zero_points = torch.round(-min_val / scales_reshaped).squeeze(-1)
            zero_points = torch.clamp(zero_points, qmin, qmax)
            
            qweight = torch.round(weight_reshaped / scales_reshaped) + zero_points.unsqueeze(-1)
        
        # 裁剪到量化范围
        qweight = torch.clamp(qweight, qmin, qmax)
        
        # 重塑回原始形状
        qweight = qweight.view(out_features, in_features)
        
        return qweight, zero_points
    
    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        config: Optional[QuantizationConfig] = None,
        input_activations: Optional[List[torch.Tensor]] = None,
    ) -> 'AWQLinear':
        """从现有线性层创建量化层"""
        awq_linear = cls(
            linear.in_features,
            linear.out_features,
            config=config,
            bias=linear.bias is not None,
        )
        
        # 复制偏置
        if linear.bias is not None:
            awq_linear.bias.copy_(linear.bias.to(torch.float32))
        
        # 执行量化
        awq_linear.quantize(linear.weight, input_activations)
        
        return awq_linear


@torch.no_grad()
def quantize_model(
    model: nn.Module,
    config: Optional[QuantizationConfig] = None,
    input_activations: Optional[dict] = None,
    layers_to_quantize: Optional[List[str]] = None,
) -> nn.Module:
    """
    量化整个模型
    
    Args:
        model: 要量化的模型
        config: 量化配置
        input_activations: 每层的激活值字典 {layer_name: [activations]}
        layers_to_quantize: 要量化的层名称列表，None表示所有线性层
    
    Returns:
        量化后的模型
    """
    config = config or QuantizationConfig()
    
    def _replace_module(module: nn.Module, prefix: str = ''):
        """递归替换模块中的线性层"""
        for name, child in module.named_children():
            full_name = f'{prefix}.{name}' if prefix else name
            
            # 检查是否需要量化这个层
            should_quantize = (
                isinstance(child, nn.Linear) and
                (layers_to_quantize is None or any(ln in full_name for ln in layers_to_quantize))
            )
            
            if should_quantize:
                # 获取该层的激活值
                acts = input_activations.get(full_name, None) if input_activations else None
                
                # 替换为量化层
                awq_linear = AWQLinear.from_linear(child, config, acts)
                setattr(module, name, awq_linear)
            else:
                # 递归处理子模块
                _replace_module(child, full_name)
    
    # 创建模型副本
    import copy
    quantized_model = copy.deepcopy(model)
    
    # 替换线性层
    _replace_module(quantized_model)
    
    return quantized_model


@torch.no_grad()
def collect_activations(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    num_samples: int = 32,
    device: Optional[torch.device] = None,
) -> dict:
    """
    收集模型各层的激活值，用于激活感知量化
    
    Args:
        model: 模型
        dataloader: 数据加载器
        num_samples: 收集的样本数
        device: 设备
    
    Returns:
        激活值字典 {layer_name: [activations]}
    """
    activations = {}
    hooks = []
    
    def make_hook(name):
        def hook(module, input, output):
            if name not in activations:
                activations[name] = []
            if len(activations[name]) < num_samples:
                # 保存输出激活值（通常是这个层的输出）
                # 注意：也可以保存 input[0]，取决于我们要捕捉什么
                if isinstance(output, tuple):
                    # 如果是 tuple，取第一个元素
                    act = output[0]
                else:
                    act = output
                activations[name].append(act.detach().cpu())
        return hook
    
    # 为所有线性层注册钩子
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hook = module.register_forward_hook(make_hook(name))
            hooks.append(hook)
    
    # 前向传播收集激活
    model.eval()
    samples_collected = 0
    
    with torch.no_grad():
        for batch in dataloader:
            if samples_collected >= num_samples:
                break
            
            # 处理批次
            if isinstance(batch, dict):
                input_ids = batch.get('input_ids')
                if input_ids is not None:
                    if device:
                        input_ids = input_ids.to(device)
                    model(input_ids)
            elif isinstance(batch, torch.Tensor):
                if device:
                    batch = batch.to(device)
                model(batch)
            elif isinstance(batch, list) and len(batch) > 0:
                # 处理列表批次
                if device:
                    batch = [b.to(device) for b in batch]
                model(batch[0])
            
            samples_collected += 1
    
    # 移除钩子
    for hook in hooks:
        hook.remove()
    
    return activations
