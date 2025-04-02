# 灵猫墨韵系统架构设计

<div align="center">

![系统架构](https://img.shields.io/badge/🏗️_系统架构-设计文档-9370DB)

*灵猫墨韵系统的整体架构设计、模块组织及核心设计决策*

</div>

## 📑 目录

- [概述](#概述)
- [设计原则](#设计原则)
- [整体架构](#整体架构)
- [核心模块](#核心模块)
  - [训练模块](#训练模块)
  - [推理模块](#推理模块)
  - [数据处理模块](#数据处理模块)
  - [分词器模块](#分词器模块)
- [数据流图](#数据流图)
- [技术选型](#技术选型)
- [性能优化设计](#性能优化设计)
- [扩展性设计](#扩展性设计)
- [关键设计决策](#关键设计决策)

## 概述

灵猫墨韵是一个轻量级的古风文言语言模型系统，专注于中国古典文学文本的生成。本文档详细说明系统的整体架构设计、各模块间的交互方式、数据流转过程以及核心设计决策。

> 💡 **设计目标**：打造一个模型体积小、资源占用低、生成质量高的古典文学特化语言模型，适合在普通消费级硬件上运行。

## 设计原则

灵猫墨韵的设计遵循以下核心原则：

1. **轻量化优先**：所有设计决策优先考虑模型轻量化和计算效率
2. **领域专精**：专注于古典文学领域，提高特定场景下的表现
3. **可扩展性**：保持模块化设计，便于功能扩展和替换
4. **易用性**：简化使用流程，提供友好的接口和详尽的文档
5. **资源适应性**：适应不同硬件环境，从高端GPU到普通CPU均可高效运行

## 整体架构

灵猫墨韵采用模块化架构设计，分为四个主要层次：

![系统架构图](../../assets/images/system_architecture.png)

### 架构层次

1. **数据层**：负责数据获取、预处理和转换
2. **模型层**：包含核心模型定义、训练和推理逻辑
3. **服务层**：封装模型能力，提供调用接口
4. **应用层**：面向用户的应用和工具

### 模块交互

模块间通过明确定义的接口和数据格式进行交互，保证系统的解耦性和可维护性。主要交互路径：

```
数据源 → 数据处理模块 → 分词器 → 训练模块 → 模型存储
                                  ↓
                用户输入 → 分词器 → 推理模块 → 输出格式化 → 用户
```

## 核心模块

### 训练模块

训练模块负责模型的训练和优化，主要组件：

- **模型定义**：基于Transformer架构的自回归语言模型
- **训练循环**：包含梯度计算、参数更新和验证逻辑
- **优化器**：支持Adam、AdamW等优化器，具有学习率调度功能
- **断点续训**：支持从检查点恢复训练的功能
- **分布式训练**：支持多GPU并行训练

```python
# 简化的训练模块结构
class TrainModule:
    def __init__(self, config):
        self.model = MoYunModel(config)
        self.optimizer = AdamW(self.model.parameters(), lr=config.learning_rate)
        self.scheduler = get_scheduler(self.optimizer, config)
    
    def train(self, dataloader, epochs):
        # 训练逻辑
        pass
        
    def save_checkpoint(self, path):
        # 保存模型检查点
        pass
        
    def load_checkpoint(self, path):
        # 加载模型检查点
        pass
```

### 推理模块

推理模块负责使用训练好的模型生成文本，主要组件：

- **模型加载**：加载训练好的模型权重
- **文本生成**：使用自回归方式生成文本
- **采样策略**：支持多种采样方法（温度采样、核采样等）
- **批量生成**：支持批处理以提高吞吐量
- **格式控制**：控制生成文本的格式和风格

```python
# 简化的推理模块结构
class InferenceModule:
    def __init__(self, model_path, config):
        self.model = MoYunModel.from_pretrained(model_path)
        self.tokenizer = MoYunTokenizer.from_pretrained(model_path)
        self.config = config
    
    def generate(self, prompt, max_length=100, temperature=0.7):
        # 文本生成逻辑
        pass
        
    def batch_generate(self, prompts, **kwargs):
        # 批量生成逻辑
        pass
```

### 数据处理模块

数据处理模块负责原始文本的清洗、格式化和转换，主要组件：

- **数据加载**：支持多种格式的数据源加载
- **数据清洗**：去除噪声、标准化格式
- **数据增强**：通过规则或模型生成增强数据
- **数据转换**：将原始文本转换为模型可用格式

```python
# 简化的数据处理模块结构
class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.tokenizer = MoYunTokenizer.from_pretrained(config.tokenizer_path)
    
    def load_data(self, data_path):
        # 加载数据
        pass
        
    def clean_text(self, text):
        # 文本清洗
        pass
        
    def tokenize(self, texts):
        # 文本分词
        pass
        
    def create_dataloader(self, tokenized_data, batch_size):
        # 创建数据加载器
        pass
```

### 分词器模块

分词器模块负责文本与模型输入的相互转换，主要组件：

- **词表管理**：维护模型词汇表
- **分词逻辑**：将文本切分为标记
- **编码解码**：文本与标记ID的相互转换
- **特殊标记**：处理特殊标记（如开始、结束标记）

```python
# 简化的分词器模块结构
class MoYunTokenizer:
    def __init__(self, vocab_file):
        self.vocab = self.load_vocab(vocab_file)
        self.ids_to_tokens = {i: t for t, i in self.vocab.items()}
    
    def tokenize(self, text):
        # 文本分词
        pass
        
    def encode(self, text, add_special_tokens=True):
        # 编码为ID
        pass
        
    def decode(self, ids):
        # ID解码为文本
        pass
        
    @classmethod
    def from_pretrained(cls, path):
        # 从预训练路径加载
        pass
```

## 数据流图

### 训练流程数据流

```
原始文本 → 数据清洗 → 数据格式化 → 分词器编码 → 创建数据批次 → 模型训练 → 验证评估 → 模型保存
```

### 推理流程数据流

```
用户输入 → 提示格式化 → 分词器编码 → 模型推理 → 标记采样 → 分词器解码 → 后处理 → 最终输出
```

## 技术选型

灵猫墨韵系统的主要技术选型及原因：

| 组件 | 技术选择 | 选择理由 |
|------|----------|---------|
| 模型架构 | Transformer Decoder-only | 平衡生成质量和模型大小 |
| 优化器 | AdamW | 对NLP任务效果更好，收敛更快 |
| 学习率调度 | Cosine with warmup | 减少过拟合，提高稳定性 |
| 分词方式 | BPE (Byte-Pair Encoding) | 适合处理中文和古文混合文本 |
| 训练框架 | PyTorch | 灵活性好，研究友好 |
| 推理加速 | TorchScript/ONNX | 提高推理速度，降低资源消耗 |
| 量化技术 | 动态量化/静态量化 | 减小模型体积，提高推理速度 |

## 性能优化设计

灵猫墨韵针对性能进行了多方面优化：

1. **模型轻量化**
   - 使用知识蒸馏减小模型规模
   - 精心设计模型层数和隐藏维度
   - 使用参数共享减少总参数量

2. **推理优化**
   - 使用KV缓存减少重复计算
   - 支持批处理以提高吞吐量
   - 提供多种精度选项(FP32/FP16/INT8)

3. **内存优化**
   - 梯度检查点技术减少训练内存使用
   - 支持模型分片加载大模型
   - 优化Attention计算减少内存占用

## 扩展性设计

灵猫墨韵设计了良好的扩展接口：

1. **模型扩展**
   - 支持不同大小的模型变体(nano/mini/base/large)
   - 插件化设计，便于添加新功能

2. **训练扩展**
   - 支持自定义数据集和训练目标
   - 可扩展的评估指标体系

3. **部署扩展**
   - 支持多种部署环境(本地/服务器/移动设备)
   - 提供多种接口形式(Python API/REST API/CLI)

## 关键设计决策

### 决策1: 专注于轻量化模型而非大型通用模型

**背景**：当前大语言模型普遍体积庞大，需要强大硬件支持。

**方案**：
- 选项A: 构建体积小、任务专精的模型
- 选项B: 构建中等规模通用模型

**决策**：选择方案A，构建专注于古典文学的轻量级模型。

**理由**：
- 极大降低硬件门槛，增加可用性
- 在特定领域（古典文学）可以达到较高质量
- 避免与大厂通用模型直接竞争，寻找差异化价值

### 决策2: 使用BPE分词而非字符级分词

**背景**：中文模型常用字符级分词，但古文中有特殊用法。

**方案**：
- 选项A: 字符级分词（每个汉字作为一个标记）
- 选项B: BPE分词（基于频率的子词分词）

**决策**：选择方案B，使用定制的BPE分词器。

**理由**：
- 能更好地捕捉古文中的固定搭配和常用词组
- 减小序列长度，降低计算复杂度
- 对罕见字和术语有更好的处理能力

### 决策3: 支持多种硬件平台而非仅优化GPU

**背景**：用户硬件环境多样，从高端GPU到普通CPU。

**方案**：
- 选项A: 仅优化高端GPU性能
- 选项B: 支持多种硬件平台，包括CPU和移动设备

**决策**：选择方案B，全面支持各种硬件平台。

**理由**：
- 大幅扩大潜在用户群体
- 符合轻量化的核心设计理念
- 提供更灵活的部署选择

---

<div align="center">

[📚 文档目录](../../summary.md) | [📝 项目概述](../../project_overview.md) | [🧠 模块设计](./module_design.md)

</div>

*最后更新时间: 2025-03-20* 