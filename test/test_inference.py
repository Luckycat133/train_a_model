#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
推理模块测试脚本
用于测试 generate.py 的功能
"""

import os
import sys
import json
import tempfile
import pytest
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).resolve().parent.parent))

# 创建模拟的测试环境
def create_mock_tokenizer_file(file_path):
    """创建一个模拟的 tokenizer 文件"""
    mock_vocab = {
        "<unk>": 0,
        "<eos>": 1,
        "床前": 2,
        "明月光": 3,
        "疑是": 4,
        "地上": 5,
        "霜": 6,
        "举头": 7,
        "望": 8,
        "明月": 9,
        "低头": 10,
        "思": 11,
        "故乡": 12
    }
    
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump({"vocab": mock_vocab}, f, ensure_ascii=False)
    
    return file_path

# 测试 GenerationConfig
def test_generation_config():
    """测试 GenerationConfig 数据类"""
    from generate import GenerationConfig
    
    # 测试默认配置
    config = GenerationConfig()
    assert config.max_length == 100
    assert config.temperature == 0.7
    assert config.top_k == 50
    assert config.top_p == 0.9
    assert config.use_cache is True
    
    # 测试自定义配置
    config = GenerationConfig(
        max_length=50,
        temperature=0.5,
        top_k=30,
        top_p=0.8,
        max_new_tokens=20
    )
    assert config.max_length == 50
    assert config.temperature == 0.5
    assert config.top_k == 30
    assert config.top_p == 0.8
    assert config.max_new_tokens == 20

# 测试 SimpleTokenizer 类
def test_simple_tokenizer():
    """测试 SimpleTokenizer 类"""
    from generate import SimpleTokenizer
    
    # 创建测试用的词汇表
    mock_tokenizer_data = {
        "vocab": {
            "<unk>": 0,
            "<eos>": 1,
            "床前": 2,
            "明月光": 3,
            "疑是": 4,
            "地上": 5,
            "霜": 6
        }
    }
    
    tokenizer = SimpleTokenizer(mock_tokenizer_data)
    
    # 测试基本属性
    assert tokenizer.unk_token_id == 0
    assert tokenizer.eos_token_id == 1
    assert len(tokenizer.token_to_id) == 7
    
    # 测试编码
    tokens = tokenizer.encode("床前明月光")
    assert isinstance(tokens, list)
    assert len(tokens) > 0
    
    # 测试解码
    # 注意：由于我们使用的是模拟数据，这里只测试函数是否正常执行
    decoded = tokenizer.decode([2, 3])
    assert isinstance(decoded, str)

# 测试加载函数
def test_load_tokenizer():
    """测试 load_tokenizer 函数"""
    from generate import load_tokenizer
    
    # 创建临时文件
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", encoding="utf-8", delete=False) as f:
        mock_vocab = {
            "<unk>": 0,
            "<eos>": 1,
            "床前": 2,
            "明月光": 3
        }
        json.dump({"vocab": mock_vocab}, f)
        temp_file = f.name
    
    try:
        # 测试加载
        tokenizer = load_tokenizer(temp_file)
        assert tokenizer is not None
        assert hasattr(tokenizer, "encode")
        assert hasattr(tokenizer, "decode")
    finally:
        # 清理临时文件
        os.unlink(temp_file)

# 测试采样函数
def test_apply_sampling():
    """测试 apply_sampling 函数"""
    import torch
    from generate import apply_sampling
    
    # 创建模拟的 logits
    batch_size = 2
    vocab_size = 10
    logits = torch.randn(batch_size, vocab_size)
    
    # 测试温度为 0（贪婪采样）
    tokens = apply_sampling(logits, temperature=0.0)
    assert tokens.shape == (batch_size,)
    
    # 测试普通采样
    tokens = apply_sampling(logits, temperature=0.7)
    assert tokens.shape == (batch_size,)
    
    # 测试 top_k 采样
    tokens = apply_sampling(logits, temperature=0.7, top_k=3)
    assert tokens.shape == (batch_size,)
    
    # 测试 top_p 采样
    tokens = apply_sampling(logits, temperature=0.7, top_p=0.8)
    assert tokens.shape == (batch_size,)

# 测试 Trie 数据结构
def test_trie():
    """测试 Trie 数据结构"""
    from generate import Trie
    
    trie = Trie()
    
    # 插入一些词汇
    trie.insert("床前")
    trie.insert("明月光")
    trie.insert("明月")
    
    # 测试查找最长匹配
    match, length = trie.find_longest_match("床前明月光", 0)
    assert match == "床前"
    assert length == 2
    
    match, length = trie.find_longest_match("床前明月光", 2)
    assert match == "明月光"
    assert length == 3

# 测试设置日志
def test_setup_logger():
    """测试 setup_logger 函数"""
    from generate import setup_logger
    
    # 这个测试只是确保函数不会抛出异常
    logger = setup_logger()
    assert logger is not None

# 运行简单测试
def run_simple_tests():
    """运行简单的测试集合"""
    print("开始推理模块简单测试...")
    
    # 测试 GenerationConfig
    print("\n1. 测试 GenerationConfig...")
    test_generation_config()
    print("   ✓ 通过")
    
    # 测试 Trie
    print("\n2. 测试 Trie 数据结构...")
    test_trie()
    print("   ✓ 通过")
    
    # 测试 SimpleTokenizer
    print("\n3. 测试 SimpleTokenizer...")
    test_simple_tokenizer()
    print("   ✓ 通过")
    
    # 测试 apply_sampling
    print("\n4. 测试 apply_sampling...")
    test_apply_sampling()
    print("   ✓ 通过")
    
    # 测试 setup_logger
    print("\n5. 测试 setup_logger...")
    test_setup_logger()
    print("   ✓ 通过")
    
    # 测试 load_tokenizer
    print("\n6. 测试 load_tokenizer...")
    test_load_tokenizer()
    print("   ✓ 通过")
    
    print("\n所有简单测试通过! 🎉")
    return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="推理模块测试")
    parser.add_argument("--simple", action="store_true", help="运行简单测试")
    parser.add_argument("--pytest", action="store_true", help="运行 pytest 测试")
    
    args = parser.parse_args()
    
    if args.simple:
        run_simple_tests()
    elif args.pytest:
        # 使用 pytest 运行测试
        pytest.main([__file__, "-v"])
    else:
        # 默认运行简单测试
        run_simple_tests()
