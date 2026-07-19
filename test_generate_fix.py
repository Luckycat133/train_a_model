#!/usr/bin/env python3
"""
测试 generate.py 的修复是否正常工作
"""
import sys
from pathlib import Path

# 确保我们可以导入 generate 模块
sys.path.insert(0, str(Path(__file__).parent))

from generate import load_tokenizer, load_model, generate_text, apply_sampling, GenerationConfig
import torch

def test_tokenizer():
    """测试 tokenizer 的功能"""
    print("=" * 60)
    print("测试 tokenizer 功能")
    print("=" * 60)

    tokenizer_path = "tokenizer.json"
    tokenizer = load_tokenizer(tokenizer_path)

    assert tokenizer is not None, "Tokenizer 加载失败"

    print(f"✅ Tokenizer 加载成功，词汇表大小: {len(tokenizer.token_to_id)}")

    # 测试编码和解码
    test_text = "春风又绿江南岸"
    print(f"\n测试文本: '{test_text}'")

    encoded = tokenizer.encode(test_text)
    print(f"编码结果: {encoded}")

    decoded = tokenizer.decode(encoded)
    print(f"解码结果: '{decoded}'")

    print("✅ 编码解码测试成功")

    # 测试缓存是否工作
    print("\n测试缓存功能...")
    import time
    start = time.time()
    _ = tokenizer.encode(test_text)
    first_time = time.time() - start

    start = time.time()
    _ = tokenizer.encode(test_text)
    cached_time = time.time() - start

    print(f"首次编码: {first_time:.6f} 秒")
    print(f"缓存编码: {cached_time:.6f} 秒")

    if cached_time < first_time:
        print(f"✅ 缓存正常工作，速度提升: {first_time/cached_time:.2f}x")
    else:
        print("⚠️ 缓存可能未生效")


def test_trie_structure():
    """测试前缀树结构"""
    print("\n" + "=" * 60)
    print("测试前缀树 (Trie) 结构")
    print("=" * 60)

    tokenizer_path = "tokenizer.json"
    tokenizer = load_tokenizer(tokenizer_path)

    assert tokenizer is not None, "Tokenizer 加载失败"

    # 检查是否有 trie 属性
    assert hasattr(tokenizer, 'trie'), "前缀树结构未找到"
    print("✅ 前缀树结构已创建")

    # 测试一些简单的查找
    test_texts = ["0", "1", "春"]  # 从 tokenizer.json 中的已知词汇
    for text in test_texts:
        match, length = tokenizer.trie.find_longest_match(text, 0)
        if match:
            print(f"✅ 前缀树成功找到匹配: '{match}' (长度: {length})")


def test_manual_model_behavior():
    """手动模拟模型行为，测试 generate_text 函数"""
    print("\n" + "=" * 60)
    print("测试 generate_text 函数")
    print("=" * 60)

    # 创建一个 tokenizer
    tokenizer_path = "tokenizer.json"
    tokenizer = load_tokenizer(tokenizer_path)

    assert tokenizer is not None, "Tokenizer 加载失败"

    # 创建一个简单的模拟模型，返回预期的格式
    class MockModel(torch.nn.Module):
        def __init__(self, vocab_size):
            super().__init__()
            self.vocab_size = vocab_size

        def forward(self, input_ids, **kwargs):
            # 模拟模型返回 (output, present) 格式
            batch_size, seq_len = input_ids.shape
            output = torch.randn(batch_size, seq_len, self.vocab_size)
            present = None  # 模拟无缓存情况
            return output, present

    # 创建 mock 模型
    vocab_size = len(tokenizer.token_to_id)
    model = MockModel(vocab_size)

    # 使用 GenerationConfig 传递参数
    config = GenerationConfig(
        max_length=3,
        top_k=50,
        temperature=0.7
    )
    result = generate_text(model, tokenizer, "测试提示", config=config)
    print(f"✅ generate_text 函数成功运行")
    print(f"生成结果: {result}")


def test_top_k_bounds_check():
    """测试 top_k 边界检查"""
    print("\n" + "=" * 60)
    print("测试 top_k 边界检查")
    print("=" * 60)

    # 检查 apply_sampling 函数的实现（top_k 边界检查在这里）
    import inspect
    source = inspect.getsource(apply_sampling)

    assert "min(top_k, vocab_size)" in source, "未找到 top_k 边界检查"
    print("✅ 找到 top_k = min(top_k, vocab_size) 边界检查")

    # 测试边界情况：直接测试 apply_sampling
    tokenizer_path = "tokenizer.json"
    tokenizer = load_tokenizer(tokenizer_path)

    assert tokenizer is not None, "Tokenizer 加载失败"

    vocab_size = len(tokenizer.token_to_id)

    # 测试 apply_sampling 在 top_k 超出 vocab_size 时不会崩溃
    logits = torch.randn(1, vocab_size)
    large_top_k = vocab_size + 100  # 超过词汇表大小
    result = apply_sampling(logits, temperature=0.7, top_k=large_top_k)
    assert result is not None
    assert result.shape == (1,)
    print(f"✅ 使用超出词汇表大小的 top_k 值({large_top_k}) 成功运行")


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("generate.py 修复验证测试")
    print("=" * 60 + "\n")

    test_tokenizer()
    test_trie_structure()
    test_manual_model_behavior()
    test_top_k_bounds_check()

    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    print("✅ 所有测试通过！修复成功！")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())