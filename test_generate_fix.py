#!/usr/bin/env python3
"""
测试 generate.py 的修复是否正常工作
"""
import sys
from pathlib import Path

# 确保我们可以导入 generate 模块
sys.path.insert(0, str(Path(__file__).parent))

from generate import load_tokenizer, load_model, generate_text
import torch

def test_tokenizer():
    """测试 tokenizer 的功能"""
    print("=" * 60)
    print("测试 tokenizer 功能")
    print("=" * 60)
    
    tokenizer_path = "tokenizer.json"
    tokenizer = load_tokenizer(tokenizer_path)
    
    if tokenizer is None:
        print("❌ Tokenizer 加载失败")
        return False
    
    print(f"✅ Tokenizer 加载成功，词汇表大小: {len(tokenizer.token_to_id)}")
    
    # 测试编码和解码
    test_text = "春风又绿江南岸"
    print(f"\n测试文本: '{test_text}'")
    
    try:
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
        
        return True
    except Exception as e:
        print(f"❌ 编码解码测试失败: {e}")
        return False


def test_trie_structure():
    """测试前缀树结构"""
    print("\n" + "=" * 60)
    print("测试前缀树 (Trie) 结构")
    print("=" * 60)
    
    tokenizer_path = "tokenizer.json"
    tokenizer = load_tokenizer(tokenizer_path)
    
    if tokenizer is None:
        print("❌ Tokenizer 加载失败")
        return False
    
    # 检查是否有 trie 属性
    if hasattr(tokenizer, 'trie'):
        print("✅ 前缀树结构已创建")
        
        # 测试一些简单的查找
        test_texts = ["0", "1", "春"]  # 从 tokenizer.json 中的已知词汇
        for text in test_texts:
            match, length = tokenizer.trie.find_longest_match(text, 0)
            if match:
                print(f"✅ 前缀树成功找到匹配: '{match}' (长度: {length})")
        
        return True
    else:
        print("❌ 前缀树结构未找到")
        return False


def test_manual_model_behavior():
    """手动模拟模型行为，测试 generate_text 函数"""
    print("\n" + "=" * 60)
    print("测试 generate_text 函数")
    print("=" * 60)
    
    # 创建一个 tokenizer
    tokenizer_path = "tokenizer.json"
    tokenizer = load_tokenizer(tokenizer_path)
    
    if tokenizer is None:
        print("❌ Tokenizer 加载失败")
        return False
    
    # 创建一个简单的模拟模型，返回预期的格式
    class MockModel(torch.nn.Module):
        def __init__(self, vocab_size):
            super().__init__()
            self.vocab_size = vocab_size
        
        def forward(self, input_ids):
            # 模拟模型返回 (output, present) 格式
            batch_size, seq_len = input_ids.shape
            output = torch.randn(batch_size, seq_len, self.vocab_size)
            present = None  # 模拟无缓存情况
            return output, present
    
    # 创建 mock 模型
    vocab_size = len(tokenizer.token_to_id)
    model = MockModel(vocab_size)
    
    # 测试 generate_text 函数
    try:
        # 用小的 max_length 快速测试
        result = generate_text(
            model,
            tokenizer,
            "测试提示",
            max_length=3,
            top_k=50,
            temperature=0.7
        )
        print(f"✅ generate_text 函数成功运行")
        print(f"生成结果: {result}")
        return True
    except Exception as e:
        print(f"❌ generate_text 函数失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_top_k_bounds_check():
    """测试 top_k 边界检查"""
    print("\n" + "=" * 60)
    print("测试 top_k 边界检查")
    print("=" * 60)
    
    # 直接检查 generate_text 函数的实现
    import inspect
    source = inspect.getsource(generate_text)
    
    if "min(top_k, vocab_size)" in source:
        print("✅ 找到 top_k = min(top_k, vocab_size) 边界检查")
    else:
        print("❌ 未找到 top_k 边界检查")
        return False
    
    # 测试边界情况
    tokenizer_path = "tokenizer.json"
    tokenizer = load_tokenizer(tokenizer_path)
    
    if tokenizer is None:
        print("❌ Tokenizer 加载失败")
        return False
    
    vocab_size = len(tokenizer.token_to_id)
    
    # 创建一个简单的模拟模型，返回预期的格式
    class MockModel(torch.nn.Module):
        def __init__(self, vocab_size):
            super().__init__()
            self.vocab_size = vocab_size
        
        def forward(self, input_ids):
            batch_size, seq_len = input_ids.shape
            output = torch.randn(batch_size, seq_len, self.vocab_size)
            present = None
            return output, present
    
    model = MockModel(vocab_size)
    
    # 测试一个很大的 top_k 值
    try:
        large_top_k = vocab_size + 100  # 超过词汇表大小
        result = generate_text(
            model,
            tokenizer,
            "测试",
            max_length=2,
            top_k=large_top_k
        )
        print(f"✅ 使用超出词汇表大小的 top_k 值({large_top_k}) 成功运行")
        return True
    except Exception as e:
        print(f"❌ 使用大 top_k 值失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("generate.py 修复验证测试")
    print("=" * 60 + "\n")
    
    results = {
        "Tokenizer 功能": test_tokenizer(),
        "前缀树结构": test_trie_structure(),
        "generate_text 函数": test_manual_model_behavior(),
        "top_k 边界检查": test_top_k_bounds_check()
    }
    
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ 所有测试通过！修复成功！")
    else:
        print("❌ 部分测试失败，请检查。")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
