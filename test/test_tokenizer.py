#!/usr/bin/env python3
"""
测试分词器功能的简单脚本
"""
import os
import logging
from pathlib import Path
from tokenizer import ClassicalTokenizer, logger

# 设置日志级别为INFO以查看详细输出
logger.setLevel(logging.INFO)

def main():
    print("开始测试分词器...")
    
    # 创建必要的目录结构
    Path("dataset/dictionaries").mkdir(parents=True, exist_ok=True)
    
    # 创建一个简单的测试词典
    dict_path = "dataset/dictionaries/test_dict.txt"
    if not os.path.exists(dict_path):
        with open(dict_path, "w", encoding="utf-8") as f:
            f.write("春风 n\n")
            f.write("杨柳 n\n")
            f.write("千里 n\n")
            f.write("江山 n\n")
            f.write("如画 v\n")
        print(f"创建测试词典: {dict_path}")
    
    # 创建测试语料
    corpus_path = "dataset/test_corpus.txt"
    if not os.path.exists(corpus_path):
        with open(corpus_path, "w", encoding="utf-8") as f:
            f.write("春风又绿江南岸，明月何时照我还。\n")
            f.write("千里江山如画，风景独好。\n")
            f.write("落霞与孤鹜齐飞，秋水共长天一色。\n")
        print(f"创建测试语料: {corpus_path}")
    
    # 初始化分词器
    print("初始化分词器...")
    tokenizer = ClassicalTokenizer(
        vocab_size=1000,  # 小词表用于快速测试
        special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[SEP]", "[MASK]"],
        dictionary_path=dict_path
    )
    
    # 训练分词器
    print("训练分词器...")
    training_files = [corpus_path]
    success = tokenizer.train(training_files)
    
    if success:
        print("分词器训练成功!")
        
        # 保存分词器 - 直接使用底层tokenizer的save方法
        save_path = "test_tokenizer.json"
        try:
            tokenizer.tokenizer.save(save_path)
            print(f"分词器已保存到 {save_path}")
        except Exception as e:
            print(f"保存分词器时出错: {e}")
        
        # 测试分词
        test_text = "春风又绿江南岸"
        print(f"\n测试分词: '{test_text}'")
        tokens = tokenizer.encode(test_text)
        print(f"分词结果: {tokens}")
        
        # 测试解码
        decoded = tokenizer.decode(tokens)
        print(f"解码结果: '{decoded}'")
    else:
        print("分词器训练失败!")

if __name__ == "__main__":
    main()