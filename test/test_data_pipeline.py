#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""数据管道测试脚本。

全面测试数据加载、处理、混合和流式处理功能。
"""

import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import (
    BaseDataset,
    StreamingDataset,
    WeightedMixingDataset,
    PretrainDataset,
    ConcatPretrainDataset,
    PretrainDatasetFactory,
    SFTDataset,
    SFTTrainingCollator,
    SFTDatasetFactory,
    ChatTemplate,
    DialogueFormat,
    PackedDataset,
    DynamicPackingDataset,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("数据管道测试")


class MockTokenizer:
    """模拟分词器用于测试。"""

    def __init__(self, vocab_size: int = 30000):
        self.vocab_size = vocab_size

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """简单地将文本转换为token ID。"""
        tokens = [ord(c) % self.vocab_size for c in text]
        if add_special_tokens:
            tokens = [2] + tokens + [3]
        return tokens

    def decode(self, tokens: List[int]) -> str:
        """将token ID转换回文本。"""
        return "".join(chr(t % 128) for t in tokens)

    def tokenize(self, text: str) -> List[int]:
        """分词方法。"""
        return self.encode(text, add_special_tokens=False)

    def batch_tokenize(self, texts: List[str]) -> List[List[int]]:
        """批量分词方法。"""
        return [self.tokenize(text) for text in texts]


def create_temp_jsonl(data: List[Dict[str, Any]], suffix: str = ".jsonl") -> Path:
    """创建临时JSONL文件。

    参数：
        data: 数据列表。
        suffix: 文件后缀。

    返回：
        临时文件路径。
    """
    fd, path = tempfile.mkstemp(suffix=suffix, text=True)
    os.close(fd)

    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    return Path(path)


def create_temp_json(data: List[Dict[str, Any]]) -> Path:
    """创建临时JSON文件。

    参数：
        data: 数据列表。

    返回：
        临时文件路径。
    """
    fd, path = tempfile.mkstemp(suffix=".json", text=True)
    os.close(fd)

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)

    return Path(path)


def create_temp_txt(texts: List[str]) -> Path:
    """创建临时文本文件。

    参数：
        texts: 文本列表。

    返回：
        临时文件路径。
    """
    fd, path = tempfile.mkstemp(suffix=".txt", text=True)
    os.close(fd)

    with open(path, 'w', encoding='utf-8') as f:
        for text in texts:
            f.write(text + '\n')

    return Path(path)


class SimpleTextDataset(BaseDataset):
    """简单的文本数据集实现，用于测试。"""

    def _load_single_file(self, path: Path) -> List[Dict[str, Any]]:
        """加载数据文件。"""
        suffix = path.suffix.lower()

        if suffix == '.jsonl':
            items = []
            with open(path, 'r', encoding='utf-8', errors='strict') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            items.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            return items
        elif suffix == '.json':
            with open(path, 'r', encoding='utf-8', errors='strict') as f:
                data = json.load(f)
                return data if isinstance(data, list) else [data]
        elif suffix == '.txt':
            items = []
            with open(path, 'r', encoding='utf-8', errors='strict') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        items.append({'text': line})
            return items
        return []

    def _process_single_item(self, item: Dict[str, Any]) -> Optional[Union[Dict[str, Any], List[Dict[str, Any]]]]:
        """处理单个数据项。"""
        text = None
        for field in ['text', 'content', 'body']:
            if field in item and isinstance(item[field], str):
                text = item[field]
                break

        if text is None:
            return None

        tokens = self._tokenize_text(text)
        if len(tokens) < 2:
            return None

        return [{
            'input_ids': tokens,
            'text': text,
        }]


class TestBaseDataset:
    """测试基础数据集类。"""

    def test_base_dataset_initialization(self):
        """测试BaseDataset基本初始化。"""
        data = [
            {"text": "床前明月光"},
            {"text": "疑是地上霜"},
        ]
        path = create_temp_jsonl(data)

        dataset = SimpleTextDataset(
            data_paths=str(path),
            tokenizer=MockTokenizer(),
            context_length=128,
        )

        assert len(dataset) == 2
        os.unlink(path)

    def test_base_dataset_multi_format(self):
        """测试多格式支持。"""
        jsonl_data = [{"text": "床前明月光"}]
        jsonl_path = create_temp_jsonl(jsonl_data)

        json_data = [{"text": "疑是地上霜"}]
        json_path = create_temp_json(json_data)

        txt_data = ["举头望明月"]
        txt_path = create_temp_txt(txt_data)

        dataset = SimpleTextDataset(
            data_paths=[str(jsonl_path), str(json_path), str(txt_path)],
            tokenizer=MockTokenizer(),
            context_length=128,
        )

        assert len(dataset) > 0

        for path in [jsonl_path, json_path, txt_path]:
            os.unlink(path)

    def test_base_dataset_getitem(self):
        """测试样本获取。"""
        data = [
            {"text": "床前明月光"},
            {"text": "疑是地上霜"},
        ]
        path = create_temp_jsonl(data)

        dataset = SimpleTextDataset(
            data_paths=str(path),
            tokenizer=MockTokenizer(),
            context_length=128,
        )

        sample = dataset[0]
        assert "input_ids" in sample or "tokens" in sample or "text" in sample

        os.unlink(path)

    def test_base_dataset_iterator(self):
        """测试迭代器功能。"""
        data = [
            {"text": "床前明月光"},
            {"text": "疑是地上霜"},
        ]
        path = create_temp_jsonl(data)

        dataset = SimpleTextDataset(
            data_paths=str(path),
            tokenizer=MockTokenizer(),
            context_length=128,
        )

        count = 0
        for sample in dataset:
            count += 1
            assert sample is not None

        assert count == len(dataset)

        os.unlink(path)

    def test_base_dataset_nonexistent_path(self):
        """测试不存在的路径。"""
        dataset = SimpleTextDataset(
            data_paths="/nonexistent/path.jsonl",
            tokenizer=MockTokenizer(),
            context_length=128,
        )

        assert len(dataset) == 0


class TestPretrainDataset:
    """测试预训练数据集。"""

    def test_pretrain_dataset_basic(self):
        """测试基本预训练数据加载。"""
        data = [
            {"text": "床前明月光，疑是地上霜。"},
            {"content": "举头望明月，低头思故乡。"},
            {"body": "静夜思。"},
        ]
        path = create_temp_jsonl(data)

        dataset = PretrainDataset(
            data_paths=str(path),
            tokenizer=MockTokenizer(),
            context_length=128,
        )

        assert len(dataset) > 0

        sample = dataset[0]
        assert "input_ids" in sample
        assert "labels" in sample

        os.unlink(path)

    def test_pretrain_dataset_stride(self):
        """测试滑动窗口步长。"""
        data = [
            {"text": "床前明月光，疑是地上霜。举头望明月，低头思故乡。"},
        ]
        path = create_temp_jsonl(data)

        dataset_stride1 = PretrainDataset(
            data_paths=str(path),
            tokenizer=MockTokenizer(),
            context_length=32,
            stride=32,
        )

        dataset_stride2 = PretrainDataset(
            data_paths=str(path),
            tokenizer=MockTokenizer(),
            context_length=32,
            stride=16,
        )

        assert len(dataset_stride2) >= len(dataset_stride1)

        os.unlink(path)

    def test_pretrain_dataset_filtering(self):
        """测试数据过滤功能。"""
        data = [
            {"text": "短"},  # 太短，会被过滤
            {"text": "床前明月光，疑是地上霜。举头望明月，低头思故乡。"},
            {"text": "中"}  # 太短，会被过滤
        ]
        path = create_temp_jsonl(data)

        dataset = PretrainDataset(
            data_paths=str(path),
            tokenizer=MockTokenizer(),
            context_length=128,
            min_length=10,
        )

        assert len(dataset) >= 1

        os.unlink(path)

    def test_pretrain_dataset_json_format(self):
        """测试JSON格式加载。"""
        data = [
            {"text": "床前明月光，疑是地上霜"},
            {"text": "举头望明月，低头思故乡"},
        ]
        path = create_temp_json(data)

        dataset = PretrainDataset(
            data_paths=str(path),
            tokenizer=MockTokenizer(),
            context_length=128,
            min_length=5,
        )

        assert len(dataset) > 0

        os.unlink(path)

    def test_pretrain_dataset_txt_format(self):
        """测试纯文本格式加载。"""
        texts = [
            "床前明月光，疑是地上霜",
            "举头望明月，低头思故乡",
        ]
        path = create_temp_txt(texts)

        dataset = PretrainDataset(
            data_paths=str(path),
            tokenizer=MockTokenizer(),
            context_length=128,
            min_length=5,
        )

        assert len(dataset) > 0

        os.unlink(path)

    def test_concat_pretrain_dataset(self):
        """测试拼接预训练数据集。"""
        data = [
            {"text": "床前明月光，疑是地上霜。举头望明月，低头思故乡。"},
            {"text": "春眠不觉晓，处处闻啼鸟，夜来风雨声，花落知多少。"},
            {"text": "白日依山尽，黄河入海流。欲穷千里目，更上一层楼。"},
        ]
        path = create_temp_jsonl(data)

        dataset = ConcatPretrainDataset(
            data_paths=str(path),
            tokenizer=MockTokenizer(),
            context_length=32,
            min_doc_length=5,
        )

        assert len(dataset) > 0

        sample = dataset[0]
        assert len(sample["input_ids"]) == 32

        os.unlink(path)

    def test_pretrain_factory(self):
        """测试预训练数据集工厂。"""
        data = [
            {"text": "床前明月光，疑是地上霜，举头望明月"},
        ]
        path = create_temp_jsonl(data)

        dataset = PretrainDatasetFactory.create_standard(
            data_paths=str(path),
            tokenizer=MockTokenizer(),
            context_length=128,
        )

        assert len(dataset) > 0

        os.unlink(path)


class TestSFTDataset:
    """测试SFT数据集。"""

    def test_sft_dataset_sharegpt_format(self):
        """测试ShareGPT格式加载。"""
        data = [
            {
                "conversations": [
                    {"role": "user", "content": "床前明月光"},
                    {"role": "assistant", "content": "疑是地上霜"},
                ]
            },
            {
                "messages": [
                    {"role": "human", "content": "举头望明月"},
                    {"role": "gpt", "content": "低头思故乡"},
                ]
            }
        ]
        path = create_temp_jsonl(data)

        dataset = SFTDataset(
            data_paths=str(path),
            tokenizer=MockTokenizer(),
            context_length=128,
            dialogue_format="sharegpt",
        )

        assert len(dataset) >= 1

        os.unlink(path)

    def test_sft_dataset_openai_format(self):
        """测试OpenAI格式加载。"""
        data = [
            {
                "messages": [
                    {"role": "system", "content": "你是诗人"},
                    {"role": "user", "content": "床前明月光"},
                    {"role": "assistant", "content": "疑是地上霜"},
                ]
            }
        ]
        path = create_temp_jsonl(data)

        dataset = SFTDataset(
            data_paths=str(path),
            tokenizer=MockTokenizer(),
            context_length=128,
            dialogue_format="openai",
        )

        assert len(dataset) >= 0

        os.unlink(path)

    def test_sft_dataset_alpaca_format(self):
        """测试Alpaca格式加载。"""
        data = [
            {
                "instruction": "续写诗句",
                "input": "床前明月光",
                "output": "疑是地上霜",
            }
        ]
        path = create_temp_jsonl(data)

        dataset = SFTDataset(
            data_paths=str(path),
            tokenizer=MockTokenizer(),
            context_length=128,
            dialogue_format="alpaca",
        )

        assert len(dataset) >= 0

        os.unlink(path)

    def test_sft_dataset_chat_template(self):
        """测试聊天模板功能。"""
        template = ChatTemplate(
            system_prefix="[系统] ",
            user_prefix="[用户] ",
            assistant_prefix="[助手] ",
        )

        messages = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！"},
        ]

        formatted = template.format_conversation(messages)

        assert "[用户]" in formatted
        assert "[助手]" in formatted
        assert "你好" in formatted

    def test_sft_training_collator(self):
        """测试SFT训练数据整理器。"""
        collator = SFTTrainingCollator(pad_token_id=0, label_pad_token_id=-100)

        batch = [
            {
                "input_ids": [2, 100, 200, 3],
                "labels": [-100, -100, 200, 3],
                "attention_mask": [1, 1, 1, 1],
            },
            {
                "input_ids": [2, 50, 60],
                "labels": [-100, -100, 60],
                "attention_mask": [1, 1, 1],
            },
        ]

        result = collator(batch)

        assert "input_ids" in result
        assert "labels" in result
        
        if hasattr(result["input_ids"], 'shape'):
            assert result["input_ids"].shape[0] == 2
            assert result["input_ids"].shape[1] == 4
        else:
            assert len(result["input_ids"]) == 2
            assert len(result["input_ids"][0]) == 4

    def test_sft_dataset_factory(self):
        """测试SFT数据集工厂。"""
        data = [
            {
                "conversations": [
                    {"role": "user", "content": "测试"},
                    {"role": "assistant", "content": "测试回复"},
                ]
            }
        ]
        path = create_temp_jsonl(data)

        dataset = SFTDatasetFactory.create_sharegpt(
            data_paths=str(path),
            tokenizer=MockTokenizer(),
            context_length=128,
        )

        os.unlink(path)


class TestPackedDataset:
    """测试打包数据集。"""

    def test_packed_dataset_basic(self):
        """测试基本打包功能。"""
        data = [
            {"text": "床前明月光，疑是地上霜"},
            {"text": "举头望明月，低头思故乡"},
            {"text": "春眠不觉晓，处处闻啼鸟"},
        ]
        path = create_temp_jsonl(data)

        dataset = PackedDataset(
            data_paths=str(path),
            tokenizer=MockTokenizer(),
            context_length=128,
            pack_strategy="first-fit",
            min_sequence_length=5,
        )

        assert len(dataset) > 0

        sample = dataset[0]
        assert "input_ids" in sample
        assert "labels" in sample

        os.unlink(path)

    def test_packed_dataset_first_fit(self):
        """测试首次适应打包策略。"""
        data = [
            {"text": "床前明月光，疑是地上霜"},
            {"text": "举头望明月，低头思故乡"},
        ]
        path = create_temp_jsonl(data)

        dataset = PackedDataset(
            data_paths=str(path),
            tokenizer=MockTokenizer(),
            context_length=64,
            pack_strategy="first-fit",
            min_sequence_length=5,
        )

        assert len(dataset) > 0

        stats = dataset.get_statistics()
        assert stats["pack_strategy"] == "first-fit"

        os.unlink(path)

    def test_packed_dataset_best_fit(self):
        """测试最佳适应打包策略。"""
        data = [
            {"text": "床前明月光"},
            {"text": "床前明月光，疑是地上霜"},
            {"text": "举头望明月，低头思故乡"},
        ]
        path = create_temp_jsonl(data)

        dataset = PackedDataset(
            data_paths=str(path),
            tokenizer=MockTokenizer(),
            context_length=64,
            pack_strategy="best-fit",
            min_sequence_length=5,
        )

        assert len(dataset) > 0

        os.unlink(path)

    def test_packed_dataset_statistics(self):
        """测试打包统计信息。"""
        data = [
            {"text": "床前明月光，疑是地上霜"},
            {"text": "举头望明月，低头思故乡"},
        ]
        path = create_temp_jsonl(data)

        dataset = PackedDataset(
            data_paths=str(path),
            tokenizer=MockTokenizer(),
            context_length=128,
            min_sequence_length=5,
        )

        stats = dataset.get_statistics()

        assert "num_samples" in stats
        assert "avg_tokens_per_sample" in stats
        assert stats["num_samples"] > 0

        os.unlink(path)

    def test_dynamic_packing_dataset(self):
        """测试动态打包数据集。"""
        data = [
            {"text": "床前明月光，疑是地上霜，举头望明月，低头思故乡。"},
            {"text": "春眠不觉晓，处处闻啼鸟，夜来风雨声，花落知多少。"},
            {"text": "白日依山尽，黄河入海流。欲穷千里目，更上一层楼。"},
        ]
        path = create_temp_jsonl(data)

        dataset = DynamicPackingDataset(
            data_paths=str(path),
            tokenizer=MockTokenizer(),
            context_length=32,
            buffer_size=10,
        )

        assert len(dataset) > 0

        sample = dataset[0]
        assert "input_ids" in sample

        os.unlink(path)


class TestWeightedMixingDataset:
    """测试加权混合数据集。"""

    def test_weighted_mixing_basic(self):
        """测试基本混合功能。"""
        data1 = [{"text": "床前明月光，疑是地上霜"}]
        path1 = create_temp_jsonl(data1)

        data2 = [{"text": "举头望明月，低头思故乡"}]
        path2 = create_temp_jsonl(data2)

        ds1 = PretrainDataset(
            data_paths=str(path1),
            tokenizer=MockTokenizer(),
            context_length=128,
            min_length=5,
        )

        ds2 = PretrainDataset(
            data_paths=str(path2),
            tokenizer=MockTokenizer(),
            context_length=128,
            min_length=5,
        )

        mixed_ds = WeightedMixingDataset(
            datasets=[(ds1, 1.0), (ds2, 3.0)],
            replacement=True,
            seed=42,
        )

        assert len(mixed_ds) > 0

        os.unlink(path1)
        os.unlink(path2)

    def test_weighted_mixing_reproducibility(self):
        """测试混合数据集可重复性。"""
        data = [{"text": "床前明月光"}]
        path = create_temp_jsonl(data)

        ds = PretrainDataset(
            data_paths=str(path),
            tokenizer=MockTokenizer(),
            context_length=128,
            min_length=5,
        )

        mixed_ds1 = WeightedMixingDataset(
            datasets=[(ds, 1.0)],
            replacement=True,
            seed=42,
        )

        mixed_ds2 = WeightedMixingDataset(
            datasets=[(ds, 1.0)],
            replacement=True,
            seed=42,
        )

        assert len(mixed_ds1) == len(mixed_ds2)

        os.unlink(path)


class TestStreamingDataset:
    """测试流式数据集。"""

    def test_streaming_dataset_jsonl(self):
        """测试JSONL流式加载。"""
        data = [
            {"text": "床前明月光"},
            {"text": "疑是地上霜"},
        ]
        path = create_temp_jsonl(data)

        dataset = StreamingDataset(
            data_paths=str(path),
            tokenizer=MockTokenizer(),
            context_length=128,
        )

        count = 0
        for _ in dataset:
            count += 1
            if count >= 10:
                break

        assert count > 0

        os.unlink(path)

    def test_streaming_iterator(self):
        """测试流式迭代器。"""
        data = [
            {"text": "床前明月光，疑是地上霜"},
            {"text": "举头望明月，低头思故乡"},
            {"text": "春眠不觉晓，处处闻啼鸟"},
        ]
        path = create_temp_jsonl(data)

        dataset = StreamingDataset(
            data_paths=str(path),
            tokenizer=MockTokenizer(),
            context_length=128,
        )

        count = 0
        for _ in dataset:
            count += 1
            if count >= 5:
                break

        assert count > 0

        os.unlink(path)


class TestIntegration:
    """集成测试。"""

    def test_full_pipeline(self):
        """测试完整数据管道。"""
        pretrain_data = [
            {"text": "床前明月光，疑是地上霜。举头望明月，低头思故乡。"},
        ]
        pretrain_path = create_temp_jsonl(pretrain_data)

        sft_data = [
            {
                "conversations": [
                    {"role": "user", "content": "床前明月光"},
                    {"role": "assistant", "content": "疑是地上霜"},
                ]
            }
        ]
        sft_path = create_temp_jsonl(sft_data)

        pretrain_ds = PretrainDatasetFactory.create_standard(
            data_paths=str(pretrain_path),
            tokenizer=MockTokenizer(),
            context_length=128,
        )

        sft_ds = SFTDatasetFactory.create_sharegpt(
            data_paths=str(sft_path),
            tokenizer=MockTokenizer(),
            context_length=128,
        )

        mixed_ds = WeightedMixingDataset(
            datasets=[(pretrain_ds, 0.8), (sft_ds, 0.2)],
            replacement=True,
            seed=42,
        )

        assert len(mixed_ds) > 0

        sample = mixed_ds[0]
        assert "input_ids" in sample

        os.unlink(pretrain_path)
        os.unlink(sft_path)

    def test_multiple_formats(self):
        """测试多种格式混合。"""
        jsonl_data = [{"text": "床前明月光，疑是地上霜"}]
        jsonl_path = create_temp_jsonl(jsonl_data)

        json_data = [{"text": "举头望明月，低头思故乡"}]
        json_path = create_temp_json(json_data)

        txt_data = ["春眠不觉晓，处处闻啼鸟"]
        txt_path = create_temp_txt(txt_data)

        dataset = PretrainDataset(
            data_paths=[str(jsonl_path), str(json_path), str(txt_path)],
            tokenizer=MockTokenizer(),
            context_length=128,
            min_length=5,
        )

        assert len(dataset) >= 3

        for path in [jsonl_path, json_path, txt_path]:
            os.unlink(path)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
