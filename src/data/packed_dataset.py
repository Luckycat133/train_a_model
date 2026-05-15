"""长序列打包数据集模块。

本模块实现了高效的长序列打包功能，支持将多个短序列打包成一个
固定长度的序列，从而提高训练效率和GPU利用率。
"""

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except Exception:
    torch = None
    Dataset = object

from src.data.base_dataset import BaseDataset
from src.logger import get_logger

logger = get_logger("LingmaoMoyun.Data.Packed")


class PackedDataset(BaseDataset):
    """打包数据集，将多个短序列合并成长序列。

    通过将多个文档打包到一个固定长度的序列中，提高训练效率。
    使用特殊分隔符标记不同文档边界，支持文档级注意力掩码。

    参数：
        data_paths: 数据文件路径或路径列表。
        tokenizer: 分词器实例。
        context_length: 打包后的目标序列长度。
        pack_strategy: 打包策略，可选 'first-fit' 或 'best-fit'。
        delimiter: 文档分隔符文本。
        delimiter_token_id: 文档分隔符的token ID。
        min_sequence_length: 最小序列长度，短于此长度的序列将被填充。
        max_sequence_length: 最大序列长度，超长序列将被截断或跳过。
        shuffle_packing: 是否在打包前打乱样本顺序。
        seed: 随机种子。
    """

    def __init__(
        self,
        data_paths: Union[str, Path, List[Union[str, Path]]],
        tokenizer: Optional[Any] = None,
        context_length: int = 2048,
        pack_strategy: str = "first-fit",
        delimiter: str = " ",
        delimiter_token_id: int = 2,
        min_sequence_length: int = 16,
        max_sequence_length: int = 100000,
        shuffle_packing: bool = False,
        seed: int = 42,
    ) -> None:
        self.pack_strategy = pack_strategy
        self.delimiter = delimiter
        self.delimiter_token_id = delimiter_token_id
        self.min_sequence_length = min_sequence_length
        self.max_sequence_length = max_sequence_length
        self.shuffle_packing = shuffle_packing
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        super().__init__(
            data_paths=data_paths,
            tokenizer=tokenizer,
            context_length=context_length,
            streaming=False,
        )

    def _load_single_file(self, path: Path) -> List[Dict[str, Any]]:
        """加载单个数据文件。

        参数：
            path: 数据文件路径。

        返回：
            数据项列表。
        """
        suffix = path.suffix.lower()

        if suffix == ".jsonl":
            return self._load_jsonl(path)
        elif suffix == ".json":
            return self._load_json(path)
        elif suffix == ".txt":
            return self._load_txt(path)
        else:
            logger.warning(f"不支持的文件格式：{suffix}")
            return []

    def _load_jsonl(self, path: Path) -> List[Dict[str, Any]]:
        """加载JSONL格式数据。

        参数：
            path: JSONL文件路径。

        返回：
            数据项列表。
        """
        items = []
        with open(path, "r", encoding="utf-8", errors="strict") as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    items.append(item)
                except json.JSONDecodeError:
                    logger.warning(f"跳过第 {line_idx + 1} 行的格式错误JSON")
        return items

    def _load_json(self, path: Path) -> List[Dict[str, Any]]:
        """加载JSON格式数据。

        参数：
            path: JSON文件路径。

        返回：
            数据项列表。
        """
        with open(path, "r", encoding="utf-8", errors="strict") as f:
            try:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                else:
                    return [data]
            except json.JSONDecodeError:
                logger.warning(f"无法解析JSON文件：{path}")
                return []

    def _load_txt(self, path: Path) -> List[Dict[str, Any]]:
        """加载纯文本格式数据。

        参数：
            path: 文本文件路径。

        返回：
            数据项列表。
        """
        items = []
        with open(path, "r", encoding="utf-8", errors="strict") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append({"text": line})
        return items

    def _extract_text(self, item: Dict[str, Any]) -> Optional[str]:
        """从数据项中提取文本内容。

        参数：
            item: 数据项字典。

        返回：
            提取的文本或None。
        """
        for field in ["text", "content", "body", "document"]:
            if field in item and isinstance(item[field], str):
                return item[field].strip()

        if "title" in item:
            return str(item.get("title", "")).strip()

        return None

    def _process_single_item(self, item: Dict[str, Any]) -> Optional[Union[Dict[str, Any], List[Dict[str, Any]]]]:
        """处理单个数据项。

        参数：
            item: 原始数据项。

        返回：
            处理后的样本或None。
        """
        text = self._extract_text(item)
        if text is None:
            return None

        tokens = self._tokenize_text(text)

        if len(tokens) < self.min_sequence_length:
            return None

        if len(tokens) > self.max_sequence_length:
            tokens = tokens[: self.max_sequence_length]

        return [{
            "tokens": tokens,
            "text": text,
        }]

    def _process_items(self, items: List[Dict[str, Any]]) -> None:
        """处理所有数据项并执行打包。

        参数：
            items: 原始数据项列表。
        """
        token_sequences = []

        for item in items:
            processed = self._process_single_item(item)
            if processed:
                token_sequences.extend(processed)

        if not token_sequences:
            logger.warning("没有找到有效的数据序列")
            self._samples = []
            return

        if self.shuffle_packing:
            self.rng.shuffle(token_sequences)

        delimiter_tokens = self._tokenize_text(self.delimiter)
        if not delimiter_tokens:
            delimiter_tokens = [self.delimiter_token_id]

        self._samples = self._pack_sequences(token_sequences, delimiter_tokens)

        logger.info(
            f"打包完成：从 {len(token_sequences)} 个序列生成 {len(self._samples)} 个打包样本"
        )

    def _pack_sequences(
        self,
        sequences: List[Dict[str, Any]],
        delimiter_tokens: List[int],
    ) -> List[Dict[str, Any]]:
        """将序列打包成固定长度的块。

        参数：
            sequences: 序列列表，每个包含 'tokens' 字段。
            delimiter_tokens: 分隔符token列表。

        返回：
            打包后的样本列表。
        """
        packed_samples = []

        if self.pack_strategy == "first-fit":
            packed_samples = self._first_fit_packing(sequences, delimiter_tokens)
        elif self.pack_strategy == "best-fit":
            packed_samples = self._best_fit_packing(sequences, delimiter_tokens)
        else:
            logger.warning(f"未知的打包策略：{self.pack_strategy}，使用first-fit")
            packed_samples = self._first_fit_packing(sequences, delimiter_tokens)

        return packed_samples

    def _first_fit_packing(
        self,
        sequences: List[Dict[str, Any]],
        delimiter_tokens: List[int],
    ) -> List[Dict[str, Any]]:
        """首次适应打包策略。

        按顺序将序列放入第一个能容纳的块中。

        参数：
            sequences: 序列列表。
            delimiter_tokens: 分隔符token。

        返回：
            打包后的样本列表。
        """
        packed = []
        current_tokens = []
        current_boundaries = [0]

        for seq in sequences:
            seq_tokens = seq["tokens"]
            needed = len(seq_tokens) + len(delimiter_tokens)

            if len(current_tokens) + needed <= self.context_length:
                if current_tokens:
                    current_tokens.extend(delimiter_tokens)
                    current_boundaries.append(len(current_tokens))
                current_tokens.extend(seq_tokens)
            else:
                if current_tokens:
                    packed.append({
                        "tokens": current_tokens,
                        "doc_boundaries": current_boundaries,
                    })

                current_tokens = list(seq_tokens)
                current_boundaries = [0]

        if current_tokens:
            packed.append({
                "tokens": current_tokens,
                "doc_boundaries": current_boundaries,
            })

        for sample in packed:
            sample["attention_mask"] = self._create_document_mask(
                sample["tokens"], sample["doc_boundaries"]
            )
            sample["labels"] = sample["tokens"][1:] + [0]

            if len(sample["tokens"]) < self.context_length:
                pad_length = self.context_length - len(sample["tokens"])
                sample["tokens"] = sample["tokens"] + [0] * pad_length
                sample["labels"] = sample["labels"] + [-100] * pad_length
                sample["attention_mask"] = sample["attention_mask"] + [0] * pad_length
            else:
                sample["tokens"] = sample["tokens"][: self.context_length]
                sample["labels"] = sample["labels"][: self.context_length]
                sample["attention_mask"] = sample["attention_mask"][: self.context_length]

        return packed

    def _best_fit_packing(
        self,
        sequences: List[Dict[str, Any]],
        delimiter_tokens: List[int],
    ) -> List[Dict[str, Any]]:
        """最佳适应打包策略。

        将序列放入剩余空间最小的块中。

        参数：
            sequences: 序列列表。
            delimiter_tokens: 分隔符token。

        返回：
            打包后的样本列表。
        """
        chunks = []
        chunk_remaining = [self.context_length] * len(sequences)
        chunk_sequences = [[] for _ in range(len(sequences))]
        chunk_boundaries = [[0] for _ in range(len(sequences))]

        for seq in sequences:
            seq_tokens = seq["tokens"]
            seq_len = len(seq_tokens)

            best_idx = -1
            best_remaining = float("inf")

            for i, remaining in enumerate(chunk_remaining):
                needed = seq_len + len(delimiter_tokens)
                if needed <= remaining and remaining - needed < best_remaining:
                    best_idx = i
                    best_remaining = remaining - needed

            if best_idx == -1:
                chunks.append({
                    "tokens": [],
                    "boundaries": [],
                    "remaining": self.context_length,
                })
                chunk_remaining.append(self.context_length)
                chunk_sequences.append([])
                chunk_boundaries.append([0])
                best_idx = len(chunks) - 1

            if chunk_sequences[best_idx]:
                chunk_sequences[best_idx].extend(delimiter_tokens)
                chunk_boundaries[best_idx].append(len(chunk_sequences[best_idx]))

            chunk_sequences[best_idx].extend(seq_tokens)
            chunk_remaining[best_idx] -= seq_len + len(delimiter_tokens)

        packed = []
        for tokens, boundaries in zip(chunk_sequences, chunk_boundaries):
            if not tokens:
                continue

            attention_mask = self._create_document_mask(tokens, boundaries)
            labels = tokens[1:] + [0]

            if len(tokens) < self.context_length:
                pad_length = self.context_length - len(tokens)
                tokens = tokens + [0] * pad_length
                labels = labels + [-100] * pad_length
                attention_mask = attention_mask + [0] * pad_length
            else:
                tokens = tokens[: self.context_length]
                labels = labels[: self.context_length]
                attention_mask = attention_mask[: self.context_length]

            packed.append({
                "tokens": tokens,
                "doc_boundaries": boundaries,
                "attention_mask": attention_mask,
                "labels": labels,
            })

        return packed

    def _create_document_mask(
        self,
        tokens: List[int],
        boundaries: List[int],
    ) -> List[int]:
        """创建文档级注意力掩码。

        为打包序列中的每个token创建注意力掩码，
        用于标识不同文档的边界。

        参数：
            tokens: token列表。
            boundaries: 文档边界位置列表。

        返回：
            注意力掩码列表。
        """
        if len(boundaries) <= 1:
            return [1] * len(tokens)

        mask = [0] * len(tokens)

        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            for j in range(start, min(end, len(mask))):
                mask[j] = 1

        if boundaries[-1] < len(tokens):
            for j in range(boundaries[-1], len(tokens)):
                mask[j] = 1

        return mask

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """获取打包后的样本。

        参数：
            idx: 样本索引。

        返回：
            包含input_ids、labels、attention_mask等的字典。
        """
        sample = self._samples[idx]

        if torch is not None:
            return {
                "input_ids": torch.tensor(sample["tokens"], dtype=torch.long),
                "labels": torch.tensor(sample["labels"], dtype=torch.long),
                "attention_mask": torch.tensor(sample["attention_mask"], dtype=torch.long),
                "doc_boundaries": sample["doc_boundaries"],
            }

        return {
            "input_ids": sample["tokens"],
            "labels": sample["labels"],
            "attention_mask": sample["attention_mask"],
            "doc_boundaries": sample["doc_boundaries"],
        }

    def get_statistics(self) -> Dict[str, Any]:
        """获取打包统计信息。

        返回：
            包含统计信息的字典。
        """
        if not self._samples:
            return {
                "num_samples": 0,
                "avg_pack_ratio": 0.0,
                "avg_docs_per_sample": 0.0,
            }

        total_tokens = sum(len(s["tokens"]) for s in self._samples)
        total_docs = sum(len(s["doc_boundaries"]) for s in self._samples)

        return {
            "num_samples": len(self._samples),
            "total_tokens": total_tokens,
            "avg_tokens_per_sample": total_tokens / len(self._samples),
            "avg_docs_per_sample": total_docs / len(self._samples),
            "context_length": self.context_length,
            "pack_strategy": self.pack_strategy,
        }


class DynamicPackingDataset(BaseDataset):
    """动态打包数据集。

    在训练时动态打包短序列，支持无限长的数据流。
    """

    def __init__(
        self,
        data_paths: Union[str, Path, List[Union[str, Path]]],
        tokenizer: Optional[Any] = None,
        context_length: int = 2048,
        buffer_size: int = 1000,
        delimiter_token_id: int = 2,
        shuffle_buffer: bool = True,
        seed: int = 42,
    ) -> None:
        self.buffer_size = buffer_size
        self.delimiter_token_id = delimiter_token_id
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        super().__init__(
            data_paths=data_paths,
            tokenizer=tokenizer,
            context_length=context_length,
            streaming=False,
        )

    def _load_single_file(self, path: Path) -> List[Dict[str, Any]]:
        """加载数据文件。"""
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            items = []
            with open(path, "r", encoding="utf-8", errors="strict") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            items.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            return items
        elif suffix == ".json":
            with open(path, "r", encoding="utf-8", errors="strict") as f:
                data = json.load(f)
                return data if isinstance(data, list) else [data]
        elif suffix == ".txt":
            items = []
            with open(path, "r", encoding="utf-8", errors="strict") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        items.append({"text": line})
            return items
        return []

    def _process_items(self, items: List[Dict[str, Any]]) -> None:
        """预处理所有数据项。"""
        self._sequences = []

        for item in items:
            text = None
            for field in ["text", "content", "body"]:
                if field in item and isinstance(item[field], str):
                    text = item[field].strip()
                    break

            if text:
                tokens = self._tokenize_text(text)
                if len(tokens) >= 16:
                    self._sequences.append(tokens)

        if self.shuffle_buffer:
            self.rng.shuffle(self._sequences)

        self._index = 0

    def __len__(self) -> int:
        """返回打包后的样本数量（估计值）。"""
        if not hasattr(self, "_sequences") or not self._sequences:
            return 0
        total_tokens = sum(len(s) for s in self._sequences)
        return (total_tokens // self.context_length) + 1

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """获取动态打包的样本。"""
        if not hasattr(self, "_sequences") or not self._sequences:
            raise IndexError("Dataset is empty")

        packed_tokens = []
        boundaries = [0]

        while len(packed_tokens) < self.context_length:
            if self._index >= len(self._sequences):
                self._index = 0
                if self.shuffle_buffer:
                    self.rng.shuffle(self._sequences)

            seq = self._sequences[self._index]
            self._index += 1

            if packed_tokens:
                packed_tokens.append(self.delimiter_token_id)
                boundaries.append(len(packed_tokens))

            packed_tokens.extend(seq)

            if len(packed_tokens) >= self.context_length:
                break

        input_ids = packed_tokens[: self.context_length]
        labels = packed_tokens[1 : self.context_length + 1]
        attention_mask = [1] * len(input_ids)

        if len(input_ids) < self.context_length:
            pad_len = self.context_length - len(input_ids)
            input_ids = input_ids + [0] * pad_len
            labels = labels + [-100] * pad_len
            attention_mask = attention_mask + [0] * pad_len

        if torch is not None:
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            }

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }
