"""Lingmao Moyun 数据分析模块。

提供数据统计和分析功能，包括：
- 词频统计
- 句子长度分布分析
- 字符分布统计
- 词汇表构建

示例：
    >>> from src.data.analyzers import DataAnalyzer, VocabularyAnalyzer
    >>>
    >>> # 数据分析
    >>> analyzer = DataAnalyzer()
    >>> stats = analyzer.analyze(data_list)
    >>>
    >>> # 词汇表构建
    >>> vocab = VocabularyAnalyzer(max_vocab_size=50000)
    >>> vocab.build_from_texts(texts)
"""

import json
import re
from pathlib import Path
from typing import Any, Counter, Dict, List, Optional, Set, Tuple, Union
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from src.logger import get_logger

logger = get_logger("LingmaoMoyun.Analyzer")


@dataclass
class AnalysisStats:
    """分析统计结果。"""
    total_samples: int = 0
    total_chars: int = 0
    total_words: int = 0
    avg_length: float = 0.0
    median_length: float = 0.0
    min_length: int = 0
    max_length: int = 0
    std_length: float = 0.0
    unique_chars: int = 0
    unique_words: int = 0

    length_distribution: Dict[str, int] = field(default_factory=dict)
    char_frequency: Dict[str, int] = field(default_factory=dict)
    word_frequency: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式。"""
        return {
            "total_samples": self.total_samples,
            "total_chars": self.total_chars,
            "total_words": self.total_words,
            "avg_length": round(self.avg_length, 2),
            "median_length": round(self.median_length, 2),
            "min_length": self.min_length,
            "max_length": self.max_length,
            "std_length": round(self.std_length, 2),
            "unique_chars": self.unique_chars,
            "unique_words": self.unique_words,
            "length_distribution": dict(list(self.length_distribution.items())[:50]),
            "top_chars": dict(list(self.char_frequency.items())[:20]),
            "top_words": dict(list(self.word_frequency.items())[:50]),
        }


class DataAnalyzer:
    """数据分析器。

    提供全面的数据统计分析功能：
    - 基本统计（样本数、长度分布等）
    - 词频统计
    - 字符分布分析
    - 语言特征检测

    示例：
        >>> analyzer = DataAnalyzer()
        >>> stats = analyzer.analyze(data_list, text_key="text")
        >>> analyzer.save_stats("stats.json")
    """

    def __init__(
        self,
        compute_word_freq: bool = True,
        compute_char_freq: bool = True,
        length_bins: Optional[List[int]] = None,
    ):
        self.compute_word_freq = compute_word_freq
        self.compute_char_freq = compute_char_freq
        self.length_bins = length_bins or [0, 50, 100, 200, 500, 1000, 2000, 5000, 10000]

        self.stats: Optional[AnalysisStats] = None

    def analyze(
        self,
        data: List[Dict[str, Any]],
        text_key: str = "text",
    ) -> AnalysisStats:
        """分析数据集。

        Args:
            data: 输入数据列表。
            text_key: 文本字段名。

        Returns:
            分析统计结果。
        """
        logger.info(f"开始分析 {len(data)} 条数据...")

        texts = []
        for item in data:
            text = item.get(text_key, "")
            if isinstance(text, str):
                texts.append(text)

        lengths = [len(text) for text in texts]

        char_counter = Counter()
        word_counter = Counter()
        total_chars = sum(lengths)

        for text in texts:
            if self.compute_char_freq:
                char_counter.update(text)

            if self.compute_word_freq:
                words = self._tokenize(text)
                word_counter.update(words)

        self.stats = AnalysisStats(
            total_samples=len(texts),
            total_chars=total_chars,
            total_words=sum(word_counter.values()),
            avg_length=np.mean(lengths) if lengths else 0,
            median_length=np.median(lengths) if lengths else 0,
            min_length=min(lengths) if lengths else 0,
            max_length=max(lengths) if lengths else 0,
            std_length=np.std(lengths) if lengths else 0,
            unique_chars=len(char_counter),
            unique_words=len(word_counter),
            length_distribution=self._compute_length_distribution(lengths),
            char_frequency=dict(char_counter.most_common(1000)),
            word_frequency=dict(word_counter.most_common(1000)),
        )

        logger.info(f"分析完成: {self.stats.total_samples} 样本, {self.stats.unique_chars} 唯一字符")
        return self.stats

    def _tokenize(self, text: str) -> List[str]:
        """简单分词（按空格和标点分割）。

        Args:
            text: 输入文本。

        Returns:
            词列表。
        """
        text = text.lower()
        words = re.findall(r"[\u4e00-\u9fff]+|[a-z]+", text, flags=re.UNICODE)
        return words

    def _compute_length_distribution(
        self,
        lengths: List[int],
    ) -> Dict[str, int]:
        """计算长度分布。

        Args:
            lengths: 长度列表。

        Returns:
            长度分布字典。
        """
        distribution = {}

        for i in range(len(self.length_bins) - 1):
            low = self.length_bins[i]
            high = self.length_bins[i + 1]
            label = f"{low}-{high}"

            count = sum(1 for length in lengths if low <= length < high)
            distribution[label] = count

        return distribution

    def get_length_stats(
        self,
        data: List[Dict[str, Any]],
        text_key: str = "text",
    ) -> Dict[str, Any]:
        """获取长度统计信息。

        Args:
            data: 输入数据。
            text_key: 文本字段名。

        Returns:
            长度统计字典。
        """
        lengths = []
        for item in data:
            text = item.get(text_key, "")
            if isinstance(text, str):
                lengths.append(len(text))

        if not lengths:
            return {}

        return {
            "min": min(lengths),
            "max": max(lengths),
            "mean": np.mean(lengths),
            "median": np.median(lengths),
            "std": np.std(lengths),
            "percentile_25": np.percentile(lengths, 25),
            "percentile_75": np.percentile(lengths, 75),
            "percentile_95": np.percentile(lengths, 95),
            "percentile_99": np.percentile(lengths, 99),
        }

    def get_char_distribution(
        self,
        data: List[Dict[str, Any]],
        text_key: str = "text",
        top_n: int = 100,
    ) -> Dict[str, int]:
        """获取字符分布。

        Args:
            data: 输入数据。
            text_key: 文本字段名。
            top_n: 返回前N个高频字符。

        Returns:
            字符频率字典。
        """
        char_counter = Counter()

        for item in data:
            text = item.get(text_key, "")
            if isinstance(text, str):
                char_counter.update(text)

        return dict(char_counter.most_common(top_n))

    def get_chinese_char_ratio(
        self,
        data: List[Dict[str, Any]],
        text_key: str = "text",
    ) -> float:
        """计算中文占比。

        Args:
            data: 输入数据。
            text_key: 文本字段名。

        Returns:
            中文字符占比（0-1）。
        """
        total_chars = 0
        chinese_chars = 0

        for item in data:
            text = item.get(text_key, "")
            if isinstance(text, str):
                total_chars += len(text)
                chinese_chars += len(re.findall(r"[\u4e00-\u9fff]", text))

        return chinese_chars / total_chars if total_chars > 0 else 0

    def detect_language(
        self,
        text: str,
    ) -> str:
        """检测文本语言。

        Args:
            text: 输入文本。

        Returns:
            语言类型（chinese/english/mixed/unknown）。
        """
        chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
        english_chars = len(re.findall(r"[a-zA-Z]", text))
        total_chars = len(text)

        if total_chars == 0:
            return "unknown"

        chinese_ratio = chinese_chars / total_chars
        english_ratio = english_chars / total_chars

        if chinese_ratio > 0.3:
            return "chinese"
        elif english_ratio > 0.5:
            return "english"
        elif chinese_ratio > 0.1 or english_ratio > 0.1:
            return "mixed"
        else:
            return "unknown"

    def save_stats(
        self,
        output_path: Union[str, Path],
        include_frequency: bool = True,
    ):
        """保存统计结果。

        Args:
            output_path: 输出文件路径。
            include_frequency: 是否包含频率统计。
        """
        if self.stats is None:
            logger.warning("没有可保存的统计信息")
            return

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        result = self.stats.to_dict()

        if not include_frequency:
            result.pop("char_frequency", None)
            result.pop("word_frequency", None)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        logger.info(f"统计结果已保存到: {output_path}")

    def print_summary(self):
        """打印统计摘要。"""
        if self.stats is None:
            print("没有可显示的统计信息")
            return

        print("\n" + "=" * 60)
        print("数据统计摘要")
        print("=" * 60)
        print(f"样本数量: {self.stats.total_samples}")
        print(f"总字符数: {self.stats.total_chars}")
        print(f"总词数: {self.stats.total_words}")
        print(f"唯一字符数: {self.stats.unique_chars}")
        print(f"唯一词数: {self.stats.unique_words}")
        print(f"\n长度统计:")
        print(f"  平均长度: {self.stats.avg_length:.2f}")
        print(f"  中位长度: {self.stats.median_length:.2f}")
        print(f"  最小长度: {self.stats.min_length}")
        print(f"  最大长度: {self.stats.max_length}")
        print(f"  标准差: {self.stats.std_length:.2f}")

        print(f"\n长度分布:")
        for label, count in list(self.stats.length_distribution.items())[:10]:
            print(f"  {label}: {count}")

        print(f"\n高频字符 Top 20:")
        for char, count in list(self.stats.char_frequency.items())[:20]:
            display_char = char if char not in "\n\t" else f"\\{ord(char):02x}"
            print(f"  '{display_char}': {count}")

        print("=" * 60)


class VocabularyAnalyzer:
    """词汇表分析器。

    构建和管理字符级或词级词汇表。
    支持词汇表统计、覆盖度分析和保存。

    示例：
        >>> vocab_analyzer = VocabularyAnalyzer(max_vocab_size=50000)
        >>> vocab_analyzer.build_from_texts(texts)
        >>> vocab_analyzer.save("vocab.json")
    """

    def __init__(
        self,
        max_vocab_size: int = 50000,
        min_freq: int = 1,
        vocab_type: str = "char",
        special_tokens: Optional[Dict[str, int]] = None,
    ):
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        self.vocab_type = vocab_type
        self.special_tokens = special_tokens or {
            "<pad>": 0,
            "<unk>": 1,
            "<bos>": 2,
            "<eos>": 3,
        }

        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.freq: Counter = Counter()

        self._build_vocab_from_tokens()

    def _build_vocab_from_tokens(self):
        """从特殊标记构建词汇表基础结构。"""
        self.token_to_id = self.special_tokens.copy()
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

    def build_from_texts(
        self,
        texts: List[str],
        update: bool = False,
    ):
        """从文本列表构建词汇表。

        Args:
            texts: 文本列表。
            update: 是否追加到现有词汇表。
        """
        if not update:
            self._build_vocab_from_tokens()
            self.freq = Counter()

        logger.info(f"从 {len(texts)} 个文本构建词汇表...")

        for text in texts:
            if self.vocab_type == "char":
                tokens = list(text)
            else:
                tokens = self._tokenize(text)

            self.freq.update(tokens)

        self._build_vocab()

        logger.info(f"词汇表构建完成: {len(self.token_to_id)} 个词条")

    def build_from_file(
        self,
        file_path: Union[str, Path],
        text_key: str = "text",
    ):
        """从文件构建词汇表。

        Args:
            file_path: 文件路径。
            text_key: 文本字段名。
        """
        file_path = Path(file_path)

        logger.info(f"从文件构建词汇表: {file_path}")

        texts = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    text = item.get(text_key, "")
                    if isinstance(text, str):
                        texts.append(text)
                except json.JSONDecodeError:
                    texts.append(line)

        self.build_from_texts(texts)

    def _tokenize(self, text: str) -> List[str]:
        """分词。

        Args:
            text: 输入文本。

        Returns:
            词列表。
        """
        words = re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z]+", text, flags=re.UNICODE)
        return words

    def _build_vocab(self):
        """根据频率构建词汇表。"""
        special_token_count = len(self.special_tokens)
        max_tokens = self.max_vocab_size - special_token_count

        if max_tokens <= 0:
            return

        filtered_tokens = [
            (token, freq)
            for token, freq in self.freq.items()
            if freq >= self.min_freq
        ]

        sorted_tokens = sorted(filtered_tokens, key=lambda x: (-x[1], x[0]))

        for idx, (token, _) in enumerate(sorted_tokens[:max_tokens]):
            token_id = special_token_count + idx
            self.token_to_id[token] = token_id
            self.id_to_token[token_id] = token

    def encode(
        self,
        text: str,
        return_tokens: bool = False,
    ) -> Union[List[int], List[Tuple[str, int]]]:
        """将文本编码为ID序列。

        Args:
            text: 输入文本。
            return_tokens: 是否返回(token, id)元组列表。

        Returns:
            ID列表或(token, id)元组列表。
        """
        if self.vocab_type == "char":
            tokens = list(text)
        else:
            tokens = self._tokenize(text)

        unk_id = self.token_to_id.get("<unk>", 1)
        result = []

        for token in tokens:
            token_id = self.token_to_id.get(token, unk_id)
            result.append(token_id)

        if return_tokens:
            return list(zip(tokens, result))

        return result

    def decode(
        self,
        ids: List[int],
        skip_special: bool = True,
    ) -> str:
        """将ID序列解码为文本。

        Args:
            ids: ID列表。
            skip_special: 是否跳过特殊标记。

        Returns:
            解码后的文本。
        """
        tokens = []
        special_ids = set(self.special_tokens.values()) if skip_special else set()

        for token_id in ids:
            if skip_special and token_id in special_ids:
                continue
            token = self.id_to_token.get(token_id, "<unk>")
            tokens.append(token)

        return "".join(tokens) if self.vocab_type == "char" else " ".join(tokens)

    def get_coverage(
        self,
        texts: List[str],
    ) -> Dict[str, float]:
        """计算词汇表覆盖度。

        Args:
            texts: 文本列表。

        Returns:
            覆盖度统计字典。
        """
        total_tokens = 0
        covered_tokens = 0
        oov_tokens = Counter()

        for text in texts:
            if self.vocab_type == "char":
                tokens = list(text)
            else:
                tokens = self._tokenize(text)

            for token in tokens:
                total_tokens += 1
                if token in self.token_to_id:
                    covered_tokens += 1
                else:
                    oov_tokens[token] += 1

        coverage = covered_tokens / total_tokens if total_tokens > 0 else 0

        return {
            "total_tokens": total_tokens,
            "covered_tokens": covered_tokens,
            "coverage": coverage,
            "oov_count": len(oov_tokens),
            "oov_ratio": 1 - coverage,
            "top_oov": dict(oov_tokens.most_common(20)),
        }

    def get_stats(self) -> Dict[str, Any]:
        """获取词汇表统计信息。

        Returns:
            统计字典。
        """
        freq_values = list(self.freq.values())
        if freq_values:
            freq_mean = np.mean(freq_values)
            freq_std = np.std(freq_values)
            freq_max = max(freq_values)
            freq_min = min(freq_values)
        else:
            freq_mean = freq_std = freq_max = freq_min = 0

        return {
            "vocab_size": len(self.token_to_id),
            "max_vocab_size": self.max_vocab_size,
            "min_freq": self.min_freq,
            "vocab_type": self.vocab_type,
            "special_tokens": list(self.special_tokens.keys()),
            "freq_mean": round(freq_mean, 2),
            "freq_std": round(freq_std, 2),
            "freq_max": freq_max,
            "freq_min": freq_min,
        }

    def save(
        self,
        output_path: Union[str, Path],
        include_freq: bool = True,
        format: str = "json",
    ):
        """保存词汇表。

        Args:
            output_path: 输出文件路径。
            include_freq: 是否包含频率信息。
            format: 保存格式（json/txt）。
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            vocab_data = {
                "token_to_id": self.token_to_id,
                "id_to_token": self.id_to_token,
                "stats": self.get_stats(),
            }

            if include_freq:
                vocab_data["frequency"] = dict(self.freq)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(vocab_data, f, ensure_ascii=False, indent=2)

        elif format == "txt":
            with open(output_path, "w", encoding="utf-8") as f:
                for token_id in sorted(self.id_to_token.keys()):
                    token = self.id_to_token[token_id]
                    freq = self.freq.get(token, 0)
                    f.write(f"{token_id}\t{token}\t{freq}\n")

        logger.info(f"词汇表已保存到: {output_path}")

    def load(
        self,
        input_path: Union[str, Path],
        format: str = "json",
    ):
        """加载词汇表。

        Args:
            input_path: 输入文件路径。
            format: 文件格式（json/txt）。
        """
        input_path = Path(input_path)

        if format == "json":
            with open(input_path, "r", encoding="utf-8") as f:
                vocab_data = json.load(f)

            self.token_to_id = vocab_data["token_to_id"]
            self.id_to_token = {int(k): v for k, v in vocab_data["id_to_token"].items()}
            if "frequency" in vocab_data:
                self.freq = Counter(vocab_data["frequency"])

        elif format == "txt":
            self.token_to_id = self.special_tokens.copy()
            self.id_to_token = {v: k for k, v in self.special_tokens.items()}

            with open(input_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        token_id = int(parts[0])
                        token = parts[1]
                        self.token_to_id[token] = token_id
                        self.id_to_token[token_id] = token
                        if len(parts) >= 3:
                            self.freq[token] = int(parts[2])

        logger.info(f"词汇表已加载: {len(self.token_to_id)} 个词条")

    def print_summary(self):
        """打印词汇表摘要。"""
        stats = self.get_stats()

        print("\n" + "=" * 60)
        print("词汇表统计摘要")
        print("=" * 60)
        print(f"词汇表大小: {stats['vocab_size']}")
        print(f"最大词汇表大小: {stats['max_vocab_size']}")
        print(f"词条类型: {stats['vocab_type']}")
        print(f"最小频率: {stats['min_freq']}")
        print(f"特殊标记: {', '.join(stats['special_tokens'])}")
        print(f"\n频率统计:")
        print(f"  平均频率: {stats['freq_mean']:.2f}")
        print(f"  标准差: {stats['freq_std']:.2f}")
        print(f"  最大频率: {stats['freq_max']}")
        print(f"  最小频率: {stats['freq_min']}")

        print(f"\n高频词 Top 30:")
        sorted_by_freq = sorted(self.freq.items(), key=lambda x: -x[1])
        for token, freq in sorted_by_freq[:30]:
            display_token = token if len(token) < 10 else token[:7] + "..."
            print(f"  '{display_token}': {freq}")

        print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Lingmao Moyun 数据分析工具")
    parser.add_argument("--input", type=str, required=True, help="输入JSONL文件")
    parser.add_argument("--output", type=str, help="输出统计文件")
    parser.add_argument("--type", type=str, default="data", choices=["data", "vocab"])
    parser.add_argument("--text-key", type=str, default="text", help="文本字段名")
    parser.add_argument("--max-vocab", type=int, default=50000, help="最大词汇表大小")

    args = parser.parse_args()

    data = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    data.append({"text": line})

    if args.type == "data":
        analyzer = DataAnalyzer()
        stats = analyzer.analyze(data, text_key=args.text_key)
        analyzer.print_summary()

        if args.output:
            analyzer.save_stats(args.output)

    else:
        texts = [item.get(args.text_key, "") for item in data]
        vocab_analyzer = VocabularyAnalyzer(max_vocab_size=args.max_vocab)
        vocab_analyzer.build_from_texts(texts)
        vocab_analyzer.print_summary()

        if args.output:
            vocab_analyzer.save(args.output)

        coverage = vocab_analyzer.get_coverage(texts)
        print(f"\n词汇表覆盖度: {coverage['coverage']:.2%}")
