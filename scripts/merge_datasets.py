#!/usr/bin/env python3
"""数据集合并脚本。

合并多个数据集，支持：
- 预训练数据合并
- SFT数据合并
- 混合数据合并
- 权重控制
- 去重处理
- 统计报告

使用示例：
    # 基本用法 - 合并多个预训练数据集
    python scripts/merge_datasets.py \\
        --input data/pretrain_1.jsonl data/pretrain_2.jsonl \\
        --output data/pretrain_merged.jsonl

    # 带权重的SFT数据合并
    python scripts/merge_datasets.py \\
        --input data/sft_alpaca.jsonl data/sft_sharegpt.jsonl \\
        --output data/sft_merged.jsonl \\
        --weights 0.7 0.3 \\
        --dedup

    # 混合数据合并
    python scripts/merge_datasets.py \\
        --input data/pretrain/ data/sft/ \\
        --output data/merged/ \\
        --strategy mixed \\
        --split-ratio 0.8 0.2
"""

import argparse
import json
import sys
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import Counter
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.logger import get_logger

logger = get_logger("LingmaoMoyun.MergeDatasets")


@dataclass
class MergeStats:
    """合并统计信息。"""
    total_input: int = 0
    total_records: int = 0
    unique_records: int = 0
    duplicates_removed: int = 0
    by_source: Dict[str, int] = None

    def __post_init__(self):
        if self.by_source is None:
            self.by_source = {}


class DatasetMerger:
    """数据集合并器。

    合并多个数据集，支持权重、去重等功能。

    Attributes:
        weights: 各数据集的权重。
        dedup: 是否去重。
        shuffle: 是否打乱顺序。

    Example:
        >>> merger = DatasetMerger(weights=[0.7, 0.3], dedup=True)
        >>> merger.merge(['data1.jsonl', 'data2.jsonl'], 'output.jsonl')
    """

    def __init__(
        self,
        weights: Optional[List[float]] = None,
        dedup: bool = True,
        shuffle: bool = True,
        seed: int = 42,
    ):
        """初始化数据集合并器。

        Args:
            weights: 各数据集的权重列表。
            dedup: 是否去除重复记录。
            shuffle: 是否打乱顺序。
            seed: 随机种子。
        """
        self.weights = weights
        self.dedup = dedup
        self.shuffle = shuffle
        self.seed = seed
        self.seen_hashes = set()

    def merge(
        self,
        input_paths: List[str],
        output_path: str,
        output_format: str = "jsonl",
    ) -> MergeStats:
        """合并数据集。

        Args:
            input_paths: 输入文件路径列表。
            output_path: 输出文件路径。
            output_format: 输出格式。

        Returns:
            合并统计信息。

        Example:
            >>> merger = DatasetMerger()
            >>> stats = merger.merge(['data1.jsonl', 'data2.jsonl'], 'output.jsonl')
            >>> print(f"Merged {stats.total_records} records")
        """
        import random
        random.seed(self.seed)

        stats = MergeStats()
        stats.total_input = len(input_paths)

        all_records = []

        for idx, input_path in enumerate(input_paths):
            records = self._load_records(Path(input_path))
            source_name = Path(input_path).stem

            weight = 1.0
            if self.weights and idx < len(self.weights):
                weight = self.weights[idx]

            sampled_records = self._apply_weight(records, weight)
            stats.by_source[source_name] = len(sampled_records)

            all_records.extend(sampled_records)
            logger.info(f"Loaded {len(sampled_records)} records from {input_path}")

        stats.total_records = len(all_records)

        if self.dedup:
            all_records, dup_count = self._deduplicate(all_records)
            stats.duplicates_removed = dup_count
            logger.info(f"Removed {dup_count} duplicates")

        stats.unique_records = len(all_records)

        if self.shuffle:
            random.shuffle(all_records)

        self._save_records(all_records, Path(output_path), output_format)

        logger.info(f"Merged {stats.unique_records} records to {output_path}")
        return stats

    def merge_mixed(
        self,
        input_paths: List[str],
        output_path: str,
        split_ratio: Optional[List[float]] = None,
        dedup: bool = True,
    ) -> Tuple[MergeStats, MergeStats]:
        """合并混合类型数据集（预训练+SFT）。

        Args:
            input_paths: 输入路径列表。
            output_path: 输出路径。
            split_ratio: 分割比例 [pretrain_ratio, sft_ratio]。
            dedup: 是否去重。

        Returns:
            包含两个统计信息的元组。

        Example:
            >>> merger = DatasetMerger()
            >>> pretrain_stats, sft_stats = merger.merge_mixed(
            ...     ['pretrain/', 'sft/'],
            ...     'output/',
            ...     split_ratio=[0.8, 0.2]
            ... )
        """
        pretrain_records = []
        sft_records = []

        for input_path in input_paths:
            path = Path(input_path)
            if path.is_dir():
                files = list(path.glob("**/*.jsonl")) + list(path.glob("**/*.json"))
            else:
                files = [path]

            for file_path in files:
                records = self._load_records(file_path)
                if self._is_sft_data(records):
                    sft_records.extend(records)
                else:
                    pretrain_records.extend(records)

        if dedup:
            pretrain_records, pretrain_dups = self._deduplicate(pretrain_records)
            sft_records, sft_dups = self._deduplicate(sft_records)

        import random
        random.seed(self.seed)

        if self.shuffle:
            random.shuffle(pretrain_records)
            random.shuffle(sft_records)

        pretrain_path = Path(output_path) / "pretrain.jsonl"
        sft_path = Path(output_path) / "sft.jsonl"
        pretrain_path.parent.mkdir(parents=True, exist_ok=True)

        self._save_records(pretrain_records, pretrain_path, "jsonl")
        self._save_records(sft_records, sft_path, "jsonl")

        pretrain_stats = MergeStats(
            total_input=len(pretrain_records),
            total_records=len(pretrain_records),
            unique_records=len(pretrain_records),
        )

        sft_stats = MergeStats(
            total_input=len(sft_records),
            total_records=len(sft_records),
            unique_records=len(sft_records),
        )

        return pretrain_stats, sft_stats

    def _load_records(self, file_path: Path) -> List[Dict[str, Any]]:
        """加载记录。"""
        records = []
        try:
            if file_path.suffix == '.jsonl':
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                records.append(json.loads(line))
                            except json.JSONDecodeError:
                                continue
            elif file_path.suffix == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        records.extend(data)
                    else:
                        records.append(data)
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")

        return records

    def _apply_weight(self, records: List[Dict[str, Any]], weight: float) -> List[Dict[str, Any]]:
        """根据权重采样记录。"""
        import random
        random.seed(self.seed)

        if weight >= 1.0:
            return records

        sample_size = int(len(records) * weight)
        if sample_size < 1:
            sample_size = 1

        return random.sample(records, min(sample_size, len(records)))

    def _deduplicate(self, records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
        """去除重复记录。"""
        unique_records = []
        duplicates = 0

        for record in records:
            record_hash = self._compute_hash(record)

            if record_hash not in self.seen_hashes:
                self.seen_hashes.add(record_hash)
                unique_records.append(record)
            else:
                duplicates += 1

        return unique_records, duplicates

    def _compute_hash(self, record: Dict[str, Any]) -> str:
        """计算记录哈希。"""
        text = record.get('text', '')
        if not text:
            for key in ['content', 'body', 'conversations']:
                if key in record:
                    if isinstance(record[key], list):
                        text = ' '.join(str(item) for item in record[key])
                    else:
                        text = str(record[key])
                    break

        if not text:
            text = json.dumps(record, sort_keys=True, ensure_ascii=False)

        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def _is_sft_data(self, records: List[Dict[str, Any]]) -> bool:
        """判断是否为SFT数据。"""
        if not records:
            return False

        sample = records[0]

        if 'conversations' in sample:
            return True
        if 'messages' in sample:
            return True
        if 'instruction' in sample and 'output' in sample:
            return True

        return False

    def _save_records(
        self,
        records: List[Dict[str, Any]],
        output_path: Path,
        output_format: str,
    ) -> None:
        """保存记录。"""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            if output_format == 'jsonl':
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
            elif output_format == 'json':
                json.dump(records, f, ensure_ascii=False, indent=2)


class DatasetAnalyzer:
    """数据集分析器。"""

    @staticmethod
    def analyze(input_path: str) -> Dict[str, Any]:
        """分析数据集。

        Args:
            input_path: 数据集路径。

        Returns:
            分析结果字典。

        Example:
            >>> analysis = DatasetAnalyzer.analyze('data.jsonl')
            >>> print(f"Total records: {analysis['total_records']}")
        """
        path = Path(input_path)
        records = []

        if path.is_file():
            if path.suffix == '.jsonl':
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                records.append(json.loads(line))
                            except json.JSONDecodeError:
                                continue
            elif path.suffix == '.json':
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        records = data
                    else:
                        records = [data]

        length_stats = []
        text_lengths = []
        has_conversations = 0
        has_text = 0

        for record in records:
            if 'conversations' in record or 'messages' in record:
                has_conversations += 1
            if 'text' in record:
                has_text += 1
                text_lengths.append(len(record['text']))

            if 'conversations' in record:
                conv_text = ' '.join(turn.get('content', '') for turn in record['conversations'])
                length_stats.append(len(conv_text))
            elif 'text' in record:
                length_stats.append(len(record['text']))

        return {
            'total_records': len(records),
            'has_conversations': has_conversations,
            'has_text': has_text,
            'avg_length': sum(length_stats) / len(length_stats) if length_stats else 0,
            'min_length': min(length_stats) if length_stats else 0,
            'max_length': max(length_stats) if length_stats else 0,
            'median_length': sorted(length_stats)[len(length_stats) // 2] if length_stats else 0,
        }


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="合并数据集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  %(prog)s --input data1.jsonl data2.jsonl --output merged.jsonl
  %(prog)s --input data1.jsonl data2.jsonl --output merged.jsonl --weights 0.7 0.3
  %(prog)s --input pretrain/ sft/ --output merged/ --strategy mixed
        """
    )

    parser.add_argument(
        '--input', '-i',
        type=str,
        nargs='+',
        required=True,
        help='输入文件或目录路径'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='输出文件或目录路径'
    )

    parser.add_argument(
        '--weights', '-w',
        type=float,
        nargs='+',
        help='数据集权重列表'
    )

    parser.add_argument(
        '--dedup',
        action='store_true',
        default=True,
        help='去除重复记录 (默认: True)'
    )

    parser.add_argument(
        '--no-dedup',
        action='store_false',
        dest='dedup',
        help='不去除重复记录'
    )

    parser.add_argument(
        '--shuffle',
        action='store_true',
        default=True,
        help='打乱数据顺序 (默认: True)'
    )

    parser.add_argument(
        '--no-shuffle',
        action='store_false',
        dest='shuffle',
        help='不打乱数据顺序'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子 (默认: 42)'
    )

    parser.add_argument(
        '--output-format',
        type=str,
        choices=['jsonl', 'json'],
        default='jsonl',
        help='输出格式 (默认: jsonl)'
    )

    parser.add_argument(
        '--strategy',
        type=str,
        choices=['merge', 'mixed'],
        default='merge',
        help='合并策略 (默认: merge)'
    )

    parser.add_argument(
        '--analyze-only',
        action='store_true',
        help='仅分析数据集，不合并'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='显示详细日志'
    )

    return parser.parse_args()


def main() -> int:
    """主函数。"""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.analyze_only:
        for input_path in args.input:
            print(f"\n分析: {input_path}")
            print("-" * 40)
            analysis = DatasetAnalyzer.analyze(input_path)
            for key, value in analysis.items():
                print(f"  {key}: {value}")
        return 0

    logger.info("Starting dataset merging")
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")

    merger = DatasetMerger(
        weights=args.weights,
        dedup=args.dedup,
        shuffle=args.shuffle,
        seed=args.seed,
    )

    try:
        if args.strategy == 'mixed':
            pretrain_stats, sft_stats = merger.merge_mixed(
                args.input,
                args.output,
                dedup=args.dedup,
            )

            print("\n" + "=" * 50)
            print("预训练数据统计:")
            print(f"  记录数: {pretrain_stats.total_records}")
            print("\nSFT数据统计:")
            print(f"  记录数: {sft_stats.total_records}")
            print("=" * 50)

        else:
            stats = merger.merge(
                args.input,
                args.output,
                output_format=args.output_format,
            )

            print("\n" + "=" * 50)
            print("合并完成!")
            print("=" * 50)
            print(f"输入数据集数: {stats.total_input}")
            print(f"总记录数: {stats.total_records}")
            print(f"去重后记录数: {stats.unique_records}")
            print(f"去除重复: {stats.duplicates_removed}")
            print(f"输出路径: {args.output}")
            print("\n各数据集统计:")
            for source, count in stats.by_source.items():
                print(f"  {source}: {count}")
            print("=" * 50)

        return 0

    except Exception as e:
        logger.error(f"Merging failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
