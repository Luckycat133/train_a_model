#!/usr/bin/env python3
"""预训练数据准备脚本。

完整的预训练数据处理流程，包括：
- 数据加载和格式检测
- 数据清洗
- 数据过滤
- 文本标准化
- 数据验证
- 输出格式转换

使用示例：
    # 基本用法
    python scripts/prepare_pretrain_data.py --input data/raw/ --output data/pretrain/

    # 自定义配置
    python scripts/prepare_pretrain_data.py \\
        --input data/raw/ \\
        --output data/pretrain/ \\
        --min-length 50 \\
        --max-length 5000 \\
        --simplify-chinese \\
        --output-format jsonl

    # 显示进度
    python scripts/prepare_pretrain_data.py \\
        --input data/raw/ \\
        --output data/pretrain/ \\
        --show-progress
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Iterator
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_processing import DataCleaner, DataFilter, DataNormalizer, DataFormatter, DataValidator
from src.data_processing.cleaner import CleaningConfig
from src.data_processing.filter import FilterConfig
from src.data_processing.normalizer import NormalizerConfig
from src.data_processing.formatter import FormatConfig
from src.data_processing.validator import ValidationConfig
from src.logger import get_logger

logger = get_logger("LingmaoMoyun.PreparePretrainData")


@dataclass
class ProcessingStats:
    """处理统计信息。"""

    total_input: int = 0
    cleaned: int = 0
    filtered: int = 0
    normalized: int = 0
    validated: int = 0
    output: int = 0
    start_time: float = 0
    end_time: float = 0

    def duration(self) -> float:
        """获取处理耗时（秒）。"""
        if self.end_time > 0:
            return self.end_time - self.start_time
        return time.time() - self.start_time

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            'total_input': self.total_input,
            'cleaned': self.cleaned,
            'filtered': self.filtered,
            'normalized': self.normalized,
            'validated': self.validated,
            'output': self.output,
            'duration_seconds': self.duration(),
            'records_per_second': self.output / self.duration() if self.duration() > 0 else 0,
        }


class PretrainDataProcessor:
    """预训练数据处理器。

    完整的预训练数据处理流程。

    Attributes:
        cleaner: 数据清洗器。
        data_filter: 数据过滤器。
        normalizer: 文本标准化器。
        formatter: 格式转换器。
        validator: 数据验证器。

    Example:
        >>> processor = PretrainDataProcessor()
        >>> processor.process('input/', 'output/')
    """

    def __init__(
        self,
        min_length: int = 10,
        max_length: int = 100000,
        simplify_chinese: bool = True,
        remove_html: bool = True,
        remove_urls: bool = True,
        output_format: str = "jsonl",
        show_progress: bool = False,
    ):
        """初始化预训练数据处理器。

        Args:
            min_length: 最小文本长度。
            max_length: 最大文本长度。
            simplify_chinese: 是否进行繁简转换。
            remove_html: 是否移除HTML标签。
            remove_urls: 是否移除URL。
            output_format: 输出格式。
            show_progress: 是否显示进度。
        """
        self.cleaner = DataCleaner(CleaningConfig(
            remove_html=remove_html,
            remove_urls=remove_urls,
            normalize_whitespace=True,
            remove_control_chars=True,
        ))

        self.data_filter = DataFilter(FilterConfig(
            min_length=min_length,
            max_length=max_length,
            check_length=True,
            check_quality=True,
            check_duplicates=True,
        ))

        self.normalizer = DataNormalizer(NormalizerConfig(
            simplify_chinese=simplify_chinese,
            normalize_punctuation=True,
            normalize_whitespace=True,
            unicode_normalize="NFKC",
        ))

        self.formatter = DataFormatter(
            input_format="jsonl",
            output_format=output_format,
            config=FormatConfig(
                add_id_field=True,
                id_field_name="id",
            )
        )

        self.validator = DataValidator(ValidationConfig(
            required_fields=["text"],
            min_text_length=min_length,
            max_text_length=max_length,
        ))

        self.show_progress = show_progress
        self.stats = ProcessingStats()

    def process(
        self,
        input_path: str,
        output_path: str,
        batch_size: int = 1000,
    ) -> ProcessingStats:
        """处理预训练数据。

        Args:
            input_path: 输入文件或目录路径。
            output_path: 输出文件或目录路径。
            batch_size: 批处理大小。

        Returns:
            处理统计信息。

        Example:
            >>> processor = PretrainDataProcessor()
            >>> stats = processor.process('data/raw/', 'data/pretrain/')
            >>> print(f"Processed {stats.output} records in {stats.duration():.2f}s")
        """
        self.stats = ProcessingStats()
        self.stats.start_time = time.time()

        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        input_files = self._get_input_files(input_path)
        self.stats.total_input = len(input_files)

        logger.info(f"Found {len(input_files)} input files")

        all_records = []
        for input_file in input_files:
            records = self._load_file(input_file)
            all_records.extend(records)

        logger.info(f"Loaded {len(all_records)} records from input")

        if self.show_progress:
            print(f"[1/5] Loaded {len(all_records)} records")

        cleaned_records = self._clean_records(all_records)
        self.stats.cleaned = len(cleaned_records)

        if self.show_progress:
            print(f"[2/5] Cleaned: {len(cleaned_records)} records")

        filtered_records = self._filter_records(cleaned_records)
        self.stats.filtered = len(filtered_records)

        if self.show_progress:
            print(f"[3/5] Filtered: {len(filtered_records)} records (removed {len(cleaned_records) - len(filtered_records)})")

        normalized_records = self._normalize_records(filtered_records)
        self.stats.normalized = len(normalized_records)

        if self.show_progress:
            print(f"[4/5] Normalized: {len(normalized_records)} records")

        validated_records = self._validate_records(normalized_records)
        self.stats.validated = len(validated_records)

        if self.show_progress:
            print(f"[5/5] Validated: {len(validated_records)} records")

        output_file = output_path / f"pretrain_data.{self.formatter.output_format}"
        self._save_records(validated_records, output_file)
        self.stats.output = len(validated_records)

        self.stats.end_time = time.time()

        logger.info(f"Processing complete: {self.stats.to_dict()}")
        return self.stats

    def _get_input_files(self, input_path: Path) -> List[Path]:
        """获取输入文件列表。"""
        if input_path.is_file():
            return [input_path]
        elif input_path.is_dir():
            files = []
            for ext in ['.jsonl', '.json', '.txt', '.csv']:
                files.extend(input_path.glob(f"**/*{ext}"))
            return sorted(files)
        else:
            raise ValueError(f"Invalid input path: {input_path}")

    def _load_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """加载文件。"""
        try:
            if file_path.suffix == '.jsonl':
                with open(file_path, 'r', encoding='utf-8') as f:
                    return [json.loads(line) for line in f if line.strip()]
            elif file_path.suffix == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return data
                    return [data]
            elif file_path.suffix == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    return [{'text': line.strip()} for line in f if line.strip()]
            elif file_path.suffix == '.csv':
                import csv
                with open(file_path, 'r', encoding='utf-8', newline='') as f:
                    reader = csv.DictReader(f)
                    return [row for row in reader]
            return []
        except Exception as e:
            logger.error(f"Failed to load file {file_path}: {e}")
            return []

    def _clean_records(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """清洗记录。"""
        cleaned = []
        for record in records:
            text = self._extract_text(record)
            if text:
                cleaned_text = self.cleaner.clean(text)
                if cleaned_text.strip():
                    record['text'] = cleaned_text
                    cleaned.append(record)
        return cleaned

    def _filter_records(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """过滤记录。"""
        return self.data_filter.filter_batch(records)

    def _normalize_records(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """标准化记录。"""
        normalized = []
        for record in records:
            text = self._extract_text(record)
            if text:
                normalized_text = self.normalizer.normalize(text)
                if normalized_text.strip():
                    record['text'] = normalized_text
                    normalized.append(record)
        return normalized

    def _validate_records(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """验证记录。"""
        valid_records = []
        for record in records:
            if self.validator.validate(record):
                valid_records.append(record)
        return valid_records

    def _save_records(self, records: List[Dict[str, Any]], output_path: Path) -> None:
        """保存记录。"""
        if self.formatter.output_format == "jsonl":
            with open(output_path, 'w', encoding='utf-8') as f:
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
        elif self.formatter.output_format == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(records, f, ensure_ascii=False, indent=2)
        elif self.formatter.output_format == "txt":
            with open(output_path, 'w', encoding='utf-8') as f:
                for record in records:
                    text = record.get('text', '')
                    f.write(text + '\n')

    def _extract_text(self, record: Dict[str, Any]) -> Optional[str]:
        """提取文本。"""
        for key in ['text', 'content', 'body', 'input', 'output']:
            if key in record and isinstance(record[key], str):
                return record[key]
        return None


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="准备预训练数据",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  %(prog)s --input data/raw/ --output data/pretrain/
  %(prog)s --input data/raw/ --output data/pretrain/ --min-length 50 --max-length 5000
  %(prog)s --input data/raw/ --output data/pretrain/ --simplify-chinese --show-progress
        """
    )

    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='输入文件或目录路径'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='输出目录路径'
    )

    parser.add_argument(
        '--min-length',
        type=int,
        default=10,
        help='最小文本长度 (默认: 10)'
    )

    parser.add_argument(
        '--max-length',
        type=int,
        default=100000,
        help='最大文本长度 (默认: 100000)'
    )

    parser.add_argument(
        '--simplify-chinese',
        action='store_true',
        default=True,
        help='进行繁体转简体 (默认: True)'
    )

    parser.add_argument(
        '--keep-traditional',
        action='store_false',
        dest='simplify_chinese',
        help='不进行繁体转简体'
    )

    parser.add_argument(
        '--remove-html',
        action='store_true',
        default=True,
        help='移除HTML标签 (默认: True)'
    )

    parser.add_argument(
        '--remove-urls',
        action='store_true',
        default=True,
        help='移除URL (默认: True)'
    )

    parser.add_argument(
        '--output-format',
        type=str,
        choices=['jsonl', 'json', 'txt'],
        default='jsonl',
        help='输出格式 (默认: jsonl)'
    )

    parser.add_argument(
        '--show-progress',
        action='store_true',
        help='显示处理进度'
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

    logger.info("Starting pretrain data preparation")
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")

    processor = PretrainDataProcessor(
        min_length=args.min_length,
        max_length=args.max_length,
        simplify_chinese=args.simplify_chinese,
        remove_html=args.remove_html,
        remove_urls=args.remove_urls,
        output_format=args.output_format,
        show_progress=args.show_progress,
    )

    try:
        stats = processor.process(args.input, args.output)

        print("\n" + "=" * 50)
        print("处理完成!")
        print("=" * 50)
        print(f"输入文件数: {stats.total_input}")
        print(f"处理记录数: {stats.output}")
        print(f"处理耗时: {stats.duration():.2f} 秒")
        print(f"处理速度: {stats.to_dict()['records_per_second']:.2f} records/s")
        print(f"输出路径: {args.output}")
        print("=" * 50)

        return 0

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
