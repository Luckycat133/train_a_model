#!/usr/bin/env python3
"""SFT数据准备脚本。

完整的SFT（监督微调）数据处理流程，包括：
- 对话格式解析
- 数据清洗
- 数据过滤
- 格式标准化
- 数据验证
- 输出格式转换

支持多种对话格式：
- ShareGPT格式
- Alpaca格式
- 自定义JSON格式

使用示例：
    # 基本用法
    python scripts/prepare_sft_data.py --input data/raw/ --output data/sft/

    # 使用ShareGPT格式
    python scripts/prepare_sft_data.py \\
        --input data/sharegpt.json \\
        --output data/sft/ \\
        --format sharegpt

    # 自定义配置
    python scripts/prepare_sft_data.py \\
        --input data/raw/ \\
        --output data/sft/ \\
        --format alpaca \\
        --min-turns 2 \\
        --show-progress
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_processing import DataCleaner, DataFilter, DataNormalizer, DataValidator
from src.data_processing.cleaner import CleaningConfig
from src.data_processing.filter import FilterConfig
from src.data_processing.normalizer import NormalizerConfig
from src.data_processing.validator import ValidationConfig
from src.logger import get_logger

logger = get_logger("LingmaoMoyun.PrepareSFTData")


class DialogueFormat(Enum):
    """对话格式枚举。"""
    SHARE_GPT = "sharegpt"
    ALPACA = "alpaca"
    CUSTOM = "custom"


@dataclass
class ProcessingStats:
    """处理统计信息。"""
    total_input: int = 0
    parsed: int = 0
    cleaned: int = 0
    filtered: int = 0
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
            'parsed': self.parsed,
            'cleaned': self.cleaned,
            'filtered': self.filtered,
            'validated': self.validated,
            'output': self.output,
            'duration_seconds': self.duration(),
            'records_per_second': self.output / self.duration() if self.duration() > 0 else 0,
        }


class SFTDataProcessor:
    """SFT数据处理器。

    完整的SFT数据处理流程。

    Attributes:
        dialogue_format: 对话格式。
        cleaner: 数据清洗器。
        data_filter: 数据过滤器。
        normalizer: 文本标准化器。
        validator: 数据验证器。

    Example:
        >>> processor = SFTDataProcessor(format=DialogueFormat.SHARE_GPT)
        >>> processor.process('input/', 'output/')
    """

    def __init__(
        self,
        dialogue_format: DialogueFormat = DialogueFormat.SHARE_GPT,
        min_turns: int = 1,
        max_turns: int = 100,
        min_response_length: int = 10,
        simplify_chinese: bool = True,
        remove_html: bool = True,
        show_progress: bool = False,
    ):
        """初始化SFT数据处理器。

        Args:
            dialogue_format: 对话格式。
            min_turns: 最小对话轮数。
            max_turns: 最大对话轮数。
            min_response_length: 最小回复长度。
            simplify_chinese: 是否进行繁简转换。
            remove_html: 是否移除HTML标签。
            show_progress: 是否显示进度。
        """
        self.dialogue_format = dialogue_format
        self.min_turns = min_turns
        self.max_turns = max_turns
        self.min_response_length = min_response_length

        self.cleaner = DataCleaner(CleaningConfig(
            remove_html=remove_html,
            remove_urls=True,
            normalize_whitespace=True,
            remove_control_chars=True,
        ))

        self.data_filter = DataFilter(FilterConfig(
            min_length=min_response_length,
            max_length=100000,
            check_length=True,
            check_quality=True,
        ))

        self.normalizer = DataNormalizer(NormalizerConfig(
            simplify_chinese=simplify_chinese,
            normalize_punctuation=True,
            normalize_whitespace=True,
            unicode_normalize="NFKC",
        ))

        self.validator = DataValidator(ValidationConfig(
            required_fields=["conversations"],
            min_text_length=min_response_length,
        ))

        self.show_progress = show_progress
        self.stats = ProcessingStats()

    def process(
        self,
        input_path: str,
        output_path: str,
        output_format: str = "jsonl",
    ) -> ProcessingStats:
        """处理SFT数据。

        Args:
            input_path: 输入文件或目录路径。
            output_path: 输出目录路径。
            output_format: 输出格式。

        Returns:
            处理统计信息。

        Example:
            >>> processor = SFTDataProcessor()
            >>> stats = processor.process('data/raw/', 'data/sft/')
            >>> print(f"Processed {stats.output} dialogues")
        """
        self.stats = ProcessingStats()
        self.stats.start_time = time.time()

        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        raw_data = self._load_input(input_path)
        self.stats.total_input = len(raw_data)

        logger.info(f"Loaded {len(raw_data)} raw records")

        if self.show_progress:
            print(f"[1/5] Loaded {len(raw_data)} records")

        parsed_dialogues = self._parse_dialogues(raw_data)
        self.stats.parsed = len(parsed_dialogues)

        if self.show_progress:
            print(f"[2/5] Parsed: {len(parsed_dialogues)} dialogues")

        cleaned_dialogues = self._clean_dialogues(parsed_dialogues)
        self.stats.cleaned = len(cleaned_dialogues)

        if self.show_progress:
            print(f"[3/5] Cleaned: {len(cleaned_dialogues)} dialogues")

        filtered_dialogues = self._filter_dialogues(cleaned_dialogues)
        self.stats.filtered = len(filtered_dialogues)

        if self.show_progress:
            print(f"[4/5] Filtered: {len(filtered_dialogues)} dialogues")

        validated_dialogues = self._validate_dialogues(filtered_dialogues)
        self.stats.validated = len(validated_dialogues)

        if self.show_progress:
            print(f"[5/5] Validated: {len(validated_dialogues)} dialogues")

        output_file = output_path / f"sft_data.{output_format}"
        self._save_dialogues(validated_dialogues, output_file, output_format)
        self.stats.output = len(validated_dialogues)

        self.stats.end_time = time.time()

        logger.info(f"Processing complete: {self.stats.to_dict()}")
        return self.stats

    def _load_input(self, input_path: Path) -> List[Dict[str, Any]]:
        """加载输入数据。"""
        try:
            if input_path.suffix == '.json':
                with open(input_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return data
                    return [data]
            elif input_path.suffix == '.jsonl':
                with open(input_path, 'r', encoding='utf-8') as f:
                    return [json.loads(line) for line in f if line.strip()]
            return []
        except Exception as e:
            logger.error(f"Failed to load input: {e}")
            return []

    def _parse_dialogues(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """解析对话数据。"""
        dialogues = []

        for record in raw_data:
            try:
                if self.dialogue_format == DialogueFormat.SHARE_GPT:
                    dialogue = self._parse_sharegpt(record)
                elif self.dialogue_format == DialogueFormat.ALPACA:
                    dialogue = self._parse_alpaca(record)
                else:
                    dialogue = self._parse_custom(record)

                if dialogue:
                    dialogues.append(dialogue)

            except Exception as e:
                logger.debug(f"Failed to parse record: {e}")
                continue

        return dialogues

    def _parse_sharegpt(self, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """解析ShareGPT格式。"""
        conversations = record.get('conversations', [])

        if not conversations or len(conversations) < 2:
            return None

        parsed_turns = []
        for turn in conversations:
            role = turn.get('from', '')
            value = turn.get('value', '')

            if role == 'human':
                role = 'user'
            elif role == 'gpt':
                role = 'assistant'
            else:
                continue

            if value.strip():
                parsed_turns.append({
                    'role': role,
                    'content': value.strip(),
                })

        if len(parsed_turns) < self.min_turns:
            return None

        return {
            'id': record.get('id', ''),
            'conversations': parsed_turns[:self.max_turns],
            'source': record.get('source', 'sharegpt'),
        }

    def _parse_alpaca(self, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """解析Alpaca格式。"""
        instruction = record.get('instruction', '')
        input_text = record.get('input', '')
        output = record.get('output', '')

        if not instruction or not output:
            return None

        conversations = [
            {'role': 'user', 'content': instruction},
            {'role': 'assistant', 'content': output},
        ]

        if input_text.strip():
            conversations[0]['content'] = f"{instruction}\n{input_text}"

        return {
            'id': record.get('id', ''),
            'conversations': conversations,
            'source': record.get('source', 'alpaca'),
        }

    def _parse_custom(self, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """解析自定义格式。"""
        if 'conversations' in record:
            return record

        messages = record.get('messages', [])
        if messages:
            parsed_turns = []
            for msg in messages:
                role = msg.get('role', '')
                content = msg.get('content', '')
                if role and content:
                    parsed_turns.append({'role': role, 'content': content})

            if parsed_turns:
                return {
                    'id': record.get('id', ''),
                    'conversations': parsed_turns,
                    'source': 'custom',
                }

        return None

    def _clean_dialogues(self, dialogues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """清洗对话数据。"""
        cleaned = []

        for dialogue in dialogues:
            cleaned_turns = []
            for turn in dialogue.get('conversations', []):
                content = turn.get('content', '')
                if content:
                    cleaned_content = self.cleaner.clean(content)
                    if cleaned_content.strip():
                        cleaned_turns.append({
                            'role': turn.get('role'),
                            'content': cleaned_content,
                        })

            if cleaned_turns:
                dialogue['conversations'] = cleaned_turns
                cleaned.append(dialogue)

        return cleaned

    def _filter_dialogues(self, dialogues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """过滤对话数据。"""
        filtered = []

        for dialogue in dialogues:
            turns = dialogue.get('conversations', [])

            if len(turns) < self.min_turns:
                continue

            has_valid_response = False
            for turn in turns:
                if turn.get('role') == 'assistant':
                    content = turn.get('content', '')
                    if len(content) >= self.min_response_length:
                        has_valid_response = True
                        break

            if has_valid_response:
                filtered.append(dialogue)

        return filtered

    def _validate_dialogues(self, dialogues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """验证对话数据。"""
        valid = []

        for dialogue in dialogues:
            if not dialogue.get('conversations'):
                continue

            is_valid = True
            for turn in dialogue['conversations']:
                if not turn.get('role') or not turn.get('content'):
                    is_valid = False
                    break

            if is_valid:
                normalized_dialogue = self._normalize_dialogue(dialogue)
                valid.append(normalized_dialogue)

        return valid

    def _normalize_dialogue(self, dialogue: Dict[str, Any]) -> Dict[str, Any]:
        """标准化对话数据。"""
        normalized_turns = []

        for turn in dialogue.get('conversations', []):
            content = self.normalizer.normalize(turn.get('content', ''))
            normalized_turns.append({
                'role': turn.get('role'),
                'content': content,
            })

        dialogue['conversations'] = normalized_turns

        if 'id' not in dialogue or not dialogue['id']:
            import uuid
            dialogue['id'] = str(uuid.uuid4())

        return dialogue

    def _save_dialogues(
        self,
        dialogues: List[Dict[str, Any]],
        output_path: Path,
        output_format: str,
    ) -> None:
        """保存对话数据。"""
        with open(output_path, 'w', encoding='utf-8') as f:
            if output_format == 'jsonl':
                for dialogue in dialogues:
                    f.write(json.dumps(dialogue, ensure_ascii=False) + '\n')
            elif output_format == 'json':
                json.dump(dialogues, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="准备SFT数据",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  %(prog)s --input data/raw/ --output data/sft/
  %(prog)s --input data/sharegpt.json --output data/sft/ --format sharegpt
  %(prog)s --input data/alpaca.json --output data/sft/ --format alpaca
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
        '--format', '-f',
        type=str,
        choices=['sharegpt', 'alpaca', 'custom'],
        default='sharegpt',
        help='对话格式 (默认: sharegpt)'
    )

    parser.add_argument(
        '--min-turns',
        type=int,
        default=1,
        help='最小对话轮数 (默认: 1)'
    )

    parser.add_argument(
        '--max-turns',
        type=int,
        default=100,
        help='最大对话轮数 (默认: 100)'
    )

    parser.add_argument(
        '--min-response-length',
        type=int,
        default=10,
        help='最小回复长度 (默认: 10)'
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
        '--output-format',
        type=str,
        choices=['jsonl', 'json'],
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

    logger.info("Starting SFT data preparation")
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Format: {args.format}")

    format_map = {
        'sharegpt': DialogueFormat.SHARE_GPT,
        'alpaca': DialogueFormat.ALPACA,
        'custom': DialogueFormat.CUSTOM,
    }

    processor = SFTDataProcessor(
        dialogue_format=format_map[args.format],
        min_turns=args.min_turns,
        max_turns=args.max_turns,
        min_response_length=args.min_response_length,
        simplify_chinese=args.simplify_chinese,
        show_progress=args.show_progress,
    )

    try:
        stats = processor.process(args.input, args.output, args.output_format)

        print("\n" + "=" * 50)
        print("处理完成!")
        print("=" * 50)
        print(f"输入记录数: {stats.total_input}")
        print(f"解析对话数: {stats.parsed}")
        print(f"输出对话数: {stats.output}")
        print(f"处理耗时: {stats.duration():.2f} 秒")
        print(f"处理速度: {stats.to_dict()['records_per_second']:.2f} dialogues/s")
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
