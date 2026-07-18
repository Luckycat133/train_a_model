#!/usr/bin/env python3
"""
数据准备脚本 - 灵猫墨韵项目
将原始数据转换为项目格式（JSONL），进行数据清洗、去重，添加特殊标记，生成训练/验证/测试集
"""

import os
import sys
import re
import json
import hashlib
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import random

try:
    from tqdm import tqdm
except ImportError:
    print("请安装依赖: pip install tqdm")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('prepare_data.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


SPECIAL_TOKENS = {
    'cls': '[CLS]',
    'sep': '[SEP]',
    'mask': '[MASK]',
    'unk': '[UNK]',
    'pad': '[PAD]',
    'eod': '[EOD]',
    'eos': '[EOS]',
    'sop': '[SOP]',
    'eop': '[EOP]'
}


class DataCleaner:
    """数据清洗器"""

    def __init__(self):
        self.unicode_pattern = re.compile(
            '[^\u4e00-\u9fff\u3400-\u4dbf\uff00-\uffef'
            '\u3000-\u303f\u2000-\u206f'
            'a-zA-Z0-9\s,.!?;:\'\"。，！？；：""''【】《》（）\(\)\[\]\{\}'
            '\u2014\u2018\u2019\u201c\u201d\u3001\u3002\uff0c\uff0e]+'
        )
        self.whitespace_pattern = re.compile(r'\s+')
        self.html_pattern = re.compile(r'<[^>]+>')
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')

    def clean_text(self, text: str, remove_urls: bool = True,
                   remove_emails: bool = True, normalize_whitespace: bool = True) -> str:
        """清洗文本"""
        if not text:
            return ""

        text = self.html_pattern.sub('', text)

        if remove_urls:
            text = self.url_pattern.sub('[URL]', text)

        if remove_emails:
            text = self.email_pattern.sub('[EMAIL]', text)

        text = self.unicode_pattern.sub('', text)

        if normalize_whitespace:
            text = self.whitespace_pattern.sub(' ', text)

        text = text.strip()

        return text

    def remove_duplicates(self, items: List[Dict]) -> Tuple[List[Dict], Dict]:
        """去重并返回统计信息"""
        seen_hashes: Set[str] = set()
        unique_items = []
        stats = {
            'total': len(items),
            'duplicates': 0,
            'unique': 0
        }

        for item in items:
            content = item.get('content', '')
            if not content:
                continue

            content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()

            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_items.append(item)
            else:
                stats['duplicates'] += 1

        stats['unique'] = len(unique_items)
        return unique_items, stats

    def filter_by_length(self, items: List[Dict], min_length: int = 10,
                         max_length: int = 4096) -> Tuple[List[Dict], Dict]:
        """按长度过滤"""
        filtered = []
        stats = {
            'total': len(items),
            'filtered': 0,
            'kept': 0
        }

        for item in items:
            content = item.get('content', '')
            if min_length <= len(content) <= max_length:
                filtered.append(item)
                stats['kept'] += 1
            else:
                stats['filtered'] += 1

        return filtered, stats


class DataPreparer:
    """数据准备器"""

    def __init__(self, input_dir: str = "dataset/raw", output_dir: str = "dataset/processed"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.cleaner = DataCleaner()

    def add_special_tokens(self, text: str, tokens_config: Optional[Dict] = None) -> str:
        """添加特殊标记"""
        if tokens_config is None:
            tokens_config = {
                'add_cls': True,
                'add_sep': True,
                'cls_token': SPECIAL_TOKENS['cls'],
                'sep_token': SPECIAL_TOKENS['sep']
            }

        text = text.strip()

        if tokens_config.get('add_cls', True):
            text = tokens_config['cls_token'] + text

        if tokens_config.get('add_sep', True):
            text = text + tokens_config['sep_token']

        return text

    def convert_to_jsonl(self, input_file: Path, output_file: Path,
                        category: str = "general") -> Dict:
        """转换单个文件为JSONL格式"""
        stats = {
            'total': 0,
            'cleaned': 0,
            'duplicates_removed': 0,
            'length_filtered': 0,
            'output': 0
        }

        items = []

        try:
            if input_file.suffix == '.json':
                with open(input_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                    if isinstance(data, list):
                        items = data
                    elif isinstance(data, dict):
                        items = [data]
            elif input_file.suffix in ['.txt', '.text']:
                with open(input_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            items.append({'content': line})
            else:
                logger.warning(f"不支持的文件格式: {input_file}")
                return stats

            stats['total'] = len(items)

            cleaned_items = []
            for item in items:
                content = item.get('content', '')
                if content:
                    cleaned = self.cleaner.clean_text(content)
                    if cleaned:
                        item['content'] = cleaned
                        cleaned_items.append(item)

            stats['cleaned'] = len(cleaned_items)

            unique_items, dup_stats = self.cleaner.remove_duplicates(cleaned_items)
            stats['duplicates_removed'] = dup_stats['duplicates']

            filtered_items, filter_stats = self.cleaner.filter_by_length(unique_items)
            stats['length_filtered'] = filter_stats['filtered']

            final_items = []
            for item in filtered_items:
                output_item = {
                    'id': hashlib.md5(item.get('content', '').encode()).hexdigest()[:16],
                    'category': category,
                    'content': self.add_special_tokens(item['content']),
                    'source': item.get('source', str(input_file.name)),
                    'metadata': item.get('metadata', {})
                }

                if 'title' in item:
                    output_item['title'] = item['title']
                if 'author' in item:
                    output_item['author'] = item['author']

                final_items.append(output_item)

            stats['output'] = len(final_items)

            with open(output_file, 'w', encoding='utf-8') as f:
                for item in final_items:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')

        except Exception as e:
            logger.error(f"转换文件失败 [{input_file}]: {e}")

        return stats

    def prepare_dataset(self, categories: List[str] = None,
                       train_ratio: float = 0.8,
                       val_ratio: float = 0.1,
                       test_ratio: float = 0.1) -> Dict:
        """准备完整数据集"""
        logger.info("=" * 60)
        logger.info("开始准备数据集...")
        logger.info("=" * 60)

        all_files = list(self.input_dir.glob('*'))
        json_files = [f for f in all_files if f.suffix in ['.json', '.txt', '.text']]

        if not json_files:
            logger.warning(f"在 {self.input_dir} 中没有找到数据文件")
            return {}

        category_map = {
            'classical': '古文',
            'modern': '现代文',
            'baike': '百科',
            'dialogue': '对话'
        }

        all_data = []

        for input_file in tqdm(json_files, desc="处理文件"):
            category = "general"
            for cat, _ in category_map.items():
                if cat in input_file.stem.lower():
                    category = cat
                    break

            if categories and category not in categories:
                continue

            output_file = self.output_dir / f"{input_file.stem}_processed.jsonl"
            stats = self.convert_to_jsonl(input_file, output_file, category)

            logger.info(f"文件处理完成: {input_file.name}")
            logger.info(f"  - 总数: {stats['total']}")
            logger.info(f"  - 清洗后: {stats['cleaned']}")
            logger.info(f"  - 去重: {stats['duplicates_removed']}")
            logger.info(f"  - 长度过滤: {stats['length_filtered']}")
            logger.info(f"  - 最终输出: {stats['output']}")

            with open(output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    all_data.append(json.loads(line))

        random.shuffle(all_data)

        total_size = len(all_data)
        train_size = int(total_size * train_ratio)
        val_size = int(total_size * val_ratio)

        train_data = all_data[:train_size]
        val_data = all_data[train_size:train_size + val_size]
        test_data = all_data[train_size + val_size:]

        split_stats = {}

        for split_name, split_data in [('train', train_data), ('val', val_data), ('test', test_data)]:
            split_file = self.output_dir / f"{split_name}.jsonl"
            with open(split_file, 'w', encoding='utf-8') as f:
                for item in split_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            split_stats[split_name] = len(split_data)
            logger.info(f"生成 {split_name} 集: {len(split_data)} 条数据 -> {split_file}")

        summary = {
            'total_processed': total_size,
            'splits': split_stats,
            'timestamp': datetime.now().isoformat()
        }

        summary_file = self.output_dir / 'data_summary.json'
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info("\n" + "=" * 60)
        logger.info("数据集准备完成!")
        logger.info(f"总数据量: {total_size}")
        logger.info(f"训练集: {split_stats['train']}")
        logger.info(f"验证集: {split_stats['val']}")
        logger.info(f"测试集: {split_stats['test']}")
        logger.info("=" * 60)

        return summary


def process_single_file(args: Tuple) -> Dict:
    """处理单个文件（用于并行处理）"""
    input_file, output_file, category = args
    preparer = DataPreparer()
    return preparer.convert_to_jsonl(input_file, output_file, category)


def main():
    parser = argparse.ArgumentParser(description='灵猫墨韵 - 数据准备工具')
    parser.add_argument('--input', default='dataset/raw', help='原始数据目录')
    parser.add_argument('--output', default='dataset/processed', help='输出目录')
    parser.add_argument('--categories', nargs='+',
                        choices=['classical', 'modern', 'baike', 'dialogue', 'all'],
                        default=['all'], help='处理的数据类别')
    parser.add_argument('--train-ratio', type=float, default=0.8, help='训练集比例')
    parser.add_argument('--val-ratio', type=float, default=0.1, help='验证集比例')
    parser.add_argument('--test-ratio', type=float, default=0.1, help='测试集比例')
    parser.add_argument('--workers', type=int, default=1, help='并行处理线程数')

    args = parser.parse_args()

    categories = None
    if 'all' not in args.categories:
        categories = args.categories

    logger.info("=" * 60)
    logger.info("灵猫墨韵 - 数据准备工具")
    logger.info("=" * 60)
    logger.info(f"输入目录: {args.input}")
    logger.info(f"输出目录: {args.output}")
    logger.info(f"数据划分: 训练 {args.train_ratio} / 验证 {args.val_ratio} / 测试 {args.test_ratio}")

    preparer = DataPreparer(input_dir=args.input, output_dir=args.output)
    summary = preparer.prepare_dataset(
        categories=categories,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio
    )

    if not summary:
        logger.warning("没有处理任何数据，请检查数据文件是否存在")
        sys.exit(1)


if __name__ == '__main__':
    main()
