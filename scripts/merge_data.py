#!/usr/bin/env python3
"""
数据合并脚本 - 灵猫墨韵项目
合并多个数据源，支持数据混合比例配置，按比例采样
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from collections import defaultdict

try:
    import yaml
    from tqdm import tqdm
except ImportError:
    print("请安装依赖: pip install pyyaml tqdm")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('merge_data.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class DataMerger:
    """数据合并器"""

    def __init__(self, input_dir: str = "dataset/processed", output_dir: str = "dataset/merged"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.data_cache: Dict[str, List[Dict]] = {}
        self.category_stats: Dict[str, Dict] = {}

    def load_jsonl(self, file_path: Path) -> List[Dict]:
        """加载JSONL文件"""
        items = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        items.append(json.loads(line))
        except Exception as e:
            logger.error(f"加载文件失败 [{file_path}]: {e}")
        return items

    def save_jsonl(self, file_path: Path, items: List[Dict]):
        """保存JSONL文件"""
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

    def get_data_by_category(self, files: List[Path]) -> Dict[str, List[Dict]]:
        """按类别分组数据"""
        category_data = defaultdict(list)

        for file_path in files:
            items = self.load_jsonl(file_path)

            for item in items:
                category = item.get('category', 'unknown')
                category_data[category].append(item)

        return dict(category_data)

    def sample_by_ratio(self, data: List[Dict], ratio: float) -> List[Dict]:
        """按比例采样"""
        sample_size = int(len(data) * ratio)
        if sample_size >= len(data):
            return data.copy()

        import random
        return random.sample(data, sample_size)

    def merge_with_ratios(
        self,
        category_data: Dict[str, List[Dict]],
        ratios: Dict[str, float],
        total_target: Optional[int] = None
    ) -> List[Dict]:
        """按比例合并数据"""
        merged = []
        total_ratio = sum(ratios.values())

        if total_ratio > 1.0:
            logger.warning(f"比例总和 {total_ratio} 超过 1.0，将进行归一化")
            ratios = {k: v / total_ratio for k, v in ratios.items()}

        for category, ratio in ratios.items():
            if category not in category_data:
                logger.warning(f"类别 {category} 没有可用数据")
                continue

            category_items = category_data[category]
            sampled_items = self.sample_by_ratio(category_items, ratio / sum(ratios.values()))

            merged.extend(sampled_items)
            self.category_stats[category] = {
                'original': len(category_items),
                'sampled': len(sampled_items),
                'ratio': ratio
            }

        return merged

    def create_balanced_dataset(
        self,
        category_data: Dict[str, List[Dict]],
        samples_per_category: Optional[Dict[str, int]] = None
    ) -> List[Dict]:
        """创建均衡数据集"""
        merged = []

        if samples_per_category is None:
            min_size = min(len(items) for items in category_data.values())
            samples_per_category = {cat: min_size for cat in category_data}

        for category, items in category_data.items():
            sample_count = samples_per_category.get(category, len(items))
            sampled = self.sample_by_ratio(items, sample_count / len(items))

            merged.extend(sampled)
            self.category_stats[category] = {
                'original': len(items),
                'sampled': len(sampled),
                'target': sample_count
            }

        return merged


def load_merge_config(config_path: str = "config/merge_config.yaml") -> Dict:
    """加载合并配置"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning(f"配置文件不存在: {config_path}，使用默认配置")
        return {
            'default_ratios': {
                'classical': 0.4,
                'modern': 0.3,
                'baike': 0.2,
                'dialogue': 0.1
            }
        }


def main():
    parser = argparse.ArgumentParser(description='灵猫墨韵 - 数据合并工具')
    parser.add_argument('--input', default='dataset/processed', help='处理后数据目录')
    parser.add_argument('--output', default='dataset/merged', help='合并输出目录')
    parser.add_argument('--config', default='config/merge_config.yaml', help='合并配置文件')
    parser.add_argument('--mode', choices=['ratio', 'balanced', 'all'],
                        default='ratio', help='合并模式')
    parser.add_argument('--total-target', type=int, help='目标总数据量')
    parser.add_argument('--ratios', nargs='+', help='手动指定比例，格式: category:ratio')

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("灵猫墨韵 - 数据合并工具")
    logger.info("=" * 60)

    merger = DataMerger(input_dir=args.input, output_dir=args.output)

    jsonl_files = list(Path(args.input).glob('*.jsonl'))
    jsonl_files = [f for f in jsonl_files if f.stem not in ['train', 'val', 'test']]

    if not jsonl_files:
        logger.warning(f"在 {args.input} 中没有找到数据文件")
        return

    logger.info(f"找到 {len(jsonl_files)} 个数据文件")

    category_data = merger.get_data_by_category(jsonl_files)

    logger.info("\n各类别数据统计:")
    for category, items in category_data.items():
        logger.info(f"  {category}: {len(items)} 条")

    if args.mode == 'ratio':
        if args.ratios:
            ratios = {}
            for r in args.ratios:
                if ':' in r:
                    cat, ratio = r.split(':')
                    ratios[cat.strip()] = float(ratio.strip())
        else:
            config = load_merge_config(args.config)
            ratios = config.get('default_ratios', {})

        logger.info(f"\n使用比例合并模式: {ratios}")

        merged_data = merger.merge_with_ratios(
            category_data,
            ratios,
            total_target=args.total_target
        )

    elif args.mode == 'balanced':
        logger.info("\n使用均衡合并模式")
        merged_data = merger.create_balanced_dataset(category_data)

    else:
        logger.info("\n合并所有数据")
        merged_data = []
        for items in category_data.values():
            merged_data.extend(items)

    import random
    random.shuffle(merged_data)

    output_file = merger.output_dir / 'merged.jsonl'
    merger.save_jsonl(output_file, merged_data)

    logger.info("\n" + "=" * 60)
    logger.info("合并完成!")
    logger.info(f"输出文件: {output_file}")
    logger.info(f"总数据量: {len(merged_data)}")

    logger.info("\n各类别统计:")
    for category, stats in merger.category_stats.items():
        logger.info(f"  {category}:")
        logger.info(f"    - 原始: {stats['original']}")
        logger.info(f"    - 采样后: {stats['sampled']}")

    summary = {
        'total': len(merged_data),
        'categories': merger.category_stats,
        'timestamp': datetime.now().isoformat(),
        'mode': args.mode
    }

    summary_file = merger.output_dir / 'merge_summary.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("=" * 60)


if __name__ == '__main__':
    main()
