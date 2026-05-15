"""
数据集构建器模块
整合多种数据源，构建统一的数据集
"""

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from .poetry_downloader import PoetryDownloader, PoetryItem
from .classical_downloader import ClassicalDownloader, ClassicalText, TextFilter
from .modern_downloader import ModernDownloader, ModernText, ModernTextFilter


@dataclass
class DatasetConfig:
    """数据集配置"""
    name: str = "default_dataset"
    version: str = "1.0.0"
    output_dir: str = "./data/datasets"
    output_format: str = "jsonl"
    compression: Optional[str] = None

    poetry_sources: List[str] = field(default_factory=list)
    classical_sources: List[str] = field(default_factory=list)
    modern_sources: List[str] = field(default_factory=list)

    include_poetry: bool = True
    include_classical: bool = True
    include_modern: bool = False

    min_length: int = 10
    max_length: int = 10000
    remove_duplicates: bool = True

    train_ratio: float = 0.9
    val_ratio: float = 0.05
    test_ratio: float = 0.05

    shuffle: bool = True
    random_seed: int = 42


@dataclass
class DataRecord:
    """统一的数据记录格式"""
    id: str
    text: str
    category: str
    subcategory: str
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class DatasetBuilder:
    """数据集构建器"""

    CATEGORIES = {
        'poetry': ['tangshi', 'songci', 'yuanqu', 'shijing', 'lunyu'],
        'classical': ['guwen', 'jingbu', 'zibu', 'shibu'],
        'modern': ['wiki', 'news', 'baike', 'qa', 'subtitles'],
    }

    def __init__(self, config: Optional[DatasetConfig] = None):
        self.config = config or DatasetConfig()
        self.logger = logging.getLogger(__name__)

        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.poetry_downloader = PoetryDownloader()
        self.classical_downloader = ClassicalDownloader()
        self.modern_downloader = ModernDownloader()

        self._records: List[DataRecord] = []
        self._statistics: Dict[str, Any] = {}

    def add_poetry(self, items: List[PoetryItem]) -> int:
        """添加诗词数据"""
        count = 0
        for item in items:
            if not self._check_length(item.content):
                continue

            record = DataRecord(
                id=f"poetry_{item.id or count}",
                text=item.content,
                category="poetry",
                subcategory=item.type,
                source=f"{item.dynasty}_{item.author}",
                metadata={
                    'title': item.title,
                    'author': item.author,
                    'dynasty': item.dynasty,
                    'type': item.type,
                }
            )
            self._records.append(record)
            count += 1

        self.logger.info(f"添加了 {count} 条诗词记录")
        return count

    def add_classical(self, texts: List[ClassicalText]) -> int:
        """添加古文数据"""
        count = 0
        for text in texts:
            if not self._check_length(text.content):
                continue

            record = DataRecord(
                id=f"classical_{text.id or count}",
                text=text.content,
                category="classical",
                subcategory=text.category,
                source=text.source,
                metadata={
                    'title': text.title,
                    'author': text.author,
                    'dynasty': text.dynasty,
                    'chapter': text.chapter,
                }
            )
            self._records.append(record)
            count += 1

        self.logger.info(f"添加了 {count} 条古文记录")
        return count

    def add_modern(self, texts: List[ModernText]) -> int:
        """添加现代文数据"""
        count = 0
        for text in texts:
            if not self._check_length(text.content):
                continue

            record = DataRecord(
                id=f"modern_{text.id or count}",
                text=text.content,
                category="modern",
                subcategory=text.category,
                source=text.source,
                metadata={
                    'title': text.title,
                    'author': text.author,
                    'publish_date': text.publish_date,
                    'word_count': text.word_count,
                    'tags': text.tags,
                }
            )
            self._records.append(record)
            count += 1

        self.logger.info(f"添加了 {count} 条现代文记录")
        return count

    def _check_length(self, text: str) -> bool:
        """检查文本长度"""
        length = len(text.strip())
        return self.config.min_length <= length <= self.config.max_length

    def remove_duplicates(self) -> int:
        """去除重复记录"""
        seen: Set[str] = set()
        unique_records = []

        for record in self._records:
            normalized = self._normalize_for_dedup(record.text)
            if normalized not in seen:
                seen.add(normalized)
                unique_records.append(record)

        removed = len(self._records) - len(unique_records)
        self._records = unique_records
        self.logger.info(f"去重完成，移除了 {removed} 条重复记录")
        return removed

    def _normalize_for_dedup(self, text: str) -> str:
        """标准化文本用于去重"""
        text = re.sub(r'[\u3000\xa0\s\n]+', '', text)
        return text.lower()

    def filter_by_category(self, category: str) -> int:
        """按类别过滤"""
        original_count = len(self._records)
        self._records = [r for r in self._records if r.category == category]
        removed = original_count - len(self._records)
        self.logger.info(f"过滤完成，保留了 {len(self._records)} 条记录")
        return removed

    def filter_by_length(self, min_length: Optional[int] = None, max_length: Optional[int] = None) -> int:
        """按长度过滤"""
        min_len = min_length or self.config.min_length
        max_len = max_length or self.config.max_length

        original_count = len(self._records)
        self._records = [
            r for r in self._records
            if min_len <= len(r.text) <= max_len
        ]
        removed = original_count - len(self._records)
        self.logger.info(f"按长度过滤完成，移除了 {removed} 条记录")
        return removed

    def shuffle_records(self, seed: Optional[int] = None) -> None:
        """打乱记录顺序"""
        import random
        seed = seed or self.config.random_seed
        random.seed(seed)
        random.shuffle(self._records)
        self.logger.info(f"打乱完成，种子: {seed}")

    def split_dataset(
        self,
        train_ratio: Optional[float] = None,
        val_ratio: Optional[float] = None,
        test_ratio: Optional[float] = None,
    ) -> Dict[str, List[DataRecord]]:
        """分割数据集"""
        train_r = train_ratio or self.config.train_ratio
        val_r = val_ratio or self.config.val_ratio
        test_r = test_ratio or self.config.test_ratio

        total = train_r + val_r + test_r
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"分割比例之和必须为1，当前: {total}")

        if self.config.shuffle:
            self.shuffle_records()

        total_records = len(self._records)
        train_end = int(total_records * train_r)
        val_end = train_end + int(total_records * val_r)

        splits = {
            'train': self._records[:train_end],
            'val': self._records[train_end:val_end],
            'test': self._records[val_end:],
        }

        self.logger.info(
            f"数据集分割完成: train={len(splits['train'])}, "
            f"val={len(splits['val'])}, test={len(splits['test'])}"
        )

        return splits

    def export_to_jsonl(
        self,
        filepath: Optional[Path] = None,
        records: Optional[List[DataRecord]] = None,
        include_metadata: bool = True,
    ) -> Path:
        """导出为JSONL格式"""
        filepath = filepath or self.output_dir / f"{self.config.name}.jsonl"
        records = records or self._records

        with open(filepath, 'w', encoding='utf-8') as f:
            for record in records:
                if include_metadata:
                    output = {
                        'id': record.id,
                        'text': record.text,
                        'category': record.category,
                        'subcategory': record.subcategory,
                        'source': record.source,
                        **record.metadata,
                    }
                else:
                    output = {
                        'id': record.id,
                        'text': record.text,
                        'category': record.category,
                    }
                f.write(json.dumps(output, ensure_ascii=False) + '\n')

        self.logger.info(f"导出完成: {filepath}")
        return filepath

    def export_to_json(
        self,
        filepath: Optional[Path] = None,
        records: Optional[List[DataRecord]] = None,
        include_metadata: bool = True,
    ) -> Path:
        """导出为JSON格式"""
        filepath = filepath or self.output_dir / f"{self.config.name}.json"
        records = records or self._records

        output_list = []
        for record in records:
            if include_metadata:
                output = {
                    'id': record.id,
                    'text': record.text,
                    'category': record.category,
                    'subcategory': record.subcategory,
                    'source': record.source,
                    **record.metadata,
                }
            else:
                output = {
                    'id': record.id,
                    'text': record.text,
                    'category': record.category,
                }
            output_list.append(output)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output_list, f, ensure_ascii=False, indent=2)

        self.logger.info(f"导出完成: {filepath}")
        return filepath

    def export_to_txt(
        self,
        filepath: Optional[Path] = None,
        records: Optional[List[DataRecord]] = None,
        separator: str = "\n\n---\n\n",
    ) -> Path:
        """导出为TXT格式"""
        filepath = filepath or self.output_dir / f"{self.config.name}.txt"
        records = records or self._records

        texts = [record.text for record in records]
        content = separator.join(texts)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

        self.logger.info(f"导出完成: {filepath}")
        return filepath

    def export_splits(
        self,
        prefix: Optional[str] = None,
    ) -> Dict[str, Path]:
        """导出分割后的数据集"""
        prefix = prefix or self.config.name
        splits = self.split_dataset()
        outputs = {}

        for split_name, split_records in splits.items():
            if not split_records:
                continue

            if self.config.output_format == 'jsonl':
                filepath = self.output_dir / f"{prefix}_{split_name}.jsonl"
                self.export_to_jsonl(filepath, split_records)
            elif self.config.output_format == 'json':
                filepath = self.output_dir / f"{prefix}_{split_name}.json"
                self.export_to_json(filepath, split_records)
            else:
                filepath = self.output_dir / f"{prefix}_{split_name}.txt"
                self.export_to_txt(filepath, split_records)

            outputs[split_name] = filepath

        return outputs

    def generate_statistics(self) -> Dict[str, Any]:
        """生成数据统计报告"""
        if not self._records:
            return {'total': 0}

        by_category: Dict[str, int] = {}
        by_subcategory: Dict[str, int] = {}
        by_source: Dict[str, int] = {}
        length_distribution: Dict[str, int] = {
            '0-100': 0,
            '100-500': 0,
            '500-1000': 0,
            '1000-5000': 0,
            '5000+': 0,
        }

        total_length = 0

        for record in self._records:
            by_category[record.category] = by_category.get(record.category, 0) + 1
            by_subcategory[record.subcategory] = by_subcategory.get(record.subcategory, 0) + 1
            by_source[record.source] = by_source.get(record.source, 0) + 1

            length = len(record.text)
            total_length += length

            if length < 100:
                length_distribution['0-100'] += 1
            elif length < 500:
                length_distribution['100-500'] += 1
            elif length < 1000:
                length_distribution['500-1000'] += 1
            elif length < 5000:
                length_distribution['1000-5000'] += 1
            else:
                length_distribution['5000+'] += 1

        self._statistics = {
            'dataset_name': self.config.name,
            'version': self.config.version,
            'total_records': len(self._records),
            'total_characters': total_length,
            'average_length': total_length / len(self._records),
            'by_category': by_category,
            'by_subcategory': by_subcategory,
            'by_source': by_source,
            'length_distribution': length_distribution,
            'generated_at': datetime.now().isoformat(),
        }

        return self._statistics

    def save_statistics(self, filepath: Optional[Path] = None) -> Path:
        """保存统计报告"""
        if not self._statistics:
            self.generate_statistics()

        filepath = filepath or self.output_dir / f"{self.config.name}_statistics.json"

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self._statistics, f, ensure_ascii=False, indent=2)

        self.logger.info(f"统计报告已保存: {filepath}")
        return filepath

    def get_record_count(self) -> int:
        """获取记录数量"""
        return len(self._records)

    def get_records(self, category: Optional[str] = None) -> List[DataRecord]:
        """获取记录"""
        if category:
            return [r for r in self._records if r.category == category]
        return self._records

    def clear_records(self) -> None:
        """清空所有记录"""
        self._records.clear()
        self.logger.info("已清空所有记录")

    async def build_from_sources(
        self,
        poetry_sources: Optional[List[str]] = None,
        classical_sources: Optional[List[str]] = None,
        modern_sources: Optional[List[str]] = None,
        download: bool = True,
    ) -> Dict[str, Any]:
        """
        从数据源构建数据集

        Args:
            poetry_sources: 诗词数据源列表
            classical_sources: 古文数据源列表
            modern_sources: 现代文数据源列表
            download: 是否下载数据

        Returns:
            Dict[str, Any]: 构建结果和统计信息
        """
        results = {
            'poetry_count': 0,
            'classical_count': 0,
            'modern_count': 0,
            'total_count': 0,
            'duplicates_removed': 0,
        }

        if download:
            if self.config.include_poetry and poetry_sources:
                poetry_results = await self.poetry_downloader.download_all(poetry_sources)
                self.logger.info(f"诗词下载完成: {len(poetry_results)} 个文件")

            if self.config.include_classical and classical_sources:
                classical_results = await self.classical_downloader.download_all(classical_sources)
                self.logger.info(f"古文下载完成: {len(classical_results)} 个文件")

            if self.config.include_modern and modern_sources:
                modern_results = await self.modern_downloader.download_all(modern_sources)
                self.logger.info(f"现代文下载完成: {len(modern_results)} 个文件")

        if self.config.include_poetry:
            for source in poetry_sources or self.poetry_downloader.POETRY_SOURCES.keys():
                items = self.poetry_downloader.process_raw_data(source)
                count = self.add_poetry(items)
                results['poetry_count'] += count

        if self.config.include_classical:
            text_filter = TextFilter(
                min_length=self.config.min_length,
                max_length=self.config.max_length,
            )
            texts = self.classical_downloader.process_raw_data(text_filter=text_filter)
            count = self.add_classical(texts)
            results['classical_count'] += count

        if self.config.include_modern:
            modern_filter = ModernTextFilter(
                min_length=self.config.min_length,
                max_length=self.config.max_length,
            )
            modern_texts = self.modern_downloader.process_raw_data(text_filter=modern_filter)
            count = self.add_modern(modern_texts)
            results['modern_count'] += count

        results['total_count'] = len(self._records)

        if self.config.remove_duplicates:
            removed = self.remove_duplicates()
            results['duplicates_removed'] = removed
            results['total_count'] = len(self._records)

        self.generate_statistics()

        return results

    def export_complete(
        self,
        export_splits: bool = True,
        save_stats: bool = True,
    ) -> Dict[str, Path]:
        """
        导出完整数据集

        Args:
            export_splits: 是否导出分割后的数据集
            save_stats: 是否保存统计报告

        Returns:
            Dict[str, Path]: 导出的文件路径
        """
        outputs = {}

        if export_splits:
            outputs.update(self.export_splits())
        else:
            if self.config.output_format == 'jsonl':
                outputs['main'] = self.export_to_jsonl()
            elif self.config.output_format == 'json':
                outputs['main'] = self.export_to_json()
            else:
                outputs['main'] = self.export_to_txt()

        if save_stats:
            outputs['statistics'] = self.save_statistics()

        return outputs

    def print_summary(self) -> None:
        """打印数据集摘要"""
        if not self._statistics:
            self.generate_statistics()

        print(f"\n{'='*60}")
        print(f"数据集摘要: {self.config.name}")
        print(f"{'='*60}")
        print(f"总记录数: {self._statistics['total_records']}")
        print(f"总字符数: {self._statistics['total_characters']:,}")
        print(f"平均长度: {self._statistics['average_length']:.1f}")
        print(f"\n按类别分布:")
        for cat, count in self._statistics['by_category'].items():
            pct = count / self._statistics['total_records'] * 100
            print(f"  {cat}: {count:,} ({pct:.1f}%)")
        print(f"\n长度分布:")
        for range_name, count in self._statistics['length_distribution'].items():
            pct = count / self._statistics['total_records'] * 100
            print(f"  {range_name}: {count:,} ({pct:.1f}%)")
        print(f"{'='*60}\n")


class DataLoader:
    """数据加载器"""

    @staticmethod
    def load_jsonl(filepath: Union[str, Path]) -> List[Dict[str, Any]]:
        """加载JSONL文件"""
        records = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        return records

    @staticmethod
    def load_json(filepath: Union[str, Path]) -> List[Dict[str, Any]]:
        """加载JSON文件"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            return [data]

    @staticmethod
    def load_txt(filepath: Union[str, Path], separator: str = "\n\n---\n\n") -> List[str]:
        """加载TXT文件"""
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            return content.split(separator)

    @staticmethod
    def load_dataset(
        filepath: Union[str, Path],
        format: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """智能加载数据集"""
        filepath = Path(filepath)
        format = format or filepath.suffix[1:]

        if format == 'jsonl':
            return DataLoader.load_jsonl(filepath)
        elif format == 'json':
            return DataLoader.load_json(filepath)
        elif format == 'txt':
            return [{'text': text} for text in DataLoader.load_txt(filepath)]
        else:
            raise ValueError(f"不支持的格式: {format}")
