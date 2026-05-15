"""
古诗词下载器模块
从 chinese-poetry 等源下载唐诗、宋词等古典诗词数据
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base_downloader import BaseDownloader, FormatConverter


@dataclass
class PoetryItem:
    """诗词条目"""
    title: str
    author: str
    content: str
    dynasty: str
    type: str
    id: Optional[str] = None


class PoetryDownloader(BaseDownloader):
    """古诗词下载器"""

    CHINESE_POETRY_REPO = "https://github.com/chinese-poetry/chinese-poetry"
    API_BASE = "https://raw.githubusercontent.com/chinese-poetry/chinese-poetry/master"

    POETRY_SOURCES = {
        'tangshi': {
            'url_template': f"{API_BASE}/json/poet.tang.{{i}}.json",
            'indices': list(range(0, 58)),
            'type': 'poetry',
            'dynasty': '唐',
            'author_field': 'author',
            'title_field': 'title',
            'content_field': 'paragraphs',
        },
        'songci': {
            'url_template': f"{API_BASE}/json/ci.song.{{i}}.json",
            'indices': list(range(0, 200)),
            'type': 'ci',
            'dynasty': '宋',
            'author_field': 'author',
            'title_field': 'rhythmic',
            'content_field': 'paragraphs',
        },
        'yuanqu': {
            'url_template': f"{API_BASE}/json/yuanqu/{{i}}.json",
            'indices': list(range(0, 50)),
            'type': 'qu',
            'dynasty': '元',
            'author_field': 'author',
            'title_field': 'title',
            'content_field': 'paragraphs',
        },
        'shijing': {
            'url_template': f"{API_BASE}/json/shijing.json",
            'indices': [0],
            'type': 'shi',
            'dynasty': '先秦',
            'author_field': 'author',
            'title_field': 'title',
            'content_field': 'content',
        },
        'lunyu': {
            'url_template': f"{API_BASE}/json/lunyu.json",
            'indices': [0],
            'type': 'lunyu',
            'dynasty': '先秦',
            'author_field': 'author',
            'title_field': 'chapter',
            'content_field': 'content',
        },
    }

    def __init__(
        self,
        base_dir: str = "./data/poetry",
        max_concurrent: int = 5,
        rate_limit: float = 10.0,
    ):
        super().__init__(base_dir, max_concurrent, rate_limit)
        self.logger = logging.getLogger(__name__)
        self._raw_dir = self.base_dir / "raw"
        self._processed_dir = self.base_dir / "processed"
        self._raw_dir.mkdir(parents=True, exist_ok=True)
        self._processed_dir.mkdir(parents=True, exist_ok=True)

    async def download_all(self, sources: Optional[List[str]] = None) -> Dict[str, List[Path]]:
        """
        下载所有选定的诗词数据源

        Args:
            sources: 要下载的数据源列表，None表示下载所有

        Returns:
            Dict[str, List[Path]]: 各数据源的下载文件路径
        """
        sources = sources or list(self.POETRY_SOURCES.keys())
        results = {}

        for source in sources:
            if source not in self.POETRY_SOURCES:
                self.logger.warning(f"未知数据源: {source}")
                continue

            self.logger.info(f"开始下载 {source}...")
            try:
                files = await self._download_source(source)
                results[source] = files
                self.logger.info(f"{source} 下载完成，共 {len(files)} 个文件")
            except Exception as e:
                self.logger.error(f"{source} 下载失败: {e}")

        return results

    async def _download_source(self, source: str) -> List[Path]:
        """下载单个数据源"""
        config = self.POETRY_SOURCES[source]
        files = []

        tasks = []
        filenames = []

        for i in config['indices']:
            url = config['url_template'].format(i=i)
            filename = f"{source}_{i}.json"
            tasks.append(self.download(url, filename))
            filenames.append(filename)

        downloaded_files = await asyncio.gather(*tasks, return_exceptions=True)

        for filename, result in zip(filenames, downloaded_files):
            if isinstance(result, Exception):
                self.logger.warning(f"下载失败 {filename}: {result}")
            else:
                files.append(result)

        return files

    async def download_poetry_collection(
        self,
        collection: str = "tangshi",
        part: int = 0,
    ) -> Path:
        """下载特定的诗词集合"""
        config = self.POETRY_SOURCES.get(collection)
        if not config:
            raise ValueError(f"未知的诗词集合: {collection}")

        url = config['url_template'].format(i=part)
        filename = f"{collection}_{part}.json"

        return await self.download(url, filename)

    def process_raw_data(
        self,
        source: str,
        min_length: int = 5,
        max_length: int = 500,
        remove_duplicates: bool = True,
    ) -> List[PoetryItem]:
        """
        处理原始下载的诗词数据

        Args:
            source: 数据源名称
            min_length: 最小诗句长度
            max_length: 最大诗句长度
            remove_duplicates: 是否去重

        Returns:
            List[PoetryItem]: 处理后的诗词列表
        """
        config = self.POETRY_SOURCES.get(source)
        if not config:
            raise ValueError(f"未知的诗词集合: {source}")

        all_items = []

        for json_file in self._raw_dir.glob(f"{source}_*.json"):
            try:
                items = self._process_file(
                    json_file,
                    config,
                    min_length,
                    max_length,
                )
                all_items.extend(items)
            except Exception as e:
                self.logger.warning(f"处理文件失败 {json_file}: {e}")

        if remove_duplicates:
            all_items = self._remove_duplicates(all_items)

        self.logger.info(f"处理完成，共 {len(all_items)} 条诗词")
        return all_items

    def _process_file(
        self,
        file_path: Path,
        config: Dict[str, Any],
        min_length: int,
        max_length: int,
    ) -> List[PoetryItem]:
        """处理单个JSON文件"""
        items = []

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, list):
            data = [data]

        for idx, entry in enumerate(data):
            content_field = config.get('content_field', 'paragraphs')
            content = entry.get(content_field, [])

            if isinstance(content, list):
                content = '\n'.join(content)

            content = content.strip()

            if not content or len(content) < min_length or len(content) > max_length:
                continue

            item = PoetryItem(
                title=entry.get(config.get('title_field', 'title'), ''),
                author=entry.get(config.get('author_field', 'author'), '未知'),
                content=content,
                dynasty=config['dynasty'],
                type=config['type'],
                id=entry.get('id', f"{file_path.stem}_{idx}"),
            )

            items.append(item)

        return items

    def _remove_duplicates(self, items: List[PoetryItem]) -> List[PoetryItem]:
        """去除重复诗词"""
        seen = set()
        unique_items = []

        for item in items:
            content_hash = hash(item.content.replace('\n', '').strip())
            if content_hash not in seen:
                seen.add(content_hash)
                unique_items.append(item)

        removed = len(items) - len(unique_items)
        if removed > 0:
            self.logger.info(f"去重完成，移除了 {removed} 条重复记录")

        return unique_items

    def export_to_jsonl(
        self,
        items: List[PoetryItem],
        output_file: Optional[Path] = None,
        filename: str = "poetry_dataset.jsonl",
    ) -> Path:
        """导出诗词到JSONL格式"""
        output_file = output_file or self._processed_dir / filename

        with open(output_file, 'w', encoding='utf-8') as f:
            for item in items:
                record = {
                    'title': item.title,
                    'author': item.author,
                    'content': item.content,
                    'dynasty': item.dynasty,
                    'type': item.type,
                    'id': item.id,
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')

        self.logger.info(f"导出完成: {output_file}")
        return output_file

    def export_to_json(
        self,
        items: List[PoetryItem],
        output_file: Optional[Path] = None,
        filename: str = "poetry_dataset.json",
    ) -> Path:
        """导出诗词到JSON格式"""
        output_file = output_file or self._processed_dir / filename

        records = [
            {
                'title': item.title,
                'author': item.author,
                'content': item.content,
                'dynasty': item.dynasty,
                'type': item.type,
                'id': item.id,
            }
            for item in items
        ]

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

        self.logger.info(f"导出完成: {output_file}")
        return output_file

    def generate_statistics(self, items: List[PoetryItem]) -> Dict[str, Any]:
        """生成数据统计报告"""
        total = len(items)
        if total == 0:
            return {'total': 0}

        by_dynasty = {}
        by_type = {}
        by_author = {}
        total_length = 0

        for item in items:
            by_dynasty[item.dynasty] = by_dynasty.get(item.dynasty, 0) + 1
            by_type[item.type] = by_type.get(item.type, 0) + 1
            by_author[item.author] = by_author.get(item.author, 0) + 1
            total_length += len(item.content)

        top_authors = sorted(
            by_author.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]

        return {
            'total': total,
            'by_dynasty': by_dynasty,
            'by_type': by_type,
            'unique_authors': len(by_author),
            'average_length': total_length / total,
            'top_authors': top_authors,
        }

    async def build_complete_dataset(
        self,
        sources: Optional[List[str]] = None,
        min_length: int = 5,
        max_length: int = 500,
        output_format: str = 'jsonl',
    ) -> Dict[str, Path]:
        """
        构建完整的诗词数据集

        Args:
            sources: 要包含的数据源
            min_length: 最小诗句长度
            max_length: 最大诗句长度
            output_format: 输出格式 ('jsonl', 'json')

        Returns:
            Dict[str, Path]: 输出文件路径
        """
        await self.download_all(sources)

        all_items = []
        for source in sources or self.POETRY_SOURCES.keys():
            items = self.process_raw_data(source, min_length, max_length)
            all_items.extend(items)

        all_items = self._remove_duplicates(all_items)

        stats = self.generate_statistics(all_items)
        self.logger.info(f"数据集统计: {stats}")

        outputs = {}

        if output_format == 'jsonl':
            output = self.export_to_jsonl(all_items)
        else:
            output = self.export_to_json(all_items)

        outputs['main'] = output

        stats_file = self._processed_dir / "statistics.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        outputs['statistics'] = stats_file

        return outputs
