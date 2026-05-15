"""
现代文下载器模块（可选）
提供现代中文语料下载和处理功能
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .base_downloader import BaseDownloader


@dataclass
class ModernText:
    """现代文本条目"""
    title: str
    author: str
    content: str
    source: str
    category: str
    publish_date: Optional[str] = None
    id: Optional[str] = None
    word_count: int = 0
    tags: List[str] = field(default_factory=list)


@dataclass
class ModernTextFilter:
    """现代文本过滤配置"""
    min_length: int = 100
    max_length: int = 50000
    min_word_count: int = 50
    categories: List[str] = field(default_factory=list)
    blocked_sources: Set[str] = field(default_factory=set)
    blocked_keywords: Set[str] = field(default_factory=set)
    require_quality: bool = True


class ModernDownloader(BaseDownloader):
    """现代文下载器"""

    MODERN_SOURCES = {
        'wiki_zh': {
            'name': '中文维基百科',
            'url': 'https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles.json.gz',
            'category': '百科',
            'format': 'json_gz',
        },
        'news_chinese': {
            'name': '中文新闻语料',
            'url': 'https://github.com/aceimnorstuvwxz/chinese_corpus/releases/download/v1.0/news2016zh.json.gz',
            'category': '新闻',
            'format': 'json_gz',
        },
        'baike': {
            'name': '百度百科',
            'url': 'https://github.com/aceimnorstuvwxz/chinese_corpus/releases/download/v1.0/baike2018.json.gz',
            'category': '百科',
            'format': 'json_gz',
        },
        'xiaohua': {
            'name': '知乎问答',
            'url': 'https://github.com/aceimnorstuvwxz/chinese_corpus/releases/download/v1.0/zhihuihuajiazhang.json.gz',
            'category': '问答',
            'format': 'json_gz',
        },
        'subtitles': {
            'name': '字幕语料',
            'url': 'https://github.com/aceimnorstuvwxz/chinese_corpus/releases/download/v1.0/subtitle2018.json.gz',
            'category': '字幕',
            'format': 'json_gz',
        },
    }

    def __init__(
        self,
        base_dir: str = "./data/modern",
        max_concurrent: int = 3,
        rate_limit: float = 5.0,
    ):
        super().__init__(base_dir, max_concurrent, rate_limit)
        self.logger = logging.getLogger(__name__)
        self._raw_dir = self.base_dir / "raw"
        self._processed_dir = self.base_dir / "processed"
        self._raw_dir.mkdir(parents=True, exist_ok=True)
        self._processed_dir.mkdir(parents=True, exist_ok=True)

    async def download_all(self, sources: Optional[List[str]] = None) -> Dict[str, Path]:
        """下载所有选定的现代文数据源"""
        sources = sources or list(self.MODERN_SOURCES.keys())
        results = {}

        for source in sources:
            if source not in self.MODERN_SOURCES:
                self.logger.warning(f"未知数据源: {source}")
                continue

            config = self.MODERN_SOURCES[source]
            self.logger.info(f"开始下载 {config['name']}...")

            try:
                filename = f"{source}.json.gz"
                path = await self.download(config['url'], filename)
                results[source] = path
                self.logger.info(f"{config['name']} 下载完成")
            except Exception as e:
                self.logger.error(f"{config['name']} 下载失败: {e}")

        return results

    def process_raw_data(
        self,
        source: Optional[str] = None,
        text_filter: Optional[ModernTextFilter] = None,
    ) -> List[ModernText]:
        """处理原始现代文数据"""
        text_filter = text_filter or ModernTextFilter()
        all_texts = []

        sources_to_process = [source] if source else list(self.MODERN_SOURCES.keys())

        for src in sources_to_process:
            if src not in self.MODERN_SOURCES:
                continue

            config = self.MODERN_SOURCES[src]
            raw_file = self._raw_dir / f"{src}.json.gz"

            if not raw_file.exists():
                self.logger.warning(f"原始文件不存在: {raw_file}")
                continue

            try:
                texts = self._parse_modern_file(raw_file, config)
                filtered_texts = self._apply_filter(texts, text_filter)
                all_texts.extend(filtered_texts)
                self.logger.info(f"处理 {config['name']}: {len(texts)} -> {len(filtered_texts)} (过滤后)")
            except Exception as e:
                self.logger.error(f"处理 {raw_file} 失败: {e}")

        return all_texts

    def _parse_modern_file(self, file_path: Path, config: Dict[str, Any]) -> List[ModernText]:
        """解析现代文文件"""
        import gzip

        texts = []

        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if not line.strip():
                    continue

                try:
                    item = json.loads(line)
                    text = self._parse_item(item, config, idx)
                    if text:
                        texts.append(text)
                except json.JSONDecodeError:
                    continue

                if idx > 0 and idx % 10000 == 0:
                    self.logger.debug(f"已处理 {idx} 条记录")

        return texts

    def _parse_item(self, item: Any, config: Dict[str, Any], idx: int) -> Optional[ModernText]:
        """解析单个条目"""
        content = ""

        if isinstance(item, dict):
            content = item.get('content', item.get('text', item.get('detail', "")))
            if isinstance(content, list):
                content = '\n'.join(str(c) for c in content)

        content = str(content).strip()
        if not content:
            return None

        title = item.get('title', item.get('question', f"第{idx + 1}篇")) if isinstance(item, dict) else f"第{idx + 1}篇"
        author = item.get('author', item.get('user', '匿名用户')) if isinstance(item, dict) else '匿名用户'
        publish_date = item.get('date', item.get('publish_date')) if isinstance(item, dict) else None

        word_count = self._count_chinese_words(content)

        return ModernText(
            title=str(title),
            author=str(author),
            content=content,
            source=config['name'],
            category=config['category'],
            publish_date=str(publish_date) if publish_date else None,
            id=item.get('id', f"{config['name']}_{idx}") if isinstance(item, dict) else f"{config['name']}_{idx}",
            word_count=word_count,
            tags=item.get('tags', []) if isinstance(item, dict) else [],
        )

    def _count_chinese_words(self, text: str) -> int:
        """统计中文字数"""
        return len(re.findall(r'[\u4e00-\u9fff]', text))

    def _apply_filter(self, texts: List[ModernText], text_filter: ModernTextFilter) -> List[ModernText]:
        """应用过滤规则"""
        filtered = []

        for text in texts:
            if len(text.content) < text_filter.min_length:
                continue
            if len(text.content) > text_filter.max_length:
                continue

            if text.word_count < text_filter.min_word_count:
                continue

            if text_filter.categories and text.category not in text_filter.categories:
                continue

            if text.source in text_filter.blocked_sources:
                continue

            if any(keyword in text.content for keyword in text_filter.blocked_keywords):
                continue

            if text_filter.require_quality:
                if not self._check_quality(text):
                    continue

            filtered.append(text)

        return filtered

    def _check_quality(self, text: ModernText) -> bool:
        """检查文本质量"""
        content = text.content

        chinese_ratio = len(re.findall(r'[\u4e00-\u9fff]', content)) / max(len(content), 1)

        if chinese_ratio < 0.3:
            return False

        repeated_chars = len(re.findall(r'(.)\1{5,}', content))
        if repeated_chars > 5:
            return False

        char_counts = {}
        for char in content:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        max_char_ratio = max(char_counts.values()) / max(len(content), 1)
        if max_char_ratio > 0.5:
            return False

        return True

    def normalize_text(self, texts: List[ModernText]) -> List[ModernText]:
        """文本标准化"""
        normalized = []

        for text in texts:
            normalized_content = self._normalize_content(text.content)
            text.content = normalized_content
            text.word_count = self._count_chinese_words(normalized_content)
            normalized.append(text)

        return normalized

    def _normalize_content(self, content: str) -> str:
        """标准化文本内容"""
        content = re.sub(r'[\u3000\xa0]+', ' ', content)
        content = re.sub(r'[ \t]+', ' ', content)
        content = re.sub(r'\n{3,}', '\n\n', content)
        content = re.sub(r'https?://\S+', '[网址]', content)
        content = re.sub(r'@\w+', '[用户]', content)
        content = re.sub(r'#\w+#', '[话题]', content)
        content = content.strip()
        return content

    def remove_duplicates(self, texts: List[ModernText]) -> List[ModernText]:
        """去除重复文本"""
        seen: Set[str] = set()
        unique_texts = []

        for text in texts:
            normalized = self._normalize_for_comparison(text.content)
            if normalized not in seen:
                seen.add(normalized)
                unique_texts.append(text)

        removed = len(texts) - len(unique_texts)
        if removed > 0:
            self.logger.info(f"去重完成，移除了 {removed} 条重复记录")

        return unique_texts

    def _normalize_for_comparison(self, content: str) -> str:
        """用于比较的标准化"""
        content = re.sub(r'[\u3000\xa0\s\n]+', '', content)
        return content

    def export_to_jsonl(
        self,
        texts: List[ModernText],
        output_file: Optional[Path] = None,
        filename: str = "modern_dataset.jsonl",
    ) -> Path:
        """导出到JSONL格式"""
        output_file = output_file or self._processed_dir / filename

        with open(output_file, 'w', encoding='utf-8') as f:
            for text in texts:
                record = {
                    'title': text.title,
                    'author': text.author,
                    'content': text.content,
                    'source': text.source,
                    'category': text.category,
                    'publish_date': text.publish_date,
                    'id': text.id,
                    'word_count': text.word_count,
                    'tags': text.tags,
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')

        self.logger.info(f"导出完成: {output_file}")
        return output_file

    def export_to_json(
        self,
        texts: List[ModernText],
        output_file: Optional[Path] = None,
        filename: str = "modern_dataset.json",
    ) -> Path:
        """导出到JSON格式"""
        output_file = output_file or self._processed_dir / filename

        records = [
            {
                'title': text.title,
                'author': text.author,
                'content': text.content,
                'source': text.source,
                'category': text.category,
                'publish_date': text.publish_date,
                'id': text.id,
                'word_count': text.word_count,
                'tags': text.tags,
            }
            for text in texts
        ]

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

        self.logger.info(f"导出完成: {output_file}")
        return output_file

    def generate_statistics(self, texts: List[ModernText]) -> Dict[str, Any]:
        """生成数据统计报告"""
        total = len(texts)
        if total == 0:
            return {'total': 0}

        by_category: Dict[str, int] = {}
        by_source: Dict[str, int] = {}
        by_author: Dict[str, int] = {}
        total_length = 0
        total_words = 0

        for text in texts:
            by_category[text.category] = by_category.get(text.category, 0) + 1
            by_source[text.source] = by_source.get(text.source, 0) + 1
            by_author[text.author] = by_author.get(text.author, 0) + 1
            total_length += len(text.content)
            total_words += text.word_count

        top_authors = sorted(by_author.items(), key=lambda x: x[1], reverse=True)[:10]

        return {
            'total': total,
            'total_characters': total_length,
            'total_words': total_words,
            'average_length': total_length / total,
            'average_words': total_words / total,
            'by_category': by_category,
            'by_source': by_source,
            'unique_authors': len(by_author),
            'top_authors': top_authors,
        }

    async def build_complete_dataset(
        self,
        sources: Optional[List[str]] = None,
        text_filter: Optional[ModernTextFilter] = None,
        output_format: str = 'jsonl',
        remove_duplicates_flag: bool = True,
    ) -> Dict[str, Path]:
        """构建完整的现代文数据集"""
        await self.download_all(sources)

        texts = self.process_raw_data(text_filter=text_filter)
        texts = self.normalize_text(texts)

        if remove_duplicates_flag:
            texts = self.remove_duplicates(texts)

        stats = self.generate_statistics(texts)
        self.logger.info(f"数据集统计: {json.dumps(stats, ensure_ascii=False, indent=2)}")

        outputs = {}

        if output_format == 'jsonl':
            output = self.export_to_jsonl(texts)
        else:
            output = self.export_to_json(texts)
        outputs['main'] = output

        stats_file = self._processed_dir / "statistics.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        outputs['statistics'] = stats_file

        return outputs


class ModernTextAnalyzer:
    """现代文本分析器"""

    @staticmethod
    def extract_topics(text: ModernText, top_n: int = 5) -> List[str]:
        """提取主题词"""
        common_words = {'的', '了', '是', '在', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}
        words = re.findall(r'[\u4e00-\u9fff]{2,}', text.content)
        word_freq: Dict[str, int] = {}
        for word in words:
            if word not in common_words:
                word_freq[word] = word_freq.get(word, 0) + 1

        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, _ in sorted_words[:top_n]]

    @staticmethod
    def calculate_readability(text: ModernText) -> float:
        """计算可读性指数（简化版）"""
        sentences = re.split(r'[。！？；]', text.content)
        num_sentences = max(len(sentences), 1)
        avg_sentence_length = text.word_count / num_sentences

        readability = 100 - (avg_sentence_length / 2)
        return max(0, min(100, readability))
