"""
古文典籍下载器模块
下载和处理古文观止、古文辞类纂等古典文献数据
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
class ClassicalText:
    """古典文献条目"""
    title: str
    author: str
    content: str
    source: str
    dynasty: str
    chapter: str = ""
    section: str = ""
    id: Optional[str] = None
    difficulty: str = "未知"
    category: str = "古文"


@dataclass
class TextFilter:
    """文本过滤配置"""
    min_length: int = 50
    max_length: int = 10000
    allowed_dynasties: List[str] = field(default_factory=list)
    blocked_authors: Set[str] = field(default_factory=set)
    blocked_keywords: Set[str] = field(default_factory=set)
    require_classical_chinese: bool = False


class ClassicalDownloader(BaseDownloader):
    """古文典籍下载器"""

    CLASSICAL_SOURCES = {
        'guwen_guanzhi': {
            'name': '古文观止',
            'dynasty': '明清',
            'category': '古文',
            'url': 'https://raw.githubusercontent.com/panguojun/chinese-classics/master/guwen_guanzhi.json',
        },
        'guwen_cilei': {
            'name': '古文辞类纂',
            'dynasty': '清代',
            'category': '古文',
            'url': 'https://raw.githubusercontent.com/panguojun/chinese-classics/master/guwen_cilei.json',
        },
        'shangshu': {
            'name': '尚书',
            'dynasty': '先秦',
            'category': '经部',
            'url': 'https://raw.githubusercontent.com/panguojun/chinese-classics/master/shangshu.json',
        },
        'liji': {
            'name': '礼记',
            'dynasty': '先秦',
            'category': '经部',
            'url': 'https://raw.githubusercontent.com/panguojun/chinese-classics/master/liji.json',
        },
        'zhuangzi': {
            'name': '庄子',
            'dynasty': '先秦',
            'category': '子部',
            'url': 'https://raw.githubusercontent.com/panguojun/chinese-classics/master/zhuangzi.json',
        },
        'mengzi': {
            'name': '孟子',
            'dynasty': '先秦',
            'category': '经部',
            'url': 'https://raw.githubusercontent.com/panguojun/chinese-classics/master/mengzi.json',
        },
        'xunzi': {
            'name': '荀子',
            'dynasty': '先秦',
            'category': '子部',
            'url': 'https://raw.githubusercontent.com/panguojun/chinese-classics/master/xunzi.json',
        },
        'han_feizi': {
            'name': '韩非子',
            'dynasty': '先秦',
            'category': '子部',
            'url': 'https://raw.githubusercontent.com/panguojun/chinese-classics/master/han_feizi.json',
        },
        'huainanzi': {
            'name': '淮南子',
            'dynasty': '西汉',
            'category': '子部',
            'url': 'https://raw.githubusercontent.com/panguojun/chinese-classics/master/huainanzi.json',
        },
        'shuowen': {
            'name': '说文解字',
            'dynasty': '东汉',
            'category': '经部',
            'url': 'https://raw.githubusercontent.com/panguojun/chinese-classics/master/shuowen.json',
        },
        'lvshi_chunqiu': {
            'name': '吕氏春秋',
            'dynasty': '先秦',
            'category': '子部',
            'url': 'https://raw.githubusercontent.com/panguojun/chinese-classics/master/lvshi_chunqiu.json',
        },
        'guoyu': {
            'name': '国语',
            'dynasty': '先秦',
            'category': '史部',
            'url': 'https://raw.githubusercontent.com/panguojun/chinese-classics/master/guoyu.json',
        },
    }

    def __init__(
        self,
        base_dir: str = "./data/classical",
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
        """下载所有选定的典籍数据"""
        sources = sources or list(self.CLASSICAL_SOURCES.keys())
        results = {}

        for source in sources:
            if source not in self.CLASSICAL_SOURCES:
                self.logger.warning(f"未知数据源: {source}")
                continue

            config = self.CLASSICAL_SOURCES[source]
            self.logger.info(f"开始下载 {config['name']}...")

            try:
                filename = f"{source}.json"
                path = await self.download(config['url'], filename)
                results[source] = path
                self.logger.info(f"{config['name']} 下载完成")
            except Exception as e:
                self.logger.error(f"{config['name']} 下载失败: {e}")

        return results

    def process_raw_data(
        self,
        source: Optional[str] = None,
        text_filter: Optional[TextFilter] = None,
    ) -> List[ClassicalText]:
        """处理原始典籍数据"""
        text_filter = text_filter or TextFilter()
        all_texts = []

        sources_to_process = [source] if source else list(self.CLASSICAL_SOURCES.keys())

        for src in sources_to_process:
            if src not in self.CLASSICAL_SOURCES:
                continue

            config = self.CLASSICAL_SOURCES[src]
            raw_file = self._raw_dir / f"{src}.json"

            if not raw_file.exists():
                self.logger.warning(f"原始文件不存在: {raw_file}")
                continue

            try:
                texts = self._parse_classical_file(raw_file, config)
                filtered_texts = self._apply_filter(texts, text_filter)
                all_texts.extend(filtered_texts)
                self.logger.info(f"处理 {config['name']}: {len(texts)} -> {len(filtered_texts)} (过滤后)")
            except Exception as e:
                self.logger.error(f"处理 {raw_file} 失败: {e}")

        return all_texts

    def _parse_classical_file(self, file_path: Path, config: Dict[str, Any]) -> List[ClassicalText]:
        """解析典籍文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        texts = []

        if isinstance(data, list):
            for idx, item in enumerate(data):
                text = self._parse_item(item, config, idx)
                if text:
                    texts.append(text)
        elif isinstance(data, dict):
            text = self._parse_item(data, config, 0)
            if text:
                texts.append(text)

        return texts

    def _parse_item(self, item: Any, config: Dict[str, Any], idx: int) -> Optional[ClassicalText]:
        """解析单个条目"""
        content = ""

        if isinstance(item, str):
            content = item
        elif isinstance(item, dict):
            content = item.get('content', item.get('text', item.get('paragraphs', "")))
            if isinstance(content, list):
                content = '\n'.join(content)

        content = content.strip()
        if not content:
            return None

        title = item.get('title', item.get('name', f"第{idx + 1}篇")) if isinstance(item, dict) else f"第{idx + 1}篇"
        author = item.get('author', config.get('name', '佚名')) if isinstance(item, dict) else config.get('name', '佚名')
        chapter = item.get('chapter', item.get('section', '')) if isinstance(item, dict) else ''

        return ClassicalText(
            title=title,
            author=author,
            content=content,
            source=config['name'],
            dynasty=config['dynasty'],
            chapter=chapter,
            category=config['category'],
            id=item.get('id', f"{config.get('name', 'unknown')}_{idx}") if isinstance(item, dict) else f"{config.get('name', 'unknown')}_{idx}",
        )

    def _apply_filter(self, texts: List[ClassicalText], text_filter: TextFilter) -> List[ClassicalText]:
        """应用过滤规则"""
        filtered = []

        for text in texts:
            if len(text.content) < text_filter.min_length:
                continue
            if len(text.content) > text_filter.max_length:
                continue

            if text_filter.allowed_dynasties and text.dynasty not in text_filter.allowed_dynasties:
                continue

            if text.author in text_filter.blocked_authors:
                continue

            if any(keyword in text.content for keyword in text_filter.blocked_keywords):
                continue

            if text_filter.require_classical_chinese:
                if not self._is_classical_chinese(text.content):
                    continue

            filtered.append(text)

        return filtered

    def _is_classical_chinese(self, text: str) -> bool:
        """判断是否为文言文"""
        classical_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        total_chars = sum(1 for c in text if c.isalpha())
        return total_chars > 0 and classical_chars / total_chars > 0.7

    def normalize_text(self, texts: List[ClassicalText]) -> List[ClassicalText]:
        """文本标准化"""
        normalized = []

        for text in texts:
            normalized_content = self._normalize_content(text.content)
            text.content = normalized_content
            text.title = self._normalize_title(text.title)
            normalized.append(text)

        return normalized

    def _normalize_content(self, content: str) -> str:
        """标准化文本内容"""
        content = re.sub(r'[\u3000\xa0]+', ' ', content)
        content = re.sub(r'[ \t]+', ' ', content)
        content = re.sub(r'\n{3,}', '\n\n', content)
        content = content.strip()
        return content

    def _normalize_title(self, title: str) -> str:
        """标准化标题"""
        title = re.sub(r'[\u3000\xa0]+', '', title)
        title = re.sub(r'[ \t]+', '', title)
        return title.strip()

    def remove_duplicates(self, texts: List[ClassicalText]) -> List[ClassicalText]:
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
        texts: List[ClassicalText],
        output_file: Optional[Path] = None,
        filename: str = "classical_dataset.jsonl",
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
                    'dynasty': text.dynasty,
                    'category': text.category,
                    'chapter': text.chapter,
                    'id': text.id,
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')

        self.logger.info(f"导出完成: {output_file}")
        return output_file

    def export_to_json(
        self,
        texts: List[ClassicalText],
        output_file: Optional[Path] = None,
        filename: str = "classical_dataset.json",
    ) -> Path:
        """导出到JSON格式"""
        output_file = output_file or self._processed_dir / filename

        records = [
            {
                'title': text.title,
                'author': text.author,
                'content': text.content,
                'source': text.source,
                'dynasty': text.dynasty,
                'category': text.category,
                'chapter': text.chapter,
                'id': text.id,
            }
            for text in texts
        ]

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

        self.logger.info(f"导出完成: {output_file}")
        return output_file

    def generate_statistics(self, texts: List[ClassicalText]) -> Dict[str, Any]:
        """生成数据统计报告"""
        total = len(texts)
        if total == 0:
            return {'total': 0}

        by_dynasty: Dict[str, int] = {}
        by_category: Dict[str, int] = {}
        by_source: Dict[str, int] = {}
        by_author: Dict[str, int] = {}
        total_length = 0

        for text in texts:
            by_dynasty[text.dynasty] = by_dynasty.get(text.dynasty, 0) + 1
            by_category[text.category] = by_category.get(text.category, 0) + 1
            by_source[text.source] = by_source.get(text.source, 0) + 1
            by_author[text.author] = by_author.get(text.author, 0) + 1
            total_length += len(text.content)

        top_authors = sorted(by_author.items(), key=lambda x: x[1], reverse=True)[:10]

        return {
            'total': total,
            'total_characters': total_length,
            'average_length': total_length / total,
            'by_dynasty': by_dynasty,
            'by_category': by_category,
            'by_source': by_source,
            'unique_authors': len(by_author),
            'top_authors': top_authors,
        }

    async def build_complete_dataset(
        self,
        sources: Optional[List[str]] = None,
        text_filter: Optional[TextFilter] = None,
        output_format: str = 'jsonl',
        remove_duplicates_flag: bool = True,
    ) -> Dict[str, Path]:
        """构建完整的古典文献数据集"""
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


class ClassicalTextAnalyzer:
    """古典文本分析器"""

    @staticmethod
    def calculate_difficulty(text: ClassicalText) -> str:
        """计算文本难度等级"""
        content = text.content
        length = len(content)

        classical_keywords = ['之', '乎', '者', '也', '矣', '焉', '哉', '夫', '盖', '然']
        keyword_count = sum(content.count(k) for k in classical_keywords)

        if length < 200:
            return '简单'
        elif length < 500:
            return '中等'
        else:
            return '困难'

    @staticmethod
    def extract_keywords(text: ClassicalText, top_n: int = 10) -> List[str]:
        """提取关键词"""
        words = re.findall(r'[\u4e00-\u9fff]+', text.content)
        word_freq: Dict[str, int] = {}
        for word in words:
            if len(word) >= 2:
                word_freq[word] = word_freq.get(word, 0) + 1

        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, _ in sorted_words[:top_n]]

    @staticmethod
    def split_sentences(text: ClassicalText) -> List[str]:
        """分割句子"""
        content = text.content
        sentences = re.split(r'[。！？；]', content)
        return [s.strip() for s in sentences if s.strip()]
