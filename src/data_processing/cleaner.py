"""数据清洗模块。

提供文本数据清洗功能，包括：
- 去除HTML标签和特殊标记
- 移除控制字符和噪声字符
- 修正编码问题
- 规范化空白字符
- 移除URL和邮箱地址
- 处理特殊符号

示例：
    >>> cleaner = DataCleaner()
    >>> cleaner.remove_html = True
    >>> cleaner.remove_urls = True
    >>> cleaned_text = cleaner.clean(raw_text)
"""

import re
import html
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field

from src.logger import get_logger

logger = get_logger("LingmaoMoyun.Data.Cleaner")


@dataclass
class CleaningConfig:
    """数据清洗配置类。"""

    remove_html: bool = True
    remove_urls: bool = True
    remove_emails: bool = True
    remove_control_chars: bool = True
    normalize_whitespace: bool = True
    remove_extra_newlines: bool = True
    strip_whitespace: bool = True
    remove_bom: bool = True
    fix_encoding_errors: bool = True
    normalize_quotes: bool = True
    normalize_brackets: bool = True
    remove_zero_width_chars: bool = True
    remove_private_use_chars: bool = True


class DataCleaner:
    """数据清洗器。

    清洗文本数据中的噪声、特殊字符和格式问题。

    Attributes:
        config: 清洗配置对象。

    Example:
        >>> cleaner = DataCleaner()
        >>> cleaner.config.remove_html = True
        >>> cleaner.config.remove_urls = True
        >>> result = cleaner.clean("Hello <b>World</b>!")
        >>> print(result)
        Hello World!
    """

    HTML_TAG_PATTERN = re.compile(r'<[^>]+>')
    HTML_ENTITY_PATTERN = re.compile(r'&[a-zA-Z]+;|&#\d+;|&#x[0-9a-fA-F]+;')
    URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
    EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    CONTROL_CHARS_PATTERN = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]')
    WHITESPACE_PATTERN = re.compile(r'[ \t]+')
    NEWLINE_PATTERN = re.compile(r'\n{3,}')
    ZERO_WIDTH_PATTERN = re.compile(r'[\u200b\u200c\u200d\ufeff]')
    PRIVATE_USE_PATTERN = re.compile(r'[\ue000-\uf8ff]')

    CHINESE_LEFT_QUOTES = {'\u201c', '\u201d'}
    CHINESE_RIGHT_QUOTES = {'\u201c', '\u201d'}
    CHINESE_PUNCTUATION = {
        '\uff0c': ',', '\u3002': '.', '\uff01': '!', '\uff1f': '?',
        '\uff1a': ':', '\uff1b': ';', '\u201c': '"', '\u201d': '"',
        '\uff08': '(', '\uff09': ')', '\u3010': '[', '\u3011': ']',
        '\u300a': '<', '\u300b': '>', '\u2014': '--', '\u2026': '...',
    }
    FULLWIDTH_TO_ASCII = {
        '\uff01': '!', '\uff1f': '?', '\uff0c': ',', '\u3002': '.',
        '\uff1a': ':', '\uff1b': ';', '\u201c': '"', '\u201d': '"',
        '\uff08': '(', '\uff09': ')', '\u3010': '[', '\u3011': ']',
        '\u300a': '<', '\u300b': '>', '\u3001': ',',
    }

    def __init__(self, config: Optional[CleaningConfig] = None):
        """初始化数据清洗器。

        Args:
            config: 清洗配置对象。如果为None，使用默认配置。
        """
        self.config = config or CleaningConfig()

    def clean(self, text: str) -> str:
        """执行完整的数据清洗流程。

        Args:
            text: 待清洗的原始文本。

        Returns:
            清洗后的文本。

        Example:
            >>> cleaner = DataCleaner()
            >>> cleaner.config.remove_html = True
            >>> result = cleaner.clean('<p>Hello   World</p>\\n\\n\\n')
            >>> print(result)
            Hello World
        """
        if not isinstance(text, str):
            text = str(text) if text is not None else ""

        if not text:
            return text

        if self.config.remove_bom and text.startswith('\ufeff'):
            text = text[1:]

        if self.config.remove_html:
            text = self._remove_html(text)

        if self.config.remove_urls:
            text = self._remove_urls(text)

        if self.config.remove_emails:
            text = self._remove_emails(text)

        if self.config.remove_control_chars:
            text = self._remove_control_chars(text)

        if self.config.remove_zero_width_chars:
            text = self._remove_zero_width_chars(text)

        if self.config.remove_private_use_chars:
            text = self._remove_private_use_chars(text)

        if self.config.normalize_whitespace:
            text = self._normalize_whitespace(text)

        if self.config.remove_extra_newlines:
            text = self._remove_extra_newlines(text)

        if self.config.strip_whitespace:
            text = text.strip()

        if self.config.normalize_quotes:
            text = self._normalize_quotes(text)

        if self.config.normalize_brackets:
            text = self._normalize_brackets(text)

        if self.config.fix_encoding_errors:
            text = self._fix_encoding_errors(text)

        return text

    def _remove_html(self, text: str) -> str:
        """移除HTML标签并解码HTML实体。"""
        text = self.HTML_TAG_PATTERN.sub('', text)
        text = html.unescape(text)
        text = self.HTML_ENTITY_PATTERN.sub('', text)
        return text

    def _remove_urls(self, text: str) -> str:
        """移除URL地址。"""
        return self.URL_PATTERN.sub('', text)

    def _remove_emails(self, text: str) -> str:
        """移除邮箱地址。"""
        return self.EMAIL_PATTERN.sub('', text)

    def _remove_control_chars(self, text: str) -> str:
        """移除控制字符。"""
        return self.CONTROL_CHARS_PATTERN.sub('', text)

    def _remove_zero_width_chars(self, text: str) -> str:
        """移除零宽字符。"""
        return self.ZERO_WIDTH_PATTERN.sub('', text)

    def _remove_private_use_chars(self, text: str) -> str:
        """移除私有使用区字符。"""
        return self.PRIVATE_USE_PATTERN.sub('', text)

    def _normalize_whitespace(self, text: str) -> str:
        """规范化空白字符。"""
        text = self.WHITESPACE_PATTERN.sub(' ', text)
        return text

    def _remove_extra_newlines(self, text: str) -> str:
        """移除多余的换行符。"""
        return self.NEWLINE_PATTERN.sub('\n\n', text)

    def _normalize_quotes(self, text: str) -> str:
        """规范化引号。"""
        for ch in self.CHINESE_LEFT_QUOTES:
            text = text.replace(ch, '"')
        for ch in self.CHINESE_RIGHT_QUOTES:
            text = text.replace(ch, '"')
        return text

    def _normalize_brackets(self, text: str) -> str:
        """规范化括号和全角标点。"""
        for full, half in self.FULLWIDTH_TO_ASCII.items():
            text = text.replace(full, half)
        return text

    def _fix_encoding_errors(self, text: str) -> str:
        """尝试修复常见编码错误。"""
        replacements = {
            '\u00c2\u00a0': ' ',
            '\u00a0': ' ',
            '\u3000': ' ',
            'â€™': "'",
            'â€œ': '"',
            'â€': '"',
            'Ã©': 'é',
            'Ã¨': 'è',
            'Ã ': 'à',
            'Ã¢': 'â',
            'Ã»': 'û',
            'Ã®': 'î',
            'Ã´': 'ô',
            'Ã§': 'ç',
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    def clean_batch(self, texts: List[str]) -> List[str]:
        """批量清洗文本。

        Args:
            texts: 待清洗的文本列表。

        Returns:
            清洗后的文本列表。

        Example:
            >>> cleaner = DataCleaner()
            >>> results = cleaner.clean_batch(['<p>Hello</p>', '<b>World</b>'])
            >>> print(results)
            ['Hello', 'World']
        """
        return [self.clean(text) for text in texts]

    def get_cleaning_stats(self) -> Dict[str, int]:
        """获取清洗统计信息。

        Returns:
            包含清洗统计信息的字典。
        """
        return {
            'total_cleaned': 0,
            'html_removed': 0,
            'urls_removed': 0,
            'emails_removed': 0,
            'control_chars_removed': 0,
        }
