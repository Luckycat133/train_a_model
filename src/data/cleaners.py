"""Lingmao Moyun 数据清洗模块。

提供文本清洗、数据去重和质量过滤功能。
支持批量处理、增量处理和详细的清洗报告。

示例：
    >>> from src.data.cleaners import TextCleaner, DataCleaner, QualityFilter
    >>>
    >>> # 文本清洗
    >>> cleaner = TextCleaner(remove_special_chars=True)
    >>> cleaned = cleaner.clean_text("示例文本✨ with emoji🎉")
    >>>
    >>> # 数据去重
    >>> deduper = DataCleaner()
    >>> unique_data = deduper.deduplicate(data_list, key="text")
    >>>
    >>> # 质量过滤
    >>> filter = QualityFilter(min_length=10, max_length=500)
    >>> filtered = filter.filter(data_list)
"""

import re
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from collections import Counter

try:
    from zhon import characters, pinyin
    HAS_ZHON = True
except ImportError:
    HAS_ZHON = False

from src.logger import get_logger

logger = get_logger("LingmaoMoyun.Cleaner")


class TextCleaner:
    """文本清洗器。

    提供全面的文本清洗功能，包括：
    - 移除特殊字符和不可见字符
    - 规范化空白字符
    - 繁简中文转换
    - URL和邮箱移除
    - HTML标签移除

    示例：
        >>> cleaner = TextCleaner(
        ...     remove_special_chars=True,
        ...     normalize_whitespace=True,
        ...     convert_to_simple=True
        ... )
        >>> cleaned = cleaner.clean_text("繁體中文　簡體中文")
    """

    def __init__(
        self,
        remove_special_chars: bool = True,
        remove_emoji: bool = True,
        remove_urls: bool = True,
        remove_emails: bool = True,
        remove_html: bool = True,
        normalize_whitespace: bool = True,
        normalize_punctuation: bool = True,
        convert_to_simple: bool = False,
        convert_to_traditional: bool = False,
        custom_replacements: Optional[Dict[str, str]] = None,
    ):
        self.remove_special_chars = remove_special_chars
        self.remove_emoji = remove_emoji
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.remove_html = remove_html
        self.normalize_whitespace = normalize_whitespace
        self.normalize_punctuation = normalize_punctuation
        self.convert_to_simple = convert_to_simple
        self.convert_to_traditional = convert_to_traditional
        self.custom_replacements = custom_replacements or {}

        self._init_patterns()

    def _init_patterns(self):
        """初始化正则表达式模式。"""
        emoji_ranges = [
            (0x1F600, 0x1F64F),
            (0x1F300, 0x1F5FF),
            (0x1F680, 0x1F6FF),
            (0x1F1E0, 0x1F1FF),
            (0x02702, 0x027B0),
            (0x024C2, 0x024FF),
            (0x1F900, 0x1F9FF),
            (0x1FA00, 0x1FA6F),
            (0x1FA70, 0x1FAFF),
            (0x02600, 0x026FF),
        ]
        emoji_pattern_str = "["
        for start, end in emoji_ranges:
            emoji_pattern_str += chr(start) + "-" + chr(end)
        emoji_pattern_str += "]+"
        self.emoji_pattern = re.compile(emoji_pattern_str, flags=re.UNICODE)

        self.url_pattern = re.compile(
            r"https?://[^\s<>\"]+|www\.[^\s<>\"]+",
            flags=re.IGNORECASE,
        )

        self.email_pattern = re.compile(
            r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        )

        self.html_pattern = re.compile(
            r"<[^>]+>|[<>&]",
        )

        self.whitespace_pattern = re.compile(r"\s+")

        self.control_chars_pattern = re.compile(
            r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]",
        )

        self.special_chars_pattern = re.compile(
            r"[\U00000080-\U000000FF]"
            r"|[\U00000800-\U00000FFF]"
            r"|[\U0000F000-\U0000FFFF]"
            r"|[\U00010000-\U00010FFFF]",
            flags=re.UNICODE,
        )

        self.simp_to_trad_map = self._load_simp_trad_map() if not self.convert_to_simple else {}
        self.trad_to_simp_map = {v: k for k, v in self.simp_to_trad_map.items()}

    def _load_simp_trad_map(self) -> Dict[str, str]:
        """加载简繁转换映射表。"""
        basic_map = {
            "与": "與",
            "为": "為",
            "当": "當",
            "发": "發",
            "开": "開",
            "无": "無",
            "见": "見",
            "长": "長",
            "东": "東",
            "丝": "絲",
            "丢": "丟",
            "两": "兩",
            "严": "嚴",
            "丧": "喪",
            "个": "個",
            "临": "臨",
            "临": "臨",
            "丸": "丸",
            "丹": "丹",
            "主": "主",
            "丽": "麗",
            "举": "舉",
            "乃": "乃",
            "久": "久",
            "之": "之",
            "乌": "烏",
            "乍": "乍",
            "乎": "乎",
            "乏": "乏",
            "乐": "樂",
            "乒乓": "乒乓",
            "亨": "亨",
            "亩": "畝",
            "京": "京",
            "亭": "亭",
            "亮": "亮",
            "亲": "親",
            "人": "人",
            "亿": "億",
            "什": "什",
            "仁": "仁",
            "仍": "仍",
            "从": "從",
            "仑": "侖",
            "仓": "倉",
            "仿": "仿",
            "伙": "夥",
            "估": "估",
            "体": "體",
            "作": "作",
            "你": "你",
            "佞": "佞",
            "佳": "佳",
            "使": "使",
            "侍": "侍",
            "供": "供",
            "依": "依",
            "侠": "俠",
            "侣": "侶",
            "侦": "偵",
            "侧": "側",
            "侨": "僑",
            "佩": "佩",
            "货": "貨",
            "侦": "偵",
            "侨": "僑",
        }

        return basic_map

    def clean_text(self, text: str) -> str:
        """清洗单条文本。

        Args:
            text: 原始文本。

        Returns:
            清洗后的文本。
        """
        if not isinstance(text, str):
            text = str(text)

        result = text

        if self.remove_html:
            result = self._remove_html(result)

        if self.remove_urls:
            result = self._remove_urls(result)

        if self.remove_emails:
            result = self._remove_emails(result)

        if self.remove_emoji:
            result = self._remove_emoji(result)

        if self.remove_special_chars:
            result = self._remove_special_chars(result)

        result = self._remove_control_chars(result)

        if self.normalize_whitespace:
            result = self._normalize_whitespace(result)

        if self.normalize_punctuation:
            result = self._normalize_punctuation(result)

        if self.convert_to_simple:
            result = self._convert_to_simple(result)
        elif self.convert_to_traditional:
            result = self._convert_to_traditional(result)

        for old, new in self.custom_replacements.items():
            result = result.replace(old, new)

        return result.strip()

    def clean_batch(self, texts: List[str]) -> List[str]:
        """批量清洗文本。

        Args:
            texts: 文本列表。

        Returns:
            清洗后的文本列表。
        """
        return [self.clean_text(text) for text in texts]

    def _remove_html(self, text: str) -> str:
        """移除HTML标签。"""
        return self.html_pattern.sub("", text)

    def _remove_urls(self, text: str) -> str:
        """移除URL。"""
        return self.url_pattern.sub("", text)

    def _remove_emails(self, text: str) -> str:
        """移除邮箱地址。"""
        return self.email_pattern.sub("", text)

    def _remove_emoji(self, text: str) -> str:
        """移除表情符号。"""
        return self.emoji_pattern.sub("", text)

    def _remove_special_chars(self, text: str) -> str:
        """移除特殊字符，保留中文字符和中文标点。"""
        result = []
        for char in text:
            if char.isalnum():
                result.append(char)
            elif char.isspace():
                result.append(char)
            elif self._is_chinese_char(char):
                result.append(char)
            elif self._is_chinese_punctuation(char):
                result.append(char)
            elif ord(char) < 128:
                result.append(char)
        return "".join(result)

    def _is_chinese_char(self, char: str) -> bool:
        """检查是否为中文字符（CJK统一表意文字）。"""
        code_point = ord(char)
        return (
            0x4E00 <= code_point <= 0x9FFF or
            0x3400 <= code_point <= 0x4DBF or
            0x20000 <= code_point <= 0x2A6DF or
            0x2A700 <= code_point <= 0x2B73F or
            0x2B740 <= code_point <= 0x2B81F or
            0x2B820 <= code_point <= 0x2CEAF or
            0x2F800 <= code_point <= 0x2FA1F
        )

    def _is_chinese_punctuation(self, char: str) -> bool:
        """检查是否为中文标点符号。"""
        code_point = ord(char)
        return (
            0x3000 <= code_point <= 0x303F or
            0xFF00 <= code_point <= 0xFFEF
        )

    def _remove_control_chars(self, text: str) -> str:
        """移除控制字符。"""
        return self.control_chars_pattern.sub("", text)

    def _normalize_whitespace(self, text: str) -> str:
        """规范化空白字符。"""
        return self.whitespace_pattern.sub(" ", text).strip()

    def _normalize_punctuation(self, text: str) -> str:
        """规范化标点符号。"""
        replacements = {
            "，": "，",
            "。": "。",
            "！": "！",
            "？": "？",
            "；": "；",
            "：": "：",
            '"': '"',
            '"': '"',
            ''': "'",
            ''': "'",
            "（": "（",
            "）": "）",
            "【": "【",
            "】": "】",
            "《": "《",
            "》": "》",
            "——": "——",
            "…": "…",
            "·": "·",
        }
        result = text
        for old, new in replacements.items():
            result = result.replace(old, new)
        return result

    def _convert_to_simple(self, text: str) -> str:
        """转换为简体中文。"""
        result = []
        for char in text:
            result.append(self.trad_to_simp_map.get(char, char))
        return "".join(result)

    def _convert_to_traditional(self, text: str) -> str:
        """转换为繁体中文。"""
        result = []
        for char in text:
            result.append(self.simp_to_trad_map.get(char, char))
        return "".join(result)

    def get_stats(self) -> Dict[str, Any]:
        """获取清洗统计信息。"""
        return {
            "remove_special_chars": self.remove_special_chars,
            "remove_emoji": self.remove_emoji,
            "remove_urls": self.remove_urls,
            "normalize_whitespace": self.normalize_whitespace,
            "convert_to_simple": self.convert_to_simple,
            "convert_to_traditional": self.convert_to_traditional,
        }


class DataCleaner:
    """数据去重器。

    支持多种去重策略：
    - 精确哈希去重
    - 相似度去重（基于编辑距离）
    - 自定义键去重

    示例：
        >>> cleaner = DataCleaner()
        >>> unique_data = cleaner.deduplicate(
        ...     data_list,
        ...     key="text",
        ...     strategy="hash"
        ... )
    """

    def __init__(
        self,
        hash_algorithm: str = "md5",
        similarity_threshold: float = 0.85,
        enable_similarity: bool = False,
    ):
        self.hash_algorithm = hash_algorithm
        self.similarity_threshold = similarity_threshold
        self.enable_similarity = enable_similarity
        self._stats = {
            "total_input": 0,
            "total_output": 0,
            "duplicates_hash": 0,
            "duplicates_similarity": 0,
        }

    def deduplicate(
        self,
        data: List[Dict[str, Any]],
        key: str = "text",
        strategy: str = "hash",
        output_path: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """数据去重。

        Args:
            data: 输入数据列表。
            key: 用作去重依据的字段名。
            strategy: 去重策略（hash/similarity/both）。
            output_path: 可选，保存去重结果的文件路径。

        Returns:
            去重后的数据列表。
        """
        self._stats["total_input"] = len(data)
        self._stats["duplicates_hash"] = 0
        self._stats["duplicates_similarity"] = 0

        if strategy in ("hash", "both"):
            data = self._deduplicate_by_hash(data, key)

        if strategy in ("similarity", "both") and self.enable_similarity:
            data = self._deduplicate_by_similarity(data, key)

        self._stats["total_output"] = len(data)

        if output_path:
            self._save_results(data, output_path)

        logger.info(
            f"去重完成：输入 {self._stats['total_input']} 条，"
            f"输出 {self._stats['total_output']} 条，"
            f"去除重复 {self._stats['total_input'] - self._stats['total_output']} 条"
        )

        return data

    def _deduplicate_by_hash(
        self,
        data: List[Dict[str, Any]],
        key: str,
    ) -> List[Dict[str, Any]]:
        """基于哈希去重。

        Args:
            data: 输入数据。
            key: 字段名。

        Returns:
            去重后的数据。
        """
        seen_hashes: Set[str] = set()
        result = []

        for item in data:
            text = item.get(key, "")
            if not isinstance(text, str):
                text = str(text)

            text_hash = self._compute_hash(text)

            if text_hash not in seen_hashes:
                seen_hashes.add(text_hash)
                result.append(item)
            else:
                self._stats["duplicates_hash"] += 1

        return result

    def _deduplicate_by_similarity(
        self,
        data: List[Dict[str, Any]],
        key: str,
    ) -> List[Dict[str, Any]]:
        """基于相似度去重。

        Args:
            data: 输入数据。
            key: 字段名。

        Returns:
            去重后的数据。
        """
        if len(data) <= 1:
            return data

        result = [data[0]]

        for item in data[1:]:
            text = item.get(key, "")
            if not isinstance(text, str):
                text = str(text)

            is_duplicate = False
            for existing_item in result:
                existing_text = existing_item.get(key, "")
                if not isinstance(existing_text, str):
                    existing_text = str(existing_text)

                similarity = self._compute_similarity(text, existing_text)

                if similarity >= self.similarity_threshold:
                    is_duplicate = True
                    self._stats["duplicates_similarity"] += 1
                    break

            if not is_duplicate:
                result.append(item)

        return result

    def _compute_hash(self, text: str) -> str:
        """计算文本哈希值。

        Args:
            text: 输入文本。

        Returns:
            哈希值字符串。
        """
        normalized = self._normalize_for_hash(text)

        if self.hash_algorithm == "md5":
            return hashlib.md5(normalized.encode()).hexdigest()
        elif self.hash_algorithm == "sha1":
            return hashlib.sha1(normalized.encode()).hexdigest()
        elif self.hash_algorithm == "sha256":
            return hashlib.sha256(normalized.encode()).hexdigest()
        else:
            return hashlib.md5(normalized.encode()).hexdigest()

    def _normalize_for_hash(self, text: str) -> str:
        """规范化文本以便哈希。

        Args:
            text: 输入文本。

        Returns:
            规范化后的文本。
        """
        text = text.lower()
        text = re.sub(r"\s+", "", text)
        text = re.sub(r"[^\w\u4e00-\u9fff]", "", text)
        return text

    def _compute_similarity(self, text1: str, text2: str) -> float:
        """计算两个文本的相似度（基于编辑距离）。

        Args:
            text1: 第一个文本。
            text2: 第二个文本。

        Returns:
            相似度分数（0-1）。
        """
        len1, len2 = len(text1), len(text2)

        if len1 == 0 and len2 == 0:
            return 1.0

        if len1 == 0 or len2 == 0:
            return 0.0

        dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]

        for i in range(len1 + 1):
            dp[i][0] = i
        for j in range(len2 + 1):
            dp[0][j] = j

        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                cost = 0 if text1[i - 1] == text2[j - 1] else 1
                dp[i][j] = min(
                    dp[i - 1][j] + 1,
                    dp[i][j - 1] + 1,
                    dp[i - 1][j - 1] + cost,
                )

        max_len = max(len1, len2)
        distance = dp[len1][len2]

        return 1.0 - (distance / max_len)

    def _save_results(self, data: List[Dict[str, Any]], output_path: str):
        """保存去重结果。

        Args:
            data: 去重后的数据。
            output_path: 输出文件路径。
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        logger.info(f"去重结果已保存到: {output_path}")

    def get_stats(self) -> Dict[str, int]:
        """获取去重统计信息。"""
        stats = self._stats.copy()
        stats["removed_total"] = stats["total_input"] - stats["total_output"]
        return stats


class QualityFilter:
    """数据质量过滤器。

    提供多维度的数据质量评估和过滤：
    - 长度过滤（最小/最大长度）
    - 语言检测（中文/英文/混合）
    - 噪声过滤（广告、无意义内容）
    - 重复率检测

    示例：
        >>> filter = QualityFilter(
        ...     min_length=10,
        ...     max_length=500,
        ...     min_chinese_ratio=0.5
        ... )
        >>> filtered = filter.filter(data_list)
    """

    def __init__(
        self,
        min_length: int = 5,
        max_length: int = 10000,
        min_chinese_ratio: float = 0.0,
        max_repeat_ratio: float = 0.5,
        remove_noise: bool = True,
        remove_incomplete: bool = True,
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.min_chinese_ratio = min_chinese_ratio
        self.max_repeat_ratio = max_repeat_ratio
        self.remove_noise = remove_noise
        self.remove_incomplete = remove_incomplete

        self._stats = {
            "total_input": 0,
            "total_output": 0,
            "removed_length": 0,
            "removed_language": 0,
            "removed_noise": 0,
            "removed_repeat": 0,
        }

        self._init_noise_patterns()

    def _init_noise_patterns(self):
        """初始化噪声模式。"""
        self.noise_patterns = [
            re.compile(r"点击.+?查看"),
            re.compile(r"登录.+?注册"),
            re.compile(r"广告"),
            re.compile(r"分享到.+?"),
            re.compile(r"http[s]?://"),
            re.compile(r"www\."),
            re.compile(r"\d{3,}[-.]?\d{3,}[-.]?\d{4,}"),
            re.compile(r"^\d+$"),
            re.compile(r"^[\W_]+$"),
        ]

        self.greeting_patterns = [
            re.compile(r"^你好[啊呀吗么?]?\s*$"),
            re.compile(r"^hello[!,.]?\s*$", re.IGNORECASE),
            re.compile(r"^hi[!,.]?\s*$", re.IGNORECASE),
        ]

    def filter(
        self,
        data: List[Dict[str, Any]],
        text_key: str = "text",
    ) -> List[Dict[str, Any]]:
        """过滤低质量数据。

        Args:
            data: 输入数据列表。
            text_key: 文本字段名。

        Returns:
            过滤后的数据列表。
        """
        self._reset_stats()
        self._stats["total_input"] = len(data)

        result = []

        for item in data:
            text = item.get(text_key, "")
            if not isinstance(text, str):
                text = str(text)

            keep = True

            if not self._check_length(text):
                self._stats["removed_length"] += 1
                keep = False
            elif not self._check_language(text):
                self._stats["removed_language"] += 1
                keep = False
            elif self.remove_noise and self._is_noise(text):
                self._stats["removed_noise"] += 1
                keep = False
            elif self._check_repeat_ratio(text):
                self._stats["removed_repeat"] += 1
                keep = False

            if keep:
                result.append(item)

        self._stats["total_output"] = len(result)

        logger.info(
            f"质量过滤完成：输入 {self._stats['total_input']} 条，"
            f"输出 {self._stats['total_output']} 条，"
            f"过滤 {self._stats['total_input'] - self._stats['total_output']} 条"
        )

        return result

    def _reset_stats(self):
        """重置统计信息。"""
        self._stats = {
            "total_input": 0,
            "total_output": 0,
            "removed_length": 0,
            "removed_language": 0,
            "removed_noise": 0,
            "removed_repeat": 0,
        }

    def _check_length(self, text: str) -> bool:
        """检查文本长度。

        Args:
            text: 输入文本。

        Returns:
            是否通过长度检查。
        """
        length = len(text)

        if length < self.min_length:
            return False

        if length > self.max_length:
            return False

        return True

    def _check_language(self, text: str) -> bool:
        """检查语言比例。

        Args:
            text: 输入文本。

        Returns:
            是否通过语言检查。
        """
        if self.min_chinese_ratio <= 0:
            return True

        chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
        chinese_ratio = len(chinese_chars) / len(text) if len(text) > 0 else 0

        return chinese_ratio >= self.min_chinese_ratio

    def _is_noise(self, text: str) -> bool:
        """检测是否为噪声文本。

        Args:
            text: 输入文本。

        Returns:
            是否为噪声。
        """
        text_lower = text.lower().strip()

        for pattern in self.noise_patterns:
            if pattern.search(text):
                return True

        for pattern in self.greeting_patterns:
            if pattern.match(text_lower):
                return True

        if self.remove_incomplete:
            incomplete_indicators = [
                "...",
                "……",
                "。",
                "，",
            ]

            if len(text) < 20:
                last_char = text[-1] if text else ""
                if last_char in incomplete_indicators:
                    return True

        return False

    def _check_repeat_ratio(self, text: str) -> bool:
        """检查重复率。

        Args:
            text: 输入文本。

        Returns:
            是否重复率过高。
        """
        if len(text) < 10:
            return False

        char_counter = Counter(text)
        most_common_count = char_counter.most_common(1)[0][1]

        repeat_ratio = most_common_count / len(text)

        return repeat_ratio > self.max_repeat_ratio

    def get_stats(self) -> Dict[str, int]:
        """获取过滤统计信息。"""
        stats = self._stats.copy()
        stats["removed_total"] = stats["total_input"] - stats["total_output"]
        return stats


class CleaningPipeline:
    """清洗流水线。

    将多个清洗步骤组合成流水线：
    1. 文本清洗
    2. 去重
    3. 质量过滤

    示例：
        >>> pipeline = CleaningPipeline(
        ...     cleaner=TextCleaner(),
        ...     deduper=DataCleaner(),
        ...     filter=QualityFilter()
        ... )
        >>> cleaned_data = pipeline.process(raw_data)
    """

    def __init__(
        self,
        cleaner: Optional[TextCleaner] = None,
        deduper: Optional[DataCleaner] = None,
        filter: Optional[QualityFilter] = None,
        text_key: str = "text",
    ):
        self.cleaner = cleaner or TextCleaner()
        self.deduper = deduper or DataCleaner()
        self.quality_filter = filter or QualityFilter()
        self.text_key = text_key

        self._stats = {}

    def process(
        self,
        data: List[Dict[str, Any]],
        output_path: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """执行完整清洗流程。

        Args:
            data: 原始数据。
            output_path: 可选，保存结果的路径。

        Returns:
            清洗后的数据。
        """
        logger.info(f"开始清洗流程，输入数据 {len(data)} 条")

        logger.info("步骤1：文本清洗")
        for item in data:
            if self.text_key in item:
                item[self.text_key] = self.cleaner.clean_text(item[self.text_key])

        cleaner_stats = self.cleaner.get_stats()
        logger.info(f"  - 文本清洗完成")

        logger.info("步骤2：数据去重")
        data = self.deduper.deduplicate(data, key=self.text_key, strategy="hash")
        deduper_stats = self.deduper.get_stats()

        logger.info("步骤3：质量过滤")
        data = self.quality_filter.filter(data, text_key=self.text_key)
        filter_stats = self.quality_filter.get_stats()

        self._stats = {
            "input": deduper_stats["total_input"],
            "output": filter_stats["total_output"],
            "cleaner": cleaner_stats,
            "deduper": deduper_stats,
            "filter": filter_stats,
        }

        if output_path:
            self._save_results(data, output_path)

        logger.info(
            f"清洗流程完成：输入 {self._stats['input']} 条，"
            f"输出 {self._stats['output']} 条，"
            f"保留率 {self._stats['output'] / self._stats['input'] * 100:.2f}%"
        )

        return data

    def _save_results(self, data: List[Dict[str, Any]], output_path: str):
        """保存结果。"""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        logger.info(f"清洗结果已保存到: {output_path}")

    def get_stats(self) -> Dict[str, Any]:
        """获取完整的清洗统计信息。"""
        return self._stats.copy()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Lingmao Moyun 数据清洗工具")
    parser.add_argument("--input", type=str, required=True, help="输入JSONL文件")
    parser.add_argument("--output", type=str, required=True, help="输出JSONL文件")
    parser.add_argument("--min-length", type=int, default=5, help="最小文本长度")
    parser.add_argument("--max-length", type=int, default=10000, help="最大文本长度")
    parser.add_argument("--remove-duplicates", action="store_true", help="去除重复")
    parser.add_argument("--enable-similarity", action="store_true", help="启用相似度去重")

    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    pipeline = CleaningPipeline(
        cleaner=TextCleaner(),
        deduper=DataCleaner(enable_similarity=args.enable_similarity),
        filter=QualityFilter(min_length=args.min_length, max_length=args.max_length),
    )

    result = pipeline.process(data, output_path=args.output)

    print(f"清洗完成：{len(result)} / {len(data)} 条数据")
