"""Lingmao Moyun 数据增强模块。

提供文本数据增强功能，用于扩充训练数据集。
支持同义词替换、随机插入、随机删除等增强策略。

示例：
    >>> from src.data.augmentors import TextAugmentor
    >>>
    >>> # 初始化增强器
    >>> augmentor = TextAugmentor()
    >>> augmented = augmentor.augment("床前明月光")
    >>> print(augmented)
"""

import random
import re
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from collections import defaultdict
import copy

from src.logger import get_logger

logger = get_logger("LingmaoMoyun.Augmentor")


class SynonymReplacer:
    """同义词替换器。

    基于同义词表进行文本替换增强。

    示例：
        >>> replacer = SynonymReplacer()
        >>> replacer.add_synonym("大", ["巨大", "庞大", "宏大"])
        >>> result = replacer.replace("这是一个大的房子")
    """

    def __init__(
        self,
        synonym_file: Optional[Union[str, Path]] = None,
        language: str = "zh",
    ):
        self.language = language
        self.synonym_map: Dict[str, List[str]] = defaultdict(list)
        self.reverse_map: Dict[str, List[str]] = defaultdict(list)

        if synonym_file:
            self.load_synonyms(synonym_file)
        else:
            self._init_default_synonyms()

    def _init_default_synonyms(self):
        """初始化默认同义词表。"""
        common_synonyms = [
            ("大", ["巨大", "庞大", "宏大", "广大"]),
            ("小", ["微小", "细小", "渺小", "矮小"]),
            ("好", ["优秀", "良好", "优良", "出色"]),
            ("坏", ["恶劣", "糟糕", "差劲", "不良"]),
            ("美", ["美丽", "漂亮", "优美", "美好"]),
            ("丑", ["丑陋", "难看", "不美"]),
            ("快", ["迅速", "快速", "高速", "敏捷"]),
            ("慢", ["缓慢", "迟缓", "怠慢", "舒缓"]),
            ("高", ["高大", "高耸", "崇高", "高超"]),
            ("低", ["低下", "低矮", "矮小", "低微"]),
            ("新", ["全新", "崭新", "新鲜", "新颖"]),
            ("旧", ["陈旧", "古老", "破旧", "老旧"]),
            ("长", ["漫长", "冗长", "修长", "狭长"]),
            ("短", ["短暂", "短促", "短小", "矮短"]),
            ("多", ["众多", "许多", "大量", "诸多"]),
            ("少", ["少量", "稀少", "少数", "不多"]),
            ("强", ["强大", "强烈", "强劲", "坚强"]),
            ("弱", ["弱小", "微弱", "脆弱", "衰弱"]),
            ("明", ["明亮", "光明", "明白", "明朗"]),
            ("暗", ["黑暗", "昏暗", "暗淡", "阴沉"]),
            ("白", ["白色", "洁白", "雪白", "惨白"]),
            ("黑", ["黑色", "漆黑", "黝黑", "黑暗"]),
            ("红", ["红色", "通红", "鲜红", "绯红"]),
            ("蓝", ["蓝色", "天蓝", "湛蓝", "蔚蓝"]),
            ("绿", ["绿色", "翠绿", "嫩绿", "碧绿"]),
            ("是", ["为", "系", "乃", "属"]),
            ("有", ["拥有", "具有", "含有", "存在"]),
            ("无", ["没有", "不存在", "缺乏", "无有"]),
            ("去", ["前往", "离去", "离开", "行走"]),
            ("来", ["到来", "来临", "归来", "回来"]),
        ]

        for word, synonyms in common_synonyms:
            self.add_synonym(word, synonyms)

    def add_synonym(self, word: str, synonyms: List[str]):
        """添加同义词对。

        Args:
            word: 原词。
            synonyms: 同义词列表。
        """
        self.synonym_map[word].extend(synonyms)
        for syn in synonyms:
            self.reverse_map[syn].append(word)

    def load_synonyms(self, file_path: Union[str, Path]):
        """从文件加载同义词表。

        Args:
            file_path: 同义词文件路径。
        """
        file_path = Path(file_path)

        if not file_path.exists():
            logger.warning(f"同义词文件不存在: {file_path}，使用默认同义词表")
            self._init_default_synonyms()
            return

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, dict):
                for word, synonyms in data.items():
                    if isinstance(synonyms, list):
                        self.add_synonym(word, synonyms)
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        word, synonyms = item[0], item[1:]
                        if isinstance(synonyms, list):
                            self.add_synonym(word, synonyms)

            logger.info(f"加载了 {len(self.synonym_map)} 个同义词组")

        except Exception as e:
            logger.error(f"加载同义词文件失败: {e}")
            self._init_default_synonyms()

    def replace(
        self,
        text: str,
        max_replacements: int = 3,
        keep_original: bool = False,
    ) -> List[str]:
        """替换文本中的词语为同义词。

        Args:
            text: 输入文本。
            max_replacements: 最大替换次数。
            keep_original: 是否保留原文。

        Returns:
            替换后的文本列表。
        """
        results = []

        words = list(text)
        candidates = []

        for idx, word in enumerate(words):
            if word in self.synonym_map:
                candidates.append((idx, word, self.synonym_map[word]))

        if not candidates:
            return [text] if keep_original else []

        num_replacements = min(len(candidates), max_replacements)
        selected = random.sample(candidates, num_replacements)

        selected_indices = {idx for idx, _, _ in selected}

        for idx, word, synonyms in selected:
            new_synonym = random.choice(synonyms)
            new_words = words.copy()
            new_words[idx] = new_synonym
            results.append("".join(new_words))

        if keep_original:
            results.insert(0, text)

        return results

    def replace_random(
        self,
        text: str,
        n: int = 1,
    ) -> str:
        """随机替换一个词为同义词。

        Args:
            text: 输入文本。
            n: 替换次数。

        Returns:
            替换后的文本。
        """
        for _ in range(n):
            words = list(text)
            candidates = [
                (idx, word)
                for idx, word in enumerate(words)
                if word in self.synonym_map
            ]

            if not candidates:
                break

            idx, word = random.choice(candidates)
            synonyms = self.synonym_map[word]
            words[idx] = random.choice(synonyms)
            text = "".join(words)

        return text


class RandomInserter:
    """随机插入增强器。

    在文本中随机位置插入词语或字符。

    示例：
        >>> inserter = RandomInserter()
        >>> result = inserter.insert("床前明月光", count=2)
    """

    def __init__(
        self,
        insert_words: Optional[List[str]] = None,
        language: str = "zh",
    ):
        self.language = language
        self.insert_words = insert_words or self._get_default_insert_words()

    def _get_default_insert_words(self) -> List[str]:
        """获取默认插入词列表。"""
        if self.language == "zh":
            return [
                "的", "了", "在", "是", "我", "有",
                "和", "就", "不", "人", "都", "一",
                "一个", "上", "也", "很", "到", "说",
                "要", "去", "你", "会", "着", "没有",
            ]
        else:
            return [
                "the", "a", "an", "and", "or", "but",
                "in", "on", "at", "to", "for",
            ]

    def insert(
        self,
        text: str,
        count: int = 1,
        positions: Optional[List[int]] = None,
    ) -> str:
        """在文本中插入词语。

        Args:
            text: 输入文本。
            count: 插入次数。
            positions: 指定插入位置列表，如果为None则随机选择。

        Returns:
            插入后的文本。
        """
        words = list(text)
        result = text

        if positions is None:
            positions = random.sample(
                range(len(words) + count),
                min(count, len(words) + count)
            )
            positions.sort()

        for i, pos in enumerate(positions):
            insert_word = random.choice(self.insert_words)
            adjusted_pos = pos + i * len(insert_word)
            if adjusted_pos <= len(result):
                result = result[:adjusted_pos] + insert_word + result[adjusted_pos:]

        return result

    def insert_between(
        self,
        text: str,
        count: int = 1,
    ) -> str:
        """在字符之间插入词语。

        Args:
            text: 输入文本。
            count: 每个位置插入的词语数量。

        Returns:
            增强后的文本。
        """
        if len(text) < 2:
            return text

        chars = list(text)
        result = []

        for i, char in enumerate(chars):
            result.append(char)
            if i < len(chars) - 1 and random.random() > 0.5:
                insert_word = random.choice(self.insert_words)
                result.append(insert_word)

        return "".join(result)


class RandomDeleter:
    """随机删除增强器。

    随机删除文本中的词语或字符。

    示例：
        >>> deleter = RandomDeleter()
        >>> result = deleter.delete("床前明月光", prob=0.2)
    """

    def __init__(
        self,
        delete_chars: bool = False,
        min_length: int = 5,
    ):
        self.delete_chars = delete_chars
        self.min_length = min_length

    def delete(
        self,
        text: str,
        prob: float = 0.1,
        count: int = 1,
    ) -> str:
        """随机删除文本中的字符或词语。

        Args:
            text: 输入文本。
            prob: 删除概率。
            count: 删除数量。

        Returns:
            删除后的文本。
        """
        if len(text) < self.min_length:
            return text

        if self.delete_chars:
            chars = list(text)
            result = []

            for char in chars:
                if random.random() > prob:
                    result.append(char)

            return "".join(result) if result else text
        else:
            words = list(text)
            if len(words) <= count:
                return text

            indices_to_delete = random.sample(
                range(len(words)),
                min(count, len(words))
            )

            result = [
                word for idx, word in enumerate(words)
                if idx not in indices_to_delete
            ]

            return "".join(result) if result else text

    def delete_repeated(
        self,
        text: str,
    ) -> str:
        """删除连续重复的字符。

        Args:
            text: 输入文本。

        Returns:
            处理后的文本。
        """
        if len(text) < 2:
            return text

        result = [text[0]]

        for char in text[1:]:
            if char != result[-1]:
                result.append(char)

        return "".join(result)


class SwapAugmentor:
    """交换增强器。

    交换文本中相邻或随机位置的字符/词语。

    示例：
        >>> swapper = SwapAugmentor()
        >>> result = swapper.swap("床前明月光", n=2)
    """

    def __init__(
        self,
        swap_chars: bool = True,
        swap_words: bool = False,
    ):
        self.swap_chars = swap_chars
        self.swap_words = swap_words

    def swap(
        self,
        text: str,
        n: int = 1,
    ) -> str:
        """随机交换文本中的字符或词语。

        Args:
            text: 输入文本。
            n: 交换次数。

        Returns:
            交换后的文本。
        """
        if self.swap_chars:
            return self._swap_chars(text, n)
        elif self.swap_words:
            return self._swap_words(text, n)
        else:
            return self._swap_chars(text, n)

    def _swap_chars(self, text: str, n: int) -> str:
        """交换字符。"""
        chars = list(text)
        length = len(chars)

        if length < 2:
            return text

        for _ in range(n):
            if length < 2:
                break

            idx = random.randint(0, length - 2)
            chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]

        return "".join(chars)

    def _swap_words(self, text: str, n: int) -> str:
        """交换词语。"""
        words = list(text)
        length = len(words)

        if length < 2:
            return text

        for _ in range(n):
            if length < 2:
                break

            idx = random.randint(0, length - 2)
            words[idx], words[idx + 1] = words[idx + 1], words[idx]

        return "".join(words)


class TextAugmentor:
    """文本数据增强器。

    综合多种增强策略的文本增强器：
    - 同义词替换
    - 随机插入
    - 随机删除
    - 字符交换

    示例：
        >>> augmentor = TextAugmentor(
        ...     enable_synonym=True,
        ...     enable_insert=True,
        ...     enable_delete=True
        ... )
        >>>
        >>> # 增强单条数据
        >>> augmented = augmentor.augment("床前明月光")
        >>>
        >>> # 批量增强
        >>> results = augmentor.augment_batch(data_list)
    """

    def __init__(
        self,
        enable_synonym: bool = True,
        enable_insert: bool = False,
        enable_delete: bool = False,
        enable_swap: bool = False,
        synonym_file: Optional[Union[str, Path]] = None,
        augmentation_ratio: float = 1.0,
        random_seed: Optional[int] = None,
    ):
        self.enable_synonym = enable_synonym
        self.enable_insert = enable_insert
        self.enable_delete = enable_delete
        self.enable_swap = enable_swap
        self.augmentation_ratio = augmentation_ratio

        if random_seed is not None:
            random.seed(random_seed)

        self.synonym_replacer = None
        if enable_synonym:
            self.synonym_replacer = SynonymReplacer(synonym_file=synonym_file)

        self.inserter = RandomInserter() if enable_insert else None
        self.deleter = RandomDeleter() if enable_delete else None
        self.swapper = SwapAugmentor() if enable_swap else None

        self._stats = {
            "total_input": 0,
            "total_output": 0,
            "synonym_replaced": 0,
            "inserted": 0,
            "deleted": 0,
            "swapped": 0,
        }

    def augment(
        self,
        text: str,
        n: int = 1,
        keep_original: bool = False,
    ) -> List[str]:
        """增强单条文本。

        Args:
            text: 输入文本。
            n: 生成增强样本的数量。
            keep_original: 是否保留原文。

        Returns:
            增强后的文本列表。
        """
        self._stats["total_input"] += 1

        results = []
        if keep_original:
            results.append(text)

        for _ in range(n):
            augmented = self._apply_augmentation(text)
            if augmented and augmented != text:
                results.append(augmented)
            elif keep_original and not results:
                results.append(text)

        self._stats["total_output"] += len(results)
        return results

    def _apply_augmentation(self, text: str) -> str:
        """应用增强策略。"""
        augmented = text

        if self.enable_swap and self.swapper and random.random() > 0.5:
            augmented = self.swapper.swap(augmented, n=1)
            self._stats["swapped"] += 1

        if self.enable_delete and self.deleter and random.random() > 0.5:
            augmented = self.deleter.delete(augmented, prob=0.1, count=1)
            self._stats["deleted"] += 1

        if self.enable_insert and self.inserter and random.random() > 0.3:
            augmented = self.inserter.insert(augmented, count=1)
            self._stats["inserted"] += 1

        if self.enable_synonym and self.synonym_replacer and random.random() > 0.3:
            replacements = self.synonym_replacer.replace(
                augmented,
                max_replacements=2,
                keep_original=False
            )
            if replacements:
                augmented = random.choice(replacements)
                self._stats["synonym_replaced"] += 1

        return augmented

    def augment_batch(
        self,
        data: List[Dict[str, Any]],
        text_key: str = "text",
        n_per_sample: int = 1,
        keep_original: bool = True,
    ) -> List[Dict[str, Any]]:
        """批量增强数据。

        Args:
            data: 输入数据列表。
            text_key: 文本字段名。
            n_per_sample: 每条数据生成的增强样本数。
            keep_original: 是否保留原文。

        Returns:
            增强后的数据列表。
        """
        results = []

        for item in data:
            text = item.get(text_key, "")
            if not isinstance(text, str):
                continue

            augmented_texts = self.augment(text, n=n_per_sample, keep_original=keep_original)

            for aug_text in augmented_texts:
                new_item = copy.deepcopy(item)
                new_item[text_key] = aug_text
                if aug_text != text:
                    new_item["augmented"] = True
                else:
                    new_item["augmented"] = False
                results.append(new_item)

        logger.info(f"批量增强完成：输入 {len(data)} 条，输出 {len(results)} 条")
        return results

    def augment_file(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        text_key: str = "text",
        n_per_sample: int = 1,
        keep_original: bool = True,
    ) -> Dict[str, int]:
        """增强文件中的数据。

        Args:
            input_path: 输入文件路径。
            output_path: 输出文件路径。
            text_key: 文本字段名。
            n_per_sample: 每条数据生成的增强样本数。
            keep_original: 是否保留原文。

        Returns:
            增强统计信息。
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        data.append({text_key: line})

        results = self.augment_batch(data, text_key, n_per_sample, keep_original)

        with open(output_path, "w", encoding="utf-8") as f:
            for item in results:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        logger.info(f"文件增强完成：输出到 {output_path}")
        return self.get_stats()

    def get_stats(self) -> Dict[str, int]:
        """获取增强统计信息。"""
        stats = self._stats.copy()
        if stats["total_input"] > 0:
            stats["avg_output_per_input"] = stats["total_output"] / stats["total_input"]
        else:
            stats["avg_output_per_input"] = 0
        return stats

    def reset_stats(self):
        """重置统计信息。"""
        self._stats = {
            "total_input": 0,
            "total_output": 0,
            "synonym_replaced": 0,
            "inserted": 0,
            "deleted": 0,
            "swapped": 0,
        }

    def add_custom_augmentation(
        self,
        name: str,
        func: Callable[[str], str],
        prob: float = 0.5,
    ):
        """添加自定义增强函数。

        Args:
            name: 增强方法名称。
            func: 增强函数，接受文本返回文本。
            prob: 应用概率。
        """
        if not hasattr(self, "_custom_augmentations"):
            self._custom_augmentations = []

        self._custom_augmentations.append({
            "name": name,
            "func": func,
            "prob": prob,
        })

        original_apply = self._apply_augmentation

        def new_apply_augmentation(text: str) -> str:
            result = original_apply(text)
            for aug in self._custom_augmentations:
                if random.random() < aug["prob"]:
                    result = aug["func"](result)
            return result

        self._apply_augmentation = new_apply_augmentation


class BackTranslationAugmentor:
    """回译增强器（可选，需要翻译API）。

    通过翻译成其他语言再翻译回来进行数据增强。

    示例：
        >>> # 需要配置翻译API
        >>> augmentor = BackTranslationAugmentor(
        ...     translator=your_translator,
        ...     languages=["en", "ja"]
        ... )
        >>> result = augmentor.augment("床前明月光")
    """

    def __init__(
        self,
        translator: Optional[Any] = None,
        languages: Optional[List[str]] = None,
    ):
        self.translator = translator
        self.languages = languages or ["en"]

        if translator is None:
            logger.warning("未配置翻译器，回译增强不可用")

    def augment(self, text: str) -> List[str]:
        """回译增强。

        Args:
            text: 输入文本。

        Returns:
            增强后的文本列表。
        """
        if self.translator is None:
            return [text]

        results = []

        for lang in self.languages:
            try:
                translated = self.translator.translate(text, target_lang=lang)
                back_translated = self.translator.translate(translated, target_lang="zh")

                if back_translated and back_translated != text:
                    results.append(back_translated)

            except Exception as e:
                logger.warning(f"回译失败: {e}")
                continue

        return results if results else [text]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Lingmao Moyun 数据增强工具")
    parser.add_argument("--input", type=str, required=True, help="输入JSONL文件")
    parser.add_argument("--output", type=str, required=True, help="输出JSONL文件")
    parser.add_argument("--text-key", type=str, default="text", help="文本字段名")
    parser.add_argument("--n-per-sample", type=int, default=2, help="每条样本生成的增强数量")
    parser.add_argument("--keep-original", action="store_true", help="保留原文")
    parser.add_argument("--enable-synonym", action="store_true", default=True, help="启用同义词替换")
    parser.add_argument("--enable-insert", action="store_true", help="启用随机插入")
    parser.add_argument("--enable-delete", action="store_true", help="启用随机删除")
    parser.add_argument("--enable-swap", action="store_true", help="启用字符交换")
    parser.add_argument("--seed", type=int, help="随机种子")

    args = parser.parse_args()

    augmentor = TextAugmentor(
        enable_synonym=args.enable_synonym,
        enable_insert=args.enable_insert,
        enable_delete=args.enable_delete,
        enable_swap=args.enable_swap,
        random_seed=args.seed,
    )

    augmentor.augment_file(
        args.input,
        args.output,
        text_key=args.text_key,
        n_per_sample=args.n_per_sample,
        keep_original=args.keep_original,
    )

    stats = augmentor.get_stats()
    print(f"\n增强统计：")
    print(f"  输入样本: {stats['total_input']}")
    print(f"  输出样本: {stats['total_output']}")
    print(f"  同义词替换: {stats['synonym_replaced']}")
    print(f"  随机插入: {stats['inserted']}")
    print(f"  随机删除: {stats['deleted']}")
    print(f"  字符交换: {stats['swapped']}")
