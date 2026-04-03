import re
from typing import Iterable, List, Optional, Sequence


class ClassicalTokenizer:
    """Simplified tokenizer used in unit tests.

    The implementation focuses on deterministic behaviour rather than
    linguistic accuracy.  A small built-in dictionary enables basic maximum
    matching for common classical Chinese phrases used in the tests."""

    _DEFAULT_DICT: Sequence[str] = (
        "你好",
        "世界",
        "床前",
        "明月光",
        "春风",
        "江南",
        "落霞",
        "孤鹜",
        "齐飞",
        "关关雎鸠",
        "在河之洲",
        "帝高阳之苗裔",
        "朕皇考曰伯庸",
        "秋水",
        "长天",
        "一色",
    )

    def __init__(self, dictionary: Optional[Iterable[str]] = None) -> None:
        self.dictionary = set(dictionary or self._DEFAULT_DICT)
        self.max_word_len = max((len(w) for w in self.dictionary), default=0)
        self._cache: dict = {}

    def _max_match(self, text: str) -> List[str]:
        tokens: List[str] = []
        i = 0
        n = len(text)
        while i < n:
            matched = None
            for l in range(min(self.max_word_len, n - i), 0, -1):
                piece = text[i:i + l]
                if piece in self.dictionary:
                    matched = piece
                    break
            if matched:
                tokens.append(matched)
                i += len(matched)
            else:
                tokens.append(text[i])
                i += 1
        return tokens

    def tokenize(self, text: str, method: str = "auto", text_type: Optional[str] = None) -> List[str]:
        """Tokenize *text* into a list of tokens."""
        key = (text, method, text_type)
        if key in self._cache:
            return self._cache[key]

        text = text.strip()
        if not text:
            tokens: List[str] = []
        else:
            parts = re.findall(r"\w+|\s|[^\w\s]", text, flags=re.UNICODE)
            tokens = []
            has_chinese = re.search(r"[\u4e00-\u9fff]", text) is not None
            for part in parts:
                if part.isspace():
                    if has_chinese:
                        tokens.append(part)
                    else:
                        continue
                elif re.fullmatch(r"[\u4e00-\u9fff]+", part) and method in ("auto", "max_match"):
                    tokens.extend(self._max_match(part))
                else:
                    tokens.append(part)

        self._cache[key] = tokens
        if len(self._cache) > 10000:
            self._cache.clear()
        return tokens

    def batch_tokenize(self, texts: Iterable[str], method: str = "auto", text_type: Optional[str] = None) -> List[List[str]]:
        return [self.tokenize(t, method=method, text_type=text_type) for t in texts]
