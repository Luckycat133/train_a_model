"""
TODO: Replace ClassicalTokenizer with an improved BPE-style tokenizer for Chinese poetry.

Current Problem:
  ClassicalTokenizer uses greedy max-match which is O(n * max_word_len) and suboptimal
  for classical Chinese. It fails on unknown phrases and doesn't capture subword patterns.

Recommended Approach: Unigram-Language-Model BPE (ULM-BPE) adapted for Chinese

  1. Base tokens: Character-level (each Chinese character = 1 token initially)
     - Classical Chinese uses ~4,000-5,000 unique characters (less than modern Chinese)
     - Each character carries semantic meaning (ideal for subword decomposition)

  2. Merge operations: Learn top-k character n-gram merges from corpus
     - Count all adjacent character pairs (bigrams) in training corpus
     - Iteratively merge most frequent pair → update pair counts → repeat
     - Stop when vocab size reaches target or frequency threshold
     - Capture common classical Chinese words/phrases: "床前", "明月光", "春风", etc.

  3. Implementation using only regex + standard library:
     - Step 1: Character-tokenize corpus, count adjacent pairs
     - Step 2: Find max-frequency pair, merge all occurrences, update counts
     - Step 3: Repeat until vocab_size target reached
     - Use efficient pair counting with dict (similar to Senrar's BPE)
     - Apply merges to new text via regex replacement pipeline

  4. Benefits for Chinese poetry:
     - Handles OOV gracefully (character-level fallback)
     - Learns domain-specific phrases from training corpus
     - O(k * n) tokenization where k = num merge operations (typically < vocab_size)
     - Captures both single characters and multi-char words as separate tokens

  5. Suggested vocab_size for Chinese poetry:
     - Minimum: 8,000 (covers ~4k chars + 4k common bigrams/phrases)
     - Recommended: 15,000-20,000 (richer phrase vocabulary)
     - See config/default.yaml tokenizer.vocab_size setting

  Reference: See config/default.yaml for updated vocab_size recommendation.
"""
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
        # token_to_id mapping (populated by load() or build_vocab())
        self.token_to_id: dict = {}
        self.id_to_token: dict = {}

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

    # ── Persistence ───────────────────────────────────────────────────────

    def load(self, path: str) -> None:
        """"Deserialize tokenizer state from a JSON file."""
        import json
        with open(path, "r", encoding="utf-8", errors="strict") as f:
            data = json.load(f)
        if "vocab" in data:
            vocab = data["vocab"]
        elif "model" in data and "vocab" in data["model"]:
            vocab = data["model"]["vocab"]
        else:
            raise ValueError(f"Unrecognised tokenizer format in {path}")
        self.token_to_id = dict(vocab)
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}


    def save(self, path: str) -> None:
        """"Serialize tokenizer state to a JSON file."""
        import json
        data = {"vocab": self.token_to_id}
        with open(path, "w", encoding="utf-8", errors="strict") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def encode(self, text: str) -> List[int]:
        """Convert text to a list of token IDs (alias for tokenize → id lookup)."""
        tokens = self.tokenize(text)
        return [self.token_to_id.get(t, self.token_to_id.get("<unk>", 0)) for t in tokens]


    def decode(self, ids: List[int]) -> str:
        """Convert a list of token IDs back to text."""
        return "".join(self.id_to_token.get(i, "<unk>") for i in ids)
