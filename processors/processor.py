import re
import yaml
import chardet
from typing import List, Dict, Generator, Iterable
from contextlib import contextmanager


class MemoryPool:
    """A tiny memory pool used for testing.

    It tracks the number of allocations and the current allocated size but
    does not manage real memory blocks.  This lightweight implementation is
    sufficient for the unit tests which only exercise its public API."""

    def __init__(self, max_size: int = 0) -> None:
        self.max_size = max_size
        self.current_size = 0
        self.allocations = 0

    @contextmanager
    def acquire(self, size: int = 0):
        self.allocations += 1
        self.current_size += size
        try:
            yield self
        finally:
            self.current_size = max(0, self.current_size - size)

    def cleanup(self) -> None:
        self.current_size = 0

    def get_stats(self) -> Dict[str, int]:
        return {
            "max_size": self.max_size,
            "current_size": self.current_size,
            "allocations": self.allocations,
        }


def load_config(config_path: str) -> Dict:
    """Load YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def detect_encoding(file_path: str) -> str:
    """Detect file encoding using chardet."""
    with open(file_path, "rb") as f:
        raw = f.read()
    if not raw:
        return "utf-8"
    result = chardet.detect(raw)
    encoding = result.get("encoding") or "utf-8"
    # chardet mis-detects some short Chinese snippets as TIS-620; map it to GBK
    if encoding.upper() == "TIS-620":
        encoding = "GBK"
    return encoding


_html_re = re.compile(r"<[^>]+>")
_url_re = re.compile(r"https?://\S+")
_control_re = re.compile(r"[\x00-\x1F\x7F]")


def clean_text(text: str, rules: Dict) -> str:
    """Clean text according to rules used in tests."""
    result = text
    if rules.get("remove_html"):
        result = _html_re.sub("", result)
    if rules.get("normalize_whitespace"):
        result = re.sub(r"\s+", " ", result).strip()
    if rules.get("remove_control_chars"):
        result = _control_re.sub("", result)
    if rules.get("remove_urls"):
        result = _url_re.sub("", result)

    if rules.get("filter_quality"):
        min_length = rules.get("min_length", 0)
        max_ratio = rules.get("max_symbol_ratio", 1.0)
        if len(result) < min_length:
            return ""
        if result:
            symbols = sum(not ch.isalnum() and not ch.isspace() for ch in result)
            if symbols / len(result) > max_ratio:
                return ""
    return result


def read_data(file_list: Iterable[str], allowed_exts: List[str], batch_size: int = 100, preview: bool = False) -> Generator[List[str], None, None]:
    """Yield lines from files in batches."""
    batch: List[str] = []
    for file in file_list:
        if allowed_exts and not any(file.endswith(ext) for ext in allowed_exts):
            continue
        encoding = detect_encoding(file)
        with open(file, "r", encoding=encoding, errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                batch.append(line)
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
    if batch:
        yield batch
