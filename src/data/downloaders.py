"""Lingmao Moyun 数据下载器模块。

提供从各种来源下载和处理数据的统一接口。
支持维基百科、诗词库等中文语料资源的批量下载和解析。

示例：
    >>> from src.data.downloaders import WikiDownloader, PoetryDownloader
    >>>
    >>> # 下载维基百科数据
    >>> wiki = WikiDownloader(language='zh', max_pages=1000)
    >>> wiki.download(output_dir='data/wiki')
    >>>
    >>> # 下载诗词数据
    >>> poetry = PoetryDownloader()
    >>> poetry.download(output_dir='data/poetry')
"""

import json
import re
import time
import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Callable
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
import urllib.parse

from src.logger import get_logger

logger = get_logger("LingmaoMoyun.Downloader")


class BaseDownloader(ABC):
    """数据下载器抽象基类。

    定义所有下载器必须实现的标准接口，支持批量下载、
    断点续传、进度追踪和错误处理。

    Args:
        output_dir: 下载文件保存目录。
        batch_size: 每批下载的数据量。
        max_retries: 最大重试次数。
        timeout: 请求超时时间（秒）。
    """

    def __init__(
        self,
        output_dir: str = "./collection",
        batch_size: int = 100,
        max_retries: int = 3,
        timeout: int = 30,
    ):
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.timeout = timeout
        self._stats = {
            "total_downloaded": 0,
            "total_failed": 0,
            "total_size_bytes": 0,
        }

    @abstractmethod
    def fetch_data(self, **kwargs) -> Iterator[Dict[str, Any]]:
        """获取数据源的迭代器。

        Yields:
            单条数据记录（字典格式）。
        """
        pass

    @abstractmethod
    def parse_item(self, raw_item: Any) -> Optional[Dict[str, Any]]:
        """解析单条原始数据为标准格式。

        Args:
            raw_item: 原始数据项。

        Returns:
            解析后的数据字典，解析失败返回None。
        """
        pass

    def download(
        self,
        output_dir: Optional[str] = None,
        filename: str = "data.jsonl",
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Dict[str, int]:
        """下载并保存数据。

        Args:
            output_dir: 输出目录，默认使用初始化时指定的目录。
            filename: 输出文件名。
            progress_callback: 进度回调函数，接收(current, total)参数。

        Returns:
            包含下载统计信息的字典。
        """
        output_path = Path(output_dir or self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        file_path = output_path / filename
        total = 0
        downloaded = 0

        logger.info(f"开始下载数据到: {file_path}")

        with open(file_path, "w", encoding="utf-8") as f:
            for item in self.fetch_data():
                total += 1
                parsed = self.parse_item(item)

                if parsed is not None:
                    f.write(json.dumps(parsed, ensure_ascii=False) + "\n")
                    downloaded += 1
                    self._stats["total_downloaded"] += 1

                    if progress_callback and total % 100 == 0:
                        progress_callback(downloaded, total)
                else:
                    self._stats["total_failed"] += 1

                if total % 1000 == 0:
                    logger.info(f"已处理 {total} 条数据，成功 {downloaded} 条")

        logger.info(f"下载完成：共处理 {total} 条，成功 {downloaded} 条，失败 {total - downloaded} 条")
        return self._stats

    def get_stats(self) -> Dict[str, int]:
        """获取下载统计信息。

        Returns:
            统计信息字典。
        """
        return self._stats.copy()

    def _make_request(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Optional[bytes]:
        """发送HTTP请求，支持重试机制。

        Args:
            url: 请求URL。
            headers: 请求头。
            params: URL参数。

        Returns:
            响应内容（字节），失败返回None。
        """
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"

        default_headers = {
            "User-Agent": "Mozilla/5.0 (compatible; LingmaoMoyunBot/1.0)",
            "Accept": "application/json, text/plain, */*",
        }
        if headers:
            default_headers.update(headers)

        for attempt in range(self.max_retries):
            try:
                req = Request(url, headers=default_headers)
                with urlopen(req, timeout=self.timeout) as response:
                    return response.read()
            except (URLError, HTTPError) as e:
                logger.warning(f"请求失败（第{attempt + 1}次尝试）: {url}, 错误: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(1 * (attempt + 1))

        self._stats["total_failed"] += 1
        return None


class WikiDownloader(BaseDownloader):
    """维基百科数据下载器。

    从MediaWiki API下载中文或英文维基百科内容。
    支持按类别、搜索或随机页面下载。

    示例：
        >>> downloader = WikiDownloader(language='zh')
        >>> downloader.download_by_category('人工智能', max_pages=500)
        >>>
        >>> downloader = WikiDownloader(language='en')
        >>> downloader.download_random(max_pages=1000)
    """

    API_BASE = {
        "zh": "https://zh.wikipedia.org/w/api.php",
        "en": "https://en.wikipedia.org/w/api.php",
        "wikisource": "https://zh.wikisource.org/w/api.php",
    }

    def __init__(
        self,
        language: str = "zh",
        output_dir: str = "./collection/wiki",
        **kwargs,
    ):
        super().__init__(output_dir=output_dir, **kwargs)
        self.language = language
        self.api_base = self.API_BASE.get(language, self.API_BASE["zh"])
        self._buffer = []

    def fetch_data(self, **kwargs) -> Iterator[Dict[str, Any]]:
        """获取维基百科数据。

        支持以下模式：
        - category: 按类别获取页面
        - search: 搜索关键词获取页面
        - random: 随机获取页面

        Yields:
            原始维基百科页面数据。
        """
        mode = kwargs.get("mode", "random")
        max_pages = kwargs.get("max_pages", 100)

        if mode == "category":
            category = kwargs.get("category", "")
            yield from self._fetch_by_category(category, max_pages)
        elif mode == "search":
            query = kwargs.get("query", "")
            yield from self._fetch_by_search(query, max_pages)
        else:
            yield from self._fetch_random(max_pages)

    def _fetch_by_category(self, category: str, max_pages: int) -> Iterator[Dict[str, Any]]:
        """按类别获取页面。

        Args:
            category: 类别名称（不含"Category:"前缀）。
            max_pages: 最大页面数。

        Yields:
            页面数据。
        """
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": f"Category:{category}",
            "cmlimit": "50",
            "cmtype": "page",
            "format": "json",
        }

        continue_token = {}
        fetched = 0

        while fetched < max_pages:
            query_params = {**params, **continue_token}
            response = self._make_request(self.api_base, params=query_params)

            if response is None:
                break

            try:
                data = json.loads(response.decode("utf-8"))
                members = data.get("query", {}).get("categorymembers", [])

                for member in members:
                    if fetched >= max_pages:
                        break
                    yield from self._fetch_page_content(member["title"])
                    fetched += 1

                continue_token = data.get("continue", {})
                if not continue_token:
                    break

            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"解析类别数据失败: {e}")
                break

    def _fetch_by_search(self, query: str, max_pages: int) -> Iterator[Dict[str, Any]]:
        """搜索并获取页面。

        Args:
            query: 搜索关键词。
            max_pages: 最大页面数。

        Yields:
            页面数据。
        """
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": "50",
            "format": "json",
        }

        continue_token = {}
        fetched = 0

        while fetched < max_pages:
            query_params = {**params, **continue_token}
            response = self._make_request(self.api_base, params=query_params)

            if response is None:
                break

            try:
                data = json.loads(response.decode("utf-8"))
                results = data.get("query", {}).get("search", [])

                for result in results:
                    if fetched >= max_pages:
                        break
                    yield from self._fetch_page_content(result["title"])
                    fetched += 1

                continue_token = data.get("continue", {})
                if not continue_token:
                    break

            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"解析搜索结果失败: {e}")
                break

    def _fetch_random(self, max_pages: int) -> Iterator[Dict[str, Any]]:
        """随机获取页面。

        Args:
            max_pages: 最大页面数。

        Yields:
            页面数据。
        """
        params = {
            "action": "query",
            "list": "random",
            "rnnamespace": "0",
            "rnlimit": "50",
            "format": "json",
        }

        fetched = 0

        while fetched < max_pages:
            response = self._make_request(self.api_base, params=params)

            if response is None:
                break

            try:
                data = json.loads(response.decode("utf-8"))
                results = data.get("query", {}).get("random", [])

                for result in results:
                    if fetched >= max_pages:
                        break
                    yield from self._fetch_page_content(result["title"])
                    fetched += 1

            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"解析随机页面失败: {e}")
                break

    def _fetch_page_content(self, title: str) -> Iterator[Dict[str, Any]]:
        """获取单个页面的详细内容。

        Args:
            title: 页面标题。

        Yields:
            页面数据。
        """
        params = {
            "action": "query",
            "titles": title,
            "prop": "extracts|info",
            "explaintext": "1",
            "exlimit": "1",
            "format": "json",
        }

        response = self._make_request(self.api_base, params=params)

        if response is None:
            return

        try:
            data = json.loads(response.decode("utf-8"))
            pages = data.get("query", {}).get("pages", {})

            for page_id, page_data in pages.items():
                if page_id == "-1":
                    continue

                yield {
                    "page_id": page_id,
                    "title": page_data.get("title", ""),
                    "content": page_data.get("extract", ""),
                    "timestamp": page_data.get("touched", ""),
                    "language": self.language,
                }

        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"解析页面内容失败 {title}: {e}")

    def parse_item(self, raw_item: Any) -> Optional[Dict[str, Any]]:
        """解析维基百科数据。

        Args:
            raw_item: 原始页面数据。

        Returns:
            标准化数据字典。
        """
        if not isinstance(raw_item, dict):
            return None

        content = raw_item.get("content", "")
        if len(content) < 100:
            return None

        return {
            "text": content,
            "title": raw_item.get("title", ""),
            "source": f"wikipedia_{self.language}",
            "page_id": raw_item.get("page_id", ""),
            "language": self.language,
        }

    def download_by_category(
        self,
        category: str,
        output_dir: Optional[str] = None,
        max_pages: int = 1000,
    ) -> Dict[str, int]:
        """按类别下载维基百科页面。

        Args:
            category: 类别名称。
            output_dir: 输出目录。
            max_pages: 最大页面数。

        Returns:
            下载统计信息。
        """
        filename = f"wiki_{self.language}_{category}.jsonl"
        return self.download(
            output_dir=output_dir,
            filename=filename,
        )

    def download_random(
        self,
        output_dir: Optional[str] = None,
        max_pages: int = 1000,
    ) -> Dict[str, int]:
        """随机下载维基百科页面。

        Args:
            output_dir: 输出目录。
            max_pages: 最大页面数。

        Returns:
            下载统计信息。
        """
        filename = f"wiki_{self.language}_random.jsonl"
        return self.download(
            output_dir=output_dir,
            filename=filename,
        )


class PoetryDownloader(BaseDownloader):
    """中国古典诗词数据下载器。

    从 chinese-poetry GitHub 仓库下载唐诗、宋词、楚辞等古典诗词。
    提供完整的诗词内容、作者、朝代等元数据。

    示例：
        >>> downloader = PoetryDownloader()
        >>> downloader.download_all(output_dir='data/poetry')
        >>>
        >>> # 只下载唐诗
        >>> downloader.download_tang(max_poems=5000)
    """

    GITHUB_API = "https://api.github.com"
    REPO_OWNER = "chinese-poetry"
    REPO_NAME = "chinese-poetry"

    POETRY_FILES = {
        "tang": [
            "poet.tang.0.json",
            "poet.tang.1.json",
            "poet.tang.2.json",
            "poet.tang.3.json",
            "poet.tang.4.json",
            "poet.tang.5.json",
        ],
        "song": [
            "poet.song.0.json",
            "poet.song.1.json",
            "poet.song.2.json",
            "poet.song.3.json",
            "poet.song.4.json",
        ],
        "chuci": ["chuci.json"],
        "shijing": ["shijing.json"],
        "yuanqu": ["yuanqu.0.json"],
    }

    def __init__(
        self,
        output_dir: str = "./collection/poetry",
        include_annotations: bool = True,
        **kwargs,
    ):
        super().__init__(output_dir=output_dir, **kwargs)
        self.include_annotations = include_annotations

    def fetch_data(self, **kwargs) -> Iterator[Dict[str, Any]]:
        """获取诗词数据。

        Args:
            dynasty: 朝代类型（tang/song/chuci等），None表示全部。

        Yields:
            原始诗词数据。
        """
        dynasty = kwargs.get("dynasty", None)
        max_poems = kwargs.get("max_poems", None)

        files_to_fetch = []
        if dynasty:
            files_to_fetch = self.POETRY_FILES.get(dynasty, [])
        else:
            for files in self.POETRY_FILES.values():
                files_to_fetch.extend(files)

        fetched_count = 0

        for filename in files_to_fetch:
            if max_poems and fetched_count >= max_poems:
                break

            repo_path = filename
            content = self._fetch_github_file(repo_path)

            if content:
                try:
                    poems = json.loads(content)
                    for poem in poems:
                        if max_poems and fetched_count >= max_poems:
                            break
                        poem["_source_file"] = filename
                        yield poem
                        fetched_count += 1
                except json.JSONDecodeError as e:
                    logger.error(f"解析诗词文件失败 {filename}: {e}")

    def _fetch_github_file(self, path: str) -> Optional[str]:
        """从GitHub获取文件内容。

        Args:
            path: 文件在仓库中的路径。

        Returns:
            文件内容（字符串），失败返回None。
        """
        import base64

        url = f"{self.GITHUB_API}/repos/{self.REPO_OWNER}/{self.REPO_NAME}/contents/{path}"

        for attempt in range(self.max_retries):
            try:
                req = Request(url, headers={"Accept": "application/vnd.github.v3+json"})
                with urlopen(req, timeout=self.timeout) as response:
                    data = json.loads(response.read().decode("utf-8"))

                    if isinstance(data, dict) and "content" in data:
                        content = base64.b64decode(data["content"]).decode("utf-8")
                        return content

            except (URLError, HTTPError, json.JSONDecodeError) as e:
                logger.warning(f"获取文件失败（第{attempt + 1}次）: {path}, 错误: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(1 * (attempt + 1))

        self._stats["total_failed"] += 1
        return None

    def parse_item(self, raw_item: Any) -> Optional[Dict[str, Any]]:
        """解析诗词数据。

        Args:
            raw_item: 原始诗词数据。

        Returns:
            标准化诗词字典。
        """
        if not isinstance(raw_item, dict):
            return None

        if "paragraphs" not in raw_item:
            return None

        paragraphs = raw_item.get("paragraphs", [])
        if not paragraphs:
            return None

        text = "\n".join(paragraphs)
        if len(text) < 10:
            return None

        dynasty = self._infer_dynasty(raw_item.get("_source_file", ""))

        result = {
            "text": text,
            "author": raw_item.get("author", "未知"),
            "dynasty": dynasty,
            "source": raw_item.get("_source_file", "unknown"),
        }

        if "title" in raw_item:
            result["title"] = raw_item["title"]

        if self.include_annotations and "notes" in raw_item:
            result["notes"] = raw_item["notes"]

        return result

    def _infer_dynasty(self, source_file: str) -> str:
        """从文件名推断朝代。

        Args:
            source_file: 源文件名。

        Returns:
            朝代名称。
        """
        if "tang" in source_file.lower():
            return "唐"
        elif "song" in source_file.lower():
            return "宋"
        elif "chuci" in source_file.lower():
            return "战国"
        elif "shijing" in source_file.lower():
            return "周"
        elif "yuanqu" in source_file.lower():
            return "元"
        return "未知"

    def download_all(
        self,
        output_dir: Optional[str] = None,
        max_poems: Optional[int] = None,
    ) -> Dict[str, int]:
        """下载所有类型的诗词。

        Args:
            output_dir: 输出目录。
            max_poems: 最大诗词数量。

        Returns:
            下载统计信息。
        """
        return self.download(
            output_dir=output_dir,
            filename="poetry_all.jsonl",
        )

    def download_tang(
        self,
        output_dir: Optional[str] = None,
        max_poems: Optional[int] = None,
    ) -> Dict[str, int]:
        """下载唐诗。

        Args:
            output_dir: 输出目录。
            max_poems: 最大诗词数量。

        Returns:
            下载统计信息。
        """
        return self.download(
            output_dir=output_dir,
            filename="poetry_tang.jsonl",
        )

    def download_song(
        self,
        output_dir: Optional[str] = None,
        max_poems: Optional[int] = None,
    ) -> Dict[str, int]:
        """下载宋词。

        Args:
            output_dir: 输出目录。
            max_poems: 最大诗词数量。

        Returns:
            下载统计信息。
        """
        return self.download(
            output_dir=output_dir,
            filename="poetry_song.jsonl",
        )


class CustomDownloader(BaseDownloader):
    """自定义下载器。

    支持从自定义URL或API获取数据。适用于：
    - 自建API服务
    - 其他开放数据集
    - 本地文件批量处理

    示例：
        >>> def custom_parser(item):
        ...     return {"text": item["content"], "label": item["tag"]}
        >>>
        >>> downloader = CustomDownloader(
        ...     base_url="https://api.example.com/data",
        ...     parser=custom_parser
        ... )
        >>> downloader.download()
    """

    def __init__(
        self,
        base_url: str,
        output_dir: str = "./collection/custom",
        parser: Optional[Callable[[Any], Optional[Dict[str, Any]]]] = None,
        page_size: int = 100,
        **kwargs,
    ):
        super().__init__(output_dir=output_dir, **kwargs)
        self.base_url = base_url
        self.parser = parser or self._default_parser
        self.page_size = page_size

    def _default_parser(self, item: Any) -> Optional[Dict[str, Any]]:
        """默认解析器。

        Args:
            item: 原始数据项。

        Returns:
            解析后的字典。
        """
        if isinstance(item, dict):
            if "text" in item:
                return item
            elif "content" in item:
                return {"text": item["content"]}
            elif "body" in item:
                return {"text": item["body"]}

        if isinstance(item, str):
            return {"text": item}

        return None

    def fetch_data(self, **kwargs) -> Iterator[Dict[str, Any]]:
        """获取自定义数据。

        支持分页获取数据。

        Yields:
            原始数据项。
        """
        page = kwargs.get("page", 1)
        max_pages = kwargs.get("max_pages", 100)

        while page <= max_pages:
            params = {
                "page": page,
                "size": self.page_size,
                **kwargs.get("params", {}),
            }

            response = self._make_request(self.base_url, params=params)

            if response is None:
                break

            try:
                data = json.loads(response.decode("utf-8"))

                items = data if isinstance(data, list) else data.get("data", [])
                if not items:
                    break

                for item in items:
                    yield item

                page += 1

            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"解析自定义数据失败: {e}")
                break

    def parse_item(self, raw_item: Any) -> Optional[Dict[str, Any]]:
        """解析自定义数据。

        Args:
            raw_item: 原始数据项。

        Returns:
            解析后的数据字典。
        """
        return self.parser(raw_item)


def create_downloader(
    source: str,
    output_dir: str = "./collection",
    **kwargs,
) -> BaseDownloader:
    """工厂函数：创建下载器实例。

    Args:
        source: 数据源类型（wiki/zh, wiki/en, poetry, custom）。
        output_dir: 输出目录。
        **kwargs: 传递给下载器的其他参数。

    Returns:
        下载器实例。

    Raises:
        ValueError: 不支持的数据源类型。
    """
    source_lower = source.lower()

    if source_lower in ("wiki", "wiki_zh", "wiki:zh"):
        return WikiDownloader(language="zh", output_dir=output_dir, **kwargs)
    elif source_lower in ("wiki_en", "wiki:en"):
        return WikiDownloader(language="en", output_dir=output_dir, **kwargs)
    elif source_lower == "poetry":
        return PoetryDownloader(output_dir=output_dir, **kwargs)
    elif source_lower == "custom":
        if "base_url" not in kwargs:
            raise ValueError("Custom downloader requires 'base_url' parameter")
        return CustomDownloader(output_dir=output_dir, **kwargs)
    else:
        raise ValueError(f"Unsupported source type: {source}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Lingmao Moyun 数据下载工具")
    parser.add_argument(
        "--source",
        type=str,
        default="poetry",
        choices=["wiki_zh", "wiki_en", "poetry", "custom"],
        help="数据源类型",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./collection",
        help="输出目录",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=1000,
        help="最大下载页面数",
    )
    parser.add_argument(
        "--category",
        type=str,
        help="维基百科类别（仅wiki源）",
    )

    args = parser.parse_args()

    downloader = create_downloader(args.source, output_dir=args.output)

    if args.source.startswith("wiki"):
        if args.category:
            downloader.download_by_category(
                args.category,
                max_pages=args.max_pages,
            )
        else:
            downloader.download_random(max_pages=args.max_pages)
    else:
        downloader.download(max_pages=args.max_pages)

    stats = downloader.get_stats()
    print(f"下载完成: {json.dumps(stats, indent=2, ensure_ascii=False)}")
