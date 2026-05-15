"""
基础下载器模块
提供异步下载、数据验证、断点续传等基础功能
"""

import asyncio
import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import httpx
except ImportError:
    raise ImportError("请安装 httpx: pip install httpx")


@dataclass
class DownloadTask:
    """下载任务配置"""
    url: str
    filename: str
    expected_hash: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    timeout: int = 300
    retry_times: int = 3
    retry_delay: float = 1.0
    chunk_size: int = 8192


@dataclass
class DownloadProgress:
    """下载进度追踪"""
    total_size: int = 0
    downloaded_size: int = 0
    status: str = "pending"
    error: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    attempts: int = 0
    speed: float = 0.0


class BaseDownloader:
    """基础下载器类，提供通用的异步下载功能"""

    def __init__(
        self,
        base_dir: str = "./data",
        max_concurrent: int = 5,
        rate_limit: float = 10.0,
        timeout: int = 300,
    ):
        self.base_dir = Path(base_dir)
        self.max_concurrent = max_concurrent
        self.rate_limit = rate_limit
        self.timeout = timeout
        self.logger = logging.getLogger(self.__class__.__name__)
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._download_history: Dict[str, DownloadProgress] = {}
        self._checkpoint_file = self.base_dir / "download_checkpoint.json"

        self._init_base_dir()
        self._load_checkpoint()

    def _init_base_dir(self):
        """初始化基础目录"""
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _load_checkpoint(self):
        """加载检查点数据"""
        if self._checkpoint_file.exists():
            try:
                with open(self._checkpoint_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for url, info in data.get('history', {}).items():
                        progress = DownloadProgress()
                        progress.total_size = info.get('total_size', 0)
                        progress.downloaded_size = info.get('downloaded_size', 0)
                        progress.status = info.get('status', 'pending')
                        progress.error = info.get('error')
                        self._download_history[url] = progress
            except Exception as e:
                self.logger.warning(f"加载检查点失败: {e}")

    def _save_checkpoint(self):
        """保存检查点数据"""
        try:
            data = {
                'history': {
                    url: {
                        'total_size': p.total_size,
                        'downloaded_size': p.downloaded_size,
                        'status': p.status,
                        'error': p.error,
                    }
                    for url, p in self._download_history.items()
                }
            }
            with open(self._checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"保存检查点失败: {e}")

    async def download(
        self,
        url: str,
        filename: str,
        expected_hash: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        resume: bool = True,
    ) -> Path:
        """
        下载单个文件

        Args:
            url: 文件URL
            filename: 保存的文件名
            expected_hash: 期望的文件哈希值
            headers: 请求头
            resume: 是否支持断点续传

        Returns:
            Path: 下载文件的路径
        """
        headers = headers or {}
        task = DownloadTask(
            url=url,
            filename=filename,
            expected_hash=expected_hash,
            headers=headers,
            timeout=self.timeout,
        )

        if url in self._download_history:
            progress = self._download_history[url]
            if progress.status == 'completed' and resume:
                file_path = self.base_dir / filename
                if file_path.exists():
                    self.logger.info(f"文件已存在，跳过下载: {filename}")
                    return file_path
                else:
                    progress.status = 'pending'
                    progress.downloaded_size = 0

        return await self._download_with_retry(task, resume)

    async def _download_with_retry(self, task: DownloadTask, resume: bool = True) -> Path:
        """带重试的下载逻辑"""
        progress = self._download_history.setdefault(task.url, DownloadProgress())
        progress.status = 'downloading'
        progress.start_time = datetime.now()
        progress.attempts += 1

        for attempt in range(task.retry_times):
            try:
                result = await self._execute_download(task, resume)
                progress.status = 'completed'
                progress.end_time = datetime.now()
                self._save_checkpoint()
                return result
            except Exception as e:
                self.logger.warning(f"下载失败 (尝试 {attempt + 1}/{task.retry_times}): {e}")
                if attempt < task.retry_times - 1:
                    await asyncio.sleep(task.retry_delay * (2 ** attempt))
                else:
                    progress.status = 'failed'
                    progress.error = str(e)
                    progress.end_time = datetime.now()
                    self._save_checkpoint()
                    raise

    async def _execute_download(self, task: DownloadTask, resume: bool = True) -> Path:
        """执行下载"""
        file_path = self.base_dir / task.filename
        file_path.parent.mkdir(parents=True, exist_ok=True)

        mode = 'ab' if resume and file_path.exists() else 'wb'
        downloaded_size = file_path.stat().st_size if resume and file_path.exists() else 0

        async with self._semaphore:
            async with httpx.AsyncClient(timeout=task.timeout) as client:
                headers = task.headers.copy()
                if resume and file_path.exists():
                    headers['Range'] = f'bytes={downloaded_size}-'

                async with client.stream('GET', task.url, headers=headers) as response:
                    response.raise_for_status()
                    total_size = int(response.headers.get('content-length', 0))
                    if resume and file_path.exists():
                        total_size += downloaded_size

                    progress = self._download_history.get(task.url)
                    if progress:
                        progress.total_size = total_size
                        progress.downloaded_size = downloaded_size

                    with open(file_path, mode) as f:
                        async for chunk in response.aiter_bytes(chunk_size=task.chunk_size):
                            f.write(chunk)
                            downloaded_size += len(chunk)
                            if progress:
                                progress.downloaded_size = downloaded_size

                            if self.rate_limit > 0:
                                await asyncio.sleep(len(chunk) / (self.rate_limit * 1024 * 1024))

        if task.expected_hash:
            actual_hash = self._calculate_hash(file_path)
            if actual_hash != task.expected_hash:
                file_path.unlink()
                raise ValueError(f"哈希校验失败: 期望 {task.expected_hash}, 实际 {actual_hash}")

        self.logger.info(f"下载完成: {task.filename}")
        return file_path

    def _calculate_hash(self, file_path: Path, algorithm: str = 'sha256') -> str:
        """计算文件哈希"""
        hash_obj = hashlib.new(algorithm)
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hash_obj.update(chunk)
        return hash_obj.hexdigest()

    def get_progress(self, url: str) -> Optional[DownloadProgress]:
        """获取下载进度"""
        return self._download_history.get(url)

    def get_statistics(self) -> Dict[str, Any]:
        """获取下载统计信息"""
        total = len(self._download_history)
        completed = sum(1 for p in self._download_history.values() if p.status == 'completed')
        failed = sum(1 for p in self._download_history.values() if p.status == 'failed')
        downloading = sum(1 for p in self._download_history.values() if p.status == 'downloading')

        total_size = sum(p.total_size for p in self._download_history.values())
        downloaded_size = sum(p.downloaded_size for p in self._download_history.values())

        return {
            'total_tasks': total,
            'completed': completed,
            'failed': failed,
            'downloading': downloading,
            'pending': total - completed - failed - downloading,
            'total_size': total_size,
            'downloaded_size': downloaded_size,
            'progress_percent': (downloaded_size / total_size * 100) if total_size > 0 else 0,
        }

    async def download_batch(
        self,
        urls: List[str],
        filenames: List[str],
        progress_callback: Optional[callable] = None,
    ) -> List[Path]:
        """批量下载"""
        tasks = [
            self.download(url, filename)
            for url, filename in zip(urls, filenames)
        ]

        results = []
        for i, task in enumerate(asyncio.as_completed(tasks)):
            result = await task
            results.append(result)
            if progress_callback:
                progress_callback(i + 1, len(tasks))

        return results

    def clean_checkpoint(self):
        """清理检查点数据"""
        completed = [
            url for url, p in self._download_history.items()
            if p.status == 'completed'
        ]
        for url in completed:
            del self._download_history[url]
        self._save_checkpoint()
        self.logger.info(f"已清理 {len(completed)} 条完成记录")

    def generate_report(self) -> Dict[str, Any]:
        """生成下载报告"""
        stats = self.get_statistics()

        report = {
            '总任务数': stats['total_tasks'],
            '已完成': stats['completed'],
            '下载中': stats['downloading'],
            '失败': stats['failed'],
            '待处理': stats['pending'],
            '总大小': self._format_size(stats['total_size']),
            '已下载': self._format_size(stats['downloaded_size']),
            '进度': f"{stats['progress_percent']:.2f}%",
            '详细记录': []
        }

        for url, progress in self._download_history.items():
            record = {
                'url': url,
                'status': progress.status,
                'size': f"{self._format_size(progress.downloaded_size)}/{self._format_size(progress.total_size)}",
                'error': progress.error
            }
            if progress.start_time and progress.end_time:
                duration = (progress.end_time - progress.start_time).total_seconds()
                record['耗时'] = self._format_duration(duration)
            report['详细记录'].append(record)

        return report

    def _format_size(self, size: int) -> str:
        """格式化文件大小"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024:
                return f"{size:.2f} {unit}"
            size /= 1024
        return f"{size:.2f} PB"

    def _format_duration(self, seconds: float) -> str:
        """格式化时间"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if hours > 0:
            return f"{hours}小时{minutes}分钟{secs}秒"
        elif minutes > 0:
            return f"{minutes}分钟{secs}秒"
        else:
            return f"{secs}秒"


class FormatConverter:
    """数据格式转换器"""

    @staticmethod
    def convert(
        source_file: Path,
        target_file: Path,
        target_format: str,
        transform_func: Optional[callable] = None
    ) -> int:
        """
        转换数据格式

        Args:
            source_file: 源文件路径
            target_file: 目标文件路径
            target_format: 目标格式 ('json', 'jsonl', 'txt')
            transform_func: 可选的转换函数

        Returns:
            int: 转换的记录数
        """
        if source_file.suffix == '.json':
            return FormatConverter._convert_json(
                source_file, target_file, target_format, transform_func
            )
        elif source_file.suffix == '.jsonl':
            return FormatConverter._convert_jsonl(
                source_file, target_file, target_format, transform_func
            )
        else:
            raise ValueError(f"不支持的源文件格式: {source_file.suffix}")

    @staticmethod
    def _convert_json(
        source_file: Path,
        target_file: Path,
        target_format: str,
        transform_func: Optional[callable]
    ) -> int:
        """转换JSON格式"""
        with open(source_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        records = data if isinstance(data, list) else [data]
        if transform_func:
            records = [transform_func(r) for r in records]

        return FormatConverter._write_records(target_file, target_format, records)

    @staticmethod
    def _convert_jsonl(
        source_file: Path,
        target_file: Path,
        target_format: str,
        transform_func: Optional[callable]
    ) -> int:
        """转换JSONL格式"""
        records = []
        with open(source_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    if transform_func:
                        record = transform_func(record)
                    records.append(record)

        return FormatConverter._write_records(target_file, target_format, records)

    @staticmethod
    def _write_records(target_file: Path, target_format: str, records: List[Any]) -> int:
        """写入记录"""
        target_file.parent.mkdir(parents=True, exist_ok=True)

        if target_format == 'json':
            with open(target_file, 'w', encoding='utf-8') as f:
                json.dump(records, f, ensure_ascii=False, indent=2)
        elif target_format == 'jsonl':
            with open(target_file, 'w', encoding='utf-8') as f:
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
        elif target_format == 'txt':
            with open(target_file, 'w', encoding='utf-8') as f:
                for record in records:
                    if isinstance(record, dict):
                        f.write(str(record) + '\n')
                    else:
                        f.write(f"{record}\n")
        else:
            raise ValueError(f"不支持的目标格式: {target_format}")

        return len(records)

    @staticmethod
    def verify_hash(file_path: Path, expected_hash: str, algorithm: str = 'sha256') -> bool:
        """验证文件哈希"""
        actual_hash = FormatConverter._calculate_hash(file_path, algorithm)
        return actual_hash == expected_hash

    @staticmethod
    def _calculate_hash(file_path: Path, algorithm: str = 'sha256') -> str:
        """计算文件哈希"""
        hash_obj = hashlib.new(algorithm)
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hash_obj.update(chunk)
        return hash_obj.hexdigest()
