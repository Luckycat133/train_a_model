#!/usr/bin/env python3
"""
数据下载脚本 - 灵猫墨韵项目
支持多源下载、断点续传、进度显示、文件完整性校验、并行下载
"""

import os
import sys
import json
import hashlib
import logging
import argparse
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple
import urllib.request
import urllib.error

try:
    import requests
    from tqdm import tqdm
except ImportError:
    print("请安装依赖: pip install requests tqdm")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('download.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class DownloadProgressTracker:
    """下载进度跟踪器"""

    def __init__(self, total_size: int, desc: str = ""):
        self.pbar = tqdm(total=total_size, unit='B', unit_scale=True, desc=desc)
        self.downloaded = 0

    def update(self, chunk_size: int):
        self.downloaded += chunk_size
        self.pbar.update(chunk_size)

    def close(self):
        self.pbar.close()


class DataDownloader:
    """数据下载器基类"""

    def __init__(self, output_dir: str = "dataset/raw", chunk_size: int = 8192):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'LingmaoMoYun-DataDownloader/1.0'
        })

        self.checkpoint_file = self.output_dir / "download_checkpoint.json"
        self.checkpoint = self._load_checkpoint()

    def _load_checkpoint(self) -> Dict:
        """加载断点续传信息"""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"无法加载断点文件: {e}")
        return {}

    def _save_checkpoint(self):
        """保存断点续传信息"""
        try:
            with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(self.checkpoint, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"无法保存断点文件: {e}")

    def _get_file_hash(self, file_path: Path, algorithm: str = 'sha256') -> str:
        """计算文件哈希值"""
        hash_func = hashlib.new(algorithm)
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(self.chunk_size), b''):
                hash_func.update(chunk)
        return hash_func.hexdigest()

    def _verify_file(self, file_path: Path, expected_hash: str, algorithm: str = 'sha256') -> bool:
        """校验文件完整性"""
        if not file_path.exists():
            return False
        actual_hash = self._get_file_hash(file_path, algorithm)
        return actual_hash.lower() == expected_hash.lower()

    def download_single(
        self,
        url: str,
        filename: str,
        expected_size: Optional[int] = None,
        expected_hash: Optional[str] = None,
        force_redownload: bool = False
    ) -> Tuple[bool, str]:
        """下载单个文件"""
        output_path = self.output_dir / filename

        if not force_redownload and output_path.exists():
            if expected_hash:
                if self._verify_file(output_path, expected_hash):
                    logger.info(f"文件已存在且校验通过: {filename}")
                    self.checkpoint[filename] = {
                        'status': 'completed',
                        'hash': expected_hash,
                        'timestamp': datetime.now().isoformat()
                    }
                    self._save_checkpoint()
                    return True, output_path
                else:
                    logger.warning(f"文件存在但校验失败，将重新下载: {filename}")
                    output_path.unlink()
            else:
                logger.info(f"文件已存在，跳过: {filename}")
                return True, output_path

        try:
            response = self.session.get(url, stream=True, timeout=60)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            if expected_size and total_size != expected_size:
                logger.warning(
                    f"文件大小不匹配: 期望 {expected_size}, 实际 {total_size}"
                )

            logger.info(f"开始下载: {filename} ({total_size / 1024 / 1024:.2f} MB)")
            tracker = DownloadProgressTracker(total_size, desc=filename)

            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=self.chunk_size):
                    if chunk:
                        f.write(chunk)
                        tracker.update(len(chunk))

            tracker.close()

            if expected_hash:
                if not self._verify_file(output_path, expected_hash):
                    logger.error(f"下载后校验失败: {filename}")
                    output_path.unlink()
                    return False, ""

            self.checkpoint[filename] = {
                'status': 'completed',
                'hash': expected_hash or self._get_file_hash(output_path),
                'size': total_size,
                'timestamp': datetime.now().isoformat()
            }
            self._save_checkpoint()

            logger.info(f"下载完成: {filename}")
            return True, output_path

        except requests.exceptions.RequestException as e:
            logger.error(f"下载失败 [{filename}]: {e}")
            if output_path.exists():
                output_path.unlink()
            return False, ""

    def download_multiple(
        self,
        sources: List[Dict],
        max_workers: int = 3,
        force_redownload: bool = False
    ) -> Dict[str, Tuple[bool, str]]:
        """并行下载多个文件"""
        results = {}

        def download_task(source: Dict) -> Tuple[str, Tuple[bool, str]]:
            filename = source.get('filename', url.split('/')[-1])
            url = source.get('url')
            expected_size = source.get('size')
            expected_hash = source.get('hash')

            success, path = self.download_single(
                url=url,
                filename=filename,
                expected_size=expected_size,
                expected_hash=expected_hash,
                force_redownload=force_redownload
            )
            return filename, (success, str(path) if success else "")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_source = {
                executor.submit(download_task, source): source
                for source in sources
            }

            for future in as_completed(future_to_source):
                try:
                    filename, result = future.result()
                    results[filename] = result
                except Exception as e:
                    source = future_to_source[future]
                    logger.error(f"任务执行失败: {e}")

        return results


class ClassicalChineseDownloader(DataDownloader):
    """古文数据下载器"""

    def __init__(self, output_dir: str = "dataset/raw"):
        super().__init__(output_dir)
        self.category = "classical"

    def get_sources(self, config: Dict) -> List[Dict]:
        """获取古文数据源"""
        sources = []
        for source in config.get('data_sources', []):
            if source.get('category') == 'classical':
                sources.append({
                    'filename': f"classical_{source['name']}.{source.get('format', 'json')}",
                    'url': source['url'],
                    'size': source.get('size'),
                    'hash': source.get('sha256')
                })
        return sources


class ModernChineseDownloader(DataDownloader):
    """现代文数据下载器"""

    def __init__(self, output_dir: str = "dataset/raw"):
        super().__init__(output_dir)
        self.category = "modern"

    def get_sources(self, config: Dict) -> List[Dict]:
        """获取现代文数据源"""
        sources = []
        for source in config.get('data_sources', []):
            if source.get('category') == 'modern':
                sources.append({
                    'filename': f"modern_{source['name']}.{source.get('format', 'json')}",
                    'url': source['url'],
                    'size': source.get('size'),
                    'hash': source.get('sha256')
                })
        return sources


class BaikeDownloader(DataDownloader):
    """百科数据下载器"""

    def __init__(self, output_dir: str = "dataset/raw"):
        super().__init__(output_dir)
        self.category = "baike"

    def get_sources(self, config: Dict) -> List[Dict]:
        """获取百科数据源"""
        sources = []
        for source in config.get('data_sources', []):
            if source.get('category') == 'baike':
                sources.append({
                    'filename': f"baike_{source['name']}.{source.get('format', 'json')}",
                    'url': source['url'],
                    'size': source.get('size'),
                    'hash': source.get('sha256')
                })
        return sources


class DialogueDownloader(DataDownloader):
    """对话数据下载器"""

    def __init__(self, output_dir: str = "dataset/raw"):
        super().__init__(output_dir)
        self.category = "dialogue"

    def get_sources(self, config: Dict) -> List[Dict]:
        """获取对话数据源"""
        sources = []
        for source in config.get('data_sources', []):
            if source.get('category') == 'dialogue':
                sources.append({
                    'filename': f"dialogue_{source['name']}.{source.get('format', 'json')}",
                    'url': source['url'],
                    'size': source.get('size'),
                    'hash': source.get('sha256')
                })
        return sources


def load_config(config_path: str = "config/data_sources.yaml") -> Dict:
    """加载数据源配置"""
    try:
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except ImportError:
        logger.warning("PyYAML 未安装，使用默认配置")
        return {'data_sources': []}
    except FileNotFoundError:
        logger.error(f"配置文件不存在: {config_path}")
        return {'data_sources': []}


def main():
    parser = argparse.ArgumentParser(description='灵猫墨韵 - 数据下载工具')
    parser.add_argument('--config', default='config/data_sources.yaml', help='数据源配置文件')
    parser.add_argument('--output', default='dataset/raw', help='输出目录')
    parser.add_argument('--category', default='all',
                        choices=['all', 'classical', 'modern', 'baike', 'dialogue'],
                        help='下载的数据类别')
    parser.add_argument('--workers', type=int, default=3, help='并行下载线程数')
    parser.add_argument('--force', action='store_true', help='强制重新下载')
    parser.add_argument('--list-only', action='store_true', help='仅列出可用数据源')

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("灵猫墨韵 - 数据下载工具")
    logger.info("=" * 60)

    config = load_config(args.config)

    if args.list_only:
        logger.info("\n可用数据源:")
        for source in config.get('data_sources', []):
            logger.info(f"  [{source['category']}] {source['name']}: {source.get('description', '')}")
        return

    downloaders = {
        'classical': ClassicalChineseDownloader(args.output),
        'modern': ModernChineseDownloader(args.output),
        'baike': BaikeDownloader(args.output),
        'dialogue': DialogueDownloader(args.output)
    }

    if args.category == 'all':
        categories = list(downloaders.keys())
    else:
        categories = [args.category]

    total_success = 0
    total_failed = 0

    for category in categories:
        logger.info(f"\n开始下载 {category} 类数据...")
        downloader = downloaders[category]
        sources = downloader.get_sources(config)

        if not sources:
            logger.warning(f"没有找到 {category} 类数据源")
            continue

        logger.info(f"找到 {len(sources)} 个数据源")

        results = downloader.download_multiple(
            sources=sources,
            max_workers=args.workers,
            force_redownload=args.force
        )

        success_count = sum(1 for success, _ in results.values() if success)
        failed_count = len(results) - success_count

        total_success += success_count
        total_failed += failed_count

        logger.info(f"{category} 类下载完成: 成功 {success_count}, 失败 {failed_count}")

    logger.info("\n" + "=" * 60)
    logger.info(f"全部下载完成! 成功: {total_success}, 失败: {total_failed}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
