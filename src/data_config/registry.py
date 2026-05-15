"""数据集注册表。

本模块提供数据集的注册、查询、下载等功能。
DatasetRegistry类管理所有可用数据集的元信息，支持本地和远程数据集的注册。
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime

from src.logger import get_logger

logger = get_logger("LingmaoMoyun.DataConfig.Registry")


@dataclass
class DatasetInfo:
    """数据集信息数据类。
    
    存储单个数据集的元信息，包括名称、来源、路径、大小、描述等。
    
    属性：
        name: 数据集名称（唯一标识符）
        source: 数据源URL
        path: 本地存储路径
        size: 数据集大小（样本数）
        description: 数据集描述
        format: 数据格式
        tags: 标签列表
        created_at: 创建时间
        last_used: 最后使用时间
        checksum: 数据校验和（可选）
        metadata: 额外的元数据
    """
    name: str
    source: str
    path: str
    size: Optional[int] = None
    description: Optional[str] = None
    format: str = "jsonl"
    tags: List[str] = field(default_factory=list)
    created_at: Optional[str] = None
    last_used: Optional[str] = None
    checksum: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式。"""
        return {
            "name": self.name,
            "source": self.source,
            "path": self.path,
            "size": self.size,
            "description": self.description,
            "format": self.format,
            "tags": self.tags,
            "created_at": self.created_at,
            "last_used": self.last_used,
            "checksum": self.checksum,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetInfo":
        """从字典创建DatasetInfo实例。"""
        return cls(**data)
    
    def exists(self) -> bool:
        """检查数据集文件是否存在。"""
        from pathlib import Path
        p = Path(self.path)
        return p.exists() and p.is_file()


class DatasetRegistry:
    """数据集注册表。
    
    管理所有可用数据集的注册、查询和下载操作。
    支持从YAML/JSON配置文件加载数据集信息。
    
    功能：
    - 注册新数据集
    - 列出所有已注册的数据集
    - 获取特定数据集的信息
    - 下载远程数据集
    - 验证数据集完整性
    - 持久化注册表到文件
    
    示例：
        >>> registry = DatasetRegistry()
        >>> registry.register_dataset(
        ...     name="poetry",
        ...     source="https://example.com/poetry.jsonl",
        ...     path="data/poetry.jsonl",
        ...     size=10000,
        ...     description="古诗词数据集"
        ... )
        >>> 
        >>> # 列出所有数据集
        >>> for name, info in registry.list_datasets():
        ...     print(f"{name}: {info.description}")
        >>> 
        >>> # 下载数据集
        >>> registry.download_dataset("poetry")
    """
    
    def __init__(self, registry_path: Optional[str] = None):
        """初始化注册表。
        
        Args:
            registry_path: 注册表文件路径。如果为None，使用默认路径。
        """
        self._datasets: Dict[str, DatasetInfo] = {}
        self.registry_path = registry_path or self._get_default_registry_path()
        self._load_registry()
    
    @staticmethod
    def _get_default_registry_path() -> str:
        """获取默认的注册表文件路径。"""
        from pathlib import Path
        config_dir = Path.home() / ".lingmao" / "data_registry"
        config_dir.mkdir(parents=True, exist_ok=True)
        return str(config_dir / "registry.json")
    
    def _load_registry(self) -> None:
        """从文件加载注册表。"""
        if not os.path.exists(self.registry_path):
            logger.info(f"注册表文件不存在，将创建新注册表: {self.registry_path}")
            return
        
        try:
            with open(self.registry_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for name, info_dict in data.get("datasets", {}).items():
                    self._datasets[name] = DatasetInfo.from_dict(info_dict)
            logger.info(f"已加载 {len(self._datasets)} 个数据集")
        except Exception as e:
            logger.error(f"加载注册表失败: {e}")
    
    def _save_registry(self) -> None:
        """保存注册表到文件。"""
        try:
            os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
            data = {
                "version": "1.0",
                "updated_at": datetime.now().isoformat(),
                "datasets": {
                    name: info.to_dict() 
                    for name, info in self._datasets.items()
                }
            }
            with open(self.registry_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.debug(f"注册表已保存到: {self.registry_path}")
        except Exception as e:
            logger.error(f"保存注册表失败: {e}")
    
    def register_dataset(
        self,
        name: str,
        source: str,
        path: str,
        size: Optional[int] = None,
        description: Optional[str] = None,
        format: str = "jsonl",
        tags: Optional[List[str]] = None,
        checksum: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DatasetInfo:
        """注册新数据集。
        
        Args:
            name: 数据集名称（唯一标识符）
            source: 数据源URL
            path: 本地存储路径
            size: 数据集大小（样本数）
            description: 数据集描述
            format: 数据格式（jsonl, json, txt等）
            tags: 标签列表
            checksum: 数据校验和（MD5/SHA256）
            metadata: 额外的元数据
            
        Returns:
            DatasetInfo: 已创建的数据集信息对象
            
        Raises:
            ValueError: 如果数据集名称已存在
        """
        name = name.strip().lower().replace(' ', '_').replace('-', '_')
        
        if name in self._datasets:
            logger.warning(f"数据集 {name} 已存在，将覆盖原有信息")
        
        info = DatasetInfo(
            name=name,
            source=source,
            path=path,
            size=size,
            description=description,
            format=format,
            tags=tags or [],
            checksum=checksum,
            metadata=metadata or {},
        )
        
        self._datasets[name] = info
        self._save_registry()
        logger.info(f"已注册数据集: {name}")
        
        return info
    
    def unregister_dataset(self, name: str) -> bool:
        """注销数据集。
        
        Args:
            name: 数据集名称
            
        Returns:
            bool: 是否成功注销
        """
        name = name.strip().lower()
        if name in self._datasets:
            del self._datasets[name]
            self._save_registry()
            logger.info(f"已注销数据集: {name}")
            return True
        return False
    
    def get_dataset(self, name: str) -> Optional[DatasetInfo]:
        """获取数据集信息。
        
        Args:
            name: 数据集名称
            
        Returns:
            DatasetInfo或None: 数据集信息，如果不存在则返回None
        """
        name = name.strip().lower()
        info = self._datasets.get(name)
        if info:
            info.last_used = datetime.now().isoformat()
        return info
    
    def list_datasets(
        self, 
        tags: Optional[List[str]] = None,
        format: Optional[str] = None,
    ) -> List[tuple]:
        """列出所有数据集。
        
        Args:
            tags: 如果指定，只返回包含这些标签的数据集
            format: 如果指定，只返回指定格式的数据集
            
        Returns:
            List[tuple]: [(name, DatasetInfo), ...] 元组列表
        """
        results = []
        for name, info in self._datasets.items():
            if tags and not any(tag in info.tags for tag in tags):
                continue
            if format and info.format != format:
                continue
            results.append((name, info))
        
        return sorted(results, key=lambda x: x[0])
    
    def search_datasets(self, query: str) -> List[tuple]:
        """搜索数据集。
        
        在名称、描述和标签中搜索匹配的 dataset。
        
        Args:
            query: 搜索关键词
            
        Returns:
            List[tuple]: 匹配的 [(name, DatasetInfo), ...] 元组列表
        """
        query = query.lower().strip()
        results = []
        
        for name, info in self._datasets.items():
            if (query in name or 
                (info.description and query in info.description.lower()) or
                any(query in tag for tag in info.tags)):
                results.append((name, info))
        
        return sorted(results, key=lambda x: x[0])
    
    def download_dataset(self, name: str, force: bool = False) -> bool:
        """下载数据集。
        
        从配置的源URL下载数据集到本地路径。
        
        Args:
            name: 数据集名称
            force: 是否强制重新下载（即使已存在）
            
        Returns:
            bool: 是否下载成功
        """
        info = self.get_dataset(name)
        if not info:
            logger.error(f"数据集 {name} 不存在")
            return False
        
        from pathlib import Path
        path = Path(info.path)
        
        if path.exists() and not force:
            logger.info(f"数据集 {name} 已存在，跳过下载")
            return True
        
        if not info.source or info.source == "local":
            logger.warning(f"数据集 {name} 没有配置下载源")
            return False
        
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            
            if info.source.startswith(('http://', 'https://')):
                self._download_http(info.source, str(path), info.checksum)
            elif info.source.startswith('git://') or '.git' in info.source:
                self._download_git(info.source, str(path.parent))
            else:
                logger.error(f"不支持的下载源格式: {info.source}")
                return False
            
            logger.info(f"数据集 {name} 下载完成")
            return True
            
        except Exception as e:
            logger.error(f"下载数据集 {name} 失败: {e}")
            return False
    
    def _download_http(self, url: str, path: str, checksum: Optional[str] = None) -> None:
        """通过HTTP下载文件。
        
        Args:
            url: 下载URL
            path: 保存路径
            checksum: 校验和（可选）
        """
        import urllib.request
        import urllib.error
        
        try:
            logger.info(f"正在下载: {url}")
            urllib.request.urlretrieve(url, path)
            
            if checksum:
                self._verify_checksum(path, checksum)
                
        except urllib.error.URLError as e:
            raise RuntimeError(f"下载失败: {e}")
    
    def _download_git(self, url: str, target_dir: str) -> None:
        """通过Git克隆仓库。
        
        Args:
            url: Git仓库URL
            target_dir: 目标目录
        """
        import subprocess
        
        try:
            logger.info(f"正在克隆: {url}")
            subprocess.run(
                ['git', 'clone', '--depth', '1', url, target_dir],
                check=True,
                capture_output=True,
                text=True
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Git克隆失败: {e}")
    
    def _verify_checksum(self, path: str, expected_checksum: str) -> bool:
        """验证文件校验和。
        
        Args:
            path: 文件路径
            expected_checksum: 期望的校验和
            
        Returns:
            bool: 校验是否通过
        """
        hash_type = "sha256" if len(expected_checksum) == 64 else "md5"
        
        try:
            if hash_type == "sha256":
                hasher = hashlib.sha256()
            else:
                hasher = hashlib.md5()
            
            with open(path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    hasher.update(chunk)
            
            actual = hasher.hexdigest()
            
            if actual != expected_checksum:
                logger.warning(f"校验和不匹配: 期望 {expected_checksum}, 实际 {actual}")
                return False
            
            logger.debug(f"校验和验证通过: {hash_type}={actual}")
            return True
            
        except Exception as e:
            logger.warning(f"校验和验证失败: {e}")
            return False
    
    def verify_dataset(self, name: str) -> bool:
        """验证数据集完整性。
        
        Args:
            name: 数据集名称
            
        Returns:
            bool: 数据集是否有效
        """
        info = self.get_dataset(name)
        if not info:
            return False
        
        path = Path(info.path)
        
        if not path.exists():
            logger.error(f"数据集文件不存在: {path}")
            return False
        
        if info.checksum:
            return self._verify_checksum(str(path), info.checksum)
        
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """获取注册表统计信息。
        
        Returns:
            Dict[str, Any]: 统计信息字典
        """
        total_size = sum(
            Path(info.path).stat().st_size 
            for info in self._datasets.values() 
            if Path(info.path).exists()
        )
        
        format_counts = {}
        for info in self._datasets.values():
            format_counts[info.format] = format_counts.get(info.format, 0) + 1
        
        return {
            "total_datasets": len(self._datasets),
            "total_size_bytes": total_size,
            "format_distribution": format_counts,
        }
    
    def load_from_config(self, config_path: str) -> int:
        """从配置文件加载数据集信息。
        
        支持YAML和JSON格式的配置文件。
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            int: 加载的数据集数量
        """
        from pathlib import Path
        
        config_path = Path(config_path)
        if not config_path.exists():
            logger.error(f"配置文件不存在: {config_path}")
            return 0
        
        try:
            if config_path.suffix in ['.yaml', '.yml']:
                import yaml
                with open(config_path, 'r', encoding='utf-8') as f:
                    configs = yaml.safe_load(f)
                    if not isinstance(configs, list):
                        configs = [configs]
            else:
                with open(config_path, 'r', encoding='utf-8') as f:
                    configs = json.load(f)
                    if not isinstance(configs, list):
                        configs = [configs]
            
            count = 0
            for cfg in configs:
                if 'name' in cfg:
                    self.register_dataset(
                        name=cfg.get('name'),
                        source=cfg.get('source', ''),
                        path=cfg.get('path', ''),
                        size=cfg.get('size'),
                        description=cfg.get('description'),
                        format=cfg.get('format', 'jsonl'),
                        tags=cfg.get('tags'),
                        checksum=cfg.get('checksum'),
                        metadata=cfg.get('metadata'),
                    )
                    count += 1
            
            logger.info(f"从配置文件加载了 {count} 个数据集")
            return count
            
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            return 0
