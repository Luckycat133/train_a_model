"""数据集加载器。

本模块提供统一的数据加载接口，支持多种数据格式（JSONL、JSON、TXT等），
流式加载，以及数据混合等功能。
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union, Callable
from collections.abc import Iterable

import numpy as np

from src.logger import get_logger
from src.data_config.schema import (
    DatasetConfig, 
    DataFormat, 
    PreprocessingStep,
    PreprocessingType,
    ValidationRule,
    ValidationType,
    FieldMapping,
)

logger = get_logger("LingmaoMoyun.DataConfig.Loader")


class DatasetLoader:
    """统一的数据加载器。
    
    提供多种数据格式的加载接口，支持流式处理和数据预处理。
    
    功能特性：
    - 支持多种格式：JSONL、JSON、TXT、CSV、Parquet
    - 流式加载：内存友好，适合大规模数据集
    - 数据验证：内置验证规则支持
    - 数据预处理：过滤、转换、增强等操作
    - 字段映射：灵活的字段名称映射
    
    示例：
        >>> loader = DatasetLoader()
        >>> 
        >>> # 基础加载
        >>> data = loader.load("data/poetry.jsonl")
        >>> 
        >>> # 流式加载
        >>> for item in loader.stream_load("data/large_file.jsonl"):
        ...     process(item)
        >>> 
        >>> # 加载并预处理
        >>> loader = DatasetLoader()
        >>> loader.add_filter("text", lambda x: len(x) > 10)
        >>> data = loader.load("data/poetry.jsonl")
    """
    
    def __init__(self, config: Optional[DatasetConfig] = None):
        """初始化数据加载器。
        
        Args:
            config: 数据集配置对象（可选）
        """
        self.config = config
        self._filters: List[Callable[[Dict], bool]] = []
        self._transforms: List[Callable[[Dict], Dict]] = []
        
        if config:
            self._setup_from_config(config)
    
    def _setup_from_config(self, config: DatasetConfig) -> None:
        """从配置对象设置加载器。
        
        Args:
            config: 数据集配置
        """
        for step in config.preprocessing:
            if step.step_type == PreprocessingType.FILTER:
                self._add_filter_from_config(step)
            elif step.step_type == PreprocessingType.TRANSFORM:
                self._add_transform_from_config(step)
    
    def _add_filter_from_config(self, step: PreprocessingStep) -> None:
        """从配置添加过滤器。
        
        Args:
            step: 预处理步骤配置
        """
        field = step.field
        condition = step.condition
        
        if condition:
            def make_filter(cond: str, fld: str):
                return lambda item: self._evaluate_condition(item, cond, fld)
            self._filters.append(make_filter(condition, field))
        else:
            params = step.params
            if 'min_length' in params:
                min_len = params['min_length']
                self._filters.append(lambda item: len(item.get(field, '')) >= min_len)
            if 'max_length' in params:
                max_len = params['max_length']
                self._filters.append(lambda item: len(item.get(field, '')) <= max_len)
    
    def _add_transform_from_config(self, step: PreprocessingStep) -> None:
        """从配置添加转换器。
        
        Args:
            step: 预处理步骤配置
        """
        field = step.field
        transform_type = step.params.get('type', 'identity')
        
        if transform_type == 'strip':
            self._transforms.append(lambda item: {**item, field: item.get(field, '').strip()})
        elif transform_type == 'lowercase':
            self._transforms.append(lambda item: {**item, field: item.get(field, '').lower()})
        elif transform_type == 'uppercase':
            self._transforms.append(lambda item: {**item, field: item.get(field, '').upper()})
        elif transform_type == 'regex_replace':
            pattern = step.params.get('pattern', '')
            replacement = step.params.get('replacement', '')
            compiled_pattern = re.compile(pattern) if pattern else None
            
            def regex_transform(item, pat=compiled_pattern, repl=replacement, fld=field):
                text = item.get(fld, '')
                if pat:
                    return {**item, fld: pat.sub(repl, text)}
                return item
            self._transforms.append(regex_transform)
    
    def _evaluate_condition(self, item: Dict, condition: str, field: str) -> bool:
        """评估过滤条件。
        
        支持简单的比较运算符：>, <, >=, <=, ==, !=
        
        Args:
            item: 数据项
            condition: 条件表达式
            field: 字段名
            
        Returns:
            bool: 条件是否满足
        """
        value = item.get(field, '')
        
        operators = [
            ('>=', lambda a, b: a >= b),
            ('<=', lambda a, b: a <= b),
            ('!=', lambda a, b: a != b),
            ('>', lambda a, b: a > b),
            ('<', lambda a, b: a < b),
            ('==', lambda a, b: a == b),
        ]
        
        for op, func in operators:
            if op in condition:
                parts = condition.split(op)
                if len(parts) == 2:
                    try:
                        target = type(value)(parts[1].strip())
                        return func(value, target)
                    except (ValueError, TypeError):
                        pass
        
        return condition in str(value)
    
    def add_filter(self, field: str, func: Callable[[Any], bool]) -> "DatasetLoader":
        """添加过滤器函数。
        
        Args:
            field: 要过滤的字段名（仅用于文档记录）
            func: 过滤函数，输入数据项，返回bool
            
        Returns:
            DatasetLoader: 返回自身以支持链式调用
        """
        self._filters.append(func)
        return self
    
    def add_transform(self, func: Callable[[Dict], Dict]) -> "DatasetLoader":
        """添加转换函数。
        
        Args:
            func: 转换函数，输入数据项，返回转换后的数据项
            
        Returns:
            DatasetLoader: 返回自身以支持链式调用
        """
        self._transforms.append(func)
        return self
    
    def load(
        self, 
        path: Union[str, Path], 
        format: Optional[DataFormat] = None,
        encoding: str = "utf-8",
    ) -> List[Dict[str, Any]]:
        """加载数据文件。
        
        Args:
            path: 数据文件路径
            format: 数据格式（如果为None，自动检测）
            encoding: 文件编码
            
        Returns:
            List[Dict[str, Any]]: 数据列表
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")
        
        if format is None:
            format = self._detect_format(path)
        
        try:
            if format == DataFormat.JSONL:
                return self._load_jsonl(path, encoding)
            elif format == DataFormat.JSON:
                return self._load_json(path, encoding)
            elif format == DataFormat.TXT:
                return self._load_txt(path, encoding)
            elif format == DataFormat.CSV:
                return self._load_csv(path, encoding)
            else:
                raise ValueError(f"不支持的格式: {format}")
        except Exception as e:
            logger.error(f"加载文件失败 {path}: {e}")
            raise
    
    def stream_load(
        self, 
        path: Union[str, Path], 
        format: Optional[DataFormat] = None,
        encoding: str = "utf-8",
    ) -> Iterator[Dict[str, Any]]:
        """流式加载数据文件。
        
        内存友好，适合大规模数据集。
        
        Args:
            path: 数据文件路径
            format: 数据格式（如果为None，自动检测）
            encoding: 文件编码
            
        Yields:
            Dict[str, Any]: 数据项
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")
        
        if format is None:
            format = self._detect_format(path)
        
        try:
            if format == DataFormat.JSONL:
                yield from self._stream_jsonl(path, encoding)
            elif format == DataFormat.JSON:
                yield from self._stream_json(path, encoding)
            elif format == DataFormat.TXT:
                yield from self._stream_txt(path, encoding)
            elif format == DataFormat.CSV:
                yield from self._stream_csv(path, encoding)
            else:
                raise ValueError(f"不支持的格式: {format}")
        except Exception as e:
            logger.error(f"流式加载文件失败 {path}: {e}")
            raise
    
    def _detect_format(self, path: Path) -> DataFormat:
        """自动检测文件格式。
        
        Args:
            path: 文件路径
            
        Returns:
            DataFormat: 检测到的格式
        """
        suffix = path.suffix.lower().lstrip('.')
        
        format_map = {
            'jsonl': DataFormat.JSONL,
            'json': DataFormat.JSON,
            'txt': DataFormat.TXT,
            'csv': DataFormat.CSV,
            'parquet': DataFormat.PARQUET,
        }
        
        return format_map.get(suffix, DataFormat.JSONL)
    
    def _load_jsonl(self, path: Path, encoding: str) -> List[Dict[str, Any]]:
        """加载JSONL文件。
        
        Args:
            path: 文件路径
            encoding: 编码
            
        Returns:
            List[Dict[str, Any]]: 数据列表
        """
        data = []
        for item in self._stream_jsonl(path, encoding):
            processed = self._apply_transforms(item)
            if self._apply_filters(processed):
                data.append(processed)
        return data
    
    def _load_json(self, path: Path, encoding: str) -> List[Dict[str, Any]]:
        """加载JSON文件。
        
        Args:
            path: 文件路径
            encoding: 编码
            
        Returns:
            List[Dict[str, Any]]: 数据列表
        """
        with open(path, 'r', encoding=encoding) as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            data = [data]
        
        result = []
        for item in data:
            processed = self._apply_transforms(item)
            if self._apply_filters(processed):
                result.append(processed)
        return result
    
    def _load_txt(self, path: Path, encoding: str) -> List[Dict[str, Any]]:
        """加载文本文件。
        
        每行作为一个样本，字段名为'text'。
        
        Args:
            path: 文件路径
            encoding: 编码
            
        Returns:
            List[Dict[str, Any]]: 数据列表
        """
        result = []
        for item in self._stream_txt(path, encoding):
            processed = self._apply_transforms(item)
            if self._apply_filters(processed):
                result.append(processed)
        return result
    
    def _load_csv(self, path: Path, encoding: str) -> List[Dict[str, Any]]:
        """加载CSV文件。
        
        Args:
            path: 文件路径
            encoding: 编码
            
        Returns:
            List[Dict[str, Any]]: 数据列表
        """
        import csv
        
        result = []
        with open(path, 'r', encoding=encoding, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                processed = self._apply_transforms(dict(row))
                if self._apply_filters(processed):
                    result.append(processed)
        return result
    
    def _stream_jsonl(self, path: Path, encoding: str) -> Iterator[Dict[str, Any]]:
        """流式加载JSONL文件。
        
        Args:
            path: 文件路径
            encoding: 编码
            
        Yields:
            Dict[str, Any]: 数据项
        """
        with open(path, 'r', encoding=encoding) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON解析失败 (行 {line_num}): {e}")
    
    def _stream_json(self, path: Path, encoding: str) -> Iterator[Dict[str, Any]]:
        """流式加载JSON文件。
        
        Args:
            path: 文件路径
            encoding: 编码
            
        Yields:
            Dict[str, Any]: 数据项
        """
        with open(path, 'r', encoding=encoding) as f:
            data = json.load(f)
        
        if isinstance(data, list):
            yield from data
        else:
            yield data
    
    def _stream_txt(self, path: Path, encoding: str) -> Iterator[Dict[str, Any]]:
        """流式加载文本文件。
        
        Args:
            path: 文件路径
            encoding: 编码
            
        Yields:
            Dict[str, Any]]: 数据项，字段名为'text'
        """
        with open(path, 'r', encoding=encoding) as f:
            for line in f:
                line = line.strip()
                if line:
                    yield {'text': line}
    
    def _stream_csv(self, path: Path, encoding: str) -> Iterator[Dict[str, Any]]:
        """流式加载CSV文件。
        
        Args:
            path: 文件路径
            encoding: 编码
            
        Yields:
            Dict[str, Any]: 数据项
        """
        import csv
        
        with open(path, 'r', encoding=encoding, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                yield dict(row)
    
    def _apply_filters(self, item: Dict[str, Any]) -> bool:
        """应用所有过滤器。
        
        Args:
            item: 数据项
            
        Returns:
            bool: 是否通过所有过滤器
        """
        return all(f(item) for f in self._filters)
    
    def _apply_transforms(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """应用所有转换函数。
        
        Args:
            item: 数据项
            
        Returns:
            Dict[str, Any]: 转换后的数据项
        """
        result = item
        for transform in self._transforms:
            result = transform(result)
        return result
    
    def validate_data(
        self, 
        data: List[Dict[str, Any]], 
        rules: List[ValidationRule]
    ) -> tuple:
        """验证数据集。
        
        Args:
            data: 数据列表
            rules: 验证规则列表
            
        Returns:
            tuple: (valid_items, invalid_items) 有效和无效数据
        """
        valid_items = []
        invalid_items = []
        
        for item in data:
            is_valid = True
            errors = []
            
            for rule in rules:
                if not self._validate_item(item, rule):
                    is_valid = False
                    errors.append(f"{rule.field}: {rule.message or '验证失败'}")
            
            if is_valid:
                valid_items.append(item)
            else:
                invalid_items.append({'item': item, 'errors': errors})
        
        return valid_items, invalid_items
    
    def _validate_item(self, item: Dict[str, Any], rule: ValidationRule) -> bool:
        """验证单个数据项。
        
        Args:
            item: 数据项
            rule: 验证规则
            
        Returns:
            bool: 是否通过验证
        """
        if rule.rule_type == ValidationType.REQUIRED:
            return rule.field in item and item[rule.field] is not None
        
        if rule.field not in item:
            return True
        
        value = item[rule.field]
        
        if rule.rule_type == ValidationType.RANGE and rule.value:
            min_val, max_val = rule.value.get('min'), rule.value.get('max')
            if min_val is not None and value < min_val:
                return False
            if max_val is not None and value > max_val:
                return False
        
        elif rule.rule_type == ValidationType.REGEX and rule.value:
            pattern = re.compile(rule.value)
            if isinstance(value, str):
                return bool(pattern.match(value))
        
        return True


class StreamingDataLoader:
    """流式数据加载器。
    
    专为大规模数据集设计，支持批量加载和预处理。
    
    示例：
        >>> loader = StreamingDataLoader(batch_size=1000)
        >>> for batch in loader.iter_batches("data/large.jsonl"):
        ...     process_batch(batch)
    """
    
    def __init__(
        self, 
        batch_size: int = 100,
        buffer_size: int = 1000,
    ):
        """初始化流式加载器。
        
        Args:
            batch_size: 批处理大小
            buffer_size: 缓冲区大小
        """
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self._base_loader = DatasetLoader()
    
    def iter_batches(
        self, 
        path: Union[str, Path],
        format: Optional[DataFormat] = None,
    ) -> Iterator[List[Dict[str, Any]]]:
        """迭代生成批次数据。
        
        Args:
            path: 数据文件路径
            format: 数据格式
            
        Yields:
            List[Dict[str, Any]]: 数据批次
        """
        buffer = []
        
        for item in self._base_loader.stream_load(path, format):
            buffer.append(item)
            
            if len(buffer) >= self.batch_size:
                yield buffer
                buffer = []
        
        if buffer:
            yield buffer
    
    def iter_samples(
        self, 
        path: Union[str, Path],
        format: Optional[DataFormat] = None,
    ) -> Iterator[Dict[str, Any]]:
        """迭代生成单个样本。
        
        Args:
            path: 数据文件路径
            format: 数据格式
            
        Yields:
            Dict[str, Any]: 数据样本
        """
        yield from self._base_loader.stream_load(path, format)


class DataMixer:
    """数据混合器。
    
    支持多个数据集的加权混合，用于平衡训练数据。
    
    示例：
        >>> mixer = DataMixer(seed=42)
        >>> mixer.add_dataset("data/poetry.jsonl", weight=0.6)
        >>> mixer.add_dataset("data/classical.jsonl", weight=0.4)
        >>> 
        >>> # 获取混合后的数据迭代器
        >>> for item in mixer.iter_mixed():
        ...     process(item)
    """
    
    def __init__(
        self, 
        seed: int = 42,
        shuffle: bool = True,
        replacement: bool = True,
    ):
        """初始化数据混合器。
        
        Args:
            seed: 随机种子
            shuffle: 是否打乱
            replacement: 是否采用有放回抽样
        """
        self.datasets: List[tuple] = []
        self.total_weight = 0
        self.seed = seed
        self.shuffle = shuffle
        self.replacement = replacement
        self.rng = np.random.default_rng(seed)
        self._loader = DatasetLoader()
    
    def add_dataset(
        self, 
        path: Union[str, Path],
        weight: float = 1.0,
        format: Optional[DataFormat] = None,
        name: Optional[str] = None,
    ) -> "DataMixer":
        """添加数据集到混合器。
        
        Args:
            path: 数据文件路径
            weight: 采样权重
            format: 数据格式
            name: 数据集名称（可选）
            
        Returns:
            DataMixer: 返回自身以支持链式调用
        """
        if weight <= 0:
            raise ValueError("权重必须为正数")
        
        path = str(path)
        self.datasets.append({
            'path': path,
            'weight': weight,
            'format': format,
            'name': name or Path(path).stem,
        })
        self.total_weight += weight
        
        return self
    
    def get_weights(self) -> List[float]:
        """获取归一化的权重列表。
        
        Returns:
            List[float]: 归一化后的权重
        """
        return [d['weight'] / self.total_weight for d in self.datasets]
    
    def iter_mixed(
        self, 
        max_samples: Optional[int] = None,
    ) -> Iterator[Dict[str, Any]]:
        """迭代生成混合后的数据。
        
        Args:
            max_samples: 最大样本数（可选）
            
        Yields:
            Dict[str, Any]: 混合后的数据样本
        """
        weights = self.get_weights()
        dataset_loaders = []
        
        for ds_config in self.datasets:
            loader = DatasetLoader()
            data = loader.load(ds_config['path'], ds_config['format'])
            dataset_loaders.append(iter(data))
        
        sample_count = 0
        
        while True:
            if max_samples and sample_count >= max_samples:
                break
            
            try:
                idx = self.rng.choice(len(self.datasets), p=weights)
                sample = next(dataset_loaders[idx])
                
                if self.shuffle:
                    sample['_source_dataset'] = self.datasets[idx]['name']
                
                yield sample
                sample_count += 1
                
            except StopIteration:
                if not self.replacement:
                    break
                dataset_loaders[idx] = iter(self.datasets[idx]['path'])
    
    def iter_balanced_batches(
        self, 
        batch_size: int = 32,
    ) -> Iterator[List[Dict[str, Any]]]:
        """迭代生成平衡批次。
        
        每个批次尽可能包含来自各数据集的平衡样本。
        
        Args:
            batch_size: 批次大小
            
        Yields:
            List[Dict[str, Any]]: 平衡批次
        """
        weights = self.get_weights()
        dataset_loaders = []
        dataset_iterators = []
        
        for ds_config in self.datasets:
            data = self._loader.load(ds_config['path'], ds_config['format'])
            if self.shuffle:
                indices = self.rng.permutation(len(data))
                data = [data[i] for i in indices]
            dataset_iterators.append(iter(data))
        
        batch = []
        
        while True:
            if len(batch) >= batch_size:
                yield batch
                batch = []
            
            added_to_batch = False
            
            for i, ds_idx in enumerate(self.rng.choice(len(self.datasets), size=len(self.datasets), p=weights)):
                try:
                    sample = next(dataset_iterators[ds_idx])
                    if self.shuffle:
                        sample['_source_dataset'] = self.datasets[ds_idx]['name']
                    batch.append(sample)
                    added_to_batch = True
                    
                    if len(batch) >= batch_size:
                        break
                        
                except StopIteration:
                    if not self.replacement:
                        continue
                    data = self._loader.load(self.datasets[ds_idx]['path'], self.datasets[ds_idx]['format'])
                    if self.shuffle:
                        indices = self.rng.permutation(len(data))
                        data = [data[i] for i in indices]
                    dataset_iterators[ds_idx] = iter(data)
            
            if not added_to_batch and not batch:
                break
            
            if not added_to_batch and len(batch) < batch_size:
                yield batch
                break
    
    def load_all_mixed(self) -> List[Dict[str, Any]]:
        """加载所有混合数据到内存。
        
        Returns:
            List[Dict[str, Any]]: 混合后的完整数据集
        """
        return list(self.iter_mixed())
    
    def get_stats(self) -> Dict[str, Any]:
        """获取混合器统计信息。
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        stats = {
            'num_datasets': len(self.datasets),
            'total_weight': self.total_weight,
            'weights': self.get_weights(),
            'shuffle': self.shuffle,
            'replacement': self.replacement,
            'datasets': [],
        }
        
        for ds in self.datasets:
            try:
                count = len(self._loader.load(ds['path'], ds['format']))
            except:
                count = 0
            
            stats['datasets'].append({
                'name': ds['name'],
                'path': ds['path'],
                'weight': ds['weight'],
                'samples': count,
            })
        
        return stats
