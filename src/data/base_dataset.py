"""Base dataset interface for the Lingmao Moyun data pipeline."""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except Exception:
    torch = None
    Dataset = object

from src.logger import get_logger

logger = get_logger("LingmaoMoyun.Data.Base")


class BaseDataset(ABC, Dataset):
    """Abstract base class for all dataset implementations.
    
    Provides common functionality for data loading, formatting, and streaming.
    All dataset classes should inherit from this class.
    
    Args:
        data_paths: Single path or list of paths to data files/directories.
        tokenizer: Tokenizer instance for encoding text.
        context_length: Maximum sequence length in tokens.
        streaming: If True, yield samples on-demand instead of loading all.
    """
    
    SUPPORTED_FORMATS = {'.jsonl', '.json', '.txt'}
    
    def __init__(
        self,
        data_paths: Union[str, Path, List[Union[str, Path]]],
        tokenizer: Optional[Any] = None,
        context_length: int = 512,
        streaming: bool = False,
    ) -> None:
        self.data_paths = self._normalize_paths(data_paths)
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.streaming = streaming
        self._samples: List[Dict[str, Any]] = []
        
        if not self.streaming:
            self._load_all_data()
    
    @staticmethod
    def _normalize_paths(
        data_paths: Union[str, Path, List[Union[str, Path]]]
    ) -> List[Path]:
        """Normalize input paths to a list of Path objects."""
        if isinstance(data_paths, (str, Path)):
            data_paths = [data_paths]
        return [Path(p) for p in data_paths]
    
    @abstractmethod
    def _load_single_file(self, path: Path) -> List[Dict[str, Any]]:
        """Load and parse a single data file.
        
        Args:
            path: Path to the data file.
            
        Returns:
            List of parsed data items.
        """
        pass
    
    def _load_all_data(self) -> None:
        """Load all data from provided paths."""
        all_items = []
        for data_path in self.data_paths:
            if not data_path.exists():
                logger.warning(f"Data path does not exist: {data_path}")
                continue
            
            if data_path.is_dir():
                files = []
                for ext in self.SUPPORTED_FORMATS:
                    files.extend(data_path.glob(f"**/*{ext}"))
                logger.info(f"Loading {len(files)} files from directory: {data_path}")
            else:
                files = [data_path]
            
            for file_path in files:
                try:
                    items = self._load_single_file(file_path)
                    all_items.extend(items)
                    logger.info(f"Loaded {len(items)} items from {file_path}")
                except Exception as e:
                    logger.error(f"Failed to load {file_path}: {e}")
        
        self._process_items(all_items)
    
    def _process_items(self, items: List[Dict[str, Any]]) -> None:
        """Process loaded items into training samples.
        
        Args:
            items: Raw data items to process.
        """
        self._samples = []
        for item in items:
            processed = self._process_single_item(item)
            if processed is not None:
                self._samples.extend(processed if isinstance(processed, list) else [processed])
        
        logger.info(f"Created {len(self._samples)} samples from {len(items)} items")
    
    def _process_single_item(self, item: Dict[str, Any]) -> Optional[Union[Dict[str, Any], List[Dict[str, Any]]]]:
        """Process a single data item into training sample(s).
        
        Args:
            item: Raw data item.
            
        Returns:
            Processed sample or list of samples.
        """
        raise NotImplementedError("Subclasses must implement _process_single_item")
    
    def _tokenize_text(self, text: str) -> List[int]:
        """Tokenize text using the configured tokenizer.
        
        Args:
            text: Text to tokenize.
            
        Returns:
            List of token IDs.
        """
        if self.tokenizer is None:
            return [ord(c) % 30000 for c in text]
        
        if hasattr(self.tokenizer, 'encode'):
            return self.tokenizer.encode(text, add_special_tokens=True)
        elif hasattr(self.tokenizer, 'tokenize'):
            tokens = self.tokenizer.tokenize(text)
            return tokens if isinstance(tokens, list) else list(tokens)
        else:
            logger.warning("Tokenizer has no recognized method, using char-level")
            return [ord(c) % 30000 for c in text]
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self._samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single training sample by index.
        
        Args:
            idx: Sample index.
            
        Returns:
            Dictionary containing input_ids and other fields.
        """
        if idx < 0 or idx >= len(self._samples):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self._samples)}")
        return self._samples[idx]
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over samples in the dataset."""
        for i in range(len(self)):
            yield self[i]
    
    def get_stream_iterator(self, batch_size: int = 1) -> Iterator[List[Dict[str, Any]]]:
        """Get a streaming iterator over the dataset.
        
        Args:
            batch_size: Number of samples to yield per iteration.
            
        Yields:
            Batches of samples.
        """
        for i in range(0, len(self), batch_size):
            batch = []
            for j in range(i, min(i + batch_size, len(self))):
                batch.append(self[j])
            yield batch


class StreamingDataset(BaseDataset):
    """Memory-efficient streaming dataset for large-scale data.
    
    Does not load all data into memory at once, suitable for very large datasets.
    """
    
    def __init__(
        self,
        data_paths: Union[str, Path, List[Union[str, Path]]],
        tokenizer: Optional[Any] = None,
        context_length: int = 512,
        prefetch_buffer: int = 100,
    ) -> None:
        self.prefetch_buffer = prefetch_buffer
        self._file_iterators: Dict[Path, Iterator] = {}
        self._current_path_index = 0
        
        super().__init__(
            data_paths=data_paths,
            tokenizer=tokenizer,
            context_length=context_length,
            streaming=True,
        )
    
    def _load_single_file(self, path: Path) -> List[Dict[str, Any]]:
        """Streaming doesn't use this method."""
        return []
    
    def _process_single_item(self, item: Dict[str, Any]) -> Optional[Union[Dict[str, Any], List[Dict[str, Any]]]]:
        """Process streaming item - subclasses should override."""
        text = None
        for field in ['text', 'content', 'body']:
            if field in item and isinstance(item[field], str):
                text = item[field]
                break
        
        if text is None:
            return None
        
        tokens = self._tokenize_text(text)
        if len(tokens) < 2:
            return None
        
        if len(tokens) > self.context_length:
            tokens = tokens[:self.context_length]
        
        return {
            'input_ids': tokens,
            'attention_mask': [1] * len(tokens),
        }
    
    def _init_file_iterator(self, path: Path) -> Iterator[Dict[str, Any]]:
        """Initialize an iterator for a single file.
        
        Args:
            path: Path to the file.
            
        Returns:
            Iterator over file lines/items.
        """
        suffix = path.suffix.lower()
        
        if suffix == '.jsonl':
            return self._stream_jsonl(path)
        elif suffix == '.json':
            return self._stream_json(path)
        elif suffix == '.txt':
            return self._stream_txt(path)
        else:
            logger.warning(f"Unsupported file format: {suffix}")
            return iter([])
    
    def _stream_jsonl(self, path: Path) -> Iterator[Dict[str, Any]]:
        """Stream JSONL file line by line.
        
        Args:
            path: Path to JSONL file.
            
        Yields:
            Parsed JSON objects.
        """
        with open(path, 'r', encoding='utf-8', errors='strict') as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(f"Skipping malformed JSON at line {line_idx + 1} in {path}")
    
    def _stream_json(self, path: Path) -> Iterator[Dict[str, Any]]:
        """Stream JSON file (expects array format).
        
        Args:
            path: Path to JSON file.
            
        Yields:
            Items from JSON array.
        """
        with open(path, 'r', encoding='utf-8', errors='strict') as f:
            try:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        yield item
                else:
                    yield data
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON file: {path}")
    
    def _stream_txt(self, path: Path) -> Iterator[Dict[str, Any]]:
        """Stream text file line by line.
        
        Args:
            path: Path to text file.
            
        Yields:
            Dictionary with text content.
        """
        with open(path, 'r', encoding='utf-8', errors='strict') as f:
            for line in f:
                line = line.strip()
                if line:
                    yield {'text': line}
    
    def _ensure_file_open(self) -> None:
        """Ensure a file iterator is open for the current path."""
        if self._current_path_index >= len(self.data_paths):
            self._current_path_index = 0
        
        current_path = self.data_paths[self._current_path_index]
        
        if current_path not in self._file_iterators:
            self._file_iterators[current_path] = self._init_file_iterator(current_path)
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over samples in streaming mode."""
        while True:
            self._ensure_file_open()
            current_path = self.data_paths[self._current_path_index]
            
            try:
                item = next(self._file_iterators[current_path])
                processed = self._process_single_item(item)
                if processed is not None:
                    samples = processed if isinstance(processed, list) else [processed]
                    for sample in samples:
                        yield sample
            except StopIteration:
                self._file_iterators.pop(current_path, None)
                self._current_path_index += 1
                
                if self._current_path_index >= len(self.data_paths):
                    break
    
    def __len__(self) -> int:
        """Streaming dataset doesn't have a fixed length."""
        return 0


class WeightedMixingDataset(BaseDataset):
    """Dataset that mixes multiple datasets with configurable weights.
    
    Supports weighted sampling from multiple data sources for balanced training.
    
    Args:
        datasets: List of (dataset_instance, weight) tuples.
        replacement: If True, use sampling with replacement.
        seed: Random seed for reproducibility.
    """
    
    def __init__(
        self,
        datasets: List[tuple],
        replacement: bool = True,
        seed: int = 42,
    ) -> None:
        self.datasets = []
        self.weights = []
        total_weight = 0
        
        for ds, weight in datasets:
            if not isinstance(ds, BaseDataset):
                raise TypeError(f"Expected BaseDataset instance, got {type(ds)}")
            self.datasets.append(ds)
            self.weights.append(weight)
            total_weight += weight
        
        if total_weight <= 0:
            raise ValueError("Total weight must be positive")
        
        self.weights = [w / total_weight for w in self.weights]
        self.replacement = replacement
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
        self._samples = []
        self._build_weighted_samples()
    
    def _load_single_file(self, path: Path) -> List[Dict[str, Any]]:
        """WeightedMixingDataset doesn't load files directly."""
        return []
    
    def _process_single_item(self, item: Dict[str, Any]) -> Optional[Union[Dict[str, Any], List[Dict[str, Any]]]]:
        """WeightedMixingDataset doesn't process items directly."""
        return None
    
    def _build_weighted_samples(self) -> None:
        """Build index mapping for weighted sampling."""
        cumulative_indices = []
        
        for ds_idx, (ds, weight) in enumerate(zip(self.datasets, self.weights)):
            ds_indices = [(ds_idx, i) for i in range(len(ds))]
            cumulative_indices.extend(ds_indices)
        
        if not cumulative_indices:
            logger.warning("No samples available in weighted mixing")
            return
        
        total_samples = len(cumulative_indices)
        sample_weights = []
        for ds_idx, weight in enumerate(self.weights):
            num_ds_samples = len(self.datasets[ds_idx])
            sample_weights.extend([weight] * num_ds_samples)
        
        if sum(sample_weights) > 0:
            sample_weights = [w / sum(sample_weights) for w in sample_weights]
        
        if self.replacement:
            indices = self.rng.choice(
                total_samples,
                size=total_samples,
                replace=True,
                p=sample_weights if sum(sample_weights) > 0 else None,
            )
            self._samples = [cumulative_indices[i] for i in indices]
        else:
            self._samples = cumulative_indices
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get sample by mixed index."""
        ds_idx, sample_idx = self._samples[idx % len(self._samples)]
        return self.datasets[ds_idx][sample_idx]
    
    def __len__(self) -> int:
        """Return total number of samples."""
        return len(self._samples)
