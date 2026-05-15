"""Pretraining dataset implementation for Lingmao Moyun.

Handles large-scale pretraining data with support for various formats
and efficient data loading strategies.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

try:
    import torch
except Exception:
    torch = None

from src.data.base_dataset import BaseDataset, StreamingDataset
from src.logger import get_logger

logger = get_logger("LingmaoMoyun.Data.Pretrain")


class PretrainDataset(BaseDataset):
    """Dataset for pretraining with causal language modeling objective.
    
    Supports various data formats and provides efficient preprocessing
    for large-scale pretraining.
    
    Args:
        data_paths: Single path or list of paths to data files/directories.
        tokenizer: Tokenizer instance for encoding text.
        context_length: Maximum sequence length in tokens.
        stride: Sliding window stride for creating samples.
        min_length: Minimum text length to include (filter short texts).
        max_length: Maximum text length to include (truncate long texts).
        streaming: If True, use streaming mode for large datasets.
        add_special_tokens: Whether to add special tokens during tokenization.
    """
    
    def __init__(
        self,
        data_paths: Union[str, Path, List[Union[str, Path]]],
        tokenizer: Optional[Any] = None,
        context_length: int = 512,
        stride: int = 256,
        min_length: int = 10,
        max_length: int = 100000,
        streaming: bool = False,
        add_special_tokens: bool = True,
    ) -> None:
        self.stride = stride
        self.min_length = min_length
        self.max_length = max_length
        self.add_special_tokens = add_special_tokens
        
        super().__init__(
            data_paths=data_paths,
            tokenizer=tokenizer,
            context_length=context_length,
            streaming=streaming,
        )
    
    def _load_single_file(self, path: Path) -> List[Dict[str, Any]]:
        """Load a single pretraining data file.
        
        Args:
            path: Path to the data file.
            
        Returns:
            List of parsed data items.
        """
        suffix = path.suffix.lower()
        
        if suffix == '.jsonl':
            return self._load_jsonl(path)
        elif suffix == '.json':
            return self._load_json(path)
        elif suffix == '.txt':
            return self._load_txt(path)
        else:
            logger.warning(f"Unsupported file format: {suffix}")
            return []
    
    def _load_jsonl(self, path: Path) -> List[Dict[str, Any]]:
        """Load JSONL format pretraining data.
        
        Args:
            path: Path to JSONL file.
            
        Returns:
            List of parsed items.
        """
        items = []
        with open(path, 'r', encoding='utf-8', errors='strict') as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    items.append(item)
                except json.JSONDecodeError:
                    logger.warning(f"Skipping malformed JSON at line {line_idx + 1}")
        return items
    
    def _load_json(self, path: Path) -> List[Dict[str, Any]]:
        """Load JSON format pretraining data (expects array).
        
        Args:
            path: Path to JSON file.
            
        Returns:
            List of parsed items.
        """
        with open(path, 'r', encoding='utf-8', errors='strict') as f:
            try:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                else:
                    return [data]
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON file: {path}")
                return []
    
    def _load_txt(self, path: Path) -> List[Dict[str, Any]]:
        """Load plain text file as pretraining data.
        
        Args:
            path: Path to text file.
            
        Returns:
            List of items with text field.
        """
        items = []
        with open(path, 'r', encoding='utf-8', errors='strict') as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append({'text': line})
        return items
    
    def _extract_text(self, item: Dict[str, Any]) -> Optional[str]:
        """Extract text content from a data item.
        
        Args:
            item: Data item dictionary.
            
        Returns:
            Extracted text or None if no valid text found.
        """
        text = None
        
        text_fields = ['text', 'content', 'body', 'paragraphs', 'document']
        for field in text_fields:
            if field in item and isinstance(item[field], str) and item[field].strip():
                text = item[field]
                break
        
        if text is None and 'title' in item:
            text = item.get('title', '')
        
        if text is not None:
            text = text.strip()
            if len(text) < self.min_length or len(text) > self.max_length:
                return None
        
        return text
    
    def _process_single_item(self, item: Dict[str, Any]) -> Optional[Union[Dict[str, Any], List[Dict[str, Any]]]]:
        """Process a single pretraining item into training samples.
        
        Args:
            item: Raw data item.
            
        Returns:
            Processed sample or list of samples.
        """
        text = self._extract_text(item)
        if text is None:
            return None
        
        tokens = self._tokenize_text(text)
        
        if len(tokens) <= self.context_length:
            if len(tokens) < 2:
                return None
            padding_length = self.context_length - len(tokens)
            padded_tokens = tokens + [0] * padding_length
            
            return [{
                'input_ids': padded_tokens,
                'target_ids': tokens + [0] * padding_length,
                'attention_mask': [1] * len(tokens) + [0] * padding_length,
                'labels': tokens + [-100] * padding_length,
            }]
        
        samples = []
        for i in range(0, len(tokens) - self.context_length, self.stride):
            input_tokens = tokens[i:i + self.context_length]
            target_tokens = tokens[i + 1:i + self.context_length + 1]
            
            samples.append({
                'input_ids': input_tokens,
                'target_ids': target_tokens,
                'attention_mask': [1] * self.context_length,
                'labels': target_tokens,
            })
        
        return samples if samples else None
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a pretraining sample by index.
        
        Args:
            idx: Sample index.
            
        Returns:
            Dictionary containing input_ids, target_ids, attention_mask, and labels.
        """
        sample = self._samples[idx]
        
        if torch is not None:
            return {
                'input_ids': torch.tensor(sample['input_ids'], dtype=torch.long),
                'target_ids': torch.tensor(sample['target_ids'], dtype=torch.long),
                'attention_mask': torch.tensor(sample['attention_mask'], dtype=torch.long),
                'labels': torch.tensor(sample['labels'], dtype=torch.long),
            }
        
        return sample


class ConcatPretrainDataset(BaseDataset):
    """Pretraining dataset that concatenates multiple documents.
    
    Concatenates documents with separator tokens for efficient pretraining,
    then splits into fixed-length sequences.
    
    Args:
        data_paths: Single path or list of paths to data files/directories.
        tokenizer: Tokenizer instance for encoding text.
        context_length: Maximum sequence length in tokens.
        separator_id: Token ID to use as document separator.
        doc_separator: String to insert between documents.
        min_doc_length: Minimum document length to include.
    """
    
    def __init__(
        self,
        data_paths: Union[str, Path, List[Union[str, Path]]],
        tokenizer: Optional[Any] = None,
        context_length: int = 512,
        separator_id: int = 2,
        doc_separator: str = " ",
        min_doc_length: int = 10,
    ) -> None:
        self.separator_id = separator_id
        self.doc_separator = doc_separator
        self.min_doc_length = min_doc_length
        
        super().__init__(
            data_paths=data_paths,
            tokenizer=tokenizer,
            context_length=context_length,
            streaming=False,
        )
    
    def _load_single_file(self, path: Path) -> List[Dict[str, Any]]:
        """Load pretraining data file."""
        suffix = path.suffix.lower()
        
        if suffix == '.jsonl':
            items = []
            with open(path, 'r', encoding='utf-8', errors='strict') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            items.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            return items
        elif suffix == '.json':
            with open(path, 'r', encoding='utf-8', errors='strict') as f:
                data = json.load(f)
                return data if isinstance(data, list) else [data]
        elif suffix == '.txt':
            items = []
            with open(path, 'r', encoding='utf-8', errors='strict') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        items.append({'text': line})
            return items
        return []
    
    def _process_single_item(self, item: Dict[str, Any]) -> Optional[Union[Dict[str, Any], List[Dict[str, Any]]]]:
        """Items are not processed individually; see _process_items."""
        return None
    
    def _process_items(self, items: List[Dict[str, Any]]) -> None:
        """Concatenate all documents and create samples.
        
        Args:
            items: Raw data items (documents).
        """
        self._samples = []
        
        documents = []
        for item in items:
            text = self._extract_text(item)
            if text:
                tokens = self._tokenize_text(text)
                if len(tokens) >= self.min_doc_length:
                    documents.append(tokens)
        
        if not documents:
            logger.warning("No valid documents found")
            return
        
        all_tokens = []
        for i, doc_tokens in enumerate(documents):
            if i > 0:
                all_tokens.append(self.separator_id)
            all_tokens.extend(doc_tokens)
        
        logger.info(f"Concatenated {len(documents)} documents into {len(all_tokens)} tokens")
        
        for i in range(0, len(all_tokens) - self.context_length, self.context_length):
            input_tokens = all_tokens[i:i + self.context_length]
            target_tokens = all_tokens[i + 1:i + self.context_length + 1]
            
            self._samples.append({
                'input_ids': input_tokens,
                'target_ids': target_tokens,
                'attention_mask': [1] * self.context_length,
                'labels': target_tokens,
            })
        
        logger.info(f"Created {len(self._samples)} concatenated samples")
    
    def _extract_text(self, item: Dict[str, Any]) -> Optional[str]:
        """Extract text content from item."""
        for field in ['text', 'content', 'body']:
            if field in item and isinstance(item[field], str):
                return item[field].strip()
        if 'title' in item:
            return str(item.get('title', '')).strip()
        return None
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a concatenated sample by index."""
        sample = self._samples[idx]
        
        if torch is not None:
            return {
                'input_ids': torch.tensor(sample['input_ids'], dtype=torch.long),
                'target_ids': torch.tensor(sample['target_ids'], dtype=torch.long),
                'attention_mask': torch.tensor(sample['attention_mask'], dtype=torch.long),
                'labels': torch.tensor(sample['labels'], dtype=torch.long),
            }
        
        return sample


class PretrainDatasetFactory:
    """Factory for creating pretraining datasets.
    
    Provides convenient methods to create different types of pretraining datasets.
    """
    
    @staticmethod
    def create_standard(
        data_paths: Union[str, Path, List[Union[str, Path]]],
        tokenizer: Optional[Any] = None,
        context_length: int = 512,
        stride: int = 256,
        **kwargs,
    ) -> PretrainDataset:
        """Create a standard sliding-window pretraining dataset.
        
        Args:
            data_paths: Data file paths.
            tokenizer: Tokenizer instance.
            context_length: Sequence length.
            stride: Sliding window stride.
            **kwargs: Additional arguments for PretrainDataset.
            
        Returns:
            PretrainDataset instance.
        """
        return PretrainDataset(
            data_paths=data_paths,
            tokenizer=tokenizer,
            context_length=context_length,
            stride=stride,
            **kwargs,
        )
    
    @staticmethod
    def create_concat(
        data_paths: Union[str, Path, List[Union[str, Path]]],
        tokenizer: Optional[Any] = None,
        context_length: int = 512,
        **kwargs,
    ) -> ConcatPretrainDataset:
        """Create a concatenated pretraining dataset.
        
        Args:
            data_paths: Data file paths.
            tokenizer: Tokenizer instance.
            context_length: Sequence length.
            **kwargs: Additional arguments for ConcatPretrainDataset.
            
        Returns:
            ConcatPretrainDataset instance.
        """
        return ConcatPretrainDataset(
            data_paths=data_paths,
            tokenizer=tokenizer,
            context_length=context_length,
            **kwargs,
        )
    
    @staticmethod
    def create_streaming(
        data_paths: Union[str, Path, List[Union[str, Path]]],
        tokenizer: Optional[Any] = None,
        context_length: int = 512,
        **kwargs,
    ) -> StreamingDataset:
        """Create a streaming pretraining dataset for large-scale data.
        
        Args:
            data_paths: Data file paths.
            tokenizer: Tokenizer instance.
            context_length: Sequence length.
            **kwargs: Additional arguments for StreamingDataset.
            
        Returns:
            StreamingDataset instance.
        """
        dataset = StreamingDataset(
            data_paths=data_paths,
            tokenizer=tokenizer,
            context_length=context_length,
            **kwargs,
        )
        return dataset
