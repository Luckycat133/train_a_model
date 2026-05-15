"""格式转换模块。

提供数据格式转换功能，支持：
- JSONL 格式
- JSON 格式
- CSV 格式
- TXT 格式

示例：
    >>> formatter = DataFormatter(input_format='json', output_format='jsonl')
    >>> formatter.convert('input.json', 'output.jsonl')
    >>> # 批量转换
    >>> formatter.batch_convert('input_dir/', 'output_dir/')
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Iterator
from dataclasses import dataclass

from src.logger import get_logger

logger = get_logger("LingmaoMoyun.Data.Formatter")


@dataclass
class FormatConfig:
    """格式转换配置类。"""

    input_encoding: str = "utf-8"
    output_encoding: str = "utf-8"
    jsonl_separator: str = "\n"
    csv_delimiter: str = ","
    csv_quotechar: str = '"'
    add_id_field: bool = False
    id_field_name: str = "id"
    text_field: str = "text"
    include_metadata: bool = True


class DataFormatter:
    """数据格式转换器。

    支持多种数据格式之间的相互转换。

    Attributes:
        input_format: 输入格式。
        output_format: 输出格式。
        config: 格式配置对象。

    Example:
        >>> formatter = DataFormatter(input_format='json', output_format='jsonl')
        >>> formatter.convert('input.json', 'output.jsonl')
    """

    SUPPORTED_FORMATS = {'jsonl', 'json', 'csv', 'txt'}
    FORMAT_EXTENSIONS = {
        'jsonl': '.jsonl',
        'json': '.json',
        'csv': '.csv',
        'txt': '.txt',
    }

    def __init__(
        self,
        input_format: str = "jsonl",
        output_format: str = "jsonl",
        config: Optional[FormatConfig] = None,
    ):
        """初始化数据格式转换器。

        Args:
            input_format: 输入格式。
            output_format: 输出格式。
            config: 格式配置对象。
        """
        if input_format not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported input format: {input_format}")
        if output_format not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported output format: {output_format}")

        self.input_format = input_format
        self.output_format = output_format
        self.config = config or FormatConfig()

    def convert(self, input_path: Union[str, Path], output_path: Union[str, Path]) -> int:
        """转换单个文件。

        Args:
            input_path: 输入文件路径。
            output_path: 输出文件路径。

        Returns:
            转换的记录数量。

        Example:
            >>> formatter = DataFormatter(input_format='json', output_format='jsonl')
            >>> formatter.convert('input.json', 'output.jsonl')
            100
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        records = list(self._read(input_path))
        count = self._write(records, output_path)

        logger.info(f"Converted {count} records from {input_path} to {output_path}")
        return count

    def batch_convert(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        file_pattern: str = "*",
    ) -> Dict[str, int]:
        """批量转换目录中的文件。

        Args:
            input_dir: 输入目录路径。
            output_dir: 输出目录路径。
            file_pattern: 文件匹配模式。

        Returns:
            包含转换统计信息的字典。

        Example:
            >>> formatter = DataFormatter(input_format='json', output_format='jsonl')
            >>> stats = formatter.batch_convert('input/', 'output/')
            >>> print(stats)
            {'files_processed': 10, 'total_records': 1000}
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        input_ext = self.FORMAT_EXTENSIONS.get(self.input_format, f".{self.input_format}")
        input_files = list(input_dir.glob(f"**/*{input_ext}"))

        total_records = 0
        files_processed = 0

        for input_file in input_files:
            relative_path = input_file.relative_to(input_dir)
            output_ext = self.FORMAT_EXTENSIONS.get(self.output_format, f".{self.output_format}")
            output_file = output_dir / relative_path.with_suffix(output_ext)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            try:
                count = self.convert(input_file, output_file)
                total_records += count
                files_processed += 1
            except Exception as e:
                logger.error(f"Failed to convert {input_file}: {e}")

        logger.info(f"Batch conversion complete: {files_processed} files, {total_records} records")
        return {
            'files_processed': files_processed,
            'total_records': total_records,
        }

    def _read(self, path: Path) -> Iterator[Dict[str, Any]]:
        """根据格式读取数据。"""
        if self.input_format == "jsonl":
            yield from self._read_jsonl(path)
        elif self.input_format == "json":
            yield from self._read_json(path)
        elif self.input_format == "csv":
            yield from self._read_csv(path)
        elif self.input_format == "txt":
            yield from self._read_txt(path)

    def _write(self, records: List[Dict[str, Any]], path: Path) -> int:
        """根据格式写入数据。"""
        if self.output_format == "jsonl":
            return self._write_jsonl(records, path)
        elif self.output_format == "json":
            return self._write_json(records, path)
        elif self.output_format == "csv":
            return self._write_csv(records, path)
        elif self.output_format == "txt":
            return self._write_txt(records, path)
        return 0

    def _read_jsonl(self, path: Path) -> Iterator[Dict[str, Any]]:
        """读取JSONL文件。"""
        with open(path, 'r', encoding=self.config.input_encoding, errors='replace') as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    if self.config.add_id_field and self.config.id_field_name not in record:
                        record[self.config.id_field_name] = f"{path.stem}_{line_idx}"
                    yield record
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping malformed JSON at line {line_idx + 1}: {e}")

    def _read_json(self, path: Path) -> Iterator[Dict[str, Any]]:
        """读取JSON文件。"""
        with open(path, 'r', encoding=self.config.input_encoding, errors='replace') as f:
            try:
                data = json.load(f)
                if isinstance(data, list):
                    for idx, record in enumerate(data):
                        if self.config.add_id_field and self.config.id_field_name not in record:
                            record[self.config.id_field_name] = f"{path.stem}_{idx}"
                        yield record
                elif isinstance(data, dict):
                    if self.config.add_id_field and self.config.id_field_name not in data:
                        data[self.config.id_field_name] = path.stem
                    yield data
                else:
                    yield {'content': str(data), self.config.id_field_name: path.stem}
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON file {path}: {e}")

    def _read_csv(self, path: Path) -> Iterator[Dict[str, Any]]:
        """读取CSV文件。"""
        with open(path, 'r', encoding=self.config.input_encoding, errors='replace', newline='') as f:
            reader = csv.DictReader(
                f,
                delimiter=self.config.csv_delimiter,
                quotechar=self.config.csv_quotechar,
            )
            for idx, row in enumerate(reader):
                if self.config.add_id_field and self.config.id_field_name not in row:
                    row[self.config.id_field_name] = f"{path.stem}_{idx}"
                yield dict(row)

    def _read_txt(self, path: Path) -> Iterator[Dict[str, Any]]:
        """读取TXT文件。"""
        with open(path, 'r', encoding=self.config.input_encoding, errors='replace') as f:
            for line_idx, line in enumerate(f):
                line = line.rstrip('\n\r')
                if not line.strip():
                    continue
                record = {
                    self.config.text_field: line,
                }
                if self.config.add_id_field:
                    record[self.config.id_field_name] = f"{path.stem}_{line_idx}"
                if self.config.include_metadata:
                    record['source'] = str(path)
                    record['line_num'] = line_idx + 1
                yield record

    def _write_jsonl(self, records: List[Dict[str, Any]], path: Path) -> int:
        """写入JSONL文件。"""
        with open(path, 'w', encoding=self.config.output_encoding) as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + self.config.jsonl_separator)
        return len(records)

    def _write_json(self, records: List[Dict[str, Any]], path: Path) -> int:
        """写入JSON文件。"""
        with open(path, 'w', encoding=self.config.output_encoding) as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        return len(records)

    def _write_csv(self, records: List[Dict[str, Any]], path: Path) -> int:
        """写入CSV文件。"""
        if not records:
            return 0

        fieldnames = list(records[0].keys())
        with open(path, 'w', encoding=self.config.output_encoding, newline='') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
                delimiter=self.config.csv_delimiter,
                quotechar=self.config.csv_quotechar,
                quoting=csv.QUOTE_MINIMAL,
            )
            writer.writeheader()
            writer.writerows(records)
        return len(records)

    def _write_txt(self, records: List[Dict[str, Any]], path: Path) -> int:
        """写入TXT文件。"""
        with open(path, 'w', encoding=self.config.output_encoding) as f:
            for record in records:
                text = record.get(self.config.text_field, str(record))
                f.write(text + '\n')
        return len(records)

    def stream_convert(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        batch_size: int = 1000,
    ) -> int:
        """流式转换大文件。

        Args:
            input_path: 输入文件路径。
            output_path: 输出文件路径。
            batch_size: 批处理大小。

        Returns:
            转换的记录数量。
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        total_count = 0
        batch = []

        with open(output_path, 'w', encoding=self.config.output_encoding) as out_f:
            for record in self._read(input_path):
                batch.append(record)
                if len(batch) >= batch_size:
                    self._write_batch(batch, out_f)
                    total_count += len(batch)
                    batch = []

            if batch:
                self._write_batch(batch, out_f)
                total_count += len(batch)

        logger.info(f"Stream converted {total_count} records")
        return total_count

    def _write_batch(self, batch: List[Dict[str, Any]], out_f) -> None:
        """写入一批记录。"""
        for record in batch:
            if self.output_format == "jsonl":
                out_f.write(json.dumps(record, ensure_ascii=False) + self.config.jsonl_separator)
            elif self.output_format == "txt":
                text = record.get(self.config.text_field, str(record))
                out_f.write(text + '\n')


class FormatDetector:
    """格式自动检测器。"""

    @staticmethod
    def detect(path: Union[str, Path]) -> Optional[str]:
        """自动检测文件格式。

        Args:
            path: 文件路径。

        Returns:
            检测到的格式名称，如果无法检测返回None。

        Example:
            >>> format = FormatDetector.detect('data.jsonl')
            >>> print(format)
            jsonl
        """
        path = Path(path)
        suffix = path.suffix.lower()

        format_map = {
            '.jsonl': 'jsonl',
            '.json': 'json',
            '.csv': 'csv',
            '.txt': 'txt',
        }

        return format_map.get(suffix)

    @staticmethod
    def detect_and_convert(
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        target_format: str = "jsonl",
    ) -> int:
        """自动检测格式并转换。

        Args:
            input_path: 输入文件路径。
            output_path: 输出文件路径。
            target_format: 目标格式。

        Returns:
            转换的记录数量。
        """
        detected_format = FormatDetector.detect(input_path)
        if detected_format is None:
            raise ValueError(f"Cannot detect format for file: {input_path}")

        formatter = DataFormatter(input_format=detected_format, output_format=target_format)
        return formatter.convert(input_path, output_path)
