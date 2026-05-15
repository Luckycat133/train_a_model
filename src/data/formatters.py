"""Lingmao Moyun 数据格式化模块。

提供数据格式转换功能，支持JSONL、句子分割和对话格式转换。
适用于将各种来源的数据转换为模型训练所需的统一格式。

示例：
    >>> from src.data.formatters import JSONLFormatter, SentenceFormatter, DialogueFormatter
    >>>
    >>> # JSONL格式转换
    >>> formatter = JSONLFormatter()
    >>> formatter.convert(input_file, output_file)
    >>>
    >>> # 句子级别处理
    >>> sformatter = SentenceFormatter()
    >>> sentences = sformatter.format("第一句。第二句。第三句。")
"""

import json
import re
import csv
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union
from dataclasses import dataclass, asdict

from src.logger import get_logger

logger = get_logger("LingmaoMoyun.Formatter")


@dataclass
class FormatStats:
    """格式化统计信息。"""
    input_count: int = 0
    output_count: int = 0
    skipped_count: int = 0
    error_count: int = 0

    def to_dict(self) -> Dict[str, int]:
        """转换为字典格式。"""
        return {
            "input": self.input_count,
            "output": self.output_count,
            "skipped": self.skipped_count,
            "errors": self.error_count,
            "conversion_rate": (
                f"{self.output_count / self.input_count * 100:.2f}%"
                if self.input_count > 0 else "0%"
            ),
        }


class BaseFormatter:
    """格式化器基类。

    定义所有格式化器必须实现的接口。

    Args:
        encoding: 文件编码格式，默认utf-8。
        skip_invalid: 是否跳过无效数据。
    """

    def __init__(
        self,
        encoding: str = "utf-8",
        skip_invalid: bool = True,
    ):
        self.encoding = encoding
        self.skip_invalid = skip_invalid
        self.stats = FormatStats()

    def reset_stats(self):
        """重置统计信息。"""
        self.stats = FormatStats()

    @staticmethod
    def _read_jsonl(file_path: Path) -> Iterator[Dict[str, Any]]:
        """读取JSONL文件。

        Args:
            file_path: 文件路径。

        Yields:
            解析后的JSON对象。
        """
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(f"无法解析JSON行: {line[:50]}...")
                    continue

    @staticmethod
    def _read_json(file_path: Path) -> List[Dict[str, Any]]:
        """读取JSON文件。

        Args:
            file_path: 文件路径。

        Returns:
            JSON对象列表。
        """
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                return [data]
            else:
                return []

    @staticmethod
    def _read_text(file_path: Path) -> Iterator[Dict[str, Any]]:
        """读取纯文本文件。

        Args:
            file_path: 文件路径。

        Yields:
            包含文本内容的字典。
        """
        with open(file_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if line:
                    yield {"text": line, "line_id": idx}


class JSONLFormatter(BaseFormatter):
    """JSONL格式转换器。

    支持将以下格式转换为JSONL：
    - JSON数组
    - CSV文件
    - 纯文本文件
    - 多种JSON变体格式

    示例：
        >>> formatter = JSONLFormatter()
        >>>
        >>> # JSON数组转JSONL
        >>> formatter.convert_json_to_jsonl("data.json", "data.jsonl")
        >>>
        >>> # CSV转JSONL
        >>> formatter.convert_csv_to_jsonl("data.csv", "data.jsonl")
        >>>
        >>> # 纯文本转JSONL
        >>> formatter.convert_text_to_jsonl("data.txt", "data.jsonl")
    """

    def __init__(
        self,
        encoding: str = "utf-8",
        skip_invalid: bool = True,
        text_field: str = "text",
    ):
        super().__init__(encoding=encoding, skip_invalid=skip_invalid)
        self.text_field = text_field

    def convert(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        input_format: Optional[str] = None,
    ) -> FormatStats:
        """自动检测格式并转换。

        Args:
            input_path: 输入文件路径。
            output_path: 输出文件路径。
            input_format: 强制指定输入格式。

        Returns:
            格式化统计信息。
        """
        self.reset_stats()
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if input_format is None:
            input_format = self._detect_format(input_path)

        logger.info(f"开始转换：{input_path} -> {output_path} (格式: {input_format})")

        if input_format == "json":
            self.convert_json_to_jsonl(input_path, output_path)
        elif input_format == "jsonl":
            self.convert_jsonl_to_jsonl(input_path, output_path)
        elif input_format == "csv":
            self.convert_csv_to_jsonl(input_path, output_path)
        elif input_format == "text":
            self.convert_text_to_jsonl(input_path, output_path)
        else:
            logger.error(f"不支持的格式: {input_format}")
            raise ValueError(f"不支持的格式: {input_format}")

        logger.info(f"转换完成：{self.stats.to_dict()}")
        return self.stats

    def _detect_format(self, file_path: Path) -> str:
        """自动检测文件格式。

        Args:
            file_path: 文件路径。

        Returns:
            格式类型字符串。
        """
        suffix = file_path.suffix.lower()

        if suffix == ".json":
            with open(file_path, "r", encoding=self.encoding) as f:
                first_char = f.read(1)
                if first_char == "[":
                    return "json"
                elif first_char == "{":
                    try:
                        data = json.load(f)
                        if isinstance(data, list):
                            return "json"
                        for item in data:
                            if isinstance(item, dict):
                                return "jsonl"
                        return "json"
                    except:
                        pass
                elif first_char == "{":
                    return "jsonl"

        elif suffix == ".jsonl":
            return "jsonl"

        elif suffix == ".csv":
            return "csv"

        elif suffix in (".txt", ".text"):
            return "text"

        return "unknown"

    def convert_json_to_jsonl(
        self,
        input_path: Path,
        output_path: Path,
    ) -> FormatStats:
        """将JSON数组转换为JSONL。

        Args:
            input_path: 输入JSON文件路径。
            output_path: 输出JSONL文件路径。

        Returns:
            格式化统计信息。
        """
        data = self._read_json(input_path)

        with open(output_path, "w", encoding=self.encoding) as f:
            for item in data:
                self.stats.input_count += 1
                if self._validate_item(item):
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    self.stats.output_count += 1
                else:
                    self.stats.skipped_count += 1

        return self.stats

    def convert_jsonl_to_jsonl(
        self,
        input_path: Path,
        output_path: Path,
        transform: Optional[callable] = None,
    ) -> FormatStats:
        """复制或转换JSONL文件。

        Args:
            input_path: 输入JSONL文件路径。
            output_path: 输出JSONL文件路径。
            transform: 可选的数据转换函数。

        Returns:
            格式化统计信息。
        """
        with open(output_path, "w", encoding=self.encoding) as f:
            for item in self._read_jsonl(input_path):
                self.stats.input_count += 1

                if transform:
                    try:
                        item = transform(item)
                    except Exception as e:
                        logger.warning(f"转换失败: {e}")
                        self.stats.error_count += 1
                        continue

                if self._validate_item(item):
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    self.stats.output_count += 1
                else:
                    self.stats.skipped_count += 1

        return self.stats

    def convert_csv_to_jsonl(
        self,
        input_path: Path,
        output_path: Path,
        text_columns: Optional[List[str]] = None,
        delimiter: str = ",",
    ) -> FormatStats:
        """将CSV转换为JSONL。

        Args:
            input_path: 输入CSV文件路径。
            output_path: 输出JSONL文件路径。
            text_columns: 用作文本字段的列名列表。
            delimiter: CSV分隔符。

        Returns:
            格式化统计信息。
        """
        with open(input_path, "r", encoding=self.encoding, newline="") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            fieldnames = reader.fieldnames or []

            with open(output_path, "w", encoding=self.encoding) as out_f:
                for row in reader:
                    self.stats.input_count += 1

                    if text_columns:
                        row["text"] = " ".join(
                            row.get(col, "") for col in text_columns if col in row
                        )

                    if self._validate_item(row):
                        out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        self.stats.output_count += 1
                    else:
                        self.stats.skipped_count += 1

        return self.stats

    def convert_text_to_jsonl(
        self,
        input_path: Path,
        output_path: Path,
        add_line_numbers: bool = False,
    ) -> FormatStats:
        """将纯文本转换为JSONL。

        Args:
            input_path: 输入文本文件路径。
            output_path: 输出JSONL文件路径。
            add_line_numbers: 是否添加行号。

        Returns:
            格式化统计信息。
        """
        with open(output_path, "w", encoding=self.encoding) as f:
            for item in self._read_text(input_path):
                self.stats.input_count += 1

                if self._validate_item(item):
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    self.stats.output_count += 1
                else:
                    self.stats.skipped_count += 1

        return self.stats

    def _validate_item(self, item: Dict[str, Any]) -> bool:
        """验证数据项是否有效。

        Args:
            item: 数据项。

        Returns:
            是否有效。
        """
        if not isinstance(item, dict):
            return False

        if self.text_field not in item:
            return True

        text = item.get(self.text_field, "")
        return isinstance(text, str) and len(text.strip()) > 0


class SentenceFormatter(BaseFormatter):
    """句子级别格式化器。

    将文本分割成句子，支持中文和英文。
    可添加句子边界标记和元数据。

    示例：
        >>> formatter = SentenceFormatter()
        >>> sentences = formatter.format("第一句。第二句。第三句。")
        >>> print(sentences)
        ['第一句。', '第二句。', '第三句。']
    """

    CHINESE_PUNCTUATION = "。！？；"
    ENGLISH_PUNCTUATION = ".!?;"

    def __init__(
        self,
        encoding: str = "utf-8",
        skip_invalid: bool = True,
        keep_punctuation: bool = True,
        add_metadata: bool = False,
    ):
        super().__init__(encoding=encoding, skip_invalid=skip_invalid)
        self.keep_punctuation = keep_punctuation
        self.add_metadata = add_metadata

        self.chinese_pattern = re.compile(
            r"[^\n。！？；]+[。！？；]?",
            flags=re.UNICODE,
        )

        self.english_pattern = re.compile(
            r"[^\n.!?;]+[.!?;]?",
            flags=re.UNICODE,
        )

    def format(self, text: str) -> List[str]:
        """将文本分割成句子。

        Args:
            text: 输入文本。

        Returns:
            句子列表。
        """
        if not text or not isinstance(text, str):
            return []

        sentences = []

        chinese_parts = self._split_by_language(text)

        for part, lang in chinese_parts:
            if lang == "chinese":
                sentences.extend(self._split_chinese(part))
            else:
                sentences.extend(self._split_english(part))

        sentences = [s.strip() for s in sentences if s.strip()]

        if self.keep_punctuation:
            sentences = [self._ensure_punctuation(s) for s in sentences]

        return sentences

    def _split_by_language(self, text: str) -> List[tuple]:
        """按语言分割文本。

        Args:
            text: 输入文本。

        Returns:
            (文本片段, 语言) 元组列表。
        """
        chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
        chinese_ratio = len(chinese_chars) / len(text) if text else 0

        if chinese_ratio > 0.3:
            return [(text, "chinese")]
        elif chinese_ratio < 0.1:
            return [(text, "english")]
        else:
            return [(text, "mixed")]

    def _split_chinese(self, text: str) -> List[str]:
        """分割中文句子。

        Args:
            text: 中文文本。

        Returns:
            句子列表。
        """
        sentences = []
        current = ""

        for char in text:
            current += char
            if char in self.CHINESE_PUNCTUATION:
                sentences.append(current)
                current = ""

        if current.strip():
            sentences.append(current)

        return sentences

    def _split_english(self, text: str) -> List[str]:
        """分割英文句子。

        Args:
            text: 英文文本。

        Returns:
            句子列表。
        """
        sentences = self.english_pattern.findall(text)
        return [s.strip() for s in sentences if s.strip()]

    def _ensure_punctuation(self, sentence: str) -> str:
        """确保句子以标点符号结尾。

        Args:
            sentence: 输入句子。

        Returns:
            处理后的句子。
        """
        sentence = sentence.strip()
        if not sentence:
            return sentence

        last_char = sentence[-1]

        if last_char not in self.CHINESE_PUNCTUATION and last_char not in ".!?":
            return sentence + "。"

        return sentence

    def format_file(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        text_key: str = "text",
        output_key: str = "sentences",
    ) -> FormatStats:
        """格式化文件中的文本。

        Args:
            input_path: 输入文件路径。
            output_path: 输出文件路径。
            text_key: 输入文本字段名。
            output_key: 输出句子字段名。

        Returns:
            格式化统计信息。
        """
        self.reset_stats()
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"开始句子级别格式化：{input_path} -> {output_path}")

        with open(output_path, "w", encoding=self.encoding) as out_f:
            for item in self._read_jsonl(input_path):
                self.stats.input_count += 1

                text = item.get(text_key, "")
                sentences = self.format(text)

                if sentences:
                    if self.add_metadata:
                        item[output_key] = sentences
                        item["sentence_count"] = len(sentences)
                    else:
                        for sent in sentences:
                            out_f.write(
                                json.dumps(
                                    {"text": sent, "source_id": item.get("id", "")},
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                            self.stats.output_count += 1
                else:
                    self.stats.skipped_count += 1

        return self.stats

    def format_batch(
        self,
        texts: List[str],
        add_indices: bool = False,
    ) -> List[Dict[str, Any]]:
        """批量格式化文本。

        Args:
            texts: 文本列表。
            add_indices: 是否添加句子索引。

        Returns:
            格式化结果列表。
        """
        results = []

        for idx, text in enumerate(texts):
            sentences = self.format(text)

            if add_indices:
                results.append(
                    {
                        "index": idx,
                        "sentences": sentences,
                        "count": len(sentences),
                    }
                )
            else:
                results.extend(
                    {"text": sent, "source_index": idx} for sent in sentences
                )

        return results


class DialogueFormatter(BaseFormatter):
    """对话格式转换器。

    支持多种对话格式的相互转换：
    - ShareGPT格式
    - OpenAI格式
    - 简单问答格式
    - 自定义格式

    示例：
        >>> formatter = DialogueFormatter()
        >>>
        >>> # 转换为ShareGPT格式
        >>> formatter.to_sharegpt(data, output_path)
        >>>
        >>> # 转换为训练格式
        >>> formatter.to_training_format(data, system_prompt="你是一个助手")
    """

    def __init__(
        self,
        encoding: str = "utf-8",
        skip_invalid: bool = True,
        default_role: str = "user",
    ):
        super().__init__(encoding=encoding, skip_invalid=skip_invalid)
        self.default_role = default_role

    def convert(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        input_format: str,
        output_format: str = "jsonl",
    ) -> FormatStats:
        """转换对话格式。

        Args:
            input_path: 输入文件路径。
            output_path: 输出文件路径。
            input_format: 输入格式（sharegpt/openai/qa）。
            output_format: 输出格式（sharegpt/openai/training/jsonl）。

        Returns:
            格式化统计信息。
        """
        self.reset_stats()
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"开始对话格式转换：{input_path} ({input_format}) -> {output_path} ({output_format})"
        )

        data = self._load_dialogue_data(input_path, input_format)

        with open(output_path, "w", encoding=self.encoding) as f:
            for item in data:
                self.stats.input_count += 1

                if output_format == "sharegpt":
                    converted = self.to_sharegpt([item])
                elif output_format == "openai":
                    converted = self.to_openai([item])
                elif output_format == "training":
                    converted = self.to_training_format([item])
                else:
                    converted = [item]

                if converted:
                    for conv_item in converted:
                        f.write(json.dumps(conv_item, ensure_ascii=False) + "\n")
                        self.stats.output_count += 1
                else:
                    self.stats.skipped_count += 1

        return self.stats

    def _load_dialogue_data(
        self,
        file_path: Path,
        format_type: str,
    ) -> List[Dict[str, Any]]:
        """加载对话数据。

        Args:
            file_path: 文件路径。
            format_type: 格式类型。

        Returns:
            对话数据列表。
        """
        if format_type == "sharegpt":
            return self._read_json(file_path)
        elif format_type == "openai":
            return self._read_jsonl(file_path)
        elif format_type == "qa":
            return self._read_jsonl(file_path)
        else:
            return list(self._read_jsonl(file_path))

    def to_sharegpt(
        self,
        data: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """转换为ShareGPT格式。

        ShareGPT格式：
        {
            "conversations": [
                {"from": "human", "value": "..."},
                {"from": "gpt", "value": "..."}
            ]
        }

        Args:
            data: 输入数据。
            system_prompt: 系统提示。

        Returns:
            ShareGPT格式数据。
        """
        results = []

        for item in data:
            if "conversations" in item:
                results.append(item)
                continue

            conversations = []

            if system_prompt:
                conversations.append({"from": "system", "value": system_prompt})

            if "messages" in item:
                for msg in item["messages"]:
                    role = msg.get("role", self.default_role)
                    from_map = {
                        "system": "system",
                        "user": "human",
                        "assistant": "gpt",
                        "human": "human",
                        "bot": "gpt",
                    }
                    conversations.append(
                        {
                            "from": from_map.get(role, role),
                            "value": msg.get("content", ""),
                        }
                    )
            elif "dialogue" in item:
                for turn in item["dialogue"]:
                    conversations.append(
                        {
                            "from": turn.get("speaker", self.default_role),
                            "value": turn.get("text", ""),
                        }
                    )

            if conversations:
                result = {"conversations": conversations}
                if "id" in item:
                    result["id"] = item["id"]
                results.append(result)

        return results

    def to_openai(
        self,
        data: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """转换为OpenAI格式。

        OpenAI格式：
        {
            "messages": [
                {"role": "system", "content": "..."},
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."}
            ]
        }

        Args:
            data: 输入数据。
            system_prompt: 系统提示。

        Returns:
            OpenAI格式数据。
        """
        results = []

        for item in data:
            if "messages" in item:
                results.append(item)
                continue

            messages = []

            if system_prompt or "system" in str(item):
                sys_content = system_prompt or ""
                if "system" in item:
                    sys_content = item["system"]
                if sys_content:
                    messages.append({"role": "system", "content": sys_content})

            if "conversations" in item:
                for conv in item["conversations"]:
                    from_val = conv.get("from", "")
                    role_map = {
                        "human": "user",
                        "gpt": "assistant",
                        "assistant": "assistant",
                        "user": "user",
                        "system": "system",
                    }
                    messages.append(
                        {
                            "role": role_map.get(from_val, self.default_role),
                            "content": conv.get("value", ""),
                        }
                    )
            elif "dialogue" in item:
                for turn in item["dialogue"]:
                    messages.append(
                        {
                            "role": turn.get("speaker", self.default_role),
                            "content": turn.get("text", ""),
                        }
                    )

            if messages:
                result = {"messages": messages}
                if "id" in item:
                    result["id"] = item["id"]
                results.append(result)

        return results

    def to_training_format(
        self,
        data: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        format_type: str = "plain",
    ) -> List[Dict[str, Any]]:
        """转换为训练格式。

        将对话转换为模型训练所需的格式。

        Args:
            data: 输入数据。
            system_prompt: 系统提示。
            format_type: 训练格式类型（plain/prompt_response）。

        Returns:
            训练格式数据。
        """
        results = []

        openai_data = self.to_openai(data, system_prompt)

        for item in openai_data:
            messages = item.get("messages", [])

            if format_type == "plain":
                text = self._messages_to_plain_text(messages)
                results.append({"text": text})

            elif format_type == "prompt_response":
                for i in range(1, len(messages)):
                    if messages[i]["role"] == "assistant":
                        prompt = self._messages_to_plain_text(messages[:i])
                        response = messages[i]["content"]
                        results.append(
                            {
                                "prompt": prompt,
                                "response": response,
                            }
                        )

            elif format_type == "full":
                full_text = self._messages_to_plain_text(messages)
                results.append(
                    {
                        "text": full_text,
                        "messages": messages,
                    }
                )

        return results

    def _messages_to_plain_text(self, messages: List[Dict[str, str]]) -> str:
        """将消息列表转换为纯文本。

        Args:
            messages: 消息列表。

        Returns:
            纯文本格式。
        """
        parts = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            role_prefix = {
                "system": "系统：",
                "user": "用户：",
                "assistant": "助手：",
            }

            prefix = role_prefix.get(role, f"{role}：")
            parts.append(f"{prefix}{content}")

        return "\n\n".join(parts)

    def merge_conversations(
        self,
        conversations: List[Dict[str, Any]],
        delimiter: str = "\n\n",
    ) -> str:
        """合并多轮对话为单一文本。

        Args:
            conversations: 对话列表。
            delimiter: 分隔符。

        Returns:
            合并后的文本。
        """
        parts = []

        for conv in conversations:
            from_val = conv.get("from", "")
            value = conv.get("value", "")

            speaker_map = {
                "human": "用户",
                "gpt": "助手",
                "user": "用户",
                "assistant": "助手",
            }

            speaker = speaker_map.get(from_val, from_val)
            parts.append(f"{speaker}：{value}")

        return delimiter.join(parts)


def create_formatter(
    formatter_type: str,
    **kwargs,
) -> BaseFormatter:
    """工厂函数：创建格式化器实例。

    Args:
        formatter_type: 格式化器类型（jsonl/sentence/dialogue）。
        **kwargs: 传递给格式化器的参数。

    Returns:
        格式化器实例。
    """
    formatter_type = formatter_type.lower()

    if formatter_type == "jsonl":
        return JSONLFormatter(**kwargs)
    elif formatter_type == "sentence":
        return SentenceFormatter(**kwargs)
    elif formatter_type == "dialogue":
        return DialogueFormatter(**kwargs)
    else:
        raise ValueError(f"不支持的格式化器类型: {formatter_type}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Lingmao Moyun 数据格式化工具")
    parser.add_argument(
        "--type",
        type=str,
        default="jsonl",
        choices=["jsonl", "sentence", "dialogue"],
        help="格式化器类型",
    )
    parser.add_argument("--input", type=str, required=True, help="输入文件")
    parser.add_argument("--output", type=str, required=True, help="输出文件")
    parser.add_argument("--input-format", type=str, help="输入格式")
    parser.add_argument("--output-format", type=str, default="jsonl", help="输出格式")

    args = parser.parse_args()

    formatter = create_formatter(args.type)

    if args.type == "jsonl":
        formatter.convert(
            args.input,
            args.output,
            input_format=args.input_format,
        )
    elif args.type == "dialogue":
        formatter.convert(
            args.input,
            args.output,
            input_format=args.input_format or "sharegpt",
            output_format=args.output_format,
        )
    else:
        formatter.format_file(args.input, args.output)

    stats = formatter.stats.to_dict()
    print(f"格式化完成：{json.dumps(stats, indent=2, ensure_ascii=False)}")
