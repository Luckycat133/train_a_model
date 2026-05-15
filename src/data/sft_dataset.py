"""SFT（监督微调）数据集模块。

本模块实现了对话数据的加载和预处理功能，支持多种对话格式，
包括 ShareGPT、OpenAI 和自定义格式。提供了灵活的对话模板系统，
支持不同角色和特殊标记的处理。
"""

import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

try:
    import torch
except Exception:
    torch = None

from src.data.base_dataset import BaseDataset
from src.logger import get_logger

logger = get_logger("LingmaoMoyun.Data.SFT")


class DialogueFormat(Enum):
    """支持的对话格式枚举。"""
    SHAREGPT = "sharegpt"
    OPENAI = "openai"
    CUSTOM = "custom"
    ALPACA = "alpaca"
    CONVERSATION = "conversation"


@dataclass
class ChatTemplate:
    """聊天模板配置类。"""
    system_prefix: str = "系统: "
    user_prefix: str = "用户: "
    assistant_prefix: str = "助手: "
    system_token: str = ""
    user_token: str = ""
    assistant_token: str = ""
    eos_token: str = ""

    def format_turn(self, role: str, content: str) -> str:
        """格式化单个对话回合。

        参数：
            role: 角色名称（system、user、assistant）。
            content: 对话内容。

        返回：
            格式化后的对话字符串。
        """
        role_lower = role.lower()

        if role_lower == "system":
            prefix = self.system_prefix
            token = self.system_token
        elif role_lower in ("user", "human"):
            prefix = self.user_prefix
            token = self.user_token
        elif role_lower in ("assistant", "bot", "gpt"):
            prefix = self.assistant_prefix
            token = self.assistant_token
        else:
            prefix = ""
            token = ""

        formatted = f"{prefix}{content}"
        if token:
            formatted = f"{token}{formatted}"

        return formatted

    def format_conversation(self, messages: List[Dict[str, str]]) -> str:
        """将消息列表格式化为完整的对话字符串。

        参数：
            messages: 消息列表，每条消息包含 role 和 content。

        返回：
            格式化后的完整对话字符串。
        """
        turns = []
        for msg in messages:
            turn = self.format_turn(msg.get("role", "user"), msg.get("content", ""))
            turns.append(turn)

        result = "\n".join(turns)
        if self.eos_token:
            result += self.eos_token

        return result


class SFTDataset(BaseDataset):
    """监督微调（SFT）数据集。

    专门用于加载和处理对话数据，支持多种数据格式，
    自动进行对话模板应用和标签处理。

    参数：
        data_paths: 数据文件路径或路径列表。
        tokenizer: 分词器实例。
        context_length: 最大序列长度。
        dialogue_format: 对话数据格式。
        chat_template: 聊天模板配置。
        max_turns: 最大对话轮数。
        drop_incomplete: 是否丢弃不完整的对话。
    """

    DEFAULT_TEMPLATE = ChatTemplate()

    def __init__(
        self,
        data_paths: Union[str, Path, List[Union[str, Path]]],
        tokenizer: Optional[Any] = None,
        context_length: int = 512,
        dialogue_format: str = "sharegpt",
        chat_template: Optional[ChatTemplate] = None,
        max_turns: Optional[int] = None,
        drop_incomplete: bool = False,
    ) -> None:
        self.dialogue_format = DialogueFormat(dialogue_format)
        self.chat_template = chat_template or self.DEFAULT_TEMPLATE
        self.max_turns = max_turns
        self.drop_incomplete = drop_incomplete

        super().__init__(
            data_paths=data_paths,
            tokenizer=tokenizer,
            context_length=context_length,
            streaming=False,
        )

    def _load_single_file(self, path: Path) -> List[Dict[str, Any]]:
        """加载单个对话数据文件。

        参数：
            path: 数据文件路径。

        返回：
            解析后的数据项列表。
        """
        suffix = path.suffix.lower()

        if suffix == ".jsonl":
            return self._load_jsonl(path)
        elif suffix == ".json":
            return self._load_json(path)
        else:
            logger.warning(f"SFT数据集不支持的文件格式：{suffix}")
            return []

    def _load_jsonl(self, path: Path) -> List[Dict[str, Any]]:
        """加载 JSONL 格式的对话数据。

        参数：
            path: JSONL 文件路径。

        返回：
            数据项列表。
        """
        items = []
        with open(path, "r", encoding="utf-8", errors="strict") as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    items.append(item)
                except json.JSONDecodeError:
                    logger.warning(f"跳过第 {line_idx + 1} 行的格式错误JSON")
        return items

    def _load_json(self, path: Path) -> List[Dict[str, Any]]:
        """加载 JSON 格式的对话数据。

        参数：
            path: JSON 文件路径。

        返回：
            数据项列表。
        """
        with open(path, "r", encoding="utf-8", errors="strict") as f:
            try:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                else:
                    return [data]
            except json.JSONDecodeError:
                logger.warning(f"无法解析JSON文件：{path}")
                return []

    def _normalize_conversation(self, item: Dict[str, Any]) -> List[Dict[str, str]]:
        """根据配置格式标准化对话数据。

        参数：
            item: 原始数据项。

        返回：
            标准化后的消息列表。
        """
        if self.dialogue_format == DialogueFormat.SHAREGPT:
            return self._normalize_sharegpt(item)
        elif self.dialogue_format == DialogueFormat.OPENAI:
            return self._normalize_openai(item)
        elif self.dialogue_format == DialogueFormat.ALPACA:
            return self._normalize_alpaca(item)
        elif self.dialogue_format == DialogueFormat.CONVERSATION:
            return self._normalize_conversation_format(item)
        else:
            return self._normalize_custom(item)

    def _normalize_sharegpt(self, item: Dict[str, Any]) -> List[Dict[str, str]]:
        """标准化 ShareGPT 格式数据。

        参数：
            item: 原始数据项。

        返回：
            消息列表。
        """
        if "conversations" in item:
            return item["conversations"]

        if "messages" in item:
            messages = []
            for msg in item["messages"]:
                role = msg.get("role", "user")
                if role == "human":
                    role = "user"
                elif role == "gpt":
                    role = "assistant"
                messages.append({
                    "role": role,
                    "content": msg.get("content", "")
                })
            return messages

        return []

    def _normalize_openai(self, item: Dict[str, Any]) -> List[Dict[str, str]]:
        """标准化 OpenAI 格式数据。

        参数：
            item: 原始数据项。

        返回：
            消息列表。
        """
        if "messages" in item:
            messages = []
            for msg in item["messages"]:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                messages.append({
                    "role": role,
                    "content": content
                })
            return messages

        return []

    def _normalize_alpaca(self, item: Dict[str, Any]) -> List[Dict[str, str]]:
        """标准化 Alpaca 格式数据。

        参数：
            item: 原始数据项。

        返回：
            消息列表。
        """
        messages = []

        if "system" in item and item["system"]:
            messages.append({
                "role": "system",
                "content": item["system"]
            })

        instruction = item.get("instruction", "")
        input_text = item.get("input", "")

        if input_text:
            user_content = f"{instruction}\n{input_text}"
        else:
            user_content = instruction

        messages.append({
            "role": "user",
            "content": user_content
        })

        if "output" in item and item["output"]:
            messages.append({
                "role": "assistant",
                "content": item["output"]
            })

        return messages

    def _normalize_conversation_format(self, item: Dict[str, Any]) -> List[Dict[str, str]]:
        """标准化通用对话格式。

        参数：
            item: 原始数据项。

        返回：
            消息列表。
        """
        if "conversation" in item:
            return item["conversation"]

        if "dialogue" in item:
            return item["dialogue"]

        if "chat" in item:
            return item["chat"]

        messages = []
        for key in ["user", "assistant", "question", "answer", "response"]:
            if key in item:
                messages.append({
                    "role": "assistant" if key in ["answer", "response"] else "user",
                    "content": item[key]
                })

        return messages

    def _normalize_custom(self, item: Dict[str, Any]) -> List[Dict[str, str]]:
        """标准化自定义格式数据。

        参数：
            item: 原始数据项。

        返回：
            消息列表。
        """
        messages = []

        for role in ["system", "user", "assistant"]:
            if role in item:
                content = item[role]
                if isinstance(content, str):
                    messages.append({
                        "role": role,
                        "content": content
                    })

        if not messages and "text" in item:
            messages.append({
                "role": "user",
                "content": item.get("instruction", item["text"])
            })
            if "output" in item:
                messages.append({
                    "role": "assistant",
                    "content": item["output"]
                })

        return messages

    def _process_single_item(
        self, item: Dict[str, Any]
    ) -> Optional[Union[Dict[str, Any], List[Dict[str, Any]]]]:
        """处理单个对话数据项。

        参数：
            item: 原始对话数据。

        返回：
            处理后的样本或样本列表。
        """
        messages = self._normalize_conversation(item)

        if len(messages) < 2:
            return None

        if self.max_turns is not None and len(messages) > self.max_turns:
            if self.drop_incomplete:
                return None
            messages = messages[:self.max_turns]

        if messages[-1].get("role") != "assistant":
            if self.drop_incomplete:
                return None
            messages = messages[:-1]

        formatted = self.chat_template.format_conversation(messages)
        tokens = self._tokenize_text(formatted)

        if len(tokens) <= 2:
            return None

        prompt_tokens, response_tokens = self._split_prompt_response(messages, tokens)

        if len(response_tokens) == 0:
            return None

        if len(prompt_tokens) + len(response_tokens) > self.context_length:
            tokens = tokens[: self.context_length]
            prompt_len = min(len(prompt_tokens), self.context_length - 1)
            prompt_tokens = tokens[:prompt_len]
            response_tokens = tokens[prompt_len:]

        padding_length = self.context_length - len(prompt_tokens) - len(response_tokens)
        input_ids = prompt_tokens + response_tokens + [0] * max(0, padding_length)
        labels = [-100] * len(prompt_tokens) + response_tokens + [-100] * max(0, padding_length)

        attention_mask = [1] * len(input_ids)

        return [{
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }]

    def _split_prompt_response(
        self,
        messages: List[Dict[str, str]],
        tokens: List[int]
    ) -> tuple[List[int], List[int]]:
        """分离提示和回复部分的token。

        参数：
            messages: 消息列表。
            tokens: 完整对话的token列表。

        返回：
            (提示token列表, 回复token列表)元组。
        """
        prompt_parts = []
        response_parts = []
        current_role = None

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                formatted = self.chat_template.format_turn(role, content)
                part_tokens = self._tokenize_text(formatted)
                prompt_parts.extend(part_tokens)
                current_role = "system"

            elif role == "user":
                formatted = self.chat_template.format_turn(role, content)
                part_tokens = self._tokenize_text(formatted)
                prompt_parts.extend(part_tokens)
                current_role = "user"

            elif role == "assistant":
                formatted = self.chat_template.format_turn(role, content)
                part_tokens = self._tokenize_text(formatted)
                response_parts.extend(part_tokens)
                current_role = "assistant"

        if len(prompt_parts) == 0:
            return [], tokens

        prompt_len = len(prompt_parts)
        return prompt_parts, tokens[prompt_len:]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """获取单个SFT样本。

        参数：
            idx: 样本索引。

        返回：
            包含input_ids、labels和attention_mask的字典。
        """
        sample = self._samples[idx]

        if torch is not None:
            return {
                "input_ids": torch.tensor(sample["input_ids"], dtype=torch.long),
                "labels": torch.tensor(sample["labels"], dtype=torch.long),
                "attention_mask": torch.tensor(sample["attention_mask"], dtype=torch.long),
            }

        return sample


class SFTTrainingCollator:
    """SFT训练数据整理器。

    用于批量处理SFT数据样本，支持动态padding和标签掩码。
    """

    def __init__(
        self,
        pad_token_id: int = 0,
        label_pad_token_id: int = -100,
        ignore_index: int = -100,
    ) -> None:
        self.pad_token_id = pad_token_id
        self.label_pad_token_id = label_pad_token_id
        self.ignore_index = ignore_index

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """整理批量数据。

        参数：
            batch: 样本列表。

        返回：
            整理后的批量数据字典。
        """
        if not batch:
            return {}

        input_ids = [item["input_ids"] for item in batch]
        labels = [item["labels"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]

        max_length = max(len(ids) for ids in input_ids)

        padded_input_ids = []
        padded_labels = []
        padded_attention_mask = []

        for ids, lbls, mask in zip(input_ids, labels, attention_mask):
            pad_length = max_length - len(ids)

            padded_ids = ids + [self.pad_token_id] * pad_length
            padded_lbls = lbls + [self.label_pad_token_id] * pad_length
            padded_mask = mask + [0] * pad_length

            padded_input_ids.append(padded_ids)
            padded_labels.append(padded_lbls)
            padded_attention_mask.append(padded_mask)

        result = {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long) if torch else padded_input_ids,
            "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long) if torch else padded_attention_mask,
            "labels": torch.tensor(padded_labels, dtype=torch.long) if torch else padded_labels,
        }

        return result


class SFTDatasetFactory:
    """SFT数据集工厂类。

    提供便捷的方法创建不同类型的SFT数据集。
    """

    @staticmethod
    def create_sharegpt(
        data_paths: Union[str, Path, List[Union[str, Path]]],
        tokenizer: Optional[Any] = None,
        context_length: int = 512,
        **kwargs,
    ) -> SFTDataset:
        """创建ShareGPT格式的SFT数据集。

        参数：
            data_paths: 数据文件路径。
            tokenizer: 分词器实例。
            context_length: 序列长度。
            **kwargs: 其他参数。

        返回：
            SFTDataset实例。
        """
        return SFTDataset(
            data_paths=data_paths,
            tokenizer=tokenizer,
            context_length=context_length,
            dialogue_format="sharegpt",
            **kwargs,
        )

    @staticmethod
    def create_alpaca(
        data_paths: Union[str, Path, List[Union[str, Path]]],
        tokenizer: Optional[Any] = None,
        context_length: int = 512,
        **kwargs,
    ) -> SFTDataset:
        """创建Alpaca格式的SFT数据集。

        参数：
            data_paths: 数据文件路径。
            tokenizer: 分词器实例。
            context_length: 序列长度。
            **kwargs: 其他参数。

        返回：
            SFTDataset实例。
        """
        return SFTDataset(
            data_paths=data_paths,
            tokenizer=tokenizer,
            context_length=context_length,
            dialogue_format="alpaca",
            **kwargs,
        )

    @staticmethod
    def create_openai(
        data_paths: Union[str, Path, List[Union[str, Path]]],
        tokenizer: Optional[Any] = None,
        context_length: int = 512,
        **kwargs,
    ) -> SFTDataset:
        """创建OpenAI格式的SFT数据集。

        参数：
            data_paths: 数据文件路径。
            tokenizer: 分词器实例。
            context_length: 序列长度。
            **kwargs: 其他参数。

        返回：
            SFTDataset实例。
        """
        return SFTDataset(
            data_paths=data_paths,
            tokenizer=tokenizer,
            context_length=context_length,
            dialogue_format="openai",
            **kwargs,
        )
