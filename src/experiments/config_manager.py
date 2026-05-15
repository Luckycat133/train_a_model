"""配置管理器 - 灵猫墨韵实验追踪系统

提供超参数管理、配置验证、配置保存和加载功能。
支持与 WandB 和 TensorBoard 集成。
"""

import json
import os
import copy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, field, asdict


@dataclass
class HyperparameterConfig:
    """超参数配置数据类。"""

    model: Dict[str, Any] = field(default_factory=dict)
    training: Dict[str, Any] = field(default_factory=dict)
    data: Dict[str, Any] = field(default_factory=dict)
    optimizer: Dict[str, Any] = field(default_factory=dict)
    scheduler: Dict[str, Any] = field(default_factory=dict)
    augmentation: Dict[str, Any] = field(default_factory=dict)
    custom: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HyperparameterConfig":
        """从字典创建配置。"""
        return cls(
            model=data.get("model", {}),
            training=data.get("training", {}),
            data=data.get("data", {}),
            optimizer=data.get("optimizer", {}),
            scheduler=data.get("scheduler", {}),
            augmentation=data.get("augmentation", {}),
            custom=data.get("custom", {}),
        )

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值，支持嵌套键访问。

        Args:
            key: 配置键，支持点分隔的路径 (如 'model.d_model')
            default: 默认值

        Returns:
            配置值或默认值
        """
        parts = key.split(".")
        current = self.to_dict()

        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default

        return current

    def set(self, key: str, value: Any) -> None:
        """设置配置值，支持嵌套键访问。

        Args:
            key: 配置键，支持点分隔的路径
            value: 配置值
        """
        parts = key.split(".")
        current = self.to_dict()

        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        current[parts[-1]] = value

        if parts[0] == "model":
            self.model = current
        elif parts[0] == "training":
            self.training = current
        elif parts[0] == "data":
            self.data = current
        elif parts[0] == "optimizer":
            self.optimizer = current
        elif parts[0] == "scheduler":
            self.scheduler = current
        elif parts[0] == "augmentation":
            self.augmentation = current
        elif parts[0] == "custom":
            self.custom = current

    def flatten(self) -> Dict[str, Any]:
        """将嵌套配置展平为单层字典。"""
        result = {}

        def flatten_dict(d: Dict[str, Any], prefix: str = "") -> None:
            for key, value in d.items():
                new_key = f"{prefix}.{key}" if prefix else key
                if isinstance(value, dict):
                    flatten_dict(value, new_key)
                else:
                    result[new_key] = value

        flatten_dict(self.to_dict())
        return result

    def merge(self, other: "HyperparameterConfig") -> "HyperparameterConfig":
        """合并另一个配置，other 的值优先。"""
        def deep_merge(base: Dict, update: Dict) -> Dict:
            result = copy.deepcopy(base)
            for key, value in update.items():
                if isinstance(value, dict) and key in result and isinstance(result[key], dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = copy.deepcopy(value)
            return result

        base_dict = self.to_dict()
        other_dict = other.to_dict()
        merged = deep_merge(base_dict, other_dict)
        return HyperparameterConfig.from_dict(merged)


class ConfigValidator:
    """配置验证器。"""

    def __init__(self):
        """初始化验证器。"""
        self.rules: Dict[str, Callable[[Any], bool]] = {}
        self.errors: List[str] = []

    def add_rule(self, key: str, validator: Callable[[Any], bool],
                 error_message: str) -> None:
        """添加验证规则。

        Args:
            key: 配置键
            validator: 验证函数
            error_message: 验证失败时的错误消息
        """
        self.rules[key] = (validator, error_message)

    def validate(self, config: HyperparameterConfig) -> bool:
        """验证配置。

        Args:
            config: 要验证的配置

        Returns:
            验证是否通过
        """
        self.errors = []
        flat_config = config.flatten()

        for key, (validator, error_message) in self.rules.items():
            if key in flat_config:
                if not validator(flat_config[key]):
                    self.errors.append(f"{key}: {error_message}")

        return len(self.errors) == 0

    def get_errors(self) -> List[str]:
        """获取验证错误列表。"""
        return self.errors


class ConfigManager:
    """配置管理器。"""

    DEFAULT_CONFIGS = {
        "model": {
            "d_model": 768,
            "nhead": 12,
            "num_layers": 12,
            "dim_feedforward": 3072,
            "dropout": 0.1,
            "max_len": 1024,
            "vocab_size": 30000,
        },
        "training": {
            "batch_size": 8,
            "learning_rate": 5e-5,
            "epochs": 10,
            "accumulation_steps": 4,
            "max_grad_norm": 1.0,
            "weight_decay": 0.01,
            "checkpoint_every": 1,
        },
        "data": {
            "context_length": 512,
            "stride": 256,
            "max_chunks": None,
        },
        "optimizer": {
            "type": "AdamW",
            "lr": 5e-5,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "fused": True,
        },
        "scheduler": {
            "type": "cosine",
            "warmup_steps": 1000,
            "min_lr_ratio": 0.1,
        },
    }

    def __init__(
        self,
        config_dir: str = "configs",
        default_config_name: str = "default",
    ):
        """初始化配置管理器。

        Args:
            config_dir: 配置保存目录
            default_config_name: 默认配置名称
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.default_config_name = default_config_name
        self.current_config: Optional[HyperparameterConfig] = None
        self.config_history: List[Dict[str, Any]] = []

        self._setup_default_validators()

    def _setup_default_validators(self) -> None:
        """设置默认验证规则。"""
        self.validator = ConfigValidator()

        self.validator.add_rule(
            "model.d_model",
            lambda x: isinstance(x, (int, float)) and x > 0,
            "d_model must be positive number"
        )
        self.validator.add_rule(
            "model.nhead",
            lambda x: isinstance(x, int) and x > 0,
            "nhead must be positive integer"
        )
        self.validator.add_rule(
            "training.batch_size",
            lambda x: isinstance(x, int) and x > 0,
            "batch_size must be positive integer"
        )
        self.validator.add_rule(
            "training.learning_rate",
            lambda x: isinstance(x, (int, float)) and 0 < x < 1,
            "learning_rate must be between 0 and 1"
        )
        self.validator.add_rule(
            "training.epochs",
            lambda x: isinstance(x, int) and x > 0,
            "epochs must be positive integer"
        )
        self.validator.add_rule(
            "model.dropout",
            lambda x: isinstance(x, (int, float)) and 0 <= x <= 1,
            "dropout must be between 0 and 1"
        )

    def create_default_config(self) -> HyperparameterConfig:
        """创建默认配置。"""
        return HyperparameterConfig.from_dict(copy.deepcopy(self.DEFAULT_CONFIGS))

    def load_config(self, name: str) -> Optional[HyperparameterConfig]:
        """加载命名配置。

        Args:
            name: 配置名称

        Returns:
            配置对象或 None
        """
        config_file = self.config_dir / f"{name}.json"

        if not config_file.exists():
            return None

        try:
            with open(config_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                config = HyperparameterConfig.from_dict(data)
                self.current_config = config
                return config
        except Exception:
            return None

    def save_config(
        self,
        name: str,
        config: Optional[HyperparameterConfig] = None,
    ) -> bool:
        """保存配置到文件。

        Args:
            name: 配置名称
            config: 要保存的配置，如果为 None 则保存当前配置

        Returns:
            是否保存成功
        """
        config = config or self.current_config
        if config is None:
            return False

        config_file = self.config_dir / f"{name}.json"

        try:
            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(config.to_dict(), f, indent=2, ensure_ascii=False)

            self.config_history.append({
                "name": name,
                "timestamp": datetime.now().isoformat(),
                "action": "save",
            })
            return True
        except Exception:
            return False

    def create_experiment_config(
        self,
        base_name: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> HyperparameterConfig:
        """创建实验配置。

        Args:
            base_name: 基础配置名称，如果为 None 则使用默认配置
            overrides: 配置覆盖项

        Returns:
            创建的配置对象
        """
        if base_name:
            config = self.load_config(base_name)
            if config is None:
                config = self.create_default_config()
        else:
            config = self.create_default_config()

        if overrides:
            for key, value in overrides.items():
                config.set(key, value)

        self.current_config = config
        return config

    def list_configs(self) -> List[str]:
        """列出所有保存的配置。"""
        return [
            f.stem for f in self.config_dir.glob("*.json")
        ]

    def delete_config(self, name: str) -> bool:
        """删除配置。

        Args:
            name: 配置名称

        Returns:
            是否删除成功
        """
        config_file = self.config_dir / f"{name}.json"

        if not config_file.exists():
            return False

        try:
            config_file.unlink()
            self.config_history.append({
                "name": name,
                "timestamp": datetime.now().isoformat(),
                "action": "delete",
            })
            return True
        except Exception:
            return False

    def export_config(
        self,
        config: HyperparameterConfig,
        export_path: Union[str, Path],
    ) -> bool:
        """导出配置到指定路径。

        Args:
            config: 要导出的配置
            export_path: 导出路径

        Returns:
            是否导出成功
        """
        export_path = Path(export_path)

        try:
            export_path.parent.mkdir(parents=True, exist_ok=True)
            with open(export_path, "w", encoding="utf-8") as f:
                json.dump(config.to_dict(), f, indent=2, ensure_ascii=False)
            return True
        except Exception:
            return False

    def import_config(self, import_path: Union[str, Path]) -> Optional[HyperparameterConfig]:
        """从指定路径导入配置。

        Args:
            import_path: 导入路径

        Returns:
            导入的配置对象或 None
        """
        import_path = Path(import_path)

        if not import_path.exists():
            return None

        try:
            with open(import_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                config = HyperparameterConfig.from_dict(data)
                self.current_config = config
                return config
        except Exception:
            return None

    def validate_config(
        self,
        config: Optional[HyperparameterConfig] = None,
    ) -> bool:
        """验证配置。

        Args:
            config: 要验证的配置，如果为 None 则验证当前配置

        Returns:
            验证是否通过
        """
        config = config or self.current_config
        if config is None:
            return False

        return self.validator.validate(config)

    def get_validation_errors(self) -> List[str]:
        """获取验证错误列表。"""
        return self.validator.get_errors()

    def get_config_summary(self, config: Optional[HyperparameterConfig] = None) -> str:
        """获取配置摘要。

        Args:
            config: 配置对象，如果为 None 则使用当前配置

        Returns:
            配置摘要字符串
        """
        config = config or self.current_config
        if config is None:
            return "No configuration loaded"

        flat = config.flatten()
        lines = ["Configuration Summary:", "=" * 40]

        for category in ["model", "training", "data", "optimizer", "scheduler"]:
            category_keys = [k for k in flat.keys() if k.startswith(f"{category}.")]
            if category_keys:
                lines.append(f"\n{category.upper()}:")
                for key in sorted(category_keys):
                    lines.append(f"  {key}: {flat[key]}")

        return "\n".join(lines)
