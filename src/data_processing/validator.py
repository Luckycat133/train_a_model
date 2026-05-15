"""数据验证模块。

提供数据验证功能，包括：
- 数据完整性检查
- 格式正确性验证
- 必需字段检查
- 数据类型验证
- 自定义验证规则

示例：
    >>> validator = DataValidator()
    >>> validator.add_required_fields(['text', 'id'])
    >>> if validator.validate(sample):
    ...     # 数据有效
    >>> # 批量验证
    >>> results = validator.validate_batch(samples)
"""

import re
from typing import Dict, List, Any, Optional, Set, Callable, Tuple
from dataclasses import dataclass, field

from src.logger import get_logger

logger = get_logger("LingmaoMoyun.Data.Validator")


@dataclass
class ValidationConfig:
    """数据验证配置类。"""

    min_text_length: int = 1
    max_text_length: int = 1000000
    required_fields: List[str] = field(default_factory=list)
    optional_fields: List[str] = field(default_factory=list)
    allowed_formats: List[str] = field(default_factory=lambda: ['json', 'jsonl'])
    check_encoding: bool = True
    check_duplicates: bool = False
    custom_rules: List[Callable] = field(default_factory=list)


class ValidationResult:
    """验证结果类。"""

    def __init__(self):
        """初始化验证结果。"""
        self.is_valid = True
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.field_errors: Dict[str, List[str]] = {}

    def add_error(self, message: str, field: Optional[str] = None) -> None:
        """添加错误信息。

        Args:
            message: 错误信息。
            field: 关联的字段名。
        """
        self.is_valid = False
        self.errors.append(message)
        if field:
            if field not in self.field_errors:
                self.field_errors[field] = []
            self.field_errors[field].append(message)

    def add_warning(self, message: str) -> None:
        """添加警告信息。

        Args:
            message: 警告信息。
        """
        self.warnings.append(message)

    def merge(self, other: 'ValidationResult') -> None:
        """合并另一个验证结果。

        Args:
            other: 另一个验证结果。
        """
        if not other.is_valid:
            self.is_valid = False
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        for field, errors in other.field_errors.items():
            if field not in self.field_errors:
                self.field_errors[field] = []
            self.field_errors[field].extend(errors)

    def __repr__(self) -> str:
        """返回验证结果的字符串表示。"""
        status = "PASS" if self.is_valid else "FAIL"
        msg = f"ValidationResult({status}"
        if self.errors:
            msg += f", errors={len(self.errors)}"
        if self.warnings:
            msg += f", warnings={len(self.warnings)}"
        msg += ")"
        return msg


class DataValidator:
    """数据验证器。

    验证数据的完整性、格式正确性和自定义规则。

    Attributes:
        config: 验证配置对象。
        custom_validators: 自定义验证器列表。

    Example:
        >>> validator = DataValidator()
        >>> validator.config.required_fields = ['text', 'id']
        >>> validator.config.min_text_length = 10
        >>> if validator.validate(sample):
        ...     print("Valid sample")
        >>> results = validator.validate_batch(samples)
    """

    def __init__(self, config: Optional[ValidationConfig] = None):
        """初始化数据验证器。

        Args:
            config: 验证配置对象。如果为None，使用默认配置。
        """
        self.config = config or ValidationConfig()
        self.custom_validators: List[Callable[[Dict[str, Any]], Tuple[bool, str]]] = []
        self._stats = {
            'total': 0,
            'valid': 0,
            'invalid': 0,
            'warnings': 0,
        }

    def add_required_field(self, field: str) -> None:
        """添加必需字段。

        Args:
            field: 字段名称。
        """
        if field not in self.config.required_fields:
            self.config.required_fields.append(field)

    def add_custom_validator(self, validator_func: Callable[[Dict[str, Any]], Tuple[bool, str]]) -> None:
        """添加自定义验证器。

        Args:
            validator_func: 验证函数，接受样本字典，返回(bool, str)元组。

        Example:
            >>> def check_positive_age(sample):
            ...     if 'age' in sample and sample['age'] < 0:
            ...         return False, "Age must be positive"
            ...     return True, ""
            >>> validator.add_custom_validator(check_positive_age)
        """
        self.custom_validators.append(validator_func)

    def validate(self, sample: Dict[str, Any]) -> bool:
        """验证单个样本。

        Args:
            sample: 待验证的样本字典。

        Returns:
            如果样本通过验证返回True，否则返回False。

        Example:
            >>> validator = DataValidator()
            >>> validator.config.required_fields = ['text']
            >>> sample = {'text': 'Hello, World!', 'id': '1'}
            >>> validator.validate(sample)
            True
        """
        self._stats['total'] += 1
        result = self._validate_sample(sample)

        if result.is_valid:
            self._stats['valid'] += 1
        else:
            self._stats['invalid'] += 1
            for error in result.errors:
                logger.debug(f"Validation error: {error}")

        if result.warnings:
            self._stats['warnings'] += len(result.warnings)

        return result.is_valid

    def validate_with_result(self, sample: Dict[str, Any]) -> ValidationResult:
        """验证单个样本并返回详细结果。

        Args:
            sample: 待验证的样本字典。

        Returns:
            验证结果对象。

        Example:
            >>> validator = DataValidator()
            >>> result = validator.validate_with_result(sample)
            >>> if not result.is_valid:
            ...     for error in result.errors:
            ...         print(f"Error: {error}")
        """
        self._stats['total'] += 1
        result = self._validate_sample(sample)

        if result.is_valid:
            self._stats['valid'] += 1
        else:
            self._stats['invalid'] += 1

        if result.warnings:
            self._stats['warnings'] += len(result.warnings)

        return result

    def _validate_sample(self, sample: Dict[str, Any]) -> ValidationResult:
        """执行样本验证。

        Args:
            sample: 待验证的样本。

        Returns:
            验证结果对象。
        """
        result = ValidationResult()

        if not isinstance(sample, dict):
            result.add_error("Sample must be a dictionary")
            return result

        self._check_required_fields(sample, result)
        self._check_field_types(sample, result)
        self._check_text_length(sample, result)
        self._check_encoding(sample, result)
        self._run_custom_validators(sample, result)

        return result

    def _check_required_fields(self, sample: Dict[str, Any], result: ValidationResult) -> None:
        """检查必需字段。

        Args:
            sample: 样本字典。
            result: 验证结果对象。
        """
        for field in self.config.required_fields:
            if field not in sample:
                result.add_error(f"Missing required field: {field}", field=field)
            elif sample[field] is None or sample[field] == '':
                result.add_error(f"Required field '{field}' is empty", field=field)

    def _check_field_types(self, sample: Dict[str, Any], result: ValidationResult) -> None:
        """检查字段类型。

        Args:
            sample: 样本字典。
            result: 验证结果对象。
        """
        for field, value in sample.items():
            if field in ['text', 'content', 'body']:
                if not isinstance(value, str):
                    result.add_error(f"Field '{field}' must be a string", field=field)
            elif field == 'id':
                if not isinstance(value, (str, int)):
                    result.add_warning(f"Field '{field}' should be string or int")

    def _check_text_length(self, sample: Dict[str, Any], result: ValidationResult) -> None:
        """检查文本长度。

        Args:
            sample: 样本字典。
            result: 验证结果对象。
        """
        text = self._extract_text(sample)
        if text is None:
            return

        text_length = len(text)

        if text_length < self.config.min_text_length:
            result.add_error(
                f"Text length {text_length} is below minimum {self.config.min_text_length}",
                field='text'
            )

        if text_length > self.config.max_text_length:
            result.add_warning(
                f"Text length {text_length} exceeds maximum {self.config.max_text_length}",
            )

    def _check_encoding(self, sample: Dict[str, Any], result: ValidationResult) -> None:
        """检查编码问题。

        Args:
            sample: 样本字典。
            result: 验证结果对象。
        """
        if not self.config.check_encoding:
            return

        text = self._extract_text(sample)
        if text is None:
            return

        try:
            text.encode('utf-8').decode('utf-8')
        except (UnicodeEncodeError, UnicodeDecodeError):
            result.add_error("Text contains encoding errors", field='text')

        if '\ufffd' in text or '\x00' in text:
            result.add_warning("Text may contain invalid Unicode characters")

    def _run_custom_validators(self, sample: Dict[str, Any], result: ValidationResult) -> None:
        """运行自定义验证器。

        Args:
            sample: 样本字典。
            result: 验证结果对象。
        """
        for validator in self.custom_validators:
            try:
                is_valid, message = validator(sample)
                if not is_valid:
                    result.add_error(f"Custom validation failed: {message}")
            except Exception as e:
                result.add_warning(f"Custom validator raised exception: {str(e)}")

    def _extract_text(self, sample: Dict[str, Any]) -> Optional[str]:
        """从样本中提取文本。

        Args:
            sample: 样本字典。

        Returns:
            提取的文本，如果不存在返回None。
        """
        for key in ['text', 'content', 'body']:
            if key in sample and isinstance(sample[key], str):
                return sample[key]
        return None

    def validate_batch(self, samples: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Tuple[Dict[str, Any], ValidationResult]]]:
        """批量验证样本。

        Args:
            samples: 待验证的样本列表。

        Returns:
            包含有效样本列表和无效样本(样本, 结果)列表的元组。

        Example:
            >>> validator = DataValidator()
            >>> valid, invalid = validator.validate_batch(samples)
            >>> print(f"Valid: {len(valid)}, Invalid: {len(invalid)}")
        """
        valid_samples = []
        invalid_samples = []

        for sample in samples:
            result = self.validate_with_result(sample)
            if result.is_valid:
                valid_samples.append(sample)
            else:
                invalid_samples.append((sample, result))

        logger.info(f"Batch validation: {len(valid_samples)} valid, {len(invalid_samples)} invalid")
        return valid_samples, invalid_samples

    def get_stats(self) -> Dict[str, int]:
        """获取验证统计信息。

        Returns:
            包含验证统计信息的字典。

        Example:
            >>> validator = DataValidator()
            >>> # ... validate samples ...
            >>> stats = validator.get_stats()
            >>> print(f"Valid rate: {stats['valid'] / stats['total']:.2%}")
        """
        return self._stats.copy()

    def reset_stats(self) -> None:
        """重置统计信息。"""
        self._stats = {
            'total': 0,
            'valid': 0,
            'invalid': 0,
            'warnings': 0,
        }


class SchemaValidator:
    """JSON Schema 验证器（简化版本）。"""

    def __init__(self, schema: Dict[str, Any]):
        """初始化Schema验证器。

        Args:
            schema: JSON Schema定义。
        """
        self.schema = schema

    def validate(self, data: Dict[str, Any]) -> ValidationResult:
        """验证数据是否符合Schema。

        Args:
            data: 待验证的数据。

        Returns:
            验证结果对象。
        """
        result = ValidationResult()
        self._validate_object(data, self.schema, "", result)
        return result

    def _validate_object(self, data: Any, schema: Dict[str, Any], path: str, result: ValidationResult) -> None:
        """递归验证对象。

        Args:
            data: 待验证的数据。
            schema: Schema定义。
            path: 当前路径。
            result: 验证结果对象。
        """
        if 'type' in schema:
            expected_type = schema['type']
            if not self._check_type(data, expected_type):
                result.add_error(f"Type mismatch at {path}: expected {expected_type}, got {type(data).__name__}")
                return

        if 'required' in schema and isinstance(data, dict):
            for field in schema['required']:
                if field not in data:
                    result.add_error(f"Missing required field '{field}' at {path}")

        if 'properties' in schema and isinstance(data, dict):
            for field, field_schema in schema['properties'].items():
                if field in data:
                    self._validate_object(data[field], field_schema, f"{path}.{field}", result)

    def _check_type(self, data: Any, expected_type: str) -> bool:
        """检查数据类型。"""
        type_map = {
            'string': str,
            'number': (int, float),
            'integer': int,
            'boolean': bool,
            'array': list,
            'object': dict,
            'null': type(None),
        }
        expected = type_map.get(expected_type)
        if expected is None:
            return True
        return isinstance(data, expected)
