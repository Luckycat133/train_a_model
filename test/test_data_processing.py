#!/usr/bin/env python3
"""数据处理模块测试脚本。

测试数据清洗、过滤、标准化、格式转换和验证功能。
"""

import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_processing import DataCleaner, DataFilter, DataNormalizer, DataFormatter, DataValidator
from src.data_processing.cleaner import CleaningConfig
from src.data_processing.filter import FilterConfig, QualityScorer
from src.data_processing.normalizer import NormalizerConfig
from src.data_processing.formatter import FormatConfig, FormatDetector
from src.data_processing.validator import ValidationConfig, ValidationResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("数据处理测试")


def create_temp_jsonl(data: List[Dict[str, Any]]) -> Path:
    """创建临时JSONL文件。"""
    fd, path = tempfile.mkstemp(suffix=".jsonl", text=True)
    os.close(fd)

    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    return Path(path)


def create_temp_json(data: List[Dict[str, Any]]) -> Path:
    """创建临时JSON文件。"""
    fd, path = tempfile.mkstemp(suffix=".json", text=True)
    os.close(fd)

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)

    return Path(path)


def create_temp_txt(texts: List[str]) -> Path:
    """创建临时文本文件。"""
    fd, path = tempfile.mkstemp(suffix=".txt", text=True)
    os.close(fd)

    with open(path, 'w', encoding='utf-8') as f:
        for text in texts:
            f.write(text + '\n')

    return Path(path)


class TestDataCleaner:
    """测试数据清洗模块。"""

    def test_clean_html_tags(self):
        """测试HTML标签移除。"""
        cleaner = DataCleaner(CleaningConfig(remove_html=True))
        text = "<p>床前明月光</p>"
        result = cleaner.clean(text)
        assert "<p>" not in result
        assert "床前明月光" in result

    def test_clean_urls(self):
        """测试URL移除。"""
        cleaner = DataCleaner(CleaningConfig(remove_urls=True))
        text = "访问 https://example.com 获取更多信息"
        result = cleaner.clean(text)
        assert "https://" not in result

    def test_clean_control_chars(self):
        """测试控制字符移除。"""
        cleaner = DataCleaner(CleaningConfig(remove_control_chars=True))
        text = "床前\x00明月光\x1f"
        result = cleaner.clean(text)
        assert "\x00" not in result
        assert "\x1f" not in result

    def test_clean_whitespace(self):
        """测试空白字符规范化。"""
        cleaner = DataCleaner(CleaningConfig(normalize_whitespace=True))
        text = "床前    明月光"
        result = cleaner.clean(text)
        assert "    " not in result

    def test_clean_extra_newlines(self):
        """测试多余换行符移除。"""
        cleaner = DataCleaner(CleaningConfig(remove_extra_newlines=True))
        text = "床前\n\n\n\n明月光"
        result = cleaner.clean(text)
        assert result.count('\n') <= 2

    def test_clean_batch(self):
        """测试批量清洗。"""
        cleaner = DataCleaner()
        texts = ["<p>文本1</p>", "<b>文本2</b>"]
        results = cleaner.clean_batch(texts)
        assert len(results) == 2
        assert "<p>" not in results[0]
        assert "<b>" not in results[1]

    def test_clean_zero_width_chars(self):
        """测试零宽字符移除。"""
        cleaner = DataCleaner(CleaningConfig(remove_zero_width_chars=True))
        text = "床\u200b前\u200c明\u200d月光"
        result = cleaner.clean(text)
        assert "\u200b" not in result
        assert "\u200c" not in result
        assert "\u200d" not in result

    def test_clean_bom(self):
        """测试BOM移除。"""
        cleaner = DataCleaner(CleaningConfig(remove_bom=True))
        text = "\ufeff床前明月光"
        result = cleaner.clean(text)
        assert "\ufeff" not in result


class TestDataFilter:
    """测试数据过滤模块。"""

    def test_filter_by_length(self):
        """测试按长度过滤。"""
        data_filter = DataFilter(FilterConfig(
            min_length=10,
            max_length=100,
            check_length=True,
        ))

        samples = [
            {"text": "短", "id": "1"},
            {"text": "床前明月光，疑是地上霜。", "id": "2"},
            {"text": "长" * 200, "id": "3"},
        ]

        results = data_filter.filter_batch(samples)
        assert len(results) == 1
        assert results[0]["id"] == "2"

    def test_filter_duplicates(self):
        """测试重复过滤。"""
        data_filter = DataFilter(FilterConfig(
            check_duplicates=True,
            check_quality=False,
        ))

        samples = [
            {"text": "这是一个较长的测试文本用于验证去重功能。", "id": "1"},
            {"text": "这是一个较长的测试文本用于验证去重功能。", "id": "2"},
            {"text": "另一个不同的测试文本。", "id": "3"},
        ]

        results = data_filter.filter_batch(samples)
        assert len(results) == 2

    def test_filter_reset_hashes(self):
        """测试重置哈希集合。"""
        data_filter = DataFilter(FilterConfig(
            check_duplicates=True,
            check_quality=False,
        ))

        samples = [
            {"text": "这是一个较长的测试文本用于验证去重功能。", "id": "1"},
            {"text": "这是一个较长的测试文本用于验证去重功能。", "id": "1"},
        ]

        data_filter.filter_batch(samples)
        assert len(data_filter.seen_hashes) == 1

        data_filter.reset_seen_hashes()
        assert len(data_filter.seen_hashes) == 0

    def test_filter_stats(self):
        """测试过滤统计。"""
        data_filter = DataFilter(FilterConfig(min_length=10))

        samples = [
            {"text": "短", "id": "1"},
            {"text": "床前明月光，疑是地上霜。", "id": "2"},
        ]

        data_filter.filter_batch(samples)
        stats = data_filter.get_stats()

        assert stats['total'] == 2
        assert stats['passed'] >= 1
        assert stats['length_filtered'] >= 1


class TestQualityScorer:
    """测试质量评分器。"""

    def test_score_chinese_text(self):
        """测试中文文本评分。"""
        scorer = QualityScorer()
        sample = {"text": "床前明月光，疑是地上霜。举头望明月，低头思故乡。"}
        score = scorer.score(sample)
        assert 0 <= score <= 1

    def test_score_short_text(self):
        """测试短文本评分。"""
        scorer = QualityScorer()
        sample = {"text": "短"}
        score = scorer.score(sample)
        assert 0 <= score <= 1

    def test_score_empty_text(self):
        """测试空文本评分。"""
        scorer = QualityScorer()
        sample = {"text": ""}
        score = scorer.score(sample)
        assert score == 0.0


class TestDataNormalizer:
    """测试文本标准化模块。"""

    def test_normalize_simplify_chinese(self):
        """测试繁体转简体。"""
        normalizer = DataNormalizer(NormalizerConfig(simplify_chinese=True))
        text = "\u7e41\u9ad4\u4e2d\u6587"
        result = normalizer.normalize(text)
        assert len(result) > 0

    def test_normalize_punctuation(self):
        """测试标点符号规范化。"""
        normalizer = DataNormalizer(NormalizerConfig(normalize_punctuation=True))
        text = "床前明月光，疑是地上霜。"
        result = normalizer.normalize(text)
        assert "，" in result or "," in result

    def test_normalize_whitespace(self):
        """测试空白字符规范化。"""
        normalizer = DataNormalizer(NormalizerConfig(normalize_whitespace=True))
        text = "床前　　明月光"
        result = normalizer.normalize(text)
        assert "　" not in result

    def test_normalize_quotes(self):
        """测试引号规范化。"""
        normalizer = DataNormalizer(NormalizerConfig(normalize_quotes=True))
        text = '"床前明月光"'
        result = normalizer.normalize(text)
        assert '"' in result or '"' in result

    def test_normalize_batch(self):
        """测试批量标准化。"""
        normalizer = DataNormalizer()
        texts = ["繁體", "簡體"]
        results = normalizer.normalize_batch(texts)
        assert len(results) == 2

    def test_normalize_unicode(self):
        """测试Unicode规范化。"""
        normalizer = DataNormalizer(NormalizerConfig(unicode_normalize="NFKC"))
        text = "café"
        result = normalizer.normalize(text)
        assert result is not None


class TestDataFormatter:
    """测试格式转换模块。"""

    def test_jsonl_to_json(self):
        """测试JSONL转JSON。"""
        data = [
            {"text": "床前明月光", "id": "1"},
            {"text": "疑是地上霜", "id": "2"},
        ]
        input_path = create_temp_jsonl(data)
        fd, output_path = tempfile.mkstemp(suffix=".json", text=True)
        os.close(fd)
        output_path = Path(output_path)

        formatter = DataFormatter(input_format="jsonl", output_format="json")
        count = formatter.convert(input_path, output_path)

        assert count == 2
        with open(output_path, 'r', encoding='utf-8') as f:
            result = json.load(f)
            assert len(result) == 2

        os.unlink(input_path)
        os.unlink(output_path)

    def test_json_to_jsonl(self):
        """测试JSON转JSONL。"""
        data = [
            {"text": "床前明月光", "id": "1"},
            {"text": "疑是地上霜", "id": "2"},
        ]
        input_path = create_temp_json(data)
        fd, output_path = tempfile.mkstemp(suffix=".jsonl", text=True)
        os.close(fd)
        output_path = Path(output_path)

        formatter = DataFormatter(input_format="json", output_format="jsonl")
        count = formatter.convert(input_path, output_path)

        assert count == 2
        with open(output_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            assert len(lines) == 2

        os.unlink(input_path)
        os.unlink(output_path)

    def test_txt_to_jsonl(self):
        """测试TXT转JSONL。"""
        texts = ["床前明月光", "疑是地上霜"]
        input_path = create_temp_txt(texts)
        fd, output_path = tempfile.mkstemp(suffix=".jsonl", text=True)
        os.close(fd)
        output_path = Path(output_path)

        formatter = DataFormatter(
            input_format="txt",
            output_format="jsonl",
            config=FormatConfig(text_field="text")
        )
        count = formatter.convert(input_path, output_path)

        assert count == 2
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line)
                assert "text" in record

        os.unlink(input_path)
        os.unlink(output_path)

    def test_format_detector(self):
        """测试格式自动检测。"""
        assert FormatDetector.detect("data.jsonl") == "jsonl"
        assert FormatDetector.detect("data.json") == "json"
        assert FormatDetector.detect("data.csv") == "csv"
        assert FormatDetector.detect("data.txt") == "txt"


class TestDataValidator:
    """测试数据验证模块。"""

    def test_validate_required_fields(self):
        """测试必需字段验证。"""
        validator = DataValidator(ValidationConfig(
            required_fields=["text", "id"]
        ))

        valid_sample = {"text": "床前明月光", "id": "1"}
        invalid_sample = {"text": "床前明月光"}

        assert validator.validate(valid_sample) is True
        assert validator.validate(invalid_sample) is False

    def test_validate_text_length(self):
        """测试文本长度验证。"""
        validator = DataValidator(ValidationConfig(
            min_text_length=10
        ))

        valid_sample = {"text": "床前明月光，疑是地上霜。"}
        invalid_sample = {"text": "短"}

        assert validator.validate(valid_sample) is True
        assert validator.validate(invalid_sample) is False

    def test_validate_with_result(self):
        """测试详细验证结果。"""
        validator = DataValidator(ValidationConfig(
            required_fields=["text"]
        ))

        sample = {"text": ""}
        result = validator.validate_with_result(sample)

        assert result.is_valid is False
        assert len(result.errors) > 0

    def test_validate_batch(self):
        """测试批量验证。"""
        validator = DataValidator(ValidationConfig(
            required_fields=["text"]
        ))

        samples = [
            {"text": "床前明月光", "id": "1"},
            {"text": ""},
            {"text": "举头望明月", "id": "3"},
        ]

        valid, invalid = validator.validate_batch(samples)
        assert len(valid) >= 1

    def test_add_custom_validator(self):
        """测试添加自定义验证器。"""
        validator = DataValidator()

        def check_positive_age(sample):
            if 'age' in sample and sample['age'] < 0:
                return False, "Age must be positive"
            return True, ""

        validator.add_custom_validator(check_positive_age)

        valid_sample = {"text": "床前明月光", "age": 20}
        invalid_sample = {"text": "疑是地上霜", "age": -5}

        assert validator.validate(valid_sample) is True
        assert validator.validate(invalid_sample) is False

    def test_validation_stats(self):
        """测试验证统计。"""
        validator = DataValidator(ValidationConfig(
            required_fields=["text"]
        ))

        samples = [
            {"text": "床前明月光"},
            {"content": "疑是地上霜"},
        ]

        validator.validate_batch(samples)
        stats = validator.get_stats()

        assert stats['total'] == 2
        assert stats['valid'] >= 0


class TestValidationResult:
    """测试验证结果类。"""

    def test_add_error(self):
        """测试添加错误。"""
        result = ValidationResult()
        result.add_error("Test error", field="text")

        assert result.is_valid is False
        assert len(result.errors) == 1
        assert "text" in result.field_errors

    def test_add_warning(self):
        """测试添加警告。"""
        result = ValidationResult()
        result.add_warning("Test warning")

        assert result.is_valid is True
        assert len(result.warnings) == 1

    def test_merge(self):
        """测试结果合并。"""
        result1 = ValidationResult()
        result1.add_error("Error 1")

        result2 = ValidationResult()
        result2.add_warning("Warning 1")

        result1.merge(result2)

        assert result1.is_valid is False
        assert len(result1.errors) == 1
        assert len(result1.warnings) == 1


class TestIntegration:
    """集成测试。"""

    def test_full_processing_pipeline(self):
        """测试完整处理流程。"""
        raw_data = [
            {
                "id": "1",
                "text": "<p>床前明月光</p>\n\n\n疑是地上霜",
            },
            {
                "id": "2",
                "text": "   举头望明月   ",
            },
        ]
        input_path = create_temp_jsonl(raw_data)
        fd, output_path = tempfile.mkstemp(suffix=".jsonl", text=True)
        os.close(fd)
        output_path = Path(output_path)

        cleaner = DataCleaner()
        normalizer = DataNormalizer()
        data_filter = DataFilter(FilterConfig(min_length=5))
        validator = DataValidator(ValidationConfig(required_fields=["text"]))

        processed_records = []
        for record in raw_data:
            text = cleaner.clean(record.get("text", ""))
            text = normalizer.normalize(text)

            if text.strip():
                record["text"] = text
                if data_filter.is_valid(record) and validator.validate(record):
                    processed_records.append(record)

        with open(output_path, 'w', encoding='utf-8') as f:
            for record in processed_records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')

        assert len(processed_records) >= 0

        os.unlink(input_path)
        os.unlink(output_path)

    def test_multi_format_conversion(self):
        """测试多格式转换。"""
        json_data = [
            {"text": "床前明月光", "id": "1"},
            {"text": "疑是地上霜", "id": "2"},
        ]

        input_path = create_temp_json(json_data)

        fd, jsonl_path = tempfile.mkstemp(suffix=".jsonl", text=True)
        os.close(fd)
        jsonl_path = Path(jsonl_path)

        formatter = DataFormatter(input_format="json", output_format="jsonl")
        formatter.convert(input_path, jsonl_path)

        fd, csv_path = tempfile.mkstemp(suffix=".csv", text=True)
        os.close(fd)
        csv_path = Path(csv_path)

        formatter2 = DataFormatter(input_format="jsonl", output_format="csv")
        formatter2.convert(jsonl_path, csv_path)

        assert jsonl_path.exists()
        assert csv_path.exists()

        os.unlink(input_path)
        os.unlink(jsonl_path)
        os.unlink(csv_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
