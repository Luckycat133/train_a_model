#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验追踪系统测试脚本
用于测试实验追踪器、配置管理器和指标日志器的功能
"""

import os
import sys
import json
import time
import shutil
import tempfile
import logging
from pathlib import Path
from datetime import datetime

import pytest
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.experiments.tracker import (
    ExperimentTracker,
    Experiment,
    ExperimentStatus,
)
from src.experiments.config_manager import (
    ConfigManager,
    HyperparameterConfig,
    ConfigValidator,
)
from src.experiments.logger import (
    MetricsLogger,
    MetricsHistory,
)


@pytest.fixture
def temp_experiment_dir():
    """创建临时实验目录。"""
    temp_dir = tempfile.mkdtemp(prefix="test_experiments_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def temp_config_dir():
    """创建临时配置目录。"""
    temp_dir = tempfile.mkdtemp(prefix="test_configs_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def temp_log_dir():
    """创建临时日志目录。"""
    temp_dir = tempfile.mkdtemp(prefix="test_logs_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_config():
    """示例配置。"""
    return {
        "model": {
            "d_model": 512,
            "nhead": 8,
            "num_layers": 6,
            "dim_feedforward": 2048,
            "dropout": 0.1,
        },
        "training": {
            "batch_size": 16,
            "learning_rate": 1e-4,
            "epochs": 5,
        },
        "data": {
            "context_length": 256,
            "stride": 128,
        },
    }


class TestExperiment:
    """实验类测试。"""

    def test_experiment_creation(self, sample_config):
        """测试实验创建。"""
        exp = Experiment(
            experiment_id="test_001",
            name="test_experiment",
            config=sample_config,
            tags=["test", "unit"],
        )

        assert exp.experiment_id == "test_001"
        assert exp.name == "test_experiment"
        assert exp.config == sample_config
        assert exp.tags == ["test", "unit"]
        assert exp.status == ExperimentStatus.PENDING

    def test_experiment_to_dict(self, sample_config):
        """测试实验转字典。"""
        exp = Experiment(
            experiment_id="test_001",
            name="test_experiment",
            config=sample_config,
        )

        data = exp.to_dict()
        assert isinstance(data, dict)
        assert data["experiment_id"] == "test_001"
        assert data["status"] == "pending"
        assert data["config"] == sample_config

    def test_experiment_from_dict(self, sample_config):
        """测试从字典创建实验。"""
        data = {
            "experiment_id": "test_001",
            "name": "test_experiment",
            "config": sample_config,
            "status": "completed",
            "metrics": {"loss": [1.0, 0.5, 0.1]},
            "tags": ["test"],
            "notes": "",
            "created_at": datetime.now().isoformat(),
            "started_at": None,
            "completed_at": None,
            "git_commit": None,
            "checkpoint_path": None,
            "best_metric": 0.1,
            "best_metric_step": 2,
        }

        exp = Experiment.from_dict(data)
        assert exp.experiment_id == "test_001"
        assert exp.status == ExperimentStatus.COMPLETED
        assert exp.metrics["loss"] == [1.0, 0.5, 0.1]

    def test_experiment_config_hash(self, sample_config):
        """测试配置哈希生成。"""
        exp = Experiment(
            experiment_id="test_001",
            name="test_experiment",
            config=sample_config,
        )

        hash1 = exp.get_config_hash()
        assert isinstance(hash1, str)
        assert len(hash1) == 8

        exp2 = Experiment(
            experiment_id="test_002",
            name="test_experiment",
            config=sample_config,
        )
        hash2 = exp2.get_config_hash()
        assert hash1 == hash2

    def test_experiment_duration(self):
        """测试实验持续时间计算。"""
        exp = Experiment(
            experiment_id="test_001",
            name="test_experiment",
            config={},
        )

        assert exp.duration() is None

        exp.started_at = datetime.now().isoformat()
        exp.completed_at = datetime.now().isoformat()
        duration = exp.duration()
        assert duration is not None and duration < 1.0


class TestExperimentTracker:
    """实验追踪器测试。"""

    def test_tracker_initialization(self, temp_experiment_dir):
        """测试追踪器初始化。"""
        tracker = ExperimentTracker(
            experiment_dir=temp_experiment_dir,
            project_name="test_project",
        )

        assert tracker.experiment_dir == Path(temp_experiment_dir)
        assert tracker.project_name == "test_project"
        assert tracker.experiments == {}

    def test_create_experiment(self, temp_experiment_dir, sample_config):
        """测试创建实验。"""
        tracker = ExperimentTracker(experiment_dir=temp_experiment_dir)

        exp = tracker.create_experiment(
            name="test_experiment",
            config=sample_config,
            tags=["test"],
            notes="Test notes",
        )

        assert exp is not None
        assert exp.name == "test_experiment"
        assert exp.config == sample_config
        assert exp.tags == ["test"]
        assert exp.notes == "Test notes"
        assert exp.experiment_id in tracker.experiments

    def test_start_experiment(self, temp_experiment_dir, sample_config):
        """测试启动实验。"""
        tracker = ExperimentTracker(experiment_dir=temp_experiment_dir)
        exp = tracker.create_experiment(name="test_exp", config=sample_config)

        started_exp = tracker.start_experiment(exp.experiment_id)

        assert started_exp.status == ExperimentStatus.RUNNING
        assert started_exp.started_at is not None

    def test_log_metrics(self, temp_experiment_dir, sample_config):
        """测试记录指标。"""
        tracker = ExperimentTracker(experiment_dir=temp_experiment_dir)
        exp = tracker.create_experiment(name="test_exp", config=sample_config)
        tracker.start_experiment(exp.experiment_id)

        tracker.log_metrics(exp.experiment_id, {"loss": 1.0, "accuracy": 0.5})
        tracker.log_metrics(exp.experiment_id, {"loss": 0.5, "accuracy": 0.7})

        updated_exp = tracker.get_experiment(exp.experiment_id)
        assert updated_exp.metrics["loss"] == [1.0, 0.5]
        assert updated_exp.metrics["accuracy"] == [0.5, 0.7]
        assert updated_exp.best_metric == 0.5

    def test_complete_experiment(self, temp_experiment_dir, sample_config):
        """测试完成实验。"""
        tracker = ExperimentTracker(experiment_dir=temp_experiment_dir)
        exp = tracker.create_experiment(name="test_exp", config=sample_config)
        tracker.start_experiment(exp.experiment_id)

        completed_exp = tracker.complete_experiment(
            exp.experiment_id,
            status=ExperimentStatus.COMPLETED,
        )

        assert completed_exp.status == ExperimentStatus.COMPLETED
        assert completed_exp.completed_at is not None

    def test_get_experiments_by_tag(self, temp_experiment_dir, sample_config):
        """测试按标签获取实验。"""
        tracker = ExperimentTracker(experiment_dir=temp_experiment_dir)

        tracker.create_experiment(
            name="exp1", config=sample_config, tags=["baseline"]
        )
        tracker.create_experiment(
            name="exp2", config=sample_config, tags=["baseline", "improved"]
        )
        tracker.create_experiment(
            name="exp3", config=sample_config, tags=["experimental"]
        )

        baseline_exps = tracker.get_experiments_by_tag("baseline")
        assert len(baseline_exps) == 2

        improved_exps = tracker.get_experiments_by_tag("improved")
        assert len(improved_exps) == 1

    def test_list_experiments(self, temp_experiment_dir, sample_config):
        """测试列出实验。"""
        tracker = ExperimentTracker(experiment_dir=temp_experiment_dir)

        tracker.create_experiment(name="exp1", config=sample_config)
        tracker.create_experiment(name="exp2", config=sample_config)

        experiments = tracker.list_experiments()
        assert len(experiments) == 2

        running_exps = tracker.list_experiments(status=ExperimentStatus.RUNNING)
        assert len(running_exps) == 0

    def test_delete_experiment(self, temp_experiment_dir, sample_config):
        """测试删除实验。"""
        tracker = ExperimentTracker(experiment_dir=temp_experiment_dir)
        exp = tracker.create_experiment(name="test_exp", config=sample_config)

        assert exp.experiment_id in tracker.experiments

        result = tracker.delete_experiment(exp.experiment_id)
        assert result is True
        assert exp.experiment_id not in tracker.experiments

    def test_get_best_experiment(self, temp_experiment_dir, sample_config):
        """测试获取最佳实验。"""
        tracker = ExperimentTracker(experiment_dir=temp_experiment_dir)

        exp1 = tracker.create_experiment(name="exp1", config=sample_config)
        tracker.start_experiment(exp1.experiment_id)
        tracker.log_metrics(exp1.experiment_id, {"loss": 1.0})
        tracker.log_metrics(exp1.experiment_id, {"loss": 0.8})
        tracker.complete_experiment(exp1.experiment_id)

        exp2 = tracker.create_experiment(name="exp2", config=sample_config)
        tracker.start_experiment(exp2.experiment_id)
        tracker.log_metrics(exp2.experiment_id, {"loss": 0.9})
        tracker.log_metrics(exp2.experiment_id, {"loss": 0.5})
        tracker.complete_experiment(exp2.experiment_id)

        best_exp = tracker.get_best_experiment("loss")
        assert best_exp is not None
        assert best_exp.experiment_id == exp2.experiment_id

    def test_experiment_persistence(self, temp_experiment_dir, sample_config):
        """测试实验持久化。"""
        tracker1 = ExperimentTracker(experiment_dir=temp_experiment_dir)
        exp1 = tracker1.create_experiment(name="test_exp", config=sample_config)

        tracker2 = ExperimentTracker(experiment_dir=temp_experiment_dir)
        assert len(tracker2.experiments) == 1
        assert tracker2.get_experiment(exp1.experiment_id) is not None


class TestHyperparameterConfig:
    """超参数配置测试。"""

    def test_config_creation(self):
        """测试配置创建。"""
        config = HyperparameterConfig()
        assert config.model == {}
        assert config.training == {}

    def test_config_from_dict(self, sample_config):
        """测试从字典创建配置。"""
        config = HyperparameterConfig.from_dict(sample_config)
        assert config.model["d_model"] == 512
        assert config.training["batch_size"] == 16

    def test_config_get_nested(self):
        """测试嵌套值获取。"""
        config = HyperparameterConfig(
            model={"d_model": 768, "nhead": 12},
            training={"learning_rate": 1e-4},
        )

        assert config.get("model.d_model") == 768
        assert config.get("model.nhead") == 12
        assert config.get("training.learning_rate") == 1e-4
        assert config.get("nonexistent.key", "default") == "default"

    def test_config_set_nested(self):
        """测试嵌套值设置。"""
        config = HyperparameterConfig()

        config.set("model.d_model", 1024)
        assert config.model["d_model"] == 1024

        config.set("training.epochs", 20)
        assert config.training["epochs"] == 20

    def test_config_flatten(self):
        """测试配置展平。"""
        config = HyperparameterConfig(
            model={"d_model": 768, "nhead": 12},
            training={"batch_size": 8, "lr": 1e-4},
        )

        flat = config.flatten()
        assert flat["model.d_model"] == 768
        assert flat["model.nhead"] == 12
        assert flat["training.batch_size"] == 8

    def test_config_merge(self):
        """测试配置合并。"""
        config1 = HyperparameterConfig(
            model={"d_model": 768, "nhead": 12},
            training={"batch_size": 8},
        )

        config2 = HyperparameterConfig(
            model={"d_model": 1024},
            training={"epochs": 10},
        )

        merged = config1.merge(config2)
        assert merged.model["d_model"] == 1024
        assert merged.model["nhead"] == 12
        assert merged.training["batch_size"] == 8
        assert merged.training["epochs"] == 10


class TestConfigValidator:
    """配置验证器测试。"""

    def test_validator_creation(self):
        """测试验证器创建。"""
        validator = ConfigValidator()
        assert validator.rules == {}
        assert validator.errors == []

    def test_add_rule(self):
        """测试添加验证规则。"""
        validator = ConfigValidator()
        validator.add_rule(
            "model.d_model",
            lambda x: x > 0,
            "d_model must be positive",
        )

        assert "model.d_model" in validator.rules

    def test_validate_success(self):
        """测试验证成功。"""
        validator = ConfigValidator()
        validator.add_rule(
            "model.d_model",
            lambda x: isinstance(x, (int, float)) and x > 0,
            "d_model must be positive",
        )

        config = HyperparameterConfig(model={"d_model": 512})
        result = validator.validate(config)
        assert result is True
        assert validator.get_errors() == []

    def test_validate_failure(self):
        """测试验证失败。"""
        validator = ConfigValidator()
        validator.add_rule(
            "model.d_model",
            lambda x: isinstance(x, (int, float)) and x > 0,
            "d_model must be positive",
        )

        config = HyperparameterConfig(model={"d_model": -100})
        result = validator.validate(config)
        assert result is False
        assert len(validator.get_errors()) > 0


class TestConfigManager:
    """配置管理器测试。"""

    def test_manager_initialization(self, temp_config_dir):
        """测试管理器初始化。"""
        manager = ConfigManager(config_dir=temp_config_dir)
        assert manager.config_dir == Path(temp_config_dir)
        assert manager.current_config is None

    def test_create_default_config(self, temp_config_dir):
        """测试创建默认配置。"""
        manager = ConfigManager(config_dir=temp_config_dir)
        config = manager.create_default_config()

        assert config.model["d_model"] == 768
        assert config.training["batch_size"] == 8
        assert config.optimizer["type"] == "AdamW"

    def test_save_and_load_config(self, temp_config_dir, sample_config):
        """测试保存和加载配置。"""
        manager = ConfigManager(config_dir=temp_config_dir)
        config = HyperparameterConfig.from_dict(sample_config)

        save_result = manager.save_config("test_config", config)
        assert save_result is True

        loaded_config = manager.load_config("test_config")
        assert loaded_config is not None
        assert loaded_config.model["d_model"] == 512

    def test_create_experiment_config(self, temp_config_dir):
        """测试创建实验配置。"""
        manager = ConfigManager(config_dir=temp_config_dir)
        config = manager.create_experiment_config(
            overrides={
                "model.d_model": 1024,
                "training.epochs": 20,
            }
        )

        assert config.model["d_model"] == 1024
        assert config.training["epochs"] == 20

    def test_list_configs(self, temp_config_dir, sample_config):
        """测试列出配置。"""
        manager = ConfigManager(config_dir=temp_config_dir)
        config = HyperparameterConfig.from_dict(sample_config)

        manager.save_config("config1", config)
        manager.save_config("config2", config)

        configs = manager.list_configs()
        assert "config1" in configs
        assert "config2" in configs

    def test_delete_config(self, temp_config_dir, sample_config):
        """测试删除配置。"""
        manager = ConfigManager(config_dir=temp_config_dir)
        config = HyperparameterConfig.from_dict(sample_config)
        manager.save_config("test_config", config)

        result = manager.delete_config("test_config")
        assert result is True

        loaded = manager.load_config("test_config")
        assert loaded is None

    def test_validate_config(self, temp_config_dir):
        """测试配置验证。"""
        manager = ConfigManager(config_dir=temp_config_dir)

        config = HyperparameterConfig(model={"d_model": -100})
        result = manager.validate_config(config)
        assert result is False
        assert len(manager.get_validation_errors()) > 0


class TestMetricsHistory:
    """指标历史测试。"""

    def test_history_creation(self):
        """测试历史记录创建。"""
        history = MetricsHistory()
        assert history.metrics == {}
        assert history.timestamps == []

    def test_add_metrics(self):
        """测试添加指标。"""
        history = MetricsHistory()
        history.add({"loss": 1.0, "accuracy": 0.5}, step=0, epoch=1)
        history.add({"loss": 0.5, "accuracy": 0.7}, step=1, epoch=1)

        assert history.metrics["loss"] == [1.0, 0.5]
        assert history.metrics["accuracy"] == [0.5, 0.7]
        assert len(history.steps) == 2

    def test_get_latest(self):
        """测试获取最新值。"""
        history = MetricsHistory()
        history.add({"loss": 1.0}, step=0)
        history.add({"loss": 0.5}, step=1)

        latest = history.get_latest("loss")
        assert latest == 0.5

    def test_get_best(self):
        """测试获取最佳值。"""
        history = MetricsHistory()
        history.add({"loss": 1.0}, step=0)
        history.add({"loss": 0.5}, step=1)
        history.add({"loss": 0.8}, step=2)

        best_min = history.get_best("loss", mode="min")
        best_max = history.get_best("accuracy", mode="max")
        assert best_min == 0.5

    def test_get_stats(self):
        """测试获取统计信息。"""
        history = MetricsHistory()
        for i in range(5):
            history.add({"loss": 1.0 - i * 0.1}, step=i)

        stats = history.get_stats("loss")
        assert stats["count"] == 5
        assert stats["min"] == 0.6
        assert stats["max"] == 1.0


class TestMetricsLogger:
    """指标日志器测试。"""

    def test_logger_initialization(self, temp_log_dir):
        """测试日志器初始化。"""
        logger = MetricsLogger(
            experiment_name="test_experiment",
            experiment_id="test_001",
            log_dir=temp_log_dir,
            use_wandb=False,
            use_tensorboard=False,
        )

        assert logger.experiment_name == "test_experiment"
        assert logger.experiment_id == "test_001"
        assert logger.metrics_history is not None

        logger.finish()

    def test_log_metrics(self, temp_log_dir):
        """测试记录指标。"""
        logger = MetricsLogger(
            experiment_name="test_experiment",
            log_dir=temp_log_dir,
            use_wandb=False,
            use_tensorboard=False,
        )

        logger.log_metrics({"loss": 1.0, "accuracy": 0.5}, step=0, epoch=1)
        logger.log_metrics({"loss": 0.5, "accuracy": 0.7}, step=1, epoch=1)

        assert logger.get_history("loss") == [1.0, 0.5]
        assert logger.get_history("accuracy") == [0.5, 0.7]
        assert logger.get_latest("loss") == 0.5

        logger.finish()

    def test_log_summary(self, temp_log_dir):
        """测试记录摘要。"""
        logger = MetricsLogger(
            experiment_name="test_experiment",
            log_dir=temp_log_dir,
            use_wandb=False,
            use_tensorboard=False,
        )

        logger.log_metrics({"loss": 1.0}, step=0)
        logger.log_metrics({"loss": 0.5}, step=1)
        logger.log_summary()

        summary_file = Path(temp_log_dir) / "test_experiment" / "metrics_summary.json"
        assert summary_file.exists()

        logger.finish()

    def test_context_manager(self, temp_log_dir):
        """测试上下文管理器。"""
        with MetricsLogger(
            experiment_name="test_experiment",
            log_dir=temp_log_dir,
            use_wandb=False,
            use_tensorboard=False,
        ) as logger:
            logger.log_metrics({"loss": 1.0}, step=0)

        assert logger.get_latest("loss") == 1.0


class TestExperimentComparison:
    """实验对比测试。"""

    def test_compare_experiments(self, temp_experiment_dir, sample_config):
        """测试对比实验。"""
        tracker = ExperimentTracker(experiment_dir=temp_experiment_dir)

        exp1 = tracker.create_experiment(name="exp1", config=sample_config)
        tracker.start_experiment(exp1.experiment_id)
        tracker.log_metrics(exp1.experiment_id, {"loss": 1.0})
        tracker.log_metrics(exp1.experiment_id, {"loss": 0.6})
        tracker.complete_experiment(exp1.experiment_id)

        exp2 = tracker.create_experiment(name="exp2", config=sample_config)
        tracker.start_experiment(exp2.experiment_id)
        tracker.log_metrics(exp2.experiment_id, {"loss": 0.9})
        tracker.log_metrics(exp2.experiment_id, {"loss": 0.4})
        tracker.complete_experiment(exp2.experiment_id)

        comparison = tracker.compare_experiments(
            [exp1.experiment_id, exp2.experiment_id],
            metric_names=["loss"],
        )

        assert not comparison.empty
        assert "experiment_id" in comparison.columns
        assert "loss_final" in comparison.columns


def run_all_tests():
    """运行所有测试。"""
    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    run_all_tests()
