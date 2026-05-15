"""Experiment tracking system for Lingmao Moyun (灵猫墨韵) project.

This module provides comprehensive experiment tracking capabilities including:
- Automatic hyperparameter recording
- Training metrics logging
- Experiment version control
- Experiment comparison and result analysis
- WandB and TensorBoard integration
"""

from src.experiments.tracker import ExperimentTracker, Experiment
from src.experiments.config_manager import ConfigManager, HyperparameterConfig
from src.experiments.logger import MetricsLogger, MetricsHistory

__all__ = [
    "ExperimentTracker",
    "Experiment",
    "ConfigManager",
    "HyperparameterConfig",
    "MetricsLogger",
    "MetricsHistory",
]

__version__ = "0.1.0"
