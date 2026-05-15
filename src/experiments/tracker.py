"""Experiment tracker for Lingmao Moyun project.

Provides experiment version control, comparison, and result tracking.
"""

import json
import os
import hashlib
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field, asdict
from enum import Enum

try:
    import pandas as pd
except Exception:
    pd = None


class ExperimentStatus(Enum):
    """Experiment status enumeration."""
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    PENDING = "pending"


@dataclass
class Experiment:
    """Represents a single experiment run."""

    experiment_id: str
    name: str
    config: Dict[str, Any]
    status: ExperimentStatus = ExperimentStatus.PENDING
    metrics: Dict[str, List[float]] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    git_commit: Optional[str] = None
    checkpoint_path: Optional[str] = None
    best_metric: Optional[float] = None
    best_metric_step: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert experiment to dictionary."""
        data = asdict(self)
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Experiment":
        """Create experiment from dictionary."""
        if isinstance(data.get("status"), str):
            data["status"] = ExperimentStatus(data["status"])
        return cls(**data)

    def get_config_hash(self) -> str:
        """Generate hash for experiment configuration."""
        config_str = json.dumps(self.config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]

    def duration(self) -> Optional[float]:
        """Calculate experiment duration in seconds."""
        if self.started_at is None:
            return None
        start = datetime.fromisoformat(self.started_at)
        end = datetime.fromisoformat(
            self.completed_at or datetime.now().isoformat()
        )
        return (end - start).total_seconds()


class ExperimentTracker:
    """Main experiment tracking system."""

    def __init__(
        self,
        experiment_dir: str = "experiments",
        project_name: str = "lingmao_moyun",
    ):
        """Initialize experiment tracker.

        Args:
            experiment_dir: Directory to store experiment data.
            project_name: Project name for organization.
        """
        self.experiment_dir = Path(experiment_dir)
        self.project_name = project_name
        self.experiments: Dict[str, Experiment] = {}
        self.current_experiment: Optional[Experiment] = None

        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self._load_experiments()

    def _load_experiments(self) -> None:
        """Load existing experiments from disk."""
        index_file = self.experiment_dir / "experiments_index.json"
        if index_file.exists():
            try:
                with open(index_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for exp_data in data.get("experiments", []):
                        exp = Experiment.from_dict(exp_data)
                        self.experiments[exp.experiment_id] = exp
            except Exception:
                pass

    def _save_index(self) -> None:
        """Save experiments index to disk."""
        index_file = self.experiment_dir / "experiments_index.json"
        data = {
            "experiments": [
                exp.to_dict() for exp in self.experiments.values()
            ]
        }
        with open(index_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _generate_experiment_id(self, name: str) -> str:
        """Generate unique experiment ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name_hash = hashlib.md5(name.encode()).hexdigest()[:4]
        return f"{name_hash}_{timestamp}"

    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash."""
        try:
            import subprocess
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=False,
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except Exception:
            return None

    def create_experiment(
        self,
        name: str,
        config: Dict[str, Any],
        tags: Optional[List[str]] = None,
        notes: str = "",
    ) -> Experiment:
        """Create a new experiment.

        Args:
            name: Experiment name.
            config: Experiment configuration/hyperparameters.
            tags: Optional list of tags.
            notes: Optional experiment notes.

        Returns:
            Created Experiment instance.
        """
        experiment_id = self._generate_experiment_id(name)

        experiment = Experiment(
            experiment_id=experiment_id,
            name=name,
            config=config,
            status=ExperimentStatus.PENDING,
            tags=tags or [],
            notes=notes,
            git_commit=self._get_git_commit(),
        )

        self.experiments[experiment_id] = experiment
        self.current_experiment = experiment
        self._save_index()

        exp_dir = self.experiment_dir / experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        config_file = exp_dir / "config.json"
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        return experiment

    def start_experiment(self, experiment_id: str) -> Experiment:
        """Mark an experiment as started.

        Args:
            experiment_id: ID of experiment to start.

        Returns:
            Updated Experiment instance.
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")

        exp = self.experiments[experiment_id]
        exp.status = ExperimentStatus.RUNNING
        exp.started_at = datetime.now().isoformat()
        self.current_experiment = exp
        self._save_index()
        return exp

    def log_metrics(
        self,
        experiment_id: str,
        metrics: Dict[str, float],
        step: Optional[int] = None,
    ) -> None:
        """Log metrics for an experiment.

        Args:
            experiment_id: ID of experiment.
            metrics: Dictionary of metric names to values.
            step: Optional step number.
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")

        exp = self.experiments[experiment_id]

        for name, value in metrics.items():
            if name not in exp.metrics:
                exp.metrics[name] = []
            exp.metrics[name].append(value)

            if exp.best_metric is None or value < exp.best_metric:
                exp.best_metric = value
                exp.best_metric_step = step

        self._save_index()

        exp_dir = self.experiment_dir / experiment_id
        metrics_file = exp_dir / "metrics.json"
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(exp.metrics, f, indent=2)

    def complete_experiment(
        self,
        experiment_id: str,
        status: ExperimentStatus = ExperimentStatus.COMPLETED,
    ) -> Experiment:
        """Mark an experiment as completed.

        Args:
            experiment_id: ID of experiment.
            status: Final experiment status.

        Returns:
            Updated Experiment instance.
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")

        exp = self.experiments[experiment_id]
        exp.status = status
        exp.completed_at = datetime.now().isoformat()
        self._save_index()
        return exp

    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Get experiment by ID."""
        return self.experiments.get(experiment_id)

    def get_experiments_by_tag(self, tag: str) -> List[Experiment]:
        """Get all experiments with a specific tag."""
        return [
            exp for exp in self.experiments.values()
            if tag in exp.tags
        ]

    def get_experiments_by_name(self, name: str) -> List[Experiment]:
        """Get all experiments with a specific name."""
        return [
            exp for exp in self.experiments.values()
            if exp.name == name
        ]

    def compare_experiments(
        self,
        experiment_ids: List[str],
        metric_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Compare multiple experiments.

        Args:
            experiment_ids: List of experiment IDs to compare.
            metric_names: Optional list of specific metrics to compare.

        Returns:
            DataFrame with comparison results.
        """
        if pd is None:
            raise ImportError("pandas is required for experiment comparison")

        results = []
        for exp_id in experiment_ids:
            exp = self.experiments.get(exp_id)
            if exp is None:
                continue

            row = {
                "experiment_id": exp_id,
                "name": exp.name,
                "status": exp.status.value,
                "duration_seconds": exp.duration(),
                "best_metric": exp.best_metric,
                "tags": ", ".join(exp.tags),
            }

            metrics_to_compare = metric_names or list(exp.metrics.keys())
            for metric in metrics_to_compare:
                if metric in exp.metrics and exp.metrics[metric]:
                    values = exp.metrics[metric]
                    row[f"{metric}_final"] = values[-1]
                    row[f"{metric}_best"] = min(values)
                    row[f"{metric}_mean"] = sum(values) / len(values)
                else:
                    row[f"{metric}_final"] = None
                    row[f"{metric}_best"] = None
                    row[f"{metric}_mean"] = None

            results.append(row)

        return pd.DataFrame(results)

    def get_best_experiment(
        self,
        metric: str,
        name: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Optional[Experiment]:
        """Get best experiment based on a metric.

        Args:
            metric: Metric name to compare.
            name: Optional experiment name filter.
            tags: Optional tag filter.

        Returns:
            Best experiment or None.
        """
        candidates = [
            exp for exp in self.experiments.values()
            if exp.metrics.get(metric) and exp.status == ExperimentStatus.COMPLETED
        ]

        if name:
            candidates = [exp for exp in candidates if exp.name == name]
        if tags:
            candidates = [
                exp for exp in candidates
                if any(tag in exp.tags for tag in tags)
            ]

        if not candidates:
            return None

        return min(
            candidates,
            key=lambda exp: min(exp.metrics[metric])
        )

    def list_experiments(
        self,
        status: Optional[ExperimentStatus] = None,
        name: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> List[Experiment]:
        """List experiments with optional filters.

        Args:
            status: Filter by experiment status.
            name: Filter by experiment name.
            tags: Filter by tags.

        Returns:
            List of matching experiments.
        """
        results = list(self.experiments.values())

        if status:
            results = [exp for exp in results if exp.status == status]
        if name:
            results = [exp for exp in results if name in exp.name]
        if tags:
            results = [
                exp for exp in results
                if any(tag in exp.tags for tag in tags)
            ]

        return sorted(
            results,
            key=lambda exp: exp.created_at,
            reverse=True
        )

    def export_experiment(
        self,
        experiment_id: str,
        export_path: Union[str, Path],
    ) -> None:
        """Export experiment data to a directory.

        Args:
            experiment_id: ID of experiment to export.
            export_path: Path to export directory.
        """
        exp = self.experiments.get(experiment_id)
        if exp is None:
            raise ValueError(f"Experiment {experiment_id} not found")

        export_path = Path(export_path)
        export_path.mkdir(parents=True, exist_ok=True)

        exp_dir = self.experiment_dir / experiment_id
        if exp_dir.exists():
            for item in exp_dir.iterdir():
                if item.is_file():
                    shutil.copy2(item, export_path / item.name)

        metadata_file = export_path / "metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(exp.to_dict(), f, indent=2, ensure_ascii=False)

    def delete_experiment(self, experiment_id: str) -> bool:
        """Delete an experiment.

        Args:
            experiment_id: ID of experiment to delete.

        Returns:
            True if deleted, False if not found.
        """
        if experiment_id not in self.experiments:
            return False

        exp_dir = self.experiment_dir / experiment_id
        if exp_dir.exists():
            shutil.rmtree(exp_dir)

        del self.experiments[experiment_id]
        self._save_index()
        return True
