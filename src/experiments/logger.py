"""实验日志记录器 - 灵猫墨韵实验追踪系统

提供训练指标日志记录功能，支持 WandB 和 TensorBoard 集成。
"""

import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field, asdict
from collections import defaultdict

import numpy as np

try:
    import torch
except Exception:
    torch = None

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

try:
    import wandb
    wandb_available = True
except Exception:
    wandb_available = False


@dataclass
class MetricsHistory:
    """指标历史记录数据类。"""

    metrics: Dict[str, List[float]] = field(default_factory=dict)
    timestamps: List[float] = field(default_factory=list)
    steps: List[int] = field(default_factory=list)
    epochs: List[int] = field(default_factory=list)

    def add(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        epoch: Optional[int] = None,
    ) -> None:
        """添加指标记录。

        Args:
            metrics: 指标字典
            step: 步数
            epoch: 轮次
        """
        current_time = time.time()

        for name, value in metrics.items():
            if name not in self.metrics:
                self.metrics[name] = []
            self.metrics[name].append(value)

        self.timestamps.append(current_time)
        self.steps.append(step if step is not None else len(self.steps))
        self.epochs.append(epoch if epoch is not None else 0)

    def get_metric(self, name: str) -> List[float]:
        """获取指定指标的完整历史。"""
        return self.metrics.get(name, [])

    def get_latest(self, name: str) -> Optional[float]:
        """获取指定指标的最新值。"""
        values = self.metrics.get(name, [])
        return values[-1] if values else None

    def get_best(self, name: str, mode: str = "min") -> Optional[float]:
        """获取最佳指标值。

        Args:
            name: 指标名称
            mode: 'min' 或 'max'

        Returns:
            最佳值或 None
        """
        values = self.metrics.get(name, [])
        if not values:
            return None
        return min(values) if mode == "min" else max(values)

    def get_stats(self, name: str) -> Dict[str, float]:
        """获取指标统计信息。"""
        values = self.metrics.get(name, [])
        if not values:
            return {}

        return {
            "count": len(values),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "latest": values[-1],
            "best": min(values),
        }

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MetricsHistory":
        """从字典创建。"""
        return cls(
            metrics=data.get("metrics", {}),
            timestamps=data.get("timestamps", []),
            steps=data.get("steps", []),
            epochs=data.get("epochs", []),
        )


class WandbLogger:
    """WandB 日志记录器封装。"""

    def __init__(
        self,
        project_name: str = "lingmao_moyun",
        experiment_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        entity: Optional[str] = None,
        tags: Optional[List[str]] = None,
        mode: str = "online",
    ):
        """初始化 WandB 日志记录器。

        Args:
            project_name: 项目名称
            experiment_name: 实验名称
            config: 实验配置
            entity: WandB entity/username
            tags: 实验标签
            mode: 记录模式 ('online', 'offline', 'disabled')
        """
        if not wandb_available:
            raise ImportError("wandb is not installed. Run: pip install wandb")

        self.project_name = project_name
        self.experiment_name = experiment_name
        self.mode = mode

        wandb.init(
            project=project_name,
            name=experiment_name,
            config=config,
            entity=entity,
            tags=tags,
            mode=mode,
        )

        self.run = wandb.run

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        epoch: Optional[int] = None,
    ) -> None:
        """记录指标。

        Args:
            metrics: 指标字典
            step: 步数
            epoch: 轮次
        """
        log_dict = metrics.copy()
        if step is not None:
            log_dict["global_step"] = step
        if epoch is not None:
            log_dict["epoch"] = epoch

        self.run.log(log_dict, step=step)

    def log_summary(self, summary_metrics: Dict[str, float]) -> None:
        """记录摘要指标。"""
        for key, value in summary_metrics.items():
            self.run.summary[key] = value

    def watch_model(self, model: Any, log: str = "gradients") -> None:
        """监视模型参数。"""
        self.run.watch(model, log=log)

    def finish(self) -> None:
        """结束 WandB 运行。"""
        self.run.finish()

    def __enter__(self) -> "WandbLogger":
        """上下文管理器入口。"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """上下文管理器出口。"""
        self.finish()


class TensorBoardLogger:
    """TensorBoard 日志记录器封装。"""

    def __init__(
        self,
        log_dir: str = "runs",
        experiment_name: Optional[str] = None,
        flush_interval: int = 30,
    ):
        """初始化 TensorBoard 日志记录器。

        Args:
            log_dir: 日志目录
            experiment_name: 实验名称
            flush_interval: 刷新间隔（秒）
        """
        if SummaryWriter is None:
            raise ImportError(
                "TensorBoard is not available. "
                "Run: pip install tensorboard"
            )

        self.log_dir = Path(log_dir)

        if experiment_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_subdir = self.log_dir / f"{experiment_name}_{timestamp}"
        else:
            log_subdir = self.log_dir / datetime.now().strftime("%Y%m%d_%H%M%S")

        self.log_dir = log_subdir
        self.writer = SummaryWriter(
            str(self.log_dir),
            flush_secs=flush_interval,
        )

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: int,
        epoch: Optional[int] = None,
    ) -> None:
        """记录指标。

        Args:
            metrics: 指标字典
            step: 全局步数
            epoch: 当前轮次
        """
        for name, value in metrics.items():
            self.writer.add_scalar(name, value, step)

        if epoch is not None:
            self.writer.add_scalar("epoch", epoch, step)

    def log_scalars(
        self,
        main_tag: str,
        tag_scalar_dict: Dict[str, float],
        step: int,
    ) -> None:
        """记录多个标量。"""
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def log_histogram(self, tag: str, values: Any, step: int) -> None:
        """记录直方图。"""
        if torch is not None and isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        self.writer.add_histogram(tag, values, step)

    def log_model_graph(self, model: Any, input_to_model: Any) -> None:
        """记录模型计算图。"""
        self.writer.add_graph(model, input_to_model)

    def log_text(self, tag: str, text: str, step: int) -> None:
        """记录文本。"""
        self.writer.add_text(tag, text, step)

    def log_figure(self, tag: str, figure: Any, step: int) -> None:
        """记录图形。"""
        self.writer.add_figure(tag, figure, step)

    def finish(self) -> None:
        """结束记录。"""
        self.writer.close()

    def __enter__(self) -> "TensorBoardLogger":
        """上下文管理器入口。"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """上下文管理器出口。"""
        self.finish()


class MetricsLogger:
    """综合指标日志记录器。"""

    def __init__(
        self,
        experiment_name: str,
        experiment_id: Optional[str] = None,
        log_dir: str = "logs",
        use_wandb: bool = False,
        use_tensorboard: bool = True,
        wandb_config: Optional[Dict[str, Any]] = None,
        wandb_tags: Optional[List[str]] = None,
    ):
        """初始化指标日志记录器。

        Args:
            experiment_name: 实验名称
            experiment_id: 实验 ID
            log_dir: 本地日志目录
            use_wandb: 是否使用 WandB
            use_tensorboard: 是否使用 TensorBoard
            wandb_config: WandB 配置
            wandb_tags: WandB 标签
        """
        self.experiment_name = experiment_name
        self.experiment_id = experiment_id or experiment_name
        self.log_dir = Path(log_dir) / self.experiment_id
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_history = MetricsHistory()
        self.logger = logging.getLogger(f"LingmaoMoyun.Metrics.{experiment_name}")

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(asctime)s - %(name)s - %(message)s")
            )
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        self.wandb_logger: Optional[WandbLogger] = None
        self.tensorboard_logger: Optional[TensorBoardLogger] = None

        if use_wandb and wandb_available:
            try:
                self.wandb_logger = WandbLogger(
                    project_name="lingmao_moyun",
                    experiment_name=experiment_name,
                    config=wandb_config,
                    tags=wandb_tags,
                )
                self.logger.info("WandB logging enabled")
            except Exception as e:
                self.logger.warning(f"Failed to initialize WandB: {e}")

        if use_tensorboard and SummaryWriter is not None:
            try:
                self.tensorboard_logger = TensorBoardLogger(
                    log_dir=str(self.log_dir / "tensorboard"),
                    experiment_name=experiment_name,
                )
                self.logger.info("TensorBoard logging enabled")
            except Exception as e:
                self.logger.warning(f"Failed to initialize TensorBoard: {e}")

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        verbose: bool = True,
    ) -> None:
        """记录指标。

        Args:
            metrics: 指标字典
            step: 步数
            epoch: 轮次
            verbose: 是否打印详细信息
        """
        current_step = step if step is not None else len(self.metrics_history.steps)
        self.metrics_history.add(metrics, step=current_step, epoch=epoch)

        if self.wandb_logger:
            try:
                self.wandb_logger.log_metrics(metrics, step=current_step, epoch=epoch)
            except Exception as e:
                self.logger.warning(f"WandB logging failed: {e}")

        if self.tensorboard_logger:
            try:
                self.tensorboard_logger.log_metrics(
                    metrics, step=current_step, epoch=epoch
                )
            except Exception as e:
                self.logger.warning(f"TensorBoard logging failed: {e}")

        if verbose:
            metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            step_info = f"Step {current_step}"
            if epoch is not None:
                step_info += f" (Epoch {epoch})"
            self.logger.info(f"{step_info} | {metrics_str}")

    def log_summary(self, summary_metrics: Optional[Dict[str, float]] = None) -> None:
        """记录摘要指标。

        Args:
            summary_metrics: 要记录的摘要指标
        """
        if summary_metrics is None:
            summary_metrics = {}

            for name, values in self.metrics_history.metrics.items():
                if values:
                    summary_metrics[f"{name}_best"] = min(values)
                    summary_metrics[f"{name}_final"] = values[-1]
                    summary_metrics[f"{name}_mean"] = float(np.mean(values))

        if self.wandb_logger:
            try:
                self.wandb_logger.log_summary(summary_metrics)
            except Exception as e:
                self.logger.warning(f"WandB summary logging failed: {e}")

        self._save_metrics_to_file(summary_metrics)

    def _save_metrics_to_file(
        self,
        summary_metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """保存指标到文件。"""
        metrics_file = self.log_dir / "metrics.json"
        history_file = self.log_dir / "metrics_history.json"

        history_data = self.metrics_history.to_dict()

        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(history_data, f, indent=2)

        if summary_metrics:
            summary_file = self.log_dir / "metrics_summary.json"
            with open(summary_file, "w", encoding="utf-8") as f:
                json.dump(summary_metrics, f, indent=2)

    def watch_model(self, model: Any) -> None:
        """监视模型参数。"""
        if self.wandb_logger:
            try:
                self.wandb_logger.watch_model(model)
            except Exception as e:
                self.logger.warning(f"Failed to watch model: {e}")

        if self.tensorboard_logger and torch is not None:
            try:
                dummy_input = torch.zeros(1, 64, dtype=torch.long)
                self.tensorboard_logger.log_model_graph(model, dummy_input)
            except Exception as e:
                self.logger.warning(f"Failed to log model graph: {e}")

    def finish(self) -> None:
        """结束日志记录。"""
        self.log_summary()

        if self.wandb_logger:
            try:
                self.wandb_logger.finish()
            except Exception as e:
                self.logger.warning(f"Failed to finish WandB: {e}")

        if self.tensorboard_logger:
            try:
                self.tensorboard_logger.finish()
            except Exception as e:
                self.logger.warning(f"Failed to finish TensorBoard: {e}")

        self.logger.info(f"Metrics logged to {self.log_dir}")

    def __enter__(self) -> "MetricsLogger":
        """上下文管理器入口。"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """上下文管理器出口。"""
        self.finish()

    def get_history(self, metric_name: str) -> List[float]:
        """获取指标历史。"""
        return self.metrics_history.get_metric(metric_name)

    def get_latest(self, metric_name: str) -> Optional[float]:
        """获取最新指标值。"""
        return self.metrics_history.get_latest(metric_name)

    def get_best(self, metric_name: str, mode: str = "min") -> Optional[float]:
        """获取最佳指标值。"""
        return self.metrics_history.get_best(metric_name, mode)

    def get_stats(self, metric_name: str) -> Dict[str, float]:
        """获取指标统计。"""
        return self.metrics_history.get_stats(metric_name)
