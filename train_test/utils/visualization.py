import os
import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
import matplotlib.pyplot as plt
import numpy as np
import torch


_session_timestamp: Optional[str] = None


def get_session_timestamp() -> str:
    global _session_timestamp
    if _session_timestamp is None:
        _session_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return _session_timestamp


def reset_session_timestamp() -> None:
    global _session_timestamp
    _session_timestamp = None


def get_visualization_dir(pic_dir: str, create_if_not_exists: bool = True) -> str:
    timestamp = get_session_timestamp()
    vis_dir = os.path.join(pic_dir, timestamp)
    if create_if_not_exists:
        os.makedirs(vis_dir, exist_ok=True)
    return vis_dir


def plot_train_val_metrics(
    train_values: List[float],
    val_values: List[float],
    save_dir: str,
    property_name: str,
    metric_name: str,
    train_color: str = "blue",
    val_color: str = "orange",
    xlabel: str = "Epoch",
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    filename: Optional[str] = None,
) -> str:
    if ylabel is None:
        ylabel = metric_name.upper()
    if title is None:
        title = f"{property_name} - {metric_name.upper()} Over Epochs"
    if filename is None:
        filename = f"{property_name}_{metric_name}.png"

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    epochs = range(1, len(train_values) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_values, label="Train", color=train_color, linewidth=2)
    plt.plot(epochs, val_values, label="Validation", color=val_color, linewidth=2)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    return save_path


def plot_train_val_test_metrics(
    train_values: List[float],
    val_values: List[float],
    test_value: float,
    save_dir: str,
    property_name: str,
    metric_name: str,
    train_color: str = "blue",
    val_color: str = "orange",
    test_color: str = "green",
    xlabel: str = "Epoch",
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    filename: Optional[str] = None,
) -> str:
    if ylabel is None:
        ylabel = metric_name.upper()
    if title is None:
        title = f"{property_name} - {metric_name.upper()} Comparison"
    if filename is None:
        filename = f"{property_name}_{metric_name}_comparison.png"

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    epochs = range(1, len(train_values) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_values, label="Train", color=train_color, linewidth=2)
    plt.plot(epochs, val_values, label="Validation", color=val_color, linewidth=2)
    plt.axhline(
        y=test_value, color=test_color, linestyle="--", linewidth=2, label=f"Test: {test_value:.6f}"
    )
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    return save_path


def plot_prediction_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_dir: str,
    property_name: str,
    title: Optional[str] = None,
    filename: Optional[str] = None,
) -> str:
    if title is None:
        title = f"{property_name} - Prediction vs True Values"
    if filename is None:
        filename = f"{property_name}_prediction_scatter.png"

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.cpu().numpy()

    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    plt.figure(figsize=(10, 8))
    plt.scatter(y_true_flat, y_pred_flat, alpha=0.6, s=20)

    min_val = min(y_true_flat.min(), y_pred_flat.min())
    max_val = max(y_true_flat.max(), y_pred_flat.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect Prediction")

    plt.xlabel("True Values", fontsize=12)
    plt.ylabel("Predicted Values", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    return save_path


def plot_all_train_val_metrics(
    train_metrics: Dict[str, List[float]],
    val_metrics: Dict[str, List[float]],
    save_dir: str,
    property_name: str,
) -> Dict[str, str]:
    saved_paths = {}

    metric_configs = {
        "loss": {"train_color": "blue", "val_color": "orange"},
        "mae": {"train_color": "red", "val_color": "green"},
        "mse": {"train_color": "purple", "val_color": "brown"},
        "pointwise_mae": {"train_color": "cyan", "val_color": "magenta"},
        "mean_fnorm_percent_error": {"train_color": "olive", "val_color": "teal"},
        "batchwise_percent_error": {"train_color": "navy", "val_color": "maroon"},
    }

    for metric_name, config in metric_configs.items():
        train_key = f"train_{metric_name}"
        val_key = f"val_{metric_name}"

        if train_key in train_metrics and val_key in val_metrics:
            path = plot_train_val_metrics(
                train_values=train_metrics[train_key],
                val_values=val_metrics[val_key],
                save_dir=save_dir,
                property_name=property_name,
                metric_name=metric_name,
                train_color=config["train_color"],
                val_color=config["val_color"],
            )
            saved_paths[metric_name] = path

    return saved_paths


def plot_all_train_val_test_metrics(
    train_metrics: Dict[str, List[float]],
    val_metrics: Dict[str, List[float]],
    test_metrics: Dict[str, float],
    save_dir: str,
    property_name: str,
) -> Dict[str, str]:
    saved_paths = {}

    metric_configs = {
        "loss": {"train_color": "blue", "val_color": "orange", "test_color": "green"},
        "mae": {"train_color": "red", "val_color": "green", "test_color": "blue"},
        "mse": {"train_color": "purple", "val_color": "brown", "test_color": "cyan"},
        "pointwise_mae": {"train_color": "cyan", "val_color": "magenta", "test_color": "olive"},
        "mean_fnorm_percent_error": {"train_color": "olive", "val_color": "teal", "test_color": "navy"},
        "batchwise_percent_error": {"train_color": "navy", "val_color": "maroon", "test_color": "purple"},
    }

    for metric_name, config in metric_configs.items():
        train_key = f"train_{metric_name}"
        val_key = f"val_{metric_name}"
        test_key = metric_name

        if (
            train_key in train_metrics
            and val_key in val_metrics
            and test_key in test_metrics
        ):
            path = plot_train_val_test_metrics(
                train_values=train_metrics[train_key],
                val_values=val_metrics[val_key],
                test_value=test_metrics[test_key],
                save_dir=save_dir,
                property_name=property_name,
                metric_name=metric_name,
                train_color=config["train_color"],
                val_color=config["val_color"],
                test_color=config["test_color"],
            )
            saved_paths[metric_name] = path

    return saved_paths
