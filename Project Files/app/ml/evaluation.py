"""
evaluation.py - OptiCrop Model Evaluation Module
=================================================
Project : OptiCrop - Smart Agricultural Production Optimization Engine
Phase   : Machine Learning Model Development
Version : 1.0

Provides reusable evaluation functions for all supervised classifiers.
Generates confusion matrices, classification reports, and visual plots.
"""

import logging
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix

from app.ml.metrics import compute_classification_metrics, print_metric_summary

logger = logging.getLogger(__name__)

FIGURES_DIR = Path(__file__).resolve().parent.parent.parent / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    label_encoder=None
) -> dict:
    """
    Evaluate a fitted classifier on the test set.

    Computes accuracy, precision, recall, F1, confusion matrix,
    and measures prediction latency.

    Args:
        model        : Fitted sklearn classifier with .predict() method.
        X_test       : Scaled test feature array (440 x 15).
        y_test       : Encoded integer ground truth labels (440,).
        model_name   : Human-readable model name for display/logging.
        label_encoder: Optional fitted LabelEncoder for class name display.

    Returns:
        Complete metrics dictionary from compute_classification_metrics()
        plus 'prediction_time' key.
    """
    # Prediction with timing
    t_start = time.perf_counter()
    y_pred  = model.predict(X_test)
    pred_time = time.perf_counter() - t_start

    metrics = compute_classification_metrics(y_test, y_pred, model_name)
    metrics["prediction_time"] = round(pred_time, 6)
    metrics["y_pred"]          = y_pred

    print_metric_summary(metrics)
    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    class_names: list = None,
    save: bool = True
) -> None:
    """
    Plot and optionally save a colour-coded confusion matrix.

    Args:
        y_true      : Ground truth integer labels.
        y_pred      : Predicted integer labels.
        model_name  : Model name for plot title.
        class_names : List of crop name strings (22 items).
        save        : If True, saves PNG to reports/figures/.
    """
    cm   = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(16, 13))

    sns.heatmap(
        cm, annot=True, fmt="d",
        cmap="YlOrRd",
        xticklabels=class_names if class_names else "auto",
        yticklabels=class_names if class_names else "auto",
        linewidths=0.4, linecolor="white",
        annot_kws={"size": 7}, ax=ax
    )

    ax.set_title(
        f"Confusion Matrix — {model_name}",
        fontsize=14, fontweight="bold", pad=16
    )
    ax.set_xlabel("Predicted Crop", fontsize=11, labelpad=10)
    ax.set_ylabel("True Crop",      fontsize=11, labelpad=10)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0,  labelsize=8)
    plt.tight_layout()

    if save:
        fname = FIGURES_DIR / f"cm_{model_name.lower().replace(' ', '_')}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        logger.info(f"Confusion matrix saved: {fname}")

    plt.show()
    plt.close()


def plot_feature_importance(
    importance_df,
    model_name: str,
    top_n: int = 15,
    save: bool = True
) -> None:
    """
    Plot horizontal bar chart of feature importances.

    Args:
        importance_df : DataFrame with 'Feature' and 'Importance' columns.
        model_name    : Model name for plot title.
        top_n         : Number of top features to display.
        save          : If True, saves PNG to reports/figures/.
    """
    df_plot = importance_df.head(top_n).sort_values("Importance")

    fig, ax = plt.subplots(figsize=(10, 7))
    colours = plt.cm.RdYlGn(np.linspace(0.2, 0.85, len(df_plot)))

    ax.barh(
        df_plot["Feature"],
        df_plot["Importance"],
        color=colours,
        edgecolor="white",
        height=0.7
    )

    if "Std" in df_plot.columns:
        ax.barh(
            df_plot["Feature"],
            df_plot["Importance"],
            xerr=df_plot["Std"].values,
            color="none",
            ecolor="grey",
            capsize=3,
            height=0.7
        )

    ax.set_title(
        f"Feature Importance — {model_name} (Top {top_n})",
        fontsize=14, fontweight="bold", pad=14
    )
    ax.set_xlabel("Mean Decrease in Impurity (Importance)", fontsize=11)
    ax.set_ylabel("Feature", fontsize=11)
    ax.tick_params(labelsize=10)
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()

    if save:
        fname = FIGURES_DIR / f"fi_{model_name.lower().replace(' ', '_')}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        logger.info(f"Feature importance plot saved: {fname}")

    plt.show()
    plt.close()
