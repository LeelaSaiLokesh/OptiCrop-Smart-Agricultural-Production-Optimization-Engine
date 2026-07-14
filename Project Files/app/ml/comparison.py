"""
comparison.py - OptiCrop Model Comparison Module
=================================================
Project : OptiCrop - Smart Agricultural Production Optimization Engine
Phase   : Machine Learning Model Development
Version : 1.0

Aggregates all model evaluation results into a professional comparison
table and generates multi-model visualisation charts.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

logger = logging.getLogger(__name__)

FIGURES_DIR = Path(__file__).resolve().parent.parent.parent / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Complexity and interpretability ratings (expert-assigned)
MODEL_PROPERTIES = {
    "Logistic Regression": {
        "complexity":       "Low",
        "interpretability": "High",
        "scalability":      "High",
        "deployment":       "Excellent"
    },
    "K-Nearest Neighbors": {
        "complexity":       "Low-Medium",
        "interpretability": "Medium",
        "scalability":      "Low",
        "deployment":       "Fair"
    },
    "Decision Tree": {
        "complexity":       "Medium",
        "interpretability": "Very High",
        "scalability":      "Medium",
        "deployment":       "Good"
    },
    "Random Forest": {
        "complexity":       "High",
        "interpretability": "Medium",
        "scalability":      "Medium-High",
        "deployment":       "Very Good"
    }
}


def build_comparison_dataframe(all_metrics: list) -> pd.DataFrame:
    """
    Build a structured comparison DataFrame from a list of metric dictionaries.

    Args:
        all_metrics : List of metric dicts from evaluate_model().
                      Each dict must contain: model_name, accuracy, precision,
                      recall, f1, prediction_time.
                      Training times passed separately via training_times dict.

    Returns:
        DataFrame with one row per model and all comparison columns.
    """
    rows = []
    for m in all_metrics:
        name = m["model_name"]
        props = MODEL_PROPERTIES.get(name, {})
        rows.append({
            "Model":              name,
            "Accuracy":           m["accuracy"],
            "Precision":          m["precision"],
            "Recall":             m["recall"],
            "F1 Score":           m["f1"],
            "Train Time (s)":     m.get("training_time", "N/A"),
            "Predict Time (s)":   m["prediction_time"],
            "Complexity":         props.get("complexity", "N/A"),
            "Interpretability":   props.get("interpretability", "N/A"),
            "Scalability":        props.get("scalability", "N/A"),
            "Deployment Fit":     props.get("deployment", "N/A")
        })

    df = pd.DataFrame(rows).set_index("Model")
    return df


def print_comparison_table(df: pd.DataFrame) -> None:
    """
    Print the model comparison table in a formatted tabular layout.

    Args:
        df: DataFrame from build_comparison_dataframe().
    """
    print("\n" + "=" * 90)
    print("  OPTICROP — MODEL COMPARISON TABLE")
    print("=" * 90)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)
    pd.set_option("display.float_format", "{:.4f}".format)
    print(df.to_string())
    print("=" * 90)


def plot_metric_comparison(
    df: pd.DataFrame,
    metrics: list = None,
    save: bool = True
) -> None:
    """
    Generate a grouped bar chart comparing all models across 4 metrics.

    Args:
        df      : Comparison DataFrame.
        metrics : List of metric column names to plot.
        save    : If True, saves PNG to reports/figures/.
    """
    if metrics is None:
        metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]

    plot_df = df[metrics]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    palette = ["#2d6a4f", "#40916c", "#74c69d", "#95d5b2"]

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        colours = [palette[i % len(palette)] for i in range(len(plot_df))]
        bars = ax.bar(
            plot_df.index, plot_df[metric],
            color=colours, edgecolor="white", width=0.6
        )
        ax.set_title(f"{metric} Comparison", fontsize=12, fontweight="bold")
        ax.set_ylabel(metric, fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis="x", rotation=15, labelsize=9)
        ax.tick_params(axis="y", labelsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.spines[["top", "right"]].set_visible(False)

        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.005, f"{h:.3f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold"
            )

    fig.suptitle(
        "OptiCrop — Model Performance Comparison",
        fontsize=15, fontweight="bold", y=1.01
    )
    plt.tight_layout()

    if save:
        fname = FIGURES_DIR / "model_comparison_metrics.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        logger.info(f"Comparison chart saved: {fname}")

    plt.show()
    plt.close()


def plot_accuracy_ranking(df: pd.DataFrame, save: bool = True) -> None:
    """
    Plot a horizontal bar chart ranking models by accuracy.

    Args:
        df   : Comparison DataFrame.
        save : If True, saves PNG.
    """
    sorted_df = df["Accuracy"].sort_values()
    colours = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(sorted_df)))

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(sorted_df.index, sorted_df.values, color=colours,
                   edgecolor="white", height=0.6)

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{w:.4f}", va="center", ha="left", fontsize=10, fontweight="bold")

    ax.set_title("Model Accuracy Ranking", fontsize=14, fontweight="bold")
    ax.set_xlabel("Accuracy Score", fontsize=11)
    ax.set_xlim(0, 1.08)
    ax.tick_params(labelsize=10)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    if save:
        fname = FIGURES_DIR / "accuracy_ranking.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        logger.info(f"Accuracy ranking saved: {fname}")

    plt.show()
    plt.close()


def plot_training_time_comparison(df: pd.DataFrame, save: bool = True) -> None:
    """
    Plot training time comparison across all models (log scale).

    Args:
        df   : Comparison DataFrame.
        save : If True, saves PNG.
    """
    time_data = df["Train Time (s)"].copy()
    if (time_data == "N/A").any():
        time_data = time_data[time_data != "N/A"].astype(float)

    fig, ax = plt.subplots(figsize=(9, 5))
    colours = ["#264653", "#2a9d8f", "#e9c46a", "#e76f51"]
    bars = ax.bar(time_data.index, time_data.values.astype(float),
                  color=colours[:len(time_data)], edgecolor="white")

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.001,
                f"{h:.3f}s", ha="center", va="bottom", fontsize=9)

    ax.set_title("Training Time Comparison", fontsize=14, fontweight="bold")
    ax.set_ylabel("Time (seconds)", fontsize=11)
    ax.tick_params(axis="x", rotation=15, labelsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    if save:
        fname = FIGURES_DIR / "training_time_comparison.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        logger.info(f"Training time chart saved: {fname}")

    plt.show()
    plt.close()
