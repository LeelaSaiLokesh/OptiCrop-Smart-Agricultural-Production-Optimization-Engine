"""
metrics.py - OptiCrop Evaluation Metrics Module
================================================
Project : OptiCrop - Smart Agricultural Production Optimization Engine
Phase   : Machine Learning Model Development
Version : 1.0

Centralizes all classification metric computations.
Imported by evaluation.py and comparison.py.
"""

import logging
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)

logger = logging.getLogger(__name__)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model"
) -> dict:
    """
    Compute a full suite of classification evaluation metrics.

    Args:
        y_true     : Ground truth integer labels.
        y_pred     : Predicted integer labels from model.predict().
        model_name : Display name for logging (e.g. 'Random Forest').

    Returns:
        Dictionary with keys:
            accuracy, precision, recall, f1, confusion_matrix,
            classification_report, model_name
    """
    accuracy  = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall    = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1        = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    cm        = confusion_matrix(y_true, y_pred)
    report    = classification_report(y_true, y_pred, zero_division=0)

    logger.info(
        f"{model_name:25s} | "
        f"Acc: {accuracy:.4f} | "
        f"Prec: {precision:.4f} | "
        f"Rec: {recall:.4f} | "
        f"F1: {f1:.4f}"
    )

    return {
        "model_name":            model_name,
        "accuracy":              round(accuracy,  4),
        "precision":             round(precision, 4),
        "recall":                round(recall,    4),
        "f1":                    round(f1,        4),
        "confusion_matrix":      cm,
        "classification_report": report
    }


def print_metric_summary(metrics: dict) -> None:
    """
    Print a formatted metric summary to stdout.

    Args:
        metrics: Dictionary returned by compute_classification_metrics().
    """
    print("\n" + "=" * 60)
    print(f"  MODEL: {metrics['model_name']}")
    print("=" * 60)
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1 Score  : {metrics['f1']:.4f}")
    print("-" * 60)
    print("  Classification Report:")
    print(metrics['classification_report'])
    print("=" * 60)
