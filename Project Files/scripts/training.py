"""
training.py - OptiCrop Master Training Orchestrator
====================================================
Project : OptiCrop - Smart Agricultural Production Optimization Engine
Phase   : Machine Learning Model Development
Version : 1.0

Central training script that imports all model modules and executes
a unified training pipeline for all 5 algorithms.

Usage:
    python scripts/training.py
    # or import run_full_training_pipeline() directly
"""

import sys
import logging
import pickle
from pathlib import Path

import numpy as np

# --- Path setup ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# --- Internal imports ---
from scripts.preprocessing import run_preprocessing_pipeline
from app.ml.models.logistic_regression_model import train_logistic_regression
from app.ml.models.knn_model                import train_knn
from app.ml.models.decision_tree_model      import train_decision_tree
from app.ml.models.random_forest_model      import train_random_forest, get_feature_importance_df
from app.ml.models.kmeans_model             import train_kmeans, evaluate_kmeans
from app.ml.evaluation                      import evaluate_model, plot_confusion_matrix, plot_feature_importance
from app.ml.comparison                      import (
    build_comparison_dataframe, print_comparison_table,
    plot_metric_comparison, plot_accuracy_ranking, plot_training_time_comparison
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)


def run_full_training_pipeline() -> dict:
    """
    Execute the complete multi-algorithm training pipeline.

    Pipeline steps:
        1. Load preprocessed data from preprocessing pipeline
        2. Train all 4 supervised classifiers + K-Means
        3. Evaluate all supervised models on test set
        4. Generate confusion matrix plots per model
        5. Generate feature importance plots (DT, RF)
        6. Build and print comparison table
        7. Generate comparison visualisations
        8. Return all trained models and metrics

    Returns:
        Dictionary containing all trained models, evaluation metrics,
        comparison DataFrame, and label encoder.
    """
    logger.info("=" * 70)
    logger.info("OPTICROP — FULL ML TRAINING PIPELINE STARTED")
    logger.info("=" * 70)

    # ============================================================
    # STEP 1: Load preprocessed data
    # ============================================================
    logger.info("Loading preprocessed data...")
    data = run_preprocessing_pipeline()

    X_train_scaled = data["X_train_scaled"]
    X_test_scaled  = data["X_test_scaled"]
    y_train        = data["y_train"].values if hasattr(data["y_train"], "values") else data["y_train"]
    y_test         = data["y_test"].values  if hasattr(data["y_test"],  "values") else data["y_test"]
    label_encoder  = data["label_encoder"]
    feature_cols   = data["feature_columns"]
    class_names    = list(label_encoder.classes_)

    logger.info(f"Training set: {X_train_scaled.shape} | Test set: {X_test_scaled.shape}")
    logger.info(f"Feature columns ({len(feature_cols)}): {feature_cols}")

    # ============================================================
    # STEP 2: Train all models
    # ============================================================
    logger.info("\nTraining all models...")

    lr_model,  lr_time  = train_logistic_regression(X_train_scaled, y_train)
    knn_model, knn_time = train_knn(X_train_scaled, y_train)
    dt_model,  dt_time  = train_decision_tree(X_train_scaled, y_train)
    rf_model,  rf_time  = train_random_forest(X_train_scaled, y_train)
    km_model,  km_time  = train_kmeans(X_train_scaled)          # unsupervised

    trained_models = {
        "Logistic Regression": (lr_model,  lr_time),
        "K-Nearest Neighbors": (knn_model, knn_time),
        "Decision Tree":       (dt_model,  dt_time),
        "Random Forest":       (rf_model,  rf_time),
    }

    # ============================================================
    # STEP 3: Evaluate supervised models
    # ============================================================
    all_metrics = []
    for model_name, (model, train_time) in trained_models.items():
        logger.info(f"\n{'─'*50}\nEvaluating: {model_name}\n{'─'*50}")
        metrics = evaluate_model(model, X_test_scaled, y_test, model_name, label_encoder)
        metrics["training_time"] = round(train_time, 4)
        all_metrics.append(metrics)

    # ============================================================
    # STEP 4: Confusion matrices
    # ============================================================
    logger.info("Generating confusion matrix plots...")
    for metrics in all_metrics:
        plot_confusion_matrix(
            y_test, metrics["y_pred"],
            model_name=metrics["model_name"],
            class_names=class_names,
            save=True
        )

    # ============================================================
    # STEP 5: Feature importance (Decision Tree + Random Forest)
    # ============================================================
    logger.info("Generating feature importance plots...")

    # Decision Tree importance
    dt_fi = dict(zip(feature_cols, dt_model.feature_importances_))
    import pandas as pd
    dt_fi_df = pd.DataFrame(
        [{"Feature": k, "Importance": v, "Importance_%": round(v/sum(dt_fi.values())*100, 2)}
         for k, v in sorted(dt_fi.items(), key=lambda x: x[1], reverse=True)]
    )
    plot_feature_importance(dt_fi_df, "Decision Tree")

    # Random Forest importance (with std)
    rf_fi_df = get_feature_importance_df(rf_model, feature_cols)
    plot_feature_importance(rf_fi_df, "Random Forest")

    # ============================================================
    # STEP 6: Comparison table
    # ============================================================
    logger.info("Building model comparison table...")
    comp_df = build_comparison_dataframe(all_metrics)
    print_comparison_table(comp_df)

    # ============================================================
    # STEP 7: Comparison visualisations
    # ============================================================
    plot_metric_comparison(comp_df)
    plot_accuracy_ranking(comp_df)
    plot_training_time_comparison(comp_df)

    # ============================================================
    # STEP 8: K-Means exploratory evaluation
    # ============================================================
    km_metrics = evaluate_kmeans(km_model, X_test_scaled, y_test)
    logger.info(f"K-Means exploratory metrics: {km_metrics}")

    logger.info("=" * 70)
    logger.info("TRAINING PIPELINE COMPLETE")
    logger.info("=" * 70)

    return {
        "models": {
            "Logistic Regression": lr_model,
            "K-Nearest Neighbors": knn_model,
            "Decision Tree":       dt_model,
            "Random Forest":       rf_model,
            "K-Means":             km_model
        },
        "all_metrics":      all_metrics,
        "comparison_df":    comp_df,
        "label_encoder":    label_encoder,
        "feature_columns":  feature_cols,
        "dt_importance":    dt_fi_df,
        "rf_importance":    rf_fi_df,
        "kmeans_metrics":   km_metrics
    }


if __name__ == "__main__":
    results = run_full_training_pipeline()

    print("\n" + "=" * 60)
    print("RECOMMENDED MODEL FOR DEPLOYMENT")
    print("=" * 60)
    best = results["comparison_df"]["Accuracy"].idxmax()
    best_acc = results["comparison_df"].loc[best, "Accuracy"]
    print(f"  {best} — Accuracy: {best_acc:.4f}")
    print("=" * 60)
