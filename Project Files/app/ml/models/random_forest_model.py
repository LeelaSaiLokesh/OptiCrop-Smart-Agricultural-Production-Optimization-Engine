"""
random_forest_model.py - OptiCrop Random Forest Classifier
===========================================================
Project : OptiCrop - Smart Agricultural Production Optimization Engine
Phase   : Machine Learning Model Development
Version : 1.0

THEORETICAL OVERVIEW:
    Random Forest is a bagging (Bootstrap AGGregatING) ensemble of
    Decision Trees. Each tree is trained on a bootstrap sample (random
    rows with replacement) and uses a random feature subset at each split
    (sqrt(n_features) by default). The final prediction is a majority vote
    across all trees.

    Key mechanisms:
    - Bootstrap sampling: reduces variance by decorrelating trees
    - Random feature subsets: prevents dominant features from controlling
      all splits; forces the ensemble to use diverse features
    - Majority vote: averages out individual tree errors

    Feature importance: I(f) = mean decrease in node impurity from
    feature f across all trees and all nodes.

AGRICULTURAL RELEVANCE:
    Random Forest is the industry benchmark for tabular agricultural
    datasets. It handles the NPK-Humidity-Rainfall interaction effects
    that single decision boundaries cannot capture. Its built-in feature
    importance directly quantifies which soil/climate factor drives crop
    discrimination — providing actionable insight for agronomists.

ADVANTAGES:
    - State-of-the-art accuracy on tabular data
    - Robust to outliers and noise (ensembling averages them out)
    - Built-in feature importance (MDI: Mean Decrease in Impurity)
    - No feature scaling required
    - Low overfitting risk due to bagging + random feature selection
    - Parallelizable across all CPU cores

LIMITATIONS:
    - Less interpretable than a single Decision Tree
    - Slower prediction than LR/KNN for very large forests
    - Higher memory footprint than single models
    - Feature importance can be biased toward high-cardinality features
"""

import logging
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------
# HYPERPARAMETER CONFIGURATION — RANDOM FOREST
# -----------------------------------------------------------------
# n_estimators=200    : 200 trees provide stable, reproducible predictions;
#                       beyond ~300 trees, accuracy gains plateau while
#                       training time grows linearly.
# max_depth=None      : Trees grow until pure leaves; bagging + random
#                       features prevent overfitting without depth limits.
# min_samples_split=4 : Node split requires at least 4 samples; reduces
#                       splits on very small subgroups.
# min_samples_leaf=2  : Each leaf must have >= 2 samples; prevents
#                       zero-variance singleton leaves.
# max_features='sqrt' : At each split, consider sqrt(15) ~ 4 features;
#                       the gold standard for classification tasks.
# bootstrap=True      : Enable bootstrap sampling (core of bagging).
# class_weight='balanced_subsample': Re-computes class weights per bootstrap
#                       sample; most robust for balanced-but-varied classes.
# n_jobs=-1           : Use all CPU cores in parallel.
# random_state=42     : Full reproducibility.
RANDOM_FOREST_PARAMS = {
    "n_estimators":       200,
    "max_depth":          None,
    "min_samples_split":  4,
    "min_samples_leaf":   2,
    "max_features":       "sqrt",
    "bootstrap":          True,
    "class_weight":       "balanced_subsample",
    "n_jobs":             -1,
    "random_state":       42
}


def build_random_forest() -> RandomForestClassifier:
    """
    Instantiate a RandomForestClassifier with production parameters.

    Returns:
        Unfitted RandomForestClassifier instance.
    """
    model = RandomForestClassifier(**RANDOM_FOREST_PARAMS)
    logger.info(f"RandomForestClassifier instantiated | params: {RANDOM_FOREST_PARAMS}")
    return model


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray
) -> tuple:
    """
    Train Random Forest on the preprocessed training set.

    Args:
        X_train : Training feature array (1760 x 15).
        y_train : Encoded integer target array (1760,).

    Returns:
        Tuple of (fitted_model, training_time_seconds).
    """
    model = build_random_forest()

    logger.info(f"Training RandomForestClassifier ({RANDOM_FOREST_PARAMS['n_estimators']} trees)...")
    t_start = time.perf_counter()
    model.fit(X_train, y_train)
    training_time = time.perf_counter() - t_start

    logger.info(f"RandomForest trained in {training_time:.4f}s")
    return model, training_time


def get_feature_importance_df(
    model: RandomForestClassifier,
    feature_names: list
) -> pd.DataFrame:
    """
    Extract and rank feature importances from a fitted Random Forest.

    Args:
        model         : Fitted RandomForestClassifier.
        feature_names : List of feature names matching training column order.

    Returns:
        DataFrame sorted by importance descending with columns
        ['Feature', 'Importance', 'Importance_%'].
    """
    importances = model.feature_importances_
    std         = np.std(
        [tree.feature_importances_ for tree in model.estimators_], axis=0
    )

    df = pd.DataFrame({
        "Feature":      feature_names,
        "Importance":   np.round(importances, 6),
        "Std":          np.round(std, 6),
        "Importance_%": np.round(importances / importances.sum() * 100, 2)
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    logger.info(f"Top 5 features by RF importance:\n{df.head(5).to_string()}")
    return df
