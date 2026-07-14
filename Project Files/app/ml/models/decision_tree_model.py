"""
decision_tree_model.py - OptiCrop Decision Tree Classifier
===========================================================
Project : OptiCrop - Smart Agricultural Production Optimization Engine
Phase   : Machine Learning Model Development
Version : 1.0

THEORETICAL OVERVIEW:
    Decision Tree recursively partitions the feature space by selecting
    the feature and threshold that maximally reduces impurity at each node.
    Impurity is measured by the Gini coefficient or Information Gain (entropy).

    Gini impurity: G = 1 - sum(p_k^2) for all classes k
    The tree grows until nodes are pure or a stopping criterion is met.
    Predictions are made by traversing from root to leaf node.

AGRICULTURAL RELEVANCE:
    Decision Trees produce IF-THEN rules that are directly interpretable:
    "IF Humidity > 80 AND Rainfall > 200 AND N < 40 THEN RICE"
    This maps precisely to farmer decision-making logic and can be
    printed as a ruleset that agronomic extension workers can use in the
    field WITHOUT a computer — the ultimate interpretability.

ADVANTAGES:
    - Fully interpretable: visual tree and ruleset exportable
    - No feature scaling required (partition-based, not distance-based)
    - Handles non-linear and interaction effects naturally
    - Fast training and prediction
    - Feature importance natively available

LIMITATIONS:
    - Prone to overfitting when max_depth is unlimited
    - Unstable: small data changes can produce completely different trees
    - Greedy algorithm: does not guarantee globally optimal tree
    - High variance if not depth-controlled
"""

import logging
import time
import numpy as np
from sklearn.tree import DecisionTreeClassifier

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------
# HYPERPARAMETER CONFIGURATION — DECISION TREE
# -----------------------------------------------------------------
# criterion='gini'       : Gini impurity; computationally efficient vs
#                          entropy; produces similar quality splits.
# splitter='best'        : Always selects the globally best split at each
#                          node (vs 'random' which is faster but noisier).
# max_depth=10           : Hard depth cap prevents memorizing training data;
#                          22 classes need sufficient depth to separate;
#                          depth 10 allows up to 2^10=1024 leaves.
# min_samples_split=5    : Minimum 5 samples required to attempt a split;
#                          prevents tiny leaf nodes that overfit to noise.
# min_samples_leaf=2     : Each leaf must represent at least 2 samples;
#                          eliminates singleton leaves.
# class_weight='balanced': Equal misclassification penalty across all 22
#                           classes (dataset is already balanced; included
#                           as a safety measure for future imbalanced data).
# random_state=42        : Ties in split quality resolved deterministically.
DECISION_TREE_PARAMS = {
    "criterion":        "gini",
    "splitter":         "best",
    "max_depth":        10,
    "min_samples_split": 5,
    "min_samples_leaf":  2,
    "class_weight":     "balanced",
    "random_state":     42
}


def build_decision_tree() -> DecisionTreeClassifier:
    """
    Instantiate a DecisionTreeClassifier with production parameters.

    Returns:
        Unfitted DecisionTreeClassifier instance.
    """
    model = DecisionTreeClassifier(**DECISION_TREE_PARAMS)
    logger.info(f"DecisionTreeClassifier instantiated | params: {DECISION_TREE_PARAMS}")
    return model


def train_decision_tree(
    X_train: np.ndarray,
    y_train: np.ndarray
) -> tuple:
    """
    Train a Decision Tree on the preprocessed training set.

    Args:
        X_train : Training feature array (scaled or unscaled; DT is invariant).
        y_train : Encoded integer target array (1760,).

    Returns:
        Tuple of (fitted_model, training_time_seconds).
    """
    model = build_decision_tree()

    logger.info("Training DecisionTreeClassifier...")
    t_start = time.perf_counter()
    model.fit(X_train, y_train)
    training_time = time.perf_counter() - t_start

    logger.info(
        f"DecisionTree trained in {training_time:.4f}s | "
        f"tree depth: {model.get_depth()} | "
        f"n_leaves: {model.get_n_leaves()}"
    )
    return model, training_time


def extract_decision_rules(model: DecisionTreeClassifier, feature_names: list) -> str:
    """
    Export the decision tree as a human-readable text ruleset.

    Args:
        model         : Fitted DecisionTreeClassifier.
        feature_names : List of feature column names in training order.

    Returns:
        String containing the full tree structure in text format.
    """
    from sklearn.tree import export_text
    rules = export_text(model, feature_names=feature_names)
    return rules
