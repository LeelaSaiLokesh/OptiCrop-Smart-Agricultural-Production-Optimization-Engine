"""
logistic_regression_model.py - OptiCrop Logistic Regression
============================================================
Project : OptiCrop - Smart Agricultural Production Optimization Engine
Phase   : Machine Learning Model Development
Version : 1.0

THEORETICAL OVERVIEW:
    Logistic Regression models the log-odds of each class as a linear
    combination of features. Extended to multi-class via the 'multinomial'
    softmax formulation (or one-vs-rest). It learns a weight vector W for
    each class such that: P(y=k|x) = softmax(W_k . x + b_k).

AGRICULTURAL RELEVANCE:
    Provides a linear baseline — if LR achieves high accuracy, the crop
    boundaries are linearly separable in the scaled 15-feature space.
    Low accuracy implies non-linear boundaries (motivating tree models).
    Coefficient magnitudes reveal which soil/climate factors have the
    strongest linear association with each crop class.

ADVANTAGES:
    - Fast training; scales well with more data
    - Probabilistic outputs (confidence scores per class)
    - Coefficients are directly interpretable
    - Regularization (C) controls overfitting

LIMITATIONS:
    - Assumes linear decision boundaries
    - Sensitive to feature scale (mitigated by StandardScaler in pipeline)
    - Less accurate than ensemble methods on complex datasets
    - Requires careful C tuning for high-class-count problems
"""

import logging
import time
import numpy as np
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------
# HYPERPARAMETER CONFIGURATION — LOGISTIC REGRESSION
# -----------------------------------------------------------------
# solver='lbfgs'  : Best general-purpose solver for multi-class;
#                   uses limited-memory BFGS; handles multinomial loss.
# max_iter=1000   : Generous upper bound; allows convergence on
#                   this 15-feature, 22-class problem.
# C=1.0           : Inverse regularization strength; default starting
#                   point — balanced between underfitting (low C) and
#                   overfitting (high C).
# multi_class='multinomial' : Proper softmax probability estimation for
#                             all 22 classes simultaneously.
# random_state=42 : Ensures deterministic weight initialization.
LOGISTIC_REGRESSION_PARAMS = {
    "solver":       "lbfgs",
    "max_iter":     1000,
    "C":            1.0,
    "random_state": 42
}
# Note: multi_class parameter removed in sklearn 1.4+ (multinomial is now default
# for lbfgs solver with multiple classes). Set explicitly if using older sklearn.


def build_logistic_regression() -> LogisticRegression:
    """
    Instantiate a LogisticRegression classifier with production parameters.

    Returns:
        Unfitted LogisticRegression instance.
    """
    model = LogisticRegression(**LOGISTIC_REGRESSION_PARAMS)
    logger.info(f"LogisticRegression instantiated | params: {LOGISTIC_REGRESSION_PARAMS}")
    return model


def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray
) -> tuple:
    """
    Train Logistic Regression on the preprocessed training set.

    Args:
        X_train : Scaled training feature array (1760 x 15).
        y_train : Encoded integer target array (1760,).

    Returns:
        Tuple of (fitted_model, training_time_seconds).
    """
    model = build_logistic_regression()

    logger.info("Training LogisticRegression...")
    t_start = time.perf_counter()
    model.fit(X_train, y_train)
    training_time = time.perf_counter() - t_start

    logger.info(f"LogisticRegression trained in {training_time:.4f}s")
    return model, training_time
