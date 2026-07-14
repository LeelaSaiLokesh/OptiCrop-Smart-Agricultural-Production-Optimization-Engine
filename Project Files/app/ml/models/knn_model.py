"""
knn_model.py - OptiCrop K-Nearest Neighbors Classifier
=======================================================
Project : OptiCrop - Smart Agricultural Production Optimization Engine
Phase   : Machine Learning Model Development
Version : 1.0

THEORETICAL OVERVIEW:
    KNN is a non-parametric, instance-based algorithm. At prediction time,
    it computes the distance between the query point and all training points,
    selects the K nearest neighbours, and returns the majority class
    (distance-weighted or uniform vote). It has NO training phase — the
    entire training set is stored in memory (lazy learner).

    Distance metric: Minkowski p=2 (Euclidean).
    Decision rule:   Weighted vote inversely proportional to distance.

AGRICULTURAL RELEVANCE:
    Crops with similar NPK profiles and climates cluster naturally in
    15-dimensional feature space. KNN exploits this cluster structure
    directly. If a new soil sample resembles 7 of 8 rice profiles,
    KNN confidently predicts rice. This mirrors how experienced agronomists
    reason: "this soil is similar to the conditions where rice thrives."

ADVANTAGES:
    - No assumptions about data distribution
    - Naturally multi-class; trivially extended to any number of classes
    - Simple to understand and explain
    - High accuracy when feature space is well-scaled and clustered

LIMITATIONS:
    - Prediction time O(n * d): slow on large datasets
    - Sensitive to irrelevant features and unscaled data
    - High memory footprint (stores entire training set)
    - Performance degrades in high-dimensional spaces (curse of dimensionality)
"""

import logging
import time
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------
# HYPERPARAMETER CONFIGURATION — KNN
# -----------------------------------------------------------------
# n_neighbors=7   : Odd number avoids ties; 7 provides a stable vote
#                   for 22 classes; too small (k=1) overfits, too large
#                   (k=50) underfits.
# metric='minkowski' with p=2 : Euclidean distance; appropriate for
#                               continuous scaled agricultural features.
# weights='distance' : Closer neighbours vote with greater weight;
#                      reduces the influence of distant training points;
#                      improves accuracy near class boundaries.
# algorithm='auto'   : Sklearn auto-selects optimal tree structure (kd-tree
#                      or ball_tree) based on data dimensionality.
# n_jobs=-1          : Use all available CPU cores for distance computation.
KNN_PARAMS = {
    "n_neighbors": 7,
    "metric":      "minkowski",
    "p":           2,
    "weights":     "distance",
    "algorithm":   "auto",
    "n_jobs":      -1
}


def build_knn() -> KNeighborsClassifier:
    """
    Instantiate a KNeighborsClassifier with production parameters.

    Returns:
        Unfitted KNeighborsClassifier instance.
    """
    model = KNeighborsClassifier(**KNN_PARAMS)
    logger.info(f"KNeighborsClassifier instantiated | params: {KNN_PARAMS}")
    return model


def train_knn(
    X_train: np.ndarray,
    y_train: np.ndarray
) -> tuple:
    """
    Train KNN (index-building phase) on the preprocessed training set.

    Note: KNN 'training' is data ingestion + spatial index construction.
    The actual computation happens at prediction time.

    Args:
        X_train : Scaled training feature array (1760 x 15).
        y_train : Encoded integer target array (1760,).

    Returns:
        Tuple of (fitted_model, training_time_seconds).
    """
    model = build_knn()

    logger.info("Training KNeighborsClassifier (index construction)...")
    t_start = time.perf_counter()
    model.fit(X_train, y_train)
    training_time = time.perf_counter() - t_start

    logger.info(f"KNN trained in {training_time:.4f}s")
    return model, training_time
