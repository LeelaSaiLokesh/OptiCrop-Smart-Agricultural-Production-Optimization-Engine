"""
kmeans_model.py - OptiCrop K-Means Clustering (Exploratory)
============================================================
Project : OptiCrop - Smart Agricultural Production Optimization Engine
Phase   : Machine Learning Model Development
Version : 1.0

IMPORTANT NOTE:
    K-Means is an UNSUPERVISED algorithm. It does not receive the
    target labels during training. It is included here for EXPLORATORY
    COMPARISON ONLY to assess:
        1. Whether the natural data clusters (discovered without labels)
           align with the 22 known crop classes.
        2. Silhouette score as a label-free cluster quality measure.
        3. Visualization of the clustering vs ground-truth labels.

    K-Means is NOT a candidate for the production prediction pipeline.
    Use supervised classifiers (Random Forest, Decision Tree, LR, KNN)
    for deployment.

THEORETICAL OVERVIEW:
    K-Means partitions n observations into K clusters by minimizing the
    Within-Cluster Sum of Squares (WCSS):
        WCSS = sum_k sum_{x in C_k} ||x - mu_k||^2
    where mu_k is the centroid of cluster k.

    Algorithm (Lloyd's):
        1. Initialize K centroids (k-means++ for stability)
        2. Assign each point to the nearest centroid
        3. Update centroids to cluster means
        4. Repeat until convergence (centroid shift < tol)

AGRICULTURAL RELEVANCE:
    Unsupervised crop clustering validates our understanding of agricultural
    ecological zones. If K-Means with K=22 produces clusters that map
    cleanly to the known crop labels, it confirms that crops genuinely
    occupy distinct regions in the soil-climate feature space —
    supporting the validity of our supervised classifier predictions.

ADVANTAGES:
    - No labels required; validates data structure
    - Fast training for moderate K and dataset size
    - Reveals natural ecological groupings

LIMITATIONS:
    - Must specify K in advance
    - Sensitive to initialization and outliers
    - Assumes spherical, equal-variance clusters
    - Cannot directly produce crop label predictions without a mapping step
    - Silhouette score may be low for complex, overlapping clusters
"""

import logging
import time
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------
# HYPERPARAMETER CONFIGURATION — K-MEANS
# -----------------------------------------------------------------
# n_clusters=22   : Match the 22 known crop classes for direct
#                   comparison between discovered and known groupings.
# init='k-means++': Smart centroid initialization reduces iterations
#                   and avoids poor local minima (vs random init).
# n_init=20       : Run 20 independent initializations; keep best WCSS.
#                   More robust than the sklearn default of 10.
# max_iter=500    : Maximum iterations per run; 500 is sufficient for
#                   this dataset size.
# random_state=42 : Fully reproducible initialization.
KMEANS_PARAMS = {
    "n_clusters":  22,
    "init":        "k-means++",
    "n_init":      20,
    "max_iter":    500,
    "random_state": 42
}


def build_kmeans() -> KMeans:
    """
    Instantiate a KMeans clustering model with production parameters.

    Returns:
        Unfitted KMeans instance.
    """
    model = KMeans(**KMEANS_PARAMS)
    logger.info(f"KMeans instantiated | params: {KMEANS_PARAMS}")
    return model


def train_kmeans(
    X_train: np.ndarray,
    y_train: np.ndarray = None
) -> tuple:
    """
    Train K-Means clustering on the training feature set.

    Note: y_train is accepted but NOT used during training.
    It is only used post-training for cluster-to-label alignment analysis.

    Args:
        X_train : Scaled training feature array (1760 x 15).
        y_train : Encoded labels (optional; used only for evaluation).

    Returns:
        Tuple of (fitted_model, training_time_seconds).
    """
    model = build_kmeans()

    logger.info(f"Training KMeans (K={KMEANS_PARAMS['n_clusters']}, unsupervised)...")
    t_start = time.perf_counter()
    model.fit(X_train)          # y_train intentionally excluded
    training_time = time.perf_counter() - t_start

    logger.info(
        f"KMeans trained in {training_time:.4f}s | "
        f"inertia (WCSS): {model.inertia_:.2f} | "
        f"iterations: {model.n_iter_}"
    )
    return model, training_time


def evaluate_kmeans(
    model: KMeans,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> dict:
    """
    Evaluate KMeans cluster quality using label-free and label-based metrics.

    Args:
        model  : Fitted KMeans model.
        X_test : Scaled test feature array (440 x 15).
        y_test : Ground truth integer labels (440,).

    Returns:
        Dictionary with silhouette_score, adjusted_rand_index, inertia.
    """
    cluster_labels = model.predict(X_test)

    sil_score = silhouette_score(X_test, cluster_labels, metric="euclidean")
    ari_score = adjusted_rand_score(y_test, cluster_labels)

    logger.info(
        f"KMeans evaluation | "
        f"Silhouette: {sil_score:.4f} | "
        f"Adjusted Rand Index: {ari_score:.4f} | "
        f"WCSS (inertia): {model.inertia_:.2f}"
    )

    return {
        "model_name":           "K-Means (Unsupervised)",
        "silhouette_score":     round(sil_score, 4),
        "adjusted_rand_index":  round(ari_score, 4),
        "inertia":              round(model.inertia_, 2),
        "n_clusters":           KMEANS_PARAMS["n_clusters"],
        "note":                 "Unsupervised — accuracy metric not applicable"
    }
