"""
train_test_split.py - OptiCrop Dataset Splitting Utilities
===========================================================
Project : OptiCrop - Smart Agricultural Production Optimization Engine
Phase   : Data Preprocessing - Train/Test Split Strategy
Version : 1.0

Provides reusable train-test splitting utilities with built-in
stratification, reproducibility controls, and split-quality reporting.

Split Strategy:
    - Ratio       : 80% training / 20% testing
    - Stratified  : Yes (preserves class balance per crop)
    - Shuffle     : Yes (randomizes before splitting)
    - Random State: 42 (reproducible across all runs)

Usage:
    from scripts.train_test_split import create_stratified_split, report_split_quality
"""

import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# ---- Split Configuration ----
DEFAULT_TEST_SIZE    = 0.20   # 20% held out for evaluation
DEFAULT_RANDOM_STATE = 42     # Seed for reproducibility


def create_stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
    shuffle: bool = True
) -> tuple:
    """
    Perform a stratified train-test split maintaining class proportions.

    Stratification guarantees that each of the 22 crop classes appears in
    both training (80 samples) and test (20 samples) sets proportionally.

    Split rationale:
        80% training (1,760 samples): Sufficient for all 5 algorithms to learn
                                       decision boundaries for 22 classes.
        20% testing  (440 samples):   Provides 20 samples per class for a
                                       statistically meaningful accuracy estimate.
                                       Larger test splits risk underfitting due
                                       to reduced training data.

    Args:
        X            : Feature DataFrame (all columns post-engineering).
        y            : Encoded integer target Series.
        test_size    : Fraction of data reserved for testing (default 0.20).
        random_state : Random seed for reproducibility (default 42).
        shuffle      : Whether to shuffle before splitting (always True).

    Returns:
        Tuple: (X_train, X_test, y_train, y_test)

    Raises:
        ValueError: If X and y have mismatched lengths.
        ValueError: If test_size is not in (0, 1).
    """
    if len(X) != len(y):
        raise ValueError(
            f"Feature matrix and target vector length mismatch: "
            f"{len(X)} vs {len(y)}."
        )
    if not (0 < test_size < 1):
        raise ValueError(
            f"test_size must be in (0, 1). Got: {test_size}."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
        shuffle=shuffle
    )

    logger.info(f"Train-test split | Test: {test_size:.0%} | Seed: {random_state}")
    logger.info(f"Training samples : {X_train.shape[0]}")
    logger.info(f"Test samples     : {X_test.shape[0]}")
    logger.info(f"Feature columns  : {X_train.shape[1]}")

    return X_train, X_test, y_train, y_test


def report_split_quality(
    y_train: pd.Series,
    y_test: pd.Series,
    label_encoder
) -> pd.DataFrame:
    """
    Generate a quality report verifying class balance in both splits.

    Confirms that stratification worked correctly: each class has
    approximately 80 training samples and 20 test samples.

    Args:
        y_train        : Encoded training target Series.
        y_test         : Encoded test target Series.
        label_encoder  : Fitted LabelEncoder for decoding class indices.

    Returns:
        DataFrame with per-class sample counts in train and test sets.
    """
    train_counts = y_train.value_counts().sort_index()
    test_counts  = y_test.value_counts().sort_index()

    report = pd.DataFrame({
        'Crop':        label_encoder.classes_,
        'Train Count': train_counts.values,
        'Test Count':  test_counts.values,
        'Total':       train_counts.values + test_counts.values,
        'Train %':     (train_counts.values / (train_counts.values + test_counts.values) * 100).round(1)
    })

    logger.info(f"\nSplit Quality Report:\n{report.to_string(index=False)}")

    # Verify stratification tolerance (each class should have 80/20 samples)
    max_train_deviation = abs(report['Train Count'] - 80).max()
    if max_train_deviation > 2:
        logger.warning(
            f"Stratification deviation detected: max {max_train_deviation} "
            f"samples from expected 80 per class."
        )
    else:
        logger.info("✓ Stratification quality verified: all classes balanced")

    return report
