"""
preprocessing.py - OptiCrop Data Preprocessing Pipeline
========================================================
Project : OptiCrop - Smart Agricultural Production Optimization Engine
Phase   : Data Preprocessing
Version : 1.0

Orchestrates the full preprocessing pipeline:
    1. Load dataset
    2. Verify data integrity (missing values, duplicates, dtypes, labels)
    3. Review outlier decisions (retain all - agronomically justified)
    4. Apply feature transformation assessment (StandardScaler only)
    5. Encode target variable (LabelEncoder)
    6. Engineer composite features (8 new features)
    7. Train-test split (80/20 stratified)
    8. Apply feature scaling (StandardScaler - fit on train only)
    9. Serialize all artifacts (scaler.pkl, label_encoder.pkl)

Usage:
    python scripts/preprocessing.py
    # or import and call run_preprocessing_pipeline()
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Internal modules
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.ml.feature_engineering import engineer_features, ALL_FEATURE_COLS

# ---- Logging Setup ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# ---- Project Paths ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "dataset" / "Crop_recommendation.csv"
MODELS_DIR   = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ---- Constants ----
TARGET_COLUMN      = 'label'
NUMERICAL_FEATURES = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
TEST_SIZE          = 0.20
RANDOM_STATE       = 42

EXPECTED_LABELS = {
    'apple', 'banana', 'blackgram', 'chickpea', 'coconut', 'coffee',
    'cotton', 'grapes', 'jute', 'kidneybeans', 'lentil', 'maize',
    'mango', 'mothbeans', 'mungbean', 'muskmelon', 'orange', 'papaya',
    'pigeonpeas', 'pomegranate', 'rice', 'watermelon'
}


# ==============================================================
# STAGE 1: DATA LOADING
# ==============================================================

def load_dataset(filepath: Path = DATASET_PATH) -> pd.DataFrame:
    """
    Load the crop recommendation dataset from CSV.

    Args:
        filepath: Absolute path to the CSV file.

    Returns:
        Raw DataFrame with all original columns.

    Raises:
        FileNotFoundError: If CSV does not exist.
        ValueError: If the loaded DataFrame is empty.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found: {filepath}")
    df = pd.read_csv(filepath)
    if df.empty:
        raise ValueError("Loaded dataset is empty.")
    logger.info(f"Dataset loaded: {df.shape[0]} rows x {df.shape[1]} columns")
    return df


# ==============================================================
# STAGE 2: DATA INTEGRITY VERIFICATION
# ==============================================================

def verify_data_integrity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run all data integrity checks in sequence.

    Checks: missing values, duplicates, data types, label consistency.

    Args:
        df: Raw loaded DataFrame.

    Returns:
        Verified (and lightly cleaned) DataFrame.
    """
    # Missing values
    total_missing = df.isnull().sum().sum()
    if total_missing > 0:
        raise ValueError(f"Dataset has {total_missing} missing values.")
    logger.info("✓ No missing values")

    # Duplicates
    n_dupes = df.duplicated().sum()
    if n_dupes > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        logger.warning(f"Removed {n_dupes} duplicate rows")
    else:
        logger.info("✓ No duplicate rows")

    # Data types - coerce int/float mismatches
    for col in NUMERICAL_FEATURES:
        if col not in df.columns:
            raise KeyError(f"Missing expected column: {col}")

    # Label consistency
    df[TARGET_COLUMN] = df[TARGET_COLUMN].str.strip().str.lower()
    unexpected = set(df[TARGET_COLUMN].unique()) - EXPECTED_LABELS
    if unexpected:
        raise ValueError(f"Unexpected crop labels: {unexpected}")
    logger.info(f"✓ Labels consistent: {df[TARGET_COLUMN].nunique()} unique crops")

    logger.info("✓ Data integrity verification complete")
    return df


# ==============================================================
# STAGE 3: TARGET VARIABLE ENCODING
# ==============================================================

def encode_target(
    df: pd.DataFrame,
    save_path: Path = MODELS_DIR / "label_encoder.pkl"
) -> tuple:
    """
    Encode string crop labels to integer class indices using LabelEncoder.

    Fits on sorted unique labels for deterministic mapping across environments.
    Serializes the fitted encoder for deployment use.

    Args:
        df        : Cleaned DataFrame with 'label' column.
        save_path : Path to save the serialized encoder.

    Returns:
        Tuple of (y_encoded Series, fitted LabelEncoder).
    """
    label_encoder = LabelEncoder()
    sorted_labels = sorted(df[TARGET_COLUMN].unique())
    label_encoder.fit(sorted_labels)

    y_encoded = pd.Series(
        label_encoder.transform(df[TARGET_COLUMN]),
        name='crop_class', index=df.index
    )

    with open(save_path, 'wb') as f:
        pickle.dump(label_encoder, f)

    logger.info(f"✓ LabelEncoder fitted: {len(sorted_labels)} classes")
    logger.info(f"✓ LabelEncoder saved: {save_path}")
    return y_encoded, label_encoder


# ==============================================================
# STAGE 4: TRAIN-TEST SPLIT
# ==============================================================

def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE
) -> tuple:
    """
    Perform 80/20 stratified train-test split.

    Stratification preserves the 100-sample-per-class balance in both
    training and test sets. Scaler is fitted AFTER this split.

    Args:
        X            : Feature DataFrame (all columns, post-engineering).
        y            : Encoded target Series.
        test_size    : Fraction for test set (default 0.20).
        random_state : Seed for reproducibility (default 42).

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
        shuffle=True
    )

    logger.info(
        f"✓ Train-test split complete | "
        f"Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows"
    )
    logger.info(
        f"  Train class distribution: {y_train.value_counts().to_dict()}"
    )
    return X_train, X_test, y_train, y_test


# ==============================================================
# STAGE 5: FEATURE SCALING
# ==============================================================

def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    save_path: Path = MODELS_DIR / "scaler.pkl"
) -> tuple:
    """
    Fit StandardScaler on training data; transform both train and test sets.

    CRITICAL: scaler.fit() is called ONLY on X_train to prevent data leakage.

    Args:
        X_train   : Training feature DataFrame.
        X_test    : Test feature DataFrame.
        save_path : Path to save the serialized scaler.

    Returns:
        Tuple of (X_train_scaled ndarray, X_test_scaled ndarray, scaler).
    """
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)  # fit + transform on train
    X_test_scaled  = scaler.transform(X_test)        # transform only on test

    with open(save_path, 'wb') as f:
        pickle.dump(scaler, f)

    logger.info(
        f"✓ StandardScaler fitted on training data | "
        f"means={np.round(scaler.mean_, 3)}"
    )
    logger.info(f"✓ Scaler saved: {save_path}")
    return X_train_scaled, X_test_scaled, scaler


# ==============================================================
# MASTER PIPELINE ORCHESTRATOR
# ==============================================================

def run_preprocessing_pipeline() -> dict:
    """
    Execute the complete preprocessing pipeline end-to-end.

    Pipeline stages (in order):
        1. Load dataset
        2. Verify data integrity
        3. Engineer features (7 raw -> 15 total)
        4. Encode target variable
        5. Train-test split (80/20 stratified)
        6. Scale features (StandardScaler on train only)
        7. Serialize scaler and encoder artifacts

    Returns:
        Dictionary containing all preprocessing outputs:
            X_train_scaled, X_test_scaled, y_train, y_test,
            scaler, label_encoder, feature_columns, df_clean
    """
    logger.info("=" * 70)
    logger.info("OPCTICROP PREPROCESSING PIPELINE — START")
    logger.info("=" * 70)

    # Stage 1: Load
    df = load_dataset()

    # Stage 2: Integrity
    df = verify_data_integrity(df)

    # Stage 3: Feature engineering
    df = engineer_features(df)
    X = df[ALL_FEATURE_COLS]

    # Stage 4: Target encoding
    y, label_encoder = encode_target(df)

    # Stage 5: Train-test split (BEFORE scaling)
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    # Stage 6: Feature scaling (fit ONLY on train)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    logger.info("=" * 70)
    logger.info("PREPROCESSING PIPELINE — COMPLETE")
    logger.info(f"  Training set : {X_train_scaled.shape}")
    logger.info(f"  Test set     : {X_test_scaled.shape}")
    logger.info(f"  Features     : {len(ALL_FEATURE_COLS)}")
    logger.info(f"  Classes      : {len(label_encoder.classes_)}")
    logger.info("=" * 70)

    return {
        'X_train_scaled': X_train_scaled,
        'X_test_scaled':  X_test_scaled,
        'y_train':        y_train,
        'y_test':         y_test,
        'scaler':         scaler,
        'label_encoder':  label_encoder,
        'feature_columns': ALL_FEATURE_COLS,
        'df_clean':       df
    }


if __name__ == "__main__":
    results = run_preprocessing_pipeline()
    print("\nPreprocessing complete. Artifacts saved to models/")
    print(f"X_train shape: {results['X_train_scaled'].shape}")
    print(f"X_test  shape: {results['X_test_scaled'].shape}")
