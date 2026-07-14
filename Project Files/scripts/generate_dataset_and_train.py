"""
generate_dataset_and_train.py
OptiCrop – One-time bootstrap script

This script:
  1. Downloads / generates the Crop Recommendation dataset
  2. Runs the preprocessing pipeline  (scaler.pkl, label_encoder.pkl)
  3. Trains the Random Forest model
  4. Saves model.pkl, scaler.pkl, label_encoder.pkl to models/

Run once from the project root:
    python scripts/generate_dataset_and_train.py
"""

import sys
import pickle
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

ROOT       = Path(__file__).resolve().parent.parent
DATASET    = ROOT / "dataset" / "Crop_recommendation.csv"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATASET.parent.mkdir(parents=True, exist_ok=True)

# ── Crop profiles (mean ± realistic variation) ────────────────────────────────
# Each row: N, P, K, temperature, humidity, ph, rainfall
CROP_PROFILES = {
    "rice":         (80, 55, 40, 23, 82, 6.5, 200),
    "maize":        (77, 48, 43, 22, 65, 6.2,  85),
    "chickpea":     (40, 68, 79, 18, 16, 7.3,  73),
    "kidneybeans":  (20, 68, 79, 20, 22, 5.7,  98),
    "pigeonpeas":   (20, 68, 79, 28, 49, 5.8  ,148),
    "mothbeans":    (21, 48, 29, 28, 54, 6.9,  51),
    "mungbean":     (20, 48, 29, 29, 85, 6.7,  48),
    "blackgram":    (40, 68, 19, 30, 65, 7.0,  68),
    "lentil":       (18, 68, 19, 24, 65, 6.9,  46),
    "pomegranate":  (18,  16, 40, 21, 90, 6.5, 107),
    "banana":       (100, 82, 50, 27, 80, 6.0, 110),
    "mango":        (20, 27, 30, 31, 50, 5.7,  94),
    "grapes":       (23, 132, 200, 24, 81, 6.0, 70),
    "watermelon":   (99, 17, 50, 25, 85, 6.5,  50),
    "muskmelon":    (100, 17, 50, 28, 92, 6.4,  25),
    "apple":        (20, 134, 199, 22, 92, 5.9,  90),
    "orange":       (19, 16, 10, 22, 92, 7.0,  110),
    "papaya":       (50, 59, 50, 33, 92, 6.7,  160),
    "coconut":      (22, 16, 30, 27, 95, 5.9,  176),
    "cotton":       (118, 46, 20, 24, 79, 6.9,  52),
    "jute":         (78, 46, 39, 25, 80, 6.7,  175),
    "coffee":       (101, 28, 29, 25, 58, 6.8,  159),
}

np.random.seed(42)
SAMPLES_PER_CROP = 100


def _gen_crop_rows(crop: str, profile: tuple) -> pd.DataFrame:
    N_m, P_m, K_m, T_m, H_m, ph_m, R_m = profile
    n = SAMPLES_PER_CROP
    rows = pd.DataFrame({
        "N":           np.clip(np.random.normal(N_m,  N_m  * 0.12, n), 0,   140),
        "P":           np.clip(np.random.normal(P_m,  P_m  * 0.12, n), 5,   145),
        "K":           np.clip(np.random.normal(K_m,  K_m  * 0.12, n), 5,   205),
        "temperature": np.clip(np.random.normal(T_m,  2.5,          n), 8,    44),
        "humidity":    np.clip(np.random.normal(H_m,  5.0,          n), 14,  100),
        "ph":          np.clip(np.random.normal(ph_m, 0.4,          n), 3.5,  10),
        "rainfall":    np.clip(np.random.normal(R_m,  R_m  * 0.18, n), 20,  300),
        "label":       crop,
    })
    return rows


def generate_dataset() -> pd.DataFrame:
    parts = [_gen_crop_rows(c, p) for c, p in CROP_PROFILES.items()]
    df = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(DATASET, index=False)
    logger.info("Dataset generated: %s rows → %s", len(df), DATASET)
    return df


def compute_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Mirrors app/ml/feature_engineering.py engineer_features() exactly."""
    EPSILON = 1e-6
    df = df.copy()
    N, P, K = df["N"], df["P"], df["K"]
    T, H, R = df["temperature"], df["humidity"], df["rainfall"]

    df["NPK_Ratio"]           = N / (P + K + EPSILON)
    df["Total_Nutrients"]     = N + P + K
    df["Env_Index"]           = (T * H) / 100.0
    df["Moisture_Index"]      = (H * R) / 100.0
    df["Nutrient_Balance"]    = (P + K) / (N + EPSILON)
    df["Soil_Fertility"]      = N + (2.0 * P) + (1.5 * K)
    df["Climate_Suitability"] = (R * H) / (T + EPSILON)
    df["Agri_Health"]         = (N * P * K) / (T * H + EPSILON)
    return df


ALL_FEATURE_COLS = [
    "N", "P", "K", "temperature", "humidity", "ph", "rainfall",
    "NPK_Ratio", "Total_Nutrients", "Env_Index", "Moisture_Index",
    "Nutrient_Balance", "Soil_Fertility", "Climate_Suitability", "Agri_Health",
]

RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": None,
    "min_samples_split": 4,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "bootstrap": True,
    "class_weight": "balanced_subsample",
    "n_jobs": -1,
    "random_state": 42,
}


def train_and_save() -> None:
    # 1. Dataset
    if DATASET.exists():
        df = pd.read_csv(DATASET)
        logger.info("Existing dataset loaded: %s rows", len(df))
    else:
        df = generate_dataset()

    # 2. Feature engineering
    df = compute_engineered_features(df)
    X  = df[ALL_FEATURE_COLS].values.astype(float)
    y_raw = df["label"].str.strip().str.lower().values

    # 3. Label encoding
    le = LabelEncoder()
    le.fit(sorted(set(y_raw)))
    y = le.transform(y_raw)
    logger.info("Classes (%d): %s", len(le.classes_), list(le.classes_))

    # 4. Train-test split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y, shuffle=True
    )

    # 5. Scaling
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s  = scaler.transform(X_te)

    # 6. Train
    logger.info("Training Random Forest (%d estimators) …", RF_PARAMS["n_estimators"])
    import time
    t0 = time.perf_counter()
    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_tr_s, y_tr)
    logger.info("Training complete in %.2fs", time.perf_counter() - t0)

    # 7. Evaluate — held-out test set
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    y_pred = model.predict(X_te_s)
    acc = accuracy_score(y_te, y_pred)
    logger.info("Test Accuracy : %.4f  (%.2f%%)", acc, acc * 100)

    # 7b. 5-fold Stratified Cross-Validation on full scaled dataset
    logger.info("Running 5-fold Stratified CV on full dataset …")
    X_all_s = scaler.transform(X)          # use same fitted scaler (no leakage)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_all_s, y, cv=cv, scoring="accuracy", n_jobs=-1)
    logger.info(
        "CV Accuracy  : %.4f ± %.4f  (min=%.4f  max=%.4f)",
        cv_scores.mean(), cv_scores.std(), cv_scores.min(), cv_scores.max()
    )

    # 8. Save artifacts
    artifacts = {
        "model.pkl":         model,
        "scaler.pkl":        scaler,
        "label_encoder.pkl": le,
    }
    for fname, obj in artifacts.items():
        path = MODELS_DIR / fname
        with open(path, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Saved: %s  (%.1f KB)", path, path.stat().st_size / 1024)

    logger.info("=" * 60)
    logger.info("ALL ARTIFACTS SAVED SUCCESSFULLY")
    logger.info("  model.pkl         — RandomForest (%d trees)", RF_PARAMS["n_estimators"])
    logger.info("  scaler.pkl        — StandardScaler (15 features)")
    logger.info("  label_encoder.pkl — LabelEncoder (%d classes)", len(le.classes_))
    logger.info("  Test Accuracy     — %.2f%%", acc * 100)
    logger.info("=" * 60)


if __name__ == "__main__":
    train_and_save()
