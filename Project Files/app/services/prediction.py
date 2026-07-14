"""
OptiCrop – app/services/prediction.py
Live Prediction Service  (Phase 14 — ML Integration)

Replaces the Phase 13 placeholder with full end-to-end ML inference.

Inference pipeline (mirrors training pipeline exactly):
  1. Receive validated inputs  { N, P, K, temperature, humidity, ph, rainfall }
  2. Apply feature engineering → 15 features
  3. Scale with StandardScaler (loaded from scaler.pkl)
  4. Run RandomForest.predict() + .predict_proba()
  5. Decode class index → crop name via LabelEncoder
  6. Delegate to formatter.format_result() for UI-ready output

CRITICAL — Training/Inference Parity:
  The SAME transformations applied during training MUST be applied here.
  Any deviation (different feature order, missing engineered features,
  or re-fitting the scaler on inference data) will corrupt predictions.

  Training pipeline (scripts/preprocessing.py + training.py):
    raw 7 features → engineer_single_input() → StandardScaler.transform()
    → RandomForest.predict() / predict_proba()

  Inference pipeline (this file):
    IDENTICAL — same function, same fitted scaler, same model.

Feature vector (15 elements, ORDER IS FIXED):
  [0]  N               — Nitrogen
  [1]  P               — Phosphorus
  [2]  K               — Potassium
  [3]  temperature     — Mean annual temperature (°C)
  [4]  humidity        — Relative humidity (%)
  [5]  ph              — Soil pH
  [6]  rainfall        — Annual rainfall (mm)
  [7]  NPK_Ratio       — N / (P + K)
  [8]  Total_Nutrients — N + P + K
  [9]  Env_Index       — (temp × humidity) / 100
  [10] Moisture_Index  — (humidity × rainfall) / 100
  [11] Nutrient_Balance— (P + K) / N
  [12] Soil_Fertility  — N + 2P + 1.5K
  [13] Climate_Suitability — (rainfall × humidity) / temperature
  [14] Agri_Health     — (N × P × K) / (temp × humidity)

Usage:
    from app.services.prediction import PredictionService
    svc = PredictionService()
    result = svc.predict(validated_inputs_dict)
"""

import logging
import time
from typing import Any, Dict

import numpy as np

from app.ml.feature_engineering import engineer_single_input
from app.services.formatter import format_result
from app.services.model_loader import ModelLoadError, get_artifacts

logger = logging.getLogger(__name__)

# ── Epsilon guard (mirrors feature_engineering.py) ────────────────────────────
_EPSILON = 1e-6


class PredictionService:
    """
    End-to-end ML crop recommendation service.

    Lazy-initialized — model artifacts are fetched on first .predict() call,
    not at instantiation, to keep the Flask test client startup model-free.

    Lifecycle in a typical request:
        routes.py → _get_service().predict(inputs) → PredictionService.predict()
                    ↓
                get_artifacts()          # loads from disk once, then cached
                    ↓
                _build_feature_vector()  # 7 → 15 features
                    ↓
                scaler.transform()       # StandardScaler
                    ↓
                model.predict_proba()    # RandomForest
                    ↓
                encoder.inverse_transform()  # int → crop name
                    ↓
                format_result()          # UI-ready dict
    """

    # ── Public API ────────────────────────────────────────────────────────────

    def predict(self, inputs: Dict[str, float]) -> Dict[str, Any]:
        """
        Generates a live AI crop recommendation.

        Args:
            inputs: Validated float dict produced by validators.py:
                    { nitrogen, phosphorus, potassium,
                      temperature, humidity, ph, rainfall }

        Returns:
            result_data dict — the full UI-ready payload.
            Shape is identical to the Phase 13 placeholder so
            result.html and result.js require no changes.

        Raises:
            ModelLoadError: If model artifacts cannot be loaded.
            RuntimeError:   If inference encounters an unexpected error.
        """
        logger.info(
            'PredictionService.predict()  |  inputs=N=%s P=%s K=%s T=%s H=%s pH=%s R=%s',
            inputs.get('nitrogen'),    inputs.get('phosphorus'), inputs.get('potassium'),
            inputs.get('temperature'), inputs.get('humidity'),   inputs.get('ph'),
            inputs.get('rainfall'),
        )

        # ── Load artifacts ────────────────────────────────────────────────────
        try:
            artifacts = get_artifacts()
        except ModelLoadError:
            raise  # Re-raise with full context for the route layer

        model   = artifacts['model']
        scaler  = artifacts['scaler']
        encoder = artifacts['encoder']

        # ── Build 15-feature vector ───────────────────────────────────────────
        try:
            feature_vector = self._build_feature_vector(inputs)
        except Exception as exc:
            logger.error('Feature engineering failed  |  %s', exc, exc_info=True)
            raise RuntimeError(
                f'Feature engineering error: {exc}'
            ) from exc

        logger.debug('Feature vector (15):  %s', feature_vector)

        # ── Scale ─────────────────────────────────────────────────────────────
        try:
            X = np.array(feature_vector, dtype=np.float64).reshape(1, -1)
            X_scaled = scaler.transform(X)
        except Exception as exc:
            logger.error('Scaling failed  |  %s', exc, exc_info=True)
            raise RuntimeError(f'Scaling error: {exc}') from exc

        # ── Inference ─────────────────────────────────────────────────────────
        try:
            t0          = time.perf_counter()
            crop_idx    = model.predict(X_scaled)[0]
            proba_arr   = model.predict_proba(X_scaled)[0]
            pred_time   = (time.perf_counter() - t0) * 1000   # ms
        except Exception as exc:
            logger.error('Model inference failed  |  %s', exc, exc_info=True)
            raise RuntimeError(f'Model inference error: {exc}') from exc

        # ── Decode ────────────────────────────────────────────────────────────
        crop_name  = encoder.inverse_transform([crop_idx])[0]
        confidence = round(float(proba_arr.max()) * 100, 2)

        logger.info(
            'Inference complete  |  crop=%-14s  |  confidence=%5.1f%%  |  time=%.1fms',
            crop_name, confidence, pred_time,
        )

        # ── Format ────────────────────────────────────────────────────────────
        try:
            result_data = format_result(
                crop          = crop_name,
                confidence    = confidence,
                inputs        = inputs,
                proba_arr     = proba_arr,
                label_classes = list(encoder.classes_),
                pred_time_ms  = round(pred_time, 1),
            )
        except Exception as exc:
            logger.error('Result formatting failed  |  %s', exc, exc_info=True)
            raise RuntimeError(f'Result formatting error: {exc}') from exc

        return result_data

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _build_feature_vector(inputs: Dict[str, float]) -> list:
        """
        Converts the 7 validated user inputs into the 15-feature vector
        used during training.

        Delegates to engineer_single_input() in app/ml/feature_engineering.py
        to guarantee training/inference parity.

        Args:
            inputs: Validated dict from validators.py

        Returns:
            List of 15 floats: [N, P, K, temp, humidity, ph, rainfall,
                                NPK_Ratio, Total_Nutrients, Env_Index,
                                Moisture_Index, Nutrient_Balance,
                                Soil_Fertility, Climate_Suitability,
                                Agri_Health]
        """
        raw = [
            float(inputs['nitrogen']),
            float(inputs['phosphorus']),
            float(inputs['potassium']),
            float(inputs['temperature']),
            float(inputs['humidity']),
            float(inputs['ph']),
            float(inputs['rainfall']),
        ]
        # engineer_single_input() mirrors engineer_features() exactly
        return engineer_single_input(raw)
