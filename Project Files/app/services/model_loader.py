"""
OptiCrop – app/services/model_loader.py
Singleton Model Loader

Loads model.pkl, scaler.pkl, and label_encoder.pkl from disk exactly
once during application startup and caches them in module-level state.

Design decisions:
  • Singleton pattern — avoids reloading on every prediction request.
  • Lazy loading — models are loaded on first call to get_artifacts(),
    not on module import, so Flask's test client starts without needing
    model files on disk.
  • Explicit failure — raises ModelLoadError with a clear message if any
    artifact is missing or corrupted, so the error page gives useful info.
  • Thread safety — Python's GIL ensures the load block executes atomically
    on CPython; for multi-threaded production (Gunicorn threads) a lock
    is included as belt-and-suspenders protection.

Expected file locations (configurable via Flask config):
    models/model.pkl          — Fitted RandomForestClassifier
    models/scaler.pkl         — Fitted StandardScaler (15 features)
    models/label_encoder.pkl  — Fitted LabelEncoder (22 crop classes)

Usage:
    from app.services.model_loader import get_artifacts, ModelLoadError

    try:
        artifacts = get_artifacts()
        model, scaler, encoder = artifacts["model"], artifacts["scaler"], artifacts["encoder"]
    except ModelLoadError as e:
        # Render error page — model not available
        ...
"""

import logging
import pickle
import threading
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ── Module-level singleton cache ──────────────────────────────────────────────
_artifacts: Optional[Dict[str, Any]] = None
_load_lock  = threading.Lock()


class ModelLoadError(Exception):
    """
    Raised when model artifacts cannot be loaded from disk.

    Attributes:
        message:   Human-readable description
        artifact:  Which file failed ('model', 'scaler', 'encoder', or 'unknown')
    """

    def __init__(self, message: str, artifact: str = 'unknown') -> None:
        super().__init__(message)
        self.artifact = artifact


def get_artifacts() -> Dict[str, Any]:
    """
    Returns the cached model artifacts dict, loading from disk if needed.

    Returns:
        {
            'model':   RandomForestClassifier (fitted)
            'scaler':  StandardScaler         (fitted, 15 features)
            'encoder': LabelEncoder           (fitted, 22 crop classes)
        }

    Raises:
        ModelLoadError: If any artifact file is missing, unreadable,
                        or fails pickle deserialization.
    """
    global _artifacts

    if _artifacts is not None:
        return _artifacts

    with _load_lock:
        # Double-checked locking — another thread may have loaded while waiting
        if _artifacts is not None:
            return _artifacts

        _artifacts = _load_artifacts_from_disk()
        logger.info(
            'Model artifacts loaded  |  model=%s  |  classes=%d  |  features=%d',
            type(_artifacts['model']).__name__,
            len(_artifacts['encoder'].classes_),
            _artifacts['scaler'].n_features_in_,
        )

    return _artifacts


def _load_artifacts_from_disk() -> Dict[str, Any]:
    """
    Loads all three model artifacts from the models/ directory.

    Paths are resolved relative to the project root (parent of app/).

    Returns:
        Dict with 'model', 'scaler', 'encoder' keys.

    Raises:
        ModelLoadError: On any IO, pickle, or attribute error.
    """
    # Resolve project root from this file's location:
    # app/services/model_loader.py → project_root/
    project_root = Path(__file__).resolve().parent.parent.parent
    models_dir   = project_root / 'models'

    spec = {
        'model':   models_dir / 'model.pkl',
        'scaler':  models_dir / 'scaler.pkl',
        'encoder': models_dir / 'label_encoder.pkl',
    }

    loaded: Dict[str, Any] = {}

    for key, path in spec.items():
        logger.debug('Loading artifact  |  key=%s  |  path=%s', key, path)

        if not path.exists():
            raise ModelLoadError(
                f"Artifact '{path.name}' not found at {path}. "
                "Run 'python scripts/generate_dataset_and_train.py' to generate it.",
                artifact=key,
            )

        try:
            with open(path, 'rb') as f:
                obj = pickle.load(f)
        except (pickle.UnpicklingError, EOFError, Exception) as exc:
            raise ModelLoadError(
                f"Failed to deserialize '{path.name}': {exc}",
                artifact=key,
            ) from exc

        loaded[key] = obj
        logger.info(
            'Artifact loaded  |  %-8s  →  %s  (%.1f KB)',
            key, type(obj).__name__, path.stat().st_size / 1024,
        )

    # Sanity-check the loaded objects
    _validate_artifacts(loaded)

    return loaded


def _validate_artifacts(artifacts: Dict[str, Any]) -> None:
    """
    Runs basic sanity checks on the loaded artifacts to catch
    version mismatches or corrupted files early.

    Raises:
        ModelLoadError: If any object lacks expected attributes.
    """
    model   = artifacts.get('model')
    scaler  = artifacts.get('scaler')
    encoder = artifacts.get('encoder')

    if not hasattr(model, 'predict'):
        raise ModelLoadError(
            "model.pkl does not expose a .predict() method. "
            "Ensure it is a fitted scikit-learn estimator.",
            artifact='model',
        )

    if not hasattr(scaler, 'transform'):
        raise ModelLoadError(
            "scaler.pkl does not expose a .transform() method.",
            artifact='scaler',
        )

    if not hasattr(encoder, 'inverse_transform'):
        raise ModelLoadError(
            "label_encoder.pkl does not expose .inverse_transform().",
            artifact='encoder',
        )

    n_feat = getattr(scaler, 'n_features_in_', None)
    if n_feat is not None and n_feat != 15:
        logger.warning(
            'Scaler was fitted on %d features; expected 15. '
            'Ensure preprocessing pipeline is consistent.',
            n_feat,
        )


def is_model_loaded() -> bool:
    """Returns True if artifacts have been loaded into memory."""
    return _artifacts is not None


def reload_artifacts() -> None:
    """
    Forces a reload of all model artifacts from disk.

    Use case: hot-swapping model.pkl in production without restarting.
    Exposed for admin / maintenance endpoints (not called in normal flow).
    """
    global _artifacts
    logger.warning('Forcing model artifact reload from disk.')
    with _load_lock:
        _artifacts = None
    get_artifacts()
    logger.info('Model artifacts reloaded successfully.')
