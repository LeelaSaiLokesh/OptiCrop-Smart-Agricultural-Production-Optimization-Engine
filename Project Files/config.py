"""
OptiCrop – config.py
Flask Application Configuration Classes

Provides three environment profiles:
  • DevelopmentConfig  – Local dev, debug ON, verbose logging
  • ProductionConfig   – Hardened, debug OFF, HTTPS cookies
  • TestingConfig      – Deterministic state for pytest

Usage:
    from config import config
    app.config.from_object(config['development'])

Environment variables (set via .env or shell) override class defaults:
    SECRET_KEY, MODEL_PATH, LOG_DIR, LOG_LEVEL, FLASK_ENV
"""

import os
from datetime import timedelta


class Config:
    """Base configuration shared across all environments."""

    # ── Security ─────────────────────────────────────────────────────────────
    SECRET_KEY: str = (
        os.environ.get('SECRET_KEY')
        or 'opticrop-dev-secret-CHANGE-THIS-IN-PRODUCTION-32-chars-min'
    )

    # ── Application metadata ──────────────────────────────────────────────────
    APP_NAME: str        = 'OptiCrop'
    APP_VERSION: str     = '1.0.0'
    APP_DESCRIPTION: str = (
        'AI-Powered Smart Agricultural Production Optimization Engine'
    )

    # ── Request limits ────────────────────────────────────────────────────────
    MAX_CONTENT_LENGTH: int = 32 * 1024   # 32 KB – reject oversized payloads

    # ── Session ───────────────────────────────────────────────────────────────
    SESSION_COOKIE_HTTPONLY: bool = True
    SESSION_COOKIE_SAMESITE: str  = 'Lax'
    PERMANENT_SESSION_LIFETIME    = timedelta(hours=2)

    # ── ML Model (Phase 14 will read this) ────────────────────────────────────
    MODEL_PATH: str    = os.environ.get('MODEL_PATH') or os.path.join('models', 'model.pkl')
    ENCODER_PATH: str  = os.environ.get('ENCODER_PATH') or os.path.join('models', 'encoder.pkl')
    MODEL_VERSION: str = '1.0'

    # ── Logging ───────────────────────────────────────────────────────────────
    LOG_DIR: str   = os.environ.get('LOG_DIR')   or 'logs'
    LOG_LEVEL: str = os.environ.get('LOG_LEVEL') or 'INFO'
    LOG_FILE: str  = 'opticrop.log'

    # ── CORS / JSON ───────────────────────────────────────────────────────────
    JSON_SORT_KEYS: bool = False  # Preserve dict insertion order in responses

    # ── Mode flags ────────────────────────────────────────────────────────────
    DEBUG: bool   = False
    TESTING: bool = False


class DevelopmentConfig(Config):
    """Local development – debug ON, HTTP cookies allowed."""

    DEBUG: bool    = True
    LOG_LEVEL: str = 'DEBUG'
    SESSION_COOKIE_SECURE: bool = False   # HTTP is fine on localhost


class ProductionConfig(Config):
    """Production deployment – security-hardened, minimal logging."""

    DEBUG: bool    = False
    LOG_LEVEL: str = 'WARNING'
    SESSION_COOKIE_SECURE: bool   = True  # Requires HTTPS
    SESSION_COOKIE_HTTPONLY: bool = True
    SESSION_COOKIE_SAMESITE: str  = 'Strict'


class TestingConfig(Config):
    """Automated testing – deterministic, no CSRF, error logs only."""

    TESTING: bool  = True
    DEBUG: bool    = True
    LOG_LEVEL: str = 'ERROR'    # Suppress noise in test output
    WTF_CSRF_ENABLED: bool = False
    SECRET_KEY: str        = 'testing-secret-key-not-for-production'


# ── Config registry ───────────────────────────────────────────────────────────
config: dict = {
    'development': DevelopmentConfig,
    'production':  ProductionConfig,
    'testing':     TestingConfig,
    'default':     DevelopmentConfig,
}
