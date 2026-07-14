"""
OptiCrop – app/utils/logger.py
Centralized Logging Configuration

Sets up a RotatingFileHandler (5 MB × 3 backups) + StreamHandler for
the Flask app logger and the Python root logger so all child loggers
(app.routes, app.services.prediction, etc.) inherit the same config.

Log events:
  INFO    — App startup, page visits, successful predictions
  WARNING — Validation failures, 4xx errors
  ERROR   — Prediction failures, 5xx errors
  DEBUG   — Full request payloads (development only)

Log file: logs/opticrop.log  (created automatically)

Usage (called once from app/__init__.py):
    from app.utils.logger import configure_logging
    configure_logging(app)
"""

import logging
import logging.handlers
import os

from flask import Flask

# ── Log format ────────────────────────────────────────────────────────────────
_FMT  = '[%(asctime)s]  %(levelname)-8s  %(name)-35s  %(message)s'
_DATE = '%Y-%m-%d %H:%M:%S'


def configure_logging(app: Flask) -> None:
    """
    Attaches rotating file + console handlers to Flask's app logger
    and to the Python root logger.

    Idempotent — duplicate handler guard prevents double-logging on
    Flask's development server auto-reload.

    Args:
        app: Flask application instance (config must already be loaded)
    """
    log_dir   = app.config.get('LOG_DIR',   'logs')
    log_level = app.config.get('LOG_LEVEL', 'INFO').upper()
    log_file  = app.config.get('LOG_FILE',  'opticrop.log')

    os.makedirs(log_dir, exist_ok=True)

    level     = getattr(logging, log_level, logging.INFO)
    formatter = logging.Formatter(_FMT, datefmt=_DATE)

    # ── Rotating file handler ─────────────────────────────────────────────────
    file_handler = logging.handlers.RotatingFileHandler(
        filename    = os.path.join(log_dir, log_file),
        maxBytes    = 5 * 1024 * 1024,   # 5 MB per file
        backupCount = 3,
        encoding    = 'utf-8',
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    # ── Console / stream handler ──────────────────────────────────────────────
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    # ── Root logger — all child loggers inherit ───────────────────────────────
    root = logging.getLogger()
    root.setLevel(level)
    if not root.handlers:                 # Prevent duplicate handlers on reload
        root.addHandler(file_handler)
        root.addHandler(console_handler)

    # ── Flask app logger (separate from root to prevent double output) ─────────
    app.logger.handlers.clear()
    app.logger.addHandler(file_handler)
    app.logger.addHandler(console_handler)
    app.logger.setLevel(level)
    app.logger.propagate = False          # Do NOT bubble up to root logger

    app.logger.info(
        'Logging configured  |  level=%-8s  |  file=%s',
        log_level,
        os.path.join(log_dir, log_file),
    )
