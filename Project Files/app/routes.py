"""
OptiCrop – app/routes.py
Main Application Blueprint — All Page Routes

Routes:
  GET  /             → home.html
  GET  /about        → about.html
  GET  /predict      → predict.html (form; pre-filled from session if available)
  POST /predict      → Validates inputs → PredictionService → returns JSON
  GET  /result       → result.html (session-backed; redirects to /predict if empty)
  GET  /health       → JSON health-check (for deployment monitoring)

IMPORTANT – Submission Pattern:
  The prediction form uses fetch() (AJAX) with e.preventDefault().
  POST /predict therefore returns JSON, not render_template.
  The frontend JS calls renderResult(data) to display the result inline.

  Phase 14 Integration Point:
    Replace the PredictionService placeholder with the live ML inference call.

Architecture:
  • Blueprint pattern  — importable, testable, modular
  • Lazy service init  — PredictionService instantiated once per process
  • Session storage    — last_inputs + last_prediction stored without DB
  • Structured logging — every route entry/exit logged with relevant context
"""

import logging
from typing import Any, Dict, Optional

from flask import (
    Blueprint, jsonify, redirect, render_template,
    request, session, url_for
)

from app.services.model_loader import ModelLoadError
from app.services.prediction import PredictionService
from app.utils.validators import ValidationError, validate_prediction_inputs

# ── Blueprint registration ────────────────────────────────────────────────────
main = Blueprint('main', __name__)

# ── Module-level logger ───────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── PredictionService singleton ───────────────────────────────────────────────
_svc: Optional[PredictionService] = None


def _get_service() -> PredictionService:
    """
    Lazy-initializes the PredictionService singleton.

    Using module-level state avoids re-loading model.pkl on every request.
    Thread safety is not a concern here because Werkzeug's dev server is
    single-threaded; production deployments should use Gunicorn + gevent.
    """
    global _svc
    if _svc is None:
        _svc = PredictionService()
    return _svc


# ═══════════════════════════════════════════════════════════
# Page Routes
# ═══════════════════════════════════════════════════════════

@main.route('/')
@main.route('/home')
def home():
    """Renders the OptiCrop Home landing page."""
    logger.info('GET /  →  home.html')
    return render_template('home.html')


@main.route('/about')
def about():
    """Renders the About / Project Information page."""
    logger.info('GET /about  →  about.html')
    return render_template('about.html')


@main.route('/predict', methods=['GET', 'POST'])
def predict():
    """
    GET  → Renders the prediction form (predict.html).
           Restores the user's last-submitted values from the Flask session
           so they don't have to re-enter data on page refresh.

    POST → Accepts JSON or form-encoded data, validates all 7 parameters,
           calls PredictionService, and returns a JSON response.
           The predict.html page uses fetch() to POST and receives JSON —
           it renders the result inline without a page navigation.

    JSON Response Shape (POST):
        Success → { "success": true, "crop": "Rice", "confidence": 92.6, ... }
        Error   → { "success": false, "errors": { "field": "message" } }
    """
    if request.method == 'POST':
        return _handle_predict_post()

    # ── GET — render the form ─────────────────────────────────────────────────
    logger.info('GET /predict  →  predict.html')
    form_data = session.get('last_inputs', {})
    return render_template('predict.html', form_data=form_data)


@main.route('/result')
def result():
    """
    GET fallback for the Prediction Result page.

    The primary path to the result display is:
      POST /predict → JSON response → renderResult() in predict.js

    This GET endpoint exists so users can:
      • Bookmark a result page
      • Refresh without re-submitting the form

    If no session prediction exists, the user is redirected to the form.
    """
    logger.info('GET /result  →  result.html')
    result_data: Optional[Dict] = session.get('last_prediction')

    if result_data is None:
        logger.info('No session prediction — redirecting to /predict')
        return redirect(url_for('main.predict'))

    return render_template('result.html', result_data=result_data)


@main.route('/health')
def health():
    """
    Health-check endpoint for deployment monitoring (Render, Railway, etc.).

    Returns:
        JSON: { "status": "ok", "app": "OptiCrop", "version": "1.0.0" }
    """
    logger.debug('GET /health  →  200 OK')
    return jsonify({
        'status':  'ok',
        'app':     'OptiCrop',
        'version': '1.0.0',
    }), 200


# ═══════════════════════════════════════════════════════════
# Private handlers
# ═══════════════════════════════════════════════════════════

def _handle_predict_post() -> Any:
    """
    Handles the POST /predict flow:

      1. Extract raw values from form-encoded OR JSON body
      2. Validate and sanitize all 7 soil/climate parameters
      3. Persist validated inputs in Flask session
      4. Call PredictionService.predict()
      5. Persist result in Flask session
      6. Return JSON response to the frontend fetch() call

    Returns:
        Flask JSON response (always JSON — the JS handles display)
    """
    # ── 1. Extract — support both form-encoded and JSON bodies ────────────────
    if request.is_json:
        raw: Dict[str, Any] = request.get_json(silent=True) or {}
    else:
        raw = {
            'nitrogen':    request.form.get('nitrogen',    '').strip(),
            'phosphorus':  request.form.get('phosphorus',  '').strip(),
            'potassium':   request.form.get('potassium',   '').strip(),
            'temperature': request.form.get('temperature', '').strip(),
            'humidity':    request.form.get('humidity',    '').strip(),
            'ph':          request.form.get('ph',          '').strip(),
            'rainfall':    request.form.get('rainfall',    '').strip(),
        }

    logger.info('POST /predict  |  raw=%s', raw)

    # ── 2. Validate ───────────────────────────────────────────────────────────
    try:
        validated = validate_prediction_inputs(raw)
    except ValidationError as exc:
        logger.warning('Validation failed  |  errors=%s', exc.errors)
        return jsonify({
            'success': False,
            'errors':  exc.errors,
            'message': str(exc),
        }), 422

    # ── 3. Persist inputs ─────────────────────────────────────────────────────
    session['last_inputs'] = validated
    session.modified = True

    # ── 4. Call prediction service ────────────────────────────────────────────
    try:
        result_data = _get_service().predict(validated)
        logger.info(
            'Prediction success  |  crop=%-12s  |  confidence=%.1f%%',
            result_data.get('crop'),
            result_data.get('confidence', 0.0),
        )
    except ModelLoadError as exc:
        logger.error('Model not available  |  artifact=%s  |  %s', exc.artifact, exc)
        return jsonify({
            'success': False,
            'errors':  {},
            'message': (
                'The AI prediction model is not available. '
                'Please contact the administrator to ensure model.pkl is deployed.'
            ),
        }), 503

    except Exception as exc:  # noqa: BLE001
        logger.error('PredictionService unexpected error  |  %s', exc, exc_info=True)
        return jsonify({
            'success': False,
            'errors':  {},
            'message': (
                'The prediction engine encountered an unexpected error. '
                'Please try again.'
            ),
        }), 500

    # ── 5. Persist result ─────────────────────────────────────────────────────
    session['last_prediction'] = result_data
    session.modified = True

    # ── 6. Return JSON to frontend fetch() ───────────────────────────────────
    return jsonify({'success': True, **result_data}), 200
