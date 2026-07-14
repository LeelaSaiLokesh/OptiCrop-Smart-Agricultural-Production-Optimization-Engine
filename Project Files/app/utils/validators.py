"""
OptiCrop – app/utils/validators.py
Input Validation & Sanitization Layer

Validates and sanitizes the 7 soil/climate parameters before they
reach the PredictionService. Field bounds mirror predict.html input
attributes and the preprocessing.py NUMERICAL_FEATURES config.

Usage:
    from app.utils.validators import validate_prediction_inputs, ValidationError

    try:
        validated = validate_prediction_inputs(request.form)
    except ValidationError as exc:
        return jsonify({"success": False, "errors": exc.errors}), 422

Validation pipeline per field:
  1. Presence check     — field must not be empty / missing
  2. String sanitization — strip whitespace, HTML-escape
  3. Numeric format guard — safe regex rejects scripts / junk
  4. Float conversion    — safe parse with ValueError catch
  5. Range validation    — dataset-derived min/max bounds
"""

import html
import re
from typing import Any, Dict, Optional


# ── Field configuration ───────────────────────────────────────────────────────
# Mirrors: predict.html input[min/max], preprocessing.py NUMERICAL_FEATURES
FIELD_CONFIG: Dict[str, Dict[str, Any]] = {
    'nitrogen':    {'label': 'Nitrogen (N)',    'min': 0.0,  'max': 140.0, 'unit': 'ppm'},
    'phosphorus':  {'label': 'Phosphorus (P)',  'min': 5.0,  'max': 145.0, 'unit': 'ppm'},
    'potassium':   {'label': 'Potassium (K)',   'min': 5.0,  'max': 205.0, 'unit': 'ppm'},
    'temperature': {'label': 'Temperature',     'min': 8.0,  'max': 44.0,  'unit': '°C'},
    'humidity':    {'label': 'Humidity',        'min': 14.0, 'max': 100.0, 'unit': '%'},
    'ph':          {'label': 'pH Level',        'min': 3.5,  'max': 10.0,  'unit': 'pH'},
    'rainfall':    {'label': 'Rainfall',        'min': 20.0, 'max': 300.0, 'unit': 'mm'},
}

# Safe numeric pattern — matches integers and floats only (no scripts)
_SAFE_NUMBER_RE = re.compile(r'^-?\d+(\.\d+)?$')


class ValidationError(Exception):
    """
    Raised when one or more prediction input fields fail validation.

    Attributes:
        message: Human-readable summary  (str(exc))
        errors:  Per-field error dict    {field_name: error_message}
    """

    def __init__(
        self,
        message: str,
        errors: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__(message)
        self.errors: Dict[str, str] = errors or {}


def _sanitize(value: Any) -> str:
    """
    Converts value to string, strips whitespace, and HTML-escapes.
    Returns empty string for None.
    """
    if value is None:
        return ''
    return html.escape(str(value).strip())


def _is_safe_number(value: str) -> bool:
    """Returns True only if the string looks like a plain float/int."""
    return bool(_SAFE_NUMBER_RE.match(value))


def validate_prediction_inputs(
    raw_inputs: Dict[str, Any],
) -> Dict[str, float]:
    """
    Validates and sanitizes the 7 soil/climate prediction parameters.

    Args:
        raw_inputs: Raw dict of values — typically request.form or
                    request.get_json(). Keys must match FIELD_CONFIG.

    Returns:
        Dict[str, float] — validated, type-converted inputs ready for
        the PredictionService.

    Raises:
        ValidationError: If any field fails, with a per-field errors dict.
                         Call exc.errors to get {field: message} for the UI.
    """
    validated: Dict[str, float] = {}
    errors: Dict[str, str]      = {}

    for field, cfg in FIELD_CONFIG.items():
        label   = cfg['label']
        min_val = cfg['min']
        max_val = cfg['max']
        unit    = cfg['unit']

        # 1. Presence check
        raw = raw_inputs.get(field)
        if raw is None or str(raw).strip() == '':
            errors[field] = f'{label} is required.'
            continue

        # 2. Sanitize
        safe = _sanitize(raw)
        if not safe:
            errors[field] = f'{label} is required.'
            continue

        # 3. Numeric format guard (prevents script injection via field value)
        if not _is_safe_number(safe):
            errors[field] = f'{label} must be a valid number (e.g. 6.5).'
            continue

        # 4. Float conversion
        try:
            value = float(safe)
        except ValueError:
            errors[field] = f'{label} must be a valid number.'
            continue

        # 5. Range validation (bounds match dataset and HTML input attributes)
        if not (min_val <= value <= max_val):
            errors[field] = (
                f'{label} must be between {min_val} and {max_val} {unit}. '
                f'You entered {value}.'
            )
            continue

        validated[field] = value

    if errors:
        raise ValidationError(
            f'Validation failed — {len(errors)} field(s) require attention.',
            errors=errors,
        )

    return validated
