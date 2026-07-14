"""
validation.py - OptiCrop Input Validation Module
=================================================
Project : OptiCrop - Smart Agricultural Production Optimization Engine
Phase   : Data Preprocessing - Validation Layer
Version : 1.0
"""

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

FEATURE_RANGES = {
    'N':           {'min': 0,    'max': 140,  'unit': 'kg/ha', 'label': 'Nitrogen (N)'},
    'P':           {'min': 5,    'max': 145,  'unit': 'kg/ha', 'label': 'Phosphorous (P)'},
    'K':           {'min': 5,    'max': 205,  'unit': 'kg/ha', 'label': 'Potassium (K)'},
    'temperature': {'min': 0.0,  'max': 50.0, 'unit': 'C',     'label': 'Temperature'},
    'humidity':    {'min': 10.0, 'max': 100.0,'unit': '%',     'label': 'Humidity'},
    'ph':          {'min': 3.0,  'max': 10.0, 'unit': 'pH',    'label': 'Soil pH'},
    'rainfall':    {'min': 20.0, 'max': 300.0,'unit': 'mm',    'label': 'Rainfall'},
}

FEATURE_ORDER = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']


@dataclass
class ValidationResult:
    """Structured result of a feature validation pass."""
    is_valid:     bool = True
    errors:       dict = field(default_factory=dict)
    cleaned_data: dict = field(default_factory=dict)

    def add_error(self, field_name: str, message: str) -> None:
        self.is_valid = False
        self.errors[field_name] = message

    def __repr__(self) -> str:
        status = "VALID" if self.is_valid else f"INVALID ({len(self.errors)} errors)"
        return f"ValidationResult(status={status})"


def validate_numeric_field(field_name: str, raw_value: Any, result: ValidationResult):
    """Validate a single feature field for presence, type, and range."""
    label = FEATURE_RANGES[field_name]['label']
    min_v = FEATURE_RANGES[field_name]['min']
    max_v = FEATURE_RANGES[field_name]['max']
    unit  = FEATURE_RANGES[field_name]['unit']

    if raw_value is None or str(raw_value).strip() == '':
        result.add_error(field_name, f"{label} is required ({min_v}-{max_v} {unit}).")
        return None

    try:
        float_value = float(raw_value)
    except (ValueError, TypeError):
        result.add_error(field_name, f"{label} must be a number. Got: '{raw_value}'.")
        return None

    if not (min_v <= float_value <= max_v):
        result.add_error(
            field_name,
            f"{label} must be between {min_v} and {max_v} {unit}. Got: {float_value:.2f}."
        )
        return None

    return float_value


def validate_all_features(form_data: dict) -> ValidationResult:
    """
    Validate all 7 input features from user-submitted form data.

    Args:
        form_data: Dict of raw inputs {'N': '90', 'P': '42', ...}

    Returns:
        ValidationResult with is_valid, errors, and cleaned_data.
    """
    result = ValidationResult()

    for feature_name in FEATURE_ORDER:
        raw_value = form_data.get(feature_name)
        cleaned   = validate_numeric_field(feature_name, raw_value, result)
        if cleaned is not None:
            result.cleaned_data[feature_name] = cleaned

    if result.is_valid:
        logger.info(f"Validation passed: {result.cleaned_data}")
    else:
        logger.warning(f"Validation failed: {result.errors}")

    return result


def format_for_model(cleaned_data: dict) -> list:
    """
    Convert validated input dict to an ordered float list for model inference.
    Order: [N, P, K, temperature, humidity, ph, rainfall]

    Args:
        cleaned_data: Dict of validated float values from ValidationResult.

    Returns:
        Ordered list of floats.

    Raises:
        KeyError: If any required feature is missing.
    """
    missing = [f for f in FEATURE_ORDER if f not in cleaned_data]
    if missing:
        raise KeyError(f"Missing features for inference: {missing}")
    return [cleaned_data[f] for f in FEATURE_ORDER]
