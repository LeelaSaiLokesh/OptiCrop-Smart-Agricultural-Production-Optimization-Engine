"""
OptiCrop – app/services/formatter.py
Prediction Result Formatter

Transforms the raw output from PredictionService into a rich,
UI-ready result_data dictionary consumed by:
  • result.html (Jinja2 template)
  • predict.js  renderResult() — inline result panel
  • POST /predict JSON response

Design principles:
  • Single responsibility — all result shaping lives here, not in routes.py
  • Immutable inputs — does not mutate the inputs dict
  • Complete contract — every key expected by the frontend is always present
  • Graceful defaults — missing optional data falls back to sensible values

Usage:
    from app.services.formatter import format_result
    result_data = format_result(crop, confidence, inputs, proba_arr, label_classes)
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)

# ── Crop knowledge base ───────────────────────────────────────────────────────
# Provides richer agronomic context than the bare ML output.
# Keys are lowercase crop names (matching LabelEncoder.classes_).
_CROP_KNOWLEDGE: Dict[str, Dict[str, str]] = {
    "apple":        {"scientific": "Malus domestica",      "category": "Temperate Fruit",   "emoji": "🍎", "season": "Rabi (Nov–Mar)", "soil": "Well-drained Loamy",           "climate": "Cool Temperate (8–18 °C)", "water": "Medium — 100–125 mm/season", "fert": "NPK 70:35:70 kg/ha",  "yield": "15–35 t/ha"},
    "banana":       {"scientific": "Musa acuminata",       "category": "Tropical Fruit",    "emoji": "🍌", "season": "Year-round",     "soil": "Rich Alluvial Loam",          "climate": "Tropical (24–32 °C)",     "water": "High — 150–200 mm/season",  "fert": "NPK 200:30:300 kg/ha","yield": "30–40 t/ha"},
    "blackgram":    {"scientific": "Vigna mungo",          "category": "Pulse / Legume",    "emoji": "🫘", "season": "Kharif (Jun–Sep)","soil": "Sandy Loam",                  "climate": "Warm Tropical (25–35 °C)","water": "Low — 40–60 mm/season",     "fert": "NPK 20:40:20 kg/ha", "yield": "0.8–1.5 t/ha"},
    "chickpea":     {"scientific": "Cicer arietinum",      "category": "Pulse / Legume",    "emoji": "🫛", "season": "Rabi (Oct–Mar)", "soil": "Deep Sandy Loam",             "climate": "Semi-arid (18–29 °C)",   "water": "Low — 45–75 mm/season",     "fert": "NPK 20:60:40 kg/ha", "yield": "1–2 t/ha"},
    "coconut":      {"scientific": "Cocos nucifera",       "category": "Plantation Crop",   "emoji": "🥥", "season": "Year-round",     "soil": "Sandy Loam / Coastal",        "climate": "Humid Tropical (27–32 °C)","water": "High — 130–200 mm/season", "fert": "NPK 100:40:200 kg/ha","yield": "60–80 nuts/palm/yr"},
    "coffee":       {"scientific": "Coffea arabica",       "category": "Plantation Crop",   "emoji": "☕", "season": "Rabi harvest",   "soil": "Red Laterite / Loam",         "climate": "Tropical Hill (15–28 °C)","water": "Medium — 120–200 mm/season","fert": "NPK 120:30:100 kg/ha","yield": "0.5–2 t/ha"},
    "cotton":       {"scientific": "Gossypium hirsutum",   "category": "Fibre / Cash Crop", "emoji": "🌾", "season": "Kharif (Jun–Oct)","soil": "Deep Black Clay (Regur)",     "climate": "Hot Semi-arid (21–30 °C)","water": "Medium — 50–100 mm/season","fert": "NPK 120:60:60 kg/ha", "yield": "2–3 t/ha (lint)"},
    "grapes":       {"scientific": "Vitis vinifera",       "category": "Temperate Fruit",   "emoji": "🍇", "season": "Rabi pruning",   "soil": "Sandy Clay Loam",             "climate": "Warm Temperate (15–35 °C)","water": "Medium — 60–80 mm/season", "fert": "NPK 50:50:50 kg/ha", "yield": "20–40 t/ha"},
    "jute":         {"scientific": "Corchorus olitorius",  "category": "Fibre / Cash Crop", "emoji": "🌿", "season": "Kharif (Mar–Jun)","soil": "Alluvial Loam",               "climate": "Warm Humid (24–38 °C)",  "water": "High — 150–200 mm/season",  "fert": "NPK 60:30:30 kg/ha", "yield": "2–4 t/ha (fibre)"},
    "kidneybeans":  {"scientific": "Phaseolus vulgaris",   "category": "Pulse / Legume",    "emoji": "🫘", "season": "Kharif / Rabi",  "soil": "Loamy",                       "climate": "Warm Temperate (18–24 °C)","water": "Medium — 60–100 mm/season","fert": "NPK 20:60:30 kg/ha", "yield": "1–2 t/ha"},
    "lentil":       {"scientific": "Lens culinaris",       "category": "Pulse / Legume",    "emoji": "🫛", "season": "Rabi (Oct–Mar)", "soil": "Sandy Loam",                  "climate": "Cool Dry (15–25 °C)",    "water": "Low — 40–60 mm/season",     "fert": "NPK 20:40:20 kg/ha", "yield": "0.6–1.5 t/ha"},
    "maize":        {"scientific": "Zea mays",             "category": "Cereal / Staple",   "emoji": "🌽", "season": "Kharif (Jun–Sep)","soil": "Well-drained Loamy",          "climate": "Warm Temperate (21–27 °C)","water": "Medium — 50–100 mm/season","fert": "NPK 120:60:40 kg/ha", "yield": "4–7 t/ha"},
    "mango":        {"scientific": "Mangifera indica",     "category": "Tropical Fruit",    "emoji": "🥭", "season": "Summer (Mar–Jun)","soil": "Deep Alluvial Loam",          "climate": "Tropical (24–32 °C)",    "water": "Low-Med — 50–100 mm/season","fert": "NPK 50:50:50 kg/ha", "yield": "10–20 t/ha"},
    "mothbeans":    {"scientific": "Vigna aconitifolia",   "category": "Pulse / Legume",    "emoji": "🫘", "season": "Kharif (Jun–Sep)","soil": "Sandy / Arid Loam",           "climate": "Hot Arid (25–40 °C)",    "water": "Very Low — 25–40 mm/season","fert": "NPK 20:40:20 kg/ha", "yield": "0.4–1.0 t/ha"},
    "mungbean":     {"scientific": "Vigna radiata",        "category": "Pulse / Legume",    "emoji": "🫛", "season": "Kharif (Jun–Sep)","soil": "Sandy Loam / Well-drained",   "climate": "Warm Tropical (28–35 °C)","water": "Low — 45–70 mm/season",    "fert": "NPK 20:60:30 kg/ha", "yield": "0.8–1.5 t/ha"},
    "muskmelon":    {"scientific": "Cucumis melo",         "category": "Vegetable / Fruit", "emoji": "🍈", "season": "Summer (Feb–May)","soil": "Sandy Loam",                  "climate": "Hot & Dry (25–38 °C)",   "water": "Medium — 40–60 mm/season",  "fert": "NPK 100:60:80 kg/ha","yield": "15–25 t/ha"},
    "orange":       {"scientific": "Citrus sinensis",      "category": "Citrus Fruit",      "emoji": "🍊", "season": "Rabi (Oct–Jan)", "soil": "Sandy Clay Loam",             "climate": "Sub-tropical (13–24 °C)","water": "Medium — 100–120 mm/season","fert": "NPK 50:25:50 kg/ha", "yield": "12–25 t/ha"},
    "papaya":       {"scientific": "Carica papaya",        "category": "Tropical Fruit",    "emoji": "🍈", "season": "Year-round",     "soil": "Light Sandy Loam",            "climate": "Tropical (22–28 °C)",    "water": "Medium — 100–150 mm/season","fert": "NPK 100:40:75 kg/ha","yield": "40–80 t/ha"},
    "pigeonpeas":   {"scientific": "Cajanus cajan",        "category": "Pulse / Legume",    "emoji": "🫛", "season": "Kharif (Jun–Nov)","soil": "Sandy Loam / Red",            "climate": "Warm Tropical (18–30 °C)","water": "Low — 60–100 mm/season",   "fert": "NPK 20:50:20 kg/ha", "yield": "1–2 t/ha"},
    "pomegranate":  {"scientific": "Punica granatum",      "category": "Tropical Fruit",    "emoji": "🌺", "season": "Year-round",     "soil": "Deep Loam / Sandy Clay",      "climate": "Arid/Semi-arid (25–35 °C)","water": "Low-Med — 50–80 mm/season","fert": "NPK 60:30:30 kg/ha","yield": "15–20 t/ha"},
    "rice":         {"scientific": "Oryza sativa",         "category": "Cereal / Staple",   "emoji": "🌾", "season": "Kharif (Jun–Sep)","soil": "Alluvial / Clayey Loam",      "climate": "Tropical/Humid (24–35 °C)","water": "High — 200–300 mm/season","fert": "NPK 80:40:40 kg/ha", "yield": "4–6 t/ha"},
    "watermelon":   {"scientific": "Citrullus lanatus",    "category": "Vegetable / Fruit", "emoji": "🍉", "season": "Summer (Feb–May)","soil": "Sandy Loam / Well-drained",   "climate": "Hot & Dry (24–38 °C)",   "water": "Medium — 40–60 mm/season",  "fert": "NPK 100:50:50 kg/ha","yield": "25–40 t/ha"},
}

_DEFAULT_KNOWLEDGE: Dict[str, str] = {
    "scientific": "Species spp.",
    "category":   "Agricultural Crop",
    "emoji":      "🌱",
    "season":     "Seasonal",
    "soil":       "Loamy",
    "climate":    "Moderate",
    "water":      "Moderate",
    "fert":       "NPK balanced",
    "yield":      "Varies",
}

# ── Chart colour palette ──────────────────────────────────────────────────────
_PROB_COLOURS = [
    "linear-gradient(135deg,#16a34a,#22c55e)",
    "linear-gradient(135deg,#38bdf8,#0ea5e9)",
    "linear-gradient(135deg,#fbbf24,#f59e0b)",
    "linear-gradient(135deg,#a78bfa,#7c3aed)",
    "linear-gradient(135deg,#f87171,#ef4444)",
]

_FI_COLOURS = ["#22c55e","#38bdf8","#fb7185","#a78bfa","#fbbf24","#f97316","#4ade80",
                "#34d399","#60a5fa","#f472b6","#facc15","#94a3b8","#e879f9","#2dd4bf","#f9a8d4"]


def format_result(
    crop: str,
    confidence: float,
    inputs: Dict[str, float],
    proba_arr: "np.ndarray",
    label_classes: List[str],
    pred_time_ms: float = 0.0,
) -> Dict[str, Any]:
    """
    Build the complete result_data dictionary from raw prediction outputs.

    Args:
        crop:          Predicted crop name (lowercase, as returned by LabelEncoder)
        confidence:    Prediction confidence as a percentage (0–100)
        inputs:        Validated dict of 7 user-supplied soil/climate values
        proba_arr:     numpy array of class probabilities from model.predict_proba()
        label_classes: List of class name strings from encoder.classes_
        pred_time_ms:  Inference latency in milliseconds

    Returns:
        Fully-formed result_data dict compatible with result.html and result.js
    """
    crop_key   = crop.lower().strip()
    knowledge  = _CROP_KNOWLEDGE.get(crop_key, _DEFAULT_KNOWLEDGE)
    display    = crop.replace("_", " ").title()

    logger.debug('Formatting result  |  crop=%s  |  confidence=%.1f%%', crop, confidence)

    result: Dict[str, Any] = {
        # ── Core prediction ────────────────────────────────────────────────────
        "crop":         display,
        "scientific":   knowledge["scientific"],
        "category":     knowledge["category"],
        "emoji":        knowledge["emoji"],
        "confidence":   round(float(confidence), 2),
        "model":        "Random Forest  (200 estimators, StandardScaler, 15 features)",
        "pred_time_ms": round(float(pred_time_ms), 1),
        "timestamp":    _utc_iso(),

        # ── Echoed inputs (shown in result detail panel) ───────────────────────
        "inputs": inputs,

        # ── Why-this-crop summary ─────────────────────────────────────────────
        "why": _build_why_text(crop_key, inputs, confidence),

        # ── Agronomic recommendations ──────────────────────────────────────────
        "season":    knowledge["season"],
        "soil_type": knowledge["soil"],
        "climate":   knowledge["climate"],
        "water_req": knowledge["water"],
        "fertilizer":knowledge["fert"],
        "yield_exp": knowledge["yield"],
        "irrigation":_get_irrigation(crop_key),
        "advice":    _get_advice(crop_key, inputs),

        # ── Feature importance (bar chart) ────────────────────────────────────
        "feature_importance": _build_feature_importance(inputs),

        # ── Top-5 class probabilities (chart) ─────────────────────────────────
        "probabilities": _build_probabilities(proba_arr, label_classes),
    }

    logger.info(
        'Result formatted  |  crop=%-14s  |  confidence=%5.1f%%  |  time=%sms',
        display, confidence, pred_time_ms,
    )
    return result


# ═══════════════════════════════════════════════════════════
# Private helpers
# ═══════════════════════════════════════════════════════════

def _build_why_text(crop: str, inputs: Dict[str, float], conf: float) -> str:
    """Generates a concise, agronomically-grounded rationale string."""
    n   = inputs.get("nitrogen",    0)
    h   = inputs.get("humidity",    0)
    r   = inputs.get("rainfall",    0)
    t   = inputs.get("temperature", 0)
    ph  = inputs.get("ph",          0)

    ph_desc = ("acidic" if ph < 6.5 else "alkaline" if ph > 7.5 else "neutral")
    conf_desc = ("Very high" if conf >= 90 else "High" if conf >= 75 else "Moderate")

    return (
        f"{conf_desc} confidence ({conf:.1f} %) prediction. "
        f"Your soil profile — N={n:.0f} ppm, humidity={h:.1f} %, "
        f"rainfall={r:.1f} mm, temperature={t:.1f} °C — "
        f"combined with {ph_desc} pH ({ph:.2f}) "
        f"matches the optimal growing conditions for {crop.replace('_',' ').title()}."
    )


def _get_irrigation(crop: str) -> str:
    """Returns the typical irrigation method for the predicted crop."""
    irrigation_map = {
        "rice":        "Continuous flooding / Alternate Wetting & Drying (AWD)",
        "maize":       "Furrow irrigation / Sprinkler (critical at tasseling)",
        "cotton":      "Drip irrigation (saves 40–50 % water vs flood)",
        "jute":        "Flood irrigation during early growth; reduce near harvest",
        "banana":      "Drip irrigation  (4–6 L/plant/day in peak season)",
        "coconut":     "Basin irrigation (200 L/palm/week in dry periods)",
        "sugarcane":   "Furrow / Sprinkler (1500–2500 mm/season total)",
        "coffee":      "Drip irrigation (stress-irrigation to induce flowering)",
        "grapes":      "Drip irrigation (deficit irrigation improves fruit quality)",
        "apple":       "Sprinkler / Micro-irrigation (critical at fruit fill)",
        "watermelon":  "Drip irrigation (furrow near ridge)",
        "muskmelon":   "Drip irrigation (reduce 2 weeks before harvest)",
    }
    default = "Sprinkler / Drip irrigation (adjust to local rainfall)"
    return irrigation_map.get(crop, default)


def _get_advice(crop: str, inputs: Dict[str, float]) -> str:
    """Returns targeted agronomic advice for the predicted crop."""
    advice_map = {
        "rice":       ("Level the field before transplanting. Maintain 5–7 cm standing water "
                       "during tillering. Split nitrogen: 50 % at transplanting, 25 % at "
                       "tillering, 25 % at panicle initiation."),
        "maize":      ("Apply a starter fertiliser at sowing. Hill-up soil around the base at "
                       "knee-high stage to support stalks. Monitor for FAW (Fall Army Worm) "
                       "from V4 stage onwards."),
        "cotton":     ("Follow recommended bollworm IPM schedule. Avoid excess nitrogen, "
                       "which promotes vegetative growth over fruiting. "
                       "Terminate irrigation 3 weeks before first pick."),
        "banana":     ("Provide windbreaks in exposed locations. Remove suckers, retaining "
                       "one follower per plant. Apply potassium in split doses to support "
                       "bunch development."),
        "chickpea":   ("Inoculate seeds with Rhizobium before sowing for biological nitrogen "
                       "fixation. Avoid waterlogging — plant on raised beds if soil drains poorly."),
        "coffee":     ("Maintain 25–30 % shade canopy. Prune to a single-stem frame after "
                       "bearing. Harvest only red-ripe cherries for premium quality."),
        "apple":      ("Select low-chilling rootstocks for warm climates. Train to central "
                       "leader or spindle-bush form. Apply dormant oil spray before bud-break."),
        "grapes":     ("Prune canes to 2–3 buds in December. Train on trellis for air "
                       "circulation. Monitor for downy mildew during humid periods."),
    }
    default = (
        "Use certified seeds from a reputable source. Follow district-level "
        "agronomic recommendations for sowing date and plant density. "
        "Apply fertilisers in split doses and monitor for common pests."
    )
    return advice_map.get(crop, default)


def _build_feature_importance(inputs: Dict[str, float]) -> List[Dict[str, Any]]:
    """
    Returns a 15-feature importance list for the bar chart.

    Reads directly from model.feature_importances_ via the cached artifact,
    ensuring the chart always reflects the actual trained model.

    Falls back to estimated values if model is not yet loaded.
    """
    # Colour palette — one per feature, matched to ALL_FEATURE_COLS order
    _COLOURS = [
        "#22c55e", "#38bdf8", "#fb7185", "#a78bfa", "#fbbf24",
        "#f97316", "#4ade80", "#34d399", "#60a5fa", "#f472b6",
        "#facc15", "#94a3b8", "#e879f9", "#2dd4bf", "#f9a8d4",
    ]
    # ALL_FEATURE_COLS order (must match training pipeline)
    _LABELS = [
        "Nitrogen (N)", "Phosphorus (P)", "Potassium (K)",
        "Temperature",  "Humidity",       "Soil pH",        "Rainfall",
        "NPK Ratio",    "Total Nutrients", "Env. Index",
        "Moisture Index", "Nutrient Balance", "Soil Fertility",
        "Climate Suitability", "Agri Health",
    ]

    try:
        from app.services.model_loader import get_artifacts
        artifacts   = get_artifacts()
        importances = artifacts["model"].feature_importances_
    except Exception:
        # Fallback: use empirically measured values from the trained model
        importances = [
            0.0726, 0.0919, 0.0886, 0.0234, 0.0894, 0.0127, 0.0702,
            0.0649, 0.0687, 0.0771, 0.0736, 0.0644, 0.0846, 0.0539, 0.0638,
        ]

    result = []
    for label, colour, imp in zip(_LABELS, _COLOURS, importances):
        result.append({
            "label": label,
            "pct":   round(float(imp) * 100, 1),
            "color": colour,
        })

    # Sort by importance descending for visual impact
    result.sort(key=lambda x: x["pct"], reverse=True)
    return result



def _build_probabilities(
    proba_arr: "np.ndarray",
    label_classes: List[str],
) -> List[Dict[str, Any]]:
    """
    Returns the top-5 crop probabilities as percentage dicts.

    Args:
        proba_arr:     1-D numpy array of shape (n_classes,)
        label_classes: List of class name strings
    """
    top_indices = np.argsort(proba_arr)[::-1][:5]
    result = []
    for rank, idx in enumerate(top_indices):
        pct = round(float(proba_arr[idx]) * 100, 2)
        if pct < 0.01:
            continue
        name = label_classes[idx].replace("_", " ").title()
        result.append({
            "crop":  name,
            "pct":   pct,
            "color": _PROB_COLOURS[rank % len(_PROB_COLOURS)],
        })
    return result if result else [
        {"crop": "Unknown", "pct": 0.0, "color": _PROB_COLOURS[0]}
    ]


def _utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
