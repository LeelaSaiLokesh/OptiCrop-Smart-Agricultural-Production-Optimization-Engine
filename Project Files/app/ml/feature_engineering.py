"""
feature_engineering.py - OptiCrop Feature Engineering Module
=============================================================
Project : OptiCrop - Smart Agricultural Production Optimization Engine
Phase   : Data Preprocessing - Feature Engineering
Version : 1.0

Creates 8 domain-knowledge composite features from the 7 raw
agronomic measurements. Each engineered feature captures a
nutrient interaction or ecological relationship.

Engineered Features:
    E-1  NPK_Ratio          : N / (P + K) - nitrogen dominance
    E-2  Total_Nutrients    : N + P + K   - total macronutrient load
    E-3  Env_Index          : temp * humidity / 100 - heat-moisture stress
    E-4  Moisture_Index     : humidity * rainfall / 100 - water availability
    E-5  Nutrient_Balance   : (P + K) / N - fruiting nutrient dominance
    E-6  Soil_Fertility     : N + 2*P + 1.5*K - weighted fertility score
    E-7  Climate_Suitability: rainfall*humidity / temperature - ecology index
    E-8  Agri_Health        : (N*P*K) / (temp*humidity) - field productivity
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

EPSILON = 1e-6  # Division-by-zero guard

ENGINEERED_FEATURE_COLS = [
    'NPK_Ratio', 'Total_Nutrients', 'Env_Index', 'Moisture_Index',
    'Nutrient_Balance', 'Soil_Fertility', 'Climate_Suitability', 'Agri_Health'
]

ALL_FEATURE_COLS = [
    'N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall',
    'NPK_Ratio', 'Total_Nutrients', 'Env_Index', 'Moisture_Index',
    'Nutrient_Balance', 'Soil_Fertility', 'Climate_Suitability', 'Agri_Health'
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create 8 domain-knowledge composite features from raw agronomic data.

    Args:
        df: Cleaned DataFrame with all 7 raw numerical features.

    Returns:
        DataFrame with 7 original + 8 engineered = 15 feature columns.

    Raises:
        KeyError: If any required source column is missing.
    """
    required = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    df = df.copy()  # Defensive copy - avoid mutating input

    # E-1: NPK Ratio - Nitrogen dominance indicator
    # High (>1.5): cereals/cash crops | Low (<0.5): legumes/fruits
    df['NPK_Ratio'] = df['N'] / (df['P'] + df['K'] + EPSILON)

    # E-2: Total Nutrient Level - Overall macronutrient load
    # Proxy for farming intensity and input cost
    df['Total_Nutrients'] = df['N'] + df['P'] + df['K']

    # E-3: Environmental Index - Heat-moisture stress composite
    # High: tropical-humid crops | Low: cool-dry crops
    df['Env_Index'] = (df['temperature'] * df['humidity']) / 100.0

    # E-4: Moisture Index - Combined water availability indicator
    # High: paddy/tropical crops | Low: dryland legumes
    df['Moisture_Index'] = (df['humidity'] * df['rainfall']) / 100.0

    # E-5: Nutrient Balance Score - Fruiting/rooting vs nitrogen dominance
    # High (>2.0): fruit crops | Low (<0.5): cereal/nitrogen systems
    df['Nutrient_Balance'] = (df['P'] + df['K']) / (df['N'] + EPSILON)

    # E-6: Soil Fertility Score - Weighted macronutrient composite
    # P double-weighted (most limiting globally); K 1.5x weighted
    df['Soil_Fertility'] = df['N'] + (2.0 * df['P']) + (1.5 * df['K'])

    # E-7: Climate Suitability Index - Rainfall-humidity-temperature ecology
    # High: cool-humid-high-rainfall | Low: warm-dry-low-rainfall
    df['Climate_Suitability'] = (
        (df['rainfall'] * df['humidity']) / (df['temperature'] + EPSILON)
    )

    # E-8: Agricultural Health Score - NPK * climatic stress interaction
    # High: nutrient-rich soil under mild climate stress (ideal conditions)
    df['Agri_Health'] = (
        (df['N'] * df['P'] * df['K']) /
        (df['temperature'] * df['humidity'] + EPSILON)
    )

    logger.info(
        f"Feature engineering complete: "
        f"{len(ENGINEERED_FEATURE_COLS)} features added | "
        f"Total columns: {df.shape[1]}"
    )
    return df


def engineer_single_input(raw_features: list) -> list:
    """
    Apply feature engineering to a single inference input vector.

    Used by the Flask prediction pipeline to transform raw user input
    before scaling and model inference.

    Args:
        raw_features: Ordered list [N, P, K, temp, humidity, ph, rainfall]

    Returns:
        Extended list with 15 features (7 original + 8 engineered).

    Raises:
        ValueError: If raw_features does not contain exactly 7 values.
    """
    if len(raw_features) != 7:
        raise ValueError(
            f"Expected 7 features [N,P,K,temp,humidity,ph,rainfall], "
            f"got {len(raw_features)}."
        )

    N, P, K, temp, humidity, ph, rainfall = raw_features

    npk_ratio          = N / (P + K + EPSILON)
    total_nutrients    = N + P + K
    env_index          = (temp * humidity) / 100.0
    moisture_index     = (humidity * rainfall) / 100.0
    nutrient_balance   = (P + K) / (N + EPSILON)
    soil_fertility     = N + (2.0 * P) + (1.5 * K)
    climate_suitability = (rainfall * humidity) / (temp + EPSILON)
    agri_health        = (N * P * K) / (temp * humidity + EPSILON)

    return [
        N, P, K, temp, humidity, ph, rainfall,
        npk_ratio, total_nutrients, env_index, moisture_index,
        nutrient_balance, soil_fertility, climate_suitability, agri_health
    ]
