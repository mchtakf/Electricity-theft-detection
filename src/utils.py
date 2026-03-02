"""
Utility functions for the Electricity Theft Detection Pipeline.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional


def normalize_scores(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize an array to [0, 1]."""
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + 1e-10)


def get_seasonal_profile(
    df: pd.DataFrame,
    tariff_col: str,
    date_col: str,
    consumption_col: str,
    day_diff_col: str,
) -> Dict[str, Dict[int, float]]:
    """
    Learn monthly consumption profiles per tariff group.
    
    Returns dict: {tariff_name: {month_number: proportion}}
    Each profile sums to 1.0 across 12 months.
    """
    DEFAULT = {m: 1 / 12 for m in range(1, 13)}
    profiles = {}

    reliable = df[
        (df[day_diff_col] >= 25)
        & (df[day_diff_col] <= 40)
        & (df[consumption_col] > 0)
    ].copy()
    reliable["month"] = reliable[date_col].dt.month
    reliable["daily"] = reliable[consumption_col] / reliable[day_diff_col].clip(lower=1)

    for tariff in df[tariff_col].unique():
        t_data = reliable[reliable[tariff_col] == tariff]
        if len(t_data) < 30:
            profiles[str(tariff)] = DEFAULT.copy()
            continue

        medians = t_data.groupby("month")["daily"].median()
        total = sum(medians.get(m, 0.01) for m in range(1, 13))
        raw = {m: max(medians.get(m, 0.01) / total, 0.005) for m in range(1, 13)}
        t = sum(raw.values())
        profiles[str(tariff)] = {k: v / t for k, v in raw.items()}

    return profiles


def classify_negative_consumption(
    df: pd.DataFrame,
    consumption_col: str = "total_active",
    meter_col: str = "SayacEndeksUN",
    type_col: str = "EndeksTipiTanimi",
) -> pd.Series:
    """
    Classify negative consumption records into 3 categories:
      - 'meter_change': Different meter ID from previous reading
      - 'rollback':     Same meter, negative reading → THEFT SIGNAL
      - 'correction':   Manual/estimate reading type → ignore
      - 'normal':       Non-negative consumption
    """
    result = pd.Series("normal", index=df.index)
    negative = df[consumption_col] < 0

    if not negative.any():
        return result

    if meter_col in df.columns:
        prev_meter = df.groupby("AboneUN")[meter_col].shift(1)
        meter_changed = (df[meter_col] != prev_meter) & prev_meter.notna()
    else:
        meter_changed = pd.Series(False, index=df.index)

    manual_types = ["Manuel", "Tahmin Endeksi"]
    is_correction = df[type_col].isin(manual_types) if type_col in df.columns else pd.Series(False, index=df.index)

    result.loc[negative & meter_changed] = "meter_change"
    result.loc[negative & ~meter_changed & ~is_correction] = "rollback"
    result.loc[negative & is_correction] = "correction"

    return result


def generate_ai_explanation(row: pd.Series) -> str:
    """
    Generate human-readable explanation for a subscriber's risk score.
    Combines signals from all 4 models into a single narrative.
    """
    signals = []

    # Consumption patterns
    zero_pct = row.get("f09_sifir_oran", 0)
    if zero_pct > 0.5:
        signals.append(f"%{zero_pct*100:.0f} zero-consumption months")

    consec = row.get("f18_max_ardisik_dusuk", 0)
    if consec >= 3:
        signals.append(f"{int(consec)} consecutive low months")

    # Profile mismatch
    cosine = row.get("f25_cosine_sim", 1)
    if cosine < 0.5:
        signals.append(f"Profile mismatch ({cosine:.0%} similarity)")

    # Meter rollback
    rollback = row.get("f42_negatif_geri_sarma", 0)
    if rollback > 0:
        signals.append(f"⚠ {int(rollback)}x meter rollback!")

    # Reactive power
    cos_phi = row.get("f45_cos_phi", 1)
    if 0 < cos_phi < 0.85:
        signals.append("Abnormal reactive power")

    # Model signals
    if row.get("iso_skor", 0) > 0.7:
        signals.append("🌲 IsoForest anomaly")
    if row.get("ts_skor", 0) > 0.4:
        signals.append("📈 Group time-series deviation")

    # Individual time-series
    ind_score = row.get("ind_skor", 0)
    baseline = row.get("ind_gecmis_ort", None)
    recent = row.get("ind_son_donem_ort", None)
    drop_pct = row.get("ind_son3_dusus_pct", 0)

    if ind_score > 0.3 and baseline and pd.notna(baseline) and drop_pct > 20:
        signals.append(
            f"📉 Self-baseline drop {drop_pct:.0f}% ({baseline:.0f}→{recent:.0f} kWh)"
        )

    # Very low consumption
    if row.get("f01_ort_tuketim", 0) < 1:
        signals.append("Avg < 1 kWh")

    return " | ".join(signals) if signals else "General anomalous pattern"


def precision_at_k(df: pd.DataFrame, score_col: str, target_col: str, k: int) -> Tuple[int, float]:
    """Calculate precision@k for a given score column."""
    top_k = df.nlargest(k, score_col)
    hits = int(top_k[target_col].sum())
    return hits, hits / k
