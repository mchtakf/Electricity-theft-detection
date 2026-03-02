"""
Stage 1-2: Data Loading, Exploration, Cleaning & Monthly Aggregation
=====================================================================
Reads raw meter reading data, handles quality issues, and produces
a clean monthly consumption dataset ready for feature engineering.

Input:  Raw Excel/CSV with meter readings (380K+ rows)
Output: output/aylik_tuketim.csv   — Monthly consumption per subscriber
        output/df_clean.csv        — Cleaned raw data
        output/kacak_aboneler.csv  — Known theft subscriber IDs
"""

import pandas as pd
import numpy as np
import os
from utils import get_seasonal_profile, classify_negative_consumption

# ── Configuration ──
INPUT_FILE = "data/your_data.xlsx"  # Update with your file path
OUTPUT_DIR = "output/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Column mapping — adjust if your data has different column names
COL_MAP = {
    "subscriber_id": "AboneUN",
    "start_date": "IlkOkumaTarihi",
    "end_date": "SonOkumaTarihi",
    "reading_type": "EndeksTipiTanimi",
    "tariff": "Guncel_Tarife",
    "region": "Dagitim_Bolgesi",
}

# Records to exclude from analysis
EXCLUDE_TYPES = ["Kesme", "Sayaç Sökme", "Açma Endeksi", "Tahmin Endeksi", "Ters Endeks"]
MAX_DAY_DIFF = 1095  # 3 years max

print("=" * 60)
print("📂 STAGE 1: Data Exploration")
print("=" * 60)

# ── 1A: Load data ──
if INPUT_FILE.endswith(".xlsx"):
    df = pd.read_excel(INPUT_FILE, engine="openpyxl")
else:
    df = pd.read_csv(INPUT_FILE, encoding="cp1254", on_bad_lines="skip")

print(f"  Rows:    {len(df):,}")
print(f"  Columns: {len(df.columns)}")
print(f"  Subscribers: {df['AboneUN'].nunique():,}")

# ── 1B: Data profiling ──
print(f"\n📋 Column Overview:")
for i, c in enumerate(df.columns):
    dtype = str(df[c].dtype)
    null_pct = df[c].isna().sum() / len(df) * 100
    print(f"  {i:2d}. {c:30s} | {dtype:10s} | {null_pct:5.1f}% null")

# ── 1C: Reading type distribution ──
print(f"\n📊 Reading Types:")
for val, cnt in df["EndeksTipiTanimi"].value_counts().items():
    flag = " ← THEFT LABEL" if "Kaçak" in str(val) else ""
    flag += " ← EXCLUDE" if val in EXCLUDE_TYPES else ""
    print(f"  {str(val):25s}: {cnt:7,}{flag}")

# ══════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("🧹 STAGE 2: Cleaning & Monthly Aggregation")
print("=" * 60)

# ── 2A: Parse dates ──
df["IlkOkumaTarihi"] = pd.to_datetime(df["IlkOkumaTarihi"], errors="coerce")
df["SonOkumaTarihi"] = pd.to_datetime(df["SonOkumaTarihi"], errors="coerce")
df["gun_farki"] = (df["SonOkumaTarihi"] - df["IlkOkumaTarihi"]).dt.days

# ── 2B: Derive theft labels ──
kacak_aboneler = set(df[df["EndeksTipiTanimi"] == "Kaçak"]["AboneUN"].unique())
print(f"\n  Known theft subscribers: {len(kacak_aboneler):,}")

# ── 2C: Calculate consumption ──
for c in ["AT0Tuketim", "AT1Tuketim", "AT2Tuketim", "AT3Tuketim", "ET0Tuketim", "KT0Tuketim", "Demand"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

# Multi-tariff: use AT1+AT2+AT3 if available, else AT0
df["total_active"] = np.where(
    (df["AT1Tuketim"] + df["AT2Tuketim"]) > 0,
    df["AT1Tuketim"] + df["AT2Tuketim"] + df["AT3Tuketim"],
    df["AT0Tuketim"],
)
df["total_reactive"] = df["ET0Tuketim"] + df["KT0Tuketim"]

# ── 2D: Handle negative consumption ──
df = df.sort_values(["AboneUN", "SonOkumaTarihi"])
df["neg_category"] = classify_negative_consumption(df, "total_active")

print(f"\n  Negative consumption breakdown:")
for cat, cnt in df["neg_category"].value_counts().items():
    flag = " ← THEFT SIGNAL" if cat == "rollback" else ""
    print(f"    {cat:20s}: {cnt:,}{flag}")

df["total_active"] = df["total_active"].clip(lower=0)

# ── 2E: Filter ──
before = len(df)
df_clean = df[
    (df["gun_farki"] >= 0)
    & (df["gun_farki"] <= MAX_DAY_DIFF)
    & (~df["EndeksTipiTanimi"].isin(EXCLUDE_TYPES))
    & (df["SonOkumaTarihi"] >= "2023-01-01")
].copy()
print(f"\n  Filtered: {before:,} → {len(df_clean):,} (-{before-len(df_clean):,})")

# ── 2F: Learn seasonal profiles ──
profiles = get_seasonal_profile(
    df_clean, "Guncel_Tarife", "SonOkumaTarihi", "total_active", "gun_farki"
)
DEFAULT_PROFILE = {m: 1 / 12 for m in range(1, 13)}

print(f"\n  Seasonal profiles learned for {len(profiles)} tariff groups")

# ── 2G: Build monthly dataset ──
print(f"\n  Building monthly dataset...")

all_records = []
for idx, (_, row) in enumerate(df_clean.iterrows()):
    if (idx + 1) % 50000 == 0:
        print(f"    Progress: {idx+1:,}/{len(df_clean):,}")

    ilk, son = row["IlkOkumaTarihi"], row["SonOkumaTarihi"]
    if pd.isna(ilk) or pd.isna(son):
        continue

    consumption = row["total_active"]
    tariff = str(row["Guncel_Tarife"])
    profile = profiles.get(tariff, DEFAULT_PROFILE)
    day_diff = (son - ilk).days

    base = {
        "ID": row["AboneUN"],
        "tarife_grup": tariff,
        "dagitim_bolgesi": row.get("Dagitim_Bolgesi", ""),
        "demand": row.get("Demand", 0),
        "endeks_sayisi": row.get("EndeksTipi", 0),
        "reaktif": row.get("total_reactive", 0),
        "isletme_kodu": row.get("IsletmeKodu", ""),
        "sozlesme_gucu": row.get("SozlesmeGucu", 0),
    }

    if day_diff <= 45:
        ds = pd.Timestamp(son.year, son.month, 1)
        all_records.append({**base, "ds": ds, "y": consumption, "gun": max(day_diff, 1)})
    else:
        # Seasonal-weighted month splitting
        cur = pd.Timestamp(ilk.year, ilk.month, 1)
        months = []
        while cur <= son:
            m_start = max(cur, ilk)
            next_m = cur + pd.offsets.MonthBegin(1)
            m_end = min(next_m, son)
            days = (m_end - m_start).days
            if days > 0:
                months.append({"ds": cur, "month": cur.month, "days": days})
            cur = next_m

        if months:
            total_w = sum(m["days"] * profile.get(m["month"], 1 / 12) for m in months)
            for m in months:
                w = m["days"] * profile.get(m["month"], 1 / 12)
                y_val = consumption * (w / total_w) if total_w > 0 else consumption / len(months)
                all_records.append({**base, "ds": m["ds"], "y": round(y_val, 2), "gun": m["days"]})

df_monthly = pd.DataFrame(all_records)
df_monthly = df_monthly.groupby(["ID", "ds"]).agg({
    "y": "sum", "gun": "sum", "tarife_grup": "first", "dagitim_bolgesi": "first",
    "demand": "max", "endeks_sayisi": "first", "reaktif": "sum",
    "isletme_kodu": "first", "sozlesme_gucu": "first",
}).reset_index()

df_monthly["kacak_mi"] = df_monthly["ID"].isin(kacak_aboneler).astype(int)

# ── Save ──
df_monthly.to_csv(f"{OUTPUT_DIR}aylik_tuketim.csv", index=False)
df_clean.to_csv(f"{OUTPUT_DIR}df_clean.csv", index=False)
pd.Series(list(kacak_aboneler)).to_csv(f"{OUTPUT_DIR}kacak_aboneler.csv", index=False, header=["AboneUN"])

print(f"\n  ✅ Monthly dataset: {len(df_monthly):,} records | {df_monthly['ID'].nunique():,} subscribers")
print(f"  ✅ Known theft: {df_monthly[df_monthly['kacak_mi']==1]['ID'].nunique():,}")
