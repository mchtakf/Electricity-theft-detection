"""
Stage 1: Data Exploration
==========================
Quick profiling of raw meter reading data.
Run this first to understand your data structure.

Usage: python src/01_data_exploration.py --input data/your_data.xlsx
"""

import pandas as pd
import numpy as np
import argparse
import os


def explore(filepath: str):
    print("=" * 60)
    print("📂 STAGE 1: Data Exploration")
    print("=" * 60)

    # Load
    if filepath.endswith(".xlsx"):
        df = pd.read_excel(filepath, engine="openpyxl")
    else:
        for enc in ["cp1254", "utf-8-sig", "utf-8", "latin-1"]:
            try:
                df = pd.read_csv(filepath, encoding=enc, on_bad_lines="skip")
                break
            except Exception:
                continue

    print(f"  File:    {filepath}")
    print(f"  Rows:    {len(df):,}")
    print(f"  Columns: {len(df.columns)}")
    if "AboneUN" in df.columns:
        print(f"  Subscribers: {df['AboneUN'].nunique():,}")

    # Column profiling
    print(f"\n📋 Column Details:")
    for i, c in enumerate(df.columns):
        dtype = str(df[c].dtype)
        null_pct = df[c].isna().sum() / len(df) * 100
        sample = str(df[c].dropna().iloc[0])[:40] if df[c].notna().any() else "EMPTY"
        print(f"  {i:2d}. {c:30s} | {dtype:10s} | {null_pct:5.1f}% null | sample: {sample}")

    # Theft labels
    print(f"\n🔍 Theft Label Detection:")
    if "KacakMi" in df.columns:
        print(f"  KacakMi column found: {df['KacakMi'].value_counts().to_dict()}")
    elif "EndeksTipiTanimi" in df.columns:
        print("  No KacakMi column — checking EndeksTipiTanimi:")
        for val, cnt in df["EndeksTipiTanimi"].value_counts().items():
            flag = " ← THEFT LABEL" if "Kaçak" in str(val) else ""
            print(f"    {str(val):25s}: {cnt:7,}{flag}")

    # Tariff distribution
    tarife_col = next((c for c in df.columns if "tarife" in c.lower() or "guncel" in c.lower()), None)
    if tarife_col:
        print(f"\n📊 Tariff Distribution ({tarife_col}):")
        for t, cnt in df[tarife_col].value_counts().items():
            subs = df[df[tarife_col] == t]["AboneUN"].nunique()
            print(f"  {str(t):30s}: {cnt:7,} rows | {subs:5,} subscribers")

    # Date range
    print(f"\n📅 Date Columns:")
    for c in df.columns:
        if "okuma" in c.lower() or "tarih" in c.lower():
            parsed = pd.to_datetime(df[c], errors="coerce")
            valid = parsed.notna().sum()
            print(f"  {c}: {valid:,} valid | {parsed.min()} → {parsed.max()}")

    print(f"\n✅ Exploration complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to raw data file")
    args = parser.parse_args()
    explore(args.input)
