"""
Stage 3: Feature Engineering (46 Leakage-Free Features)
========================================================
Extracts consumption, trend, seasonal, peer-comparison,
reactive power, and technical features per subscriber.

Input:  output/aylik_tuketim.csv, output/df_clean.csv
Output: output/abone_features_full.csv
"""

import pandas as pd
import numpy as np
from scipy import stats
from numpy.linalg import norm

OUTPUT_DIR = "output/"

df_monthly = pd.read_csv(f"{OUTPUT_DIR}aylik_tuketim.csv")
df_monthly["ds"] = pd.to_datetime(df_monthly["ds"])
df_clean = pd.read_csv(f"{OUTPUT_DIR}df_clean.csv")
kacak_aboneler = set(pd.read_csv(f"{OUTPUT_DIR}kacak_aboneler.csv")["AboneUN"].values)

print("🔧 STAGE 3: Feature Engineering")
print("=" * 60)

# ── Precompute segment statistics ──
segment_stats = {}
for (tarife, ds), grp in df_monthly.groupby(["tarife_grup", "ds"]):
    segment_stats[(tarife, ds)] = {
        "mean": grp["y"].mean(),
        "median": grp["y"].median(),
        "std": grp["y"].std(),
        "values": grp["y"].values,
    }

segment_profiles = {}
for tarife in df_monthly["tarife_grup"].unique():
    t_data = df_monthly[df_monthly["tarife_grup"] == tarife].copy()
    t_data["month"] = t_data["ds"].dt.month
    prof = t_data.groupby("month")["y"].median()
    total_p = prof.sum()
    segment_profiles[tarife] = np.array([
        prof.get(m, 0) / total_p if total_p > 0 else 1 / 12 for m in range(1, 13)
    ])

# ── Precompute per-subscriber info from raw data ──
endeks_info, neg_counts, reaktif_info, sozlesme_info = {}, {}, {}, {}

for aid, grp in df_clean.groupby("AboneUN"):
    # Reading type history
    if "EndeksTipiTanimi" in grp.columns:
        tipler = grp["EndeksTipiTanimi"].value_counts()
    else:
        tipler = pd.Series(dtype=int)
    endeks_info[aid] = {
        "kacak_kayit": tipler.get("Kaçak", 0),
        "kesme": tipler.get("Kesme", 0),
        "acma": tipler.get("Açma Endeksi", 0),
        "sokme": tipler.get("Sayaç Sökme", 0),
        "toplam_kayit": len(grp),
    }
    # Negative consumption
    if "neg_category" in grp.columns:
        neg_counts[aid] = {"rollback": (grp["neg_category"] == "rollback").sum()}
    elif "neg_kategori" in grp.columns:
        neg_counts[aid] = {"rollback": (grp["neg_kategori"] == "geri_sarma").sum()}
    # Reactive power
    aktif = grp.get("total_active", grp.get("toplam_aktif", pd.Series(0))).sum()
    reaktif = grp.get("total_reactive", grp.get("toplam_reaktif", pd.Series(0))).sum()
    reaktif_info[aid] = {"aktif": aktif, "reaktif": reaktif}
    # Contract power
    if "SozlesmeGucu" in grp.columns:
        sozlesme_info[aid] = pd.to_numeric(grp["SozlesmeGucu"], errors="coerce").max()

# ══════════════════════════════════════════════════════
# Feature Extraction Loop
# ══════════════════════════════════════════════════════
LEAKAGE = ["f33_kacak_kayit", "f34_kesme", "f35_acma", "f36_sokme", "f37_kesme_acma_dongu"]
features = []
abone_ids = df_monthly["ID"].unique()

for i, aid in enumerate(abone_ids):
    if (i + 1) % 2000 == 0:
        print(f"  Progress: {i+1:,}/{len(abone_ids):,}")

    adf = df_monthly[df_monthly["ID"] == aid].sort_values("ds")
    if len(adf) < 3:
        continue

    y = adf["y"].values
    tarife = adf["tarife_grup"].iloc[0]
    adf["month"] = adf["ds"].dt.month
    n = len(y)
    f = {"ID": aid}

    # ── f01-f12: Basic consumption stats ──
    f["f01_ort_tuketim"] = np.mean(y)
    f["f02_medyan_tuketim"] = np.median(y)
    f["f03_std_tuketim"] = np.std(y)
    f["f04_cv"] = np.std(y) / np.mean(y) if np.mean(y) > 0 else 0
    f["f05_min_tuketim"] = np.min(y)
    f["f06_max_tuketim"] = np.max(y)
    f["f07_max_min_oran"] = np.max(y) / max(np.min(y), 0.01)
    f["f08_sifir_ay"] = int((y <= 0.1).sum())
    f["f09_sifir_oran"] = (y <= 0.1).sum() / n
    f["f10_toplam"] = np.sum(y)
    f["f11_son_6ay_ort"] = np.mean(y[-min(6, n):])
    f["f12_ilk_6ay_ort"] = np.mean(y[:min(6, n)])

    # ── f13-f18: Trend features ──
    mid = n // 2
    first_half = np.mean(y[:mid]) if mid > 0 else 1
    second_half = np.mean(y[mid:]) if mid < n else 1
    f["f13_trend_ratio"] = second_half / max(first_half, 0.01)
    f["f14_trend_slope"] = stats.linregress(np.arange(n), y)[0] if n >= 4 else 0
    f["f15_son12_vs_onceki"] = (
        np.mean(y[-12:]) / max(np.mean(y[-24:-12]), 0.01) if n >= 24 else f["f13_trend_ratio"]
    )

    if n >= 2:
        changes = np.diff(y) / np.maximum(y[:-1], 0.01)
        f["f16_ani_dusus"] = int((changes < -0.5).sum())
        f["f17_ani_artis"] = int((changes > 2.0).sum())
    else:
        f["f16_ani_dusus"] = f["f17_ani_artis"] = 0

    threshold = f["f01_ort_tuketim"] * 0.1
    cons, max_cons = 0, 0
    for v in y:
        cons = cons + 1 if v < threshold else 0
        max_cons = max(max_cons, cons)
    f["f18_max_ardisik_dusuk"] = max_cons

    # ── f19-f26: Seasonality ──
    monthly_avgs = adf.groupby("month")["y"].mean()
    f["f19_kis_ort"] = np.mean([monthly_avgs.get(m, 0) for m in [12, 1, 2]])
    f["f20_yaz_ort"] = np.mean([monthly_avgs.get(m, 0) for m in [6, 7, 8]])
    f["f21_kis_yaz_oran"] = f["f19_kis_ort"] / max(f["f20_yaz_ort"], 0.01)
    f["f22_sulama_sezon_ort"] = np.mean([monthly_avgs.get(m, 0) for m in [4,5,6,7,8,9,10]])
    f["f23_sezon_disi_ort"] = np.mean([monthly_avgs.get(m, 0) for m in [11,12,1,2,3]])
    f["f24_sezon_ici_disi"] = f["f22_sulama_sezon_ort"] / max(f["f23_sezon_disi_ort"], 0.01)

    abone_profile = np.array([monthly_avgs.get(m, 0) for m in range(1, 13)])
    group_profile = segment_profiles.get(tarife, np.ones(12) / 12)
    if norm(abone_profile) > 0 and norm(group_profile) > 0:
        f["f25_cosine_sim"] = float(np.dot(abone_profile, group_profile) / (norm(abone_profile) * norm(group_profile)))
    else:
        f["f25_cosine_sim"] = 0
    f["f26_pik_ay"] = int(np.argmax(abone_profile) + 1) if np.sum(abone_profile) > 0 else 0

    # ── f27-f32: Peer comparison ──
    percentiles = []
    for _, row in adf.iterrows():
        key = (tarife, row["ds"])
        if key in segment_stats and len(segment_stats[key]["values"]) >= 5:
            pct = stats.percentileofscore(segment_stats[key]["values"], row["y"], kind="rank")
            percentiles.append(pct)

    if percentiles:
        f["f27_peer_ort_percentile"] = np.mean(percentiles)
        f["f28_peer_min_percentile"] = np.min(percentiles)
        f["f29_peer_anomaly_count"] = int(sum(1 for p in percentiles if p < 10))
        f["f30_peer_anomaly_ratio"] = sum(1 for p in percentiles if p < 10) / len(percentiles)
    else:
        f["f27_peer_ort_percentile"] = f["f28_peer_min_percentile"] = 50
        f["f29_peer_anomaly_count"] = 0
        f["f30_peer_anomaly_ratio"] = 0

    last_key = (tarife, adf["ds"].iloc[-1])
    f["f31_vs_group_mean"] = f["f01_ort_tuketim"] / max(segment_stats.get(last_key, {}).get("mean", 1), 0.01)
    f["f32_vs_group_median"] = f["f01_ort_tuketim"] / max(segment_stats.get(last_key, {}).get("median", 1), 0.01)

    # ── f33-f37: Historical labels (LEAKAGE — Set B only) ──
    ei = endeks_info.get(aid, {})
    f["f33_kacak_kayit"] = ei.get("kacak_kayit", 0)
    f["f34_kesme"] = ei.get("kesme", 0)
    f["f35_acma"] = ei.get("acma", 0)
    f["f36_sokme"] = ei.get("sokme", 0)
    f["f37_kesme_acma_dongu"] = min(ei.get("kesme", 0), ei.get("acma", 0))

    # ── f38-f43: Technical ──
    f["f38_toplam_kayit"] = ei.get("toplam_kayit", 0)
    f["f39_demand_max"] = float(adf["demand"].max())
    f["f40_demand_tuketim"] = f["f39_demand_max"] / max(f["f01_ort_tuketim"], 0.01)
    f["f41_endeks_sayisi"] = int(adf["endeks_sayisi"].mode().iloc[0]) if len(adf) > 0 else 1
    nc = neg_counts.get(aid, {})
    f["f42_negatif_geri_sarma"] = nc.get("rollback", nc.get("geri_sarma", 0))
    f["f43_sayac_degisim"] = 0

    # ── f44-f46: Reactive power ──
    ri = reaktif_info.get(aid, {"aktif": 0, "reaktif": 0})
    f["f44_reaktif_aktif_oran"] = ri["reaktif"] / max(ri["aktif"], 0.01)
    f["f45_cos_phi"] = ri["aktif"] / np.sqrt(ri["aktif"]**2 + ri["reaktif"]**2) if (ri["aktif"] + ri["reaktif"]) > 0 else 1.0
    f["f46_kapasitif_oran"] = 0

    # ── f47-f51: Segment info ──
    f["f47_tarife_grup"] = tarife
    f["f48_dagitim_bolgesi"] = adf["dagitim_bolgesi"].iloc[0]
    f["f49_isletme_prefix"] = str(adf["isletme_kodu"].iloc[0])[:10]
    f["f50_veri_ay_sayisi"] = n
    f["f51_sozlesme_gucu"] = sozlesme_info.get(aid, 0)

    f["target_kacak"] = 1 if aid in kacak_aboneler else 0
    features.append(f)

df_feat = pd.DataFrame(features)

# Encode categoricals
for col in ["f47_tarife_grup", "f48_dagitim_bolgesi", "f49_isletme_prefix"]:
    df_feat[col + "_raw"] = df_feat[col]
    df_feat[col] = df_feat[col].astype("category").cat.codes

feature_cols_A = [c for c in df_feat.columns if c.startswith("f") and not c.endswith("_raw") and c not in LEAKAGE]

df_feat.to_csv(f"{OUTPUT_DIR}abone_features_full.csv", index=False)

print(f"\n  ✅ {len(df_feat):,} subscribers | {len(feature_cols_A)} leakage-free features")
print(f"  ✅ Known theft: {df_feat['target_kacak'].sum():,} ({df_feat['target_kacak'].mean()*100:.1f}%)")
