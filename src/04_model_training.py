"""
Stage 4-5: Model Training & Mega Ensemble Scoring
===================================================
Trains 4 complementary models and combines them into a unified risk score.

Models:
  1. Supervised Ensemble (RF + XGBoost + LightGBM) — learns from past theft
  2. Isolation Forest — finds multi-dimensional anomalies without labels
  3. Group Time-Series — detects seasonal deviation from peer group
  4. Individual Time-Series — catches self-baseline behavioral changes

Input:  output/abone_features_full.csv, output/aylik_tuketim.csv
Output: output/abone_features_final_v2.csv
        output/FINAL_SAHA_RAPORU_v2.xlsx
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from utils import normalize_scores, precision_at_k, generate_ai_explanation

OUTPUT_DIR = "output/"

df_feat = pd.read_csv(f"{OUTPUT_DIR}abone_features_full.csv")
df_monthly = pd.read_csv(f"{OUTPUT_DIR}aylik_tuketim.csv")
df_monthly["ds"] = pd.to_datetime(df_monthly["ds"])

LEAKAGE = ["f33_kacak_kayit", "f34_kesme", "f35_acma", "f36_sokme", "f37_kesme_acma_dongu"]
feature_cols = [c for c in df_feat.columns if c.startswith("f") and not c.endswith("_raw") and c not in LEAKAGE]
X = df_feat[feature_cols].fillna(0)
y = df_feat["target_kacak"]

# ══════════════════════════════════════════════════════
# MODEL 1: Supervised Ensemble
# ══════════════════════════════════════════════════════
print("=" * 60)
print("🤖 MODEL 1: Supervised Ensemble (RF + XGBoost + LightGBM)")
print("=" * 60)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
sup_proba = np.zeros(len(X))

models = {
    "RF": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1),
    "XGB": XGBClassifier(
        n_estimators=200,
        scale_pos_weight=y.value_counts()[0] / y.value_counts()[1],
        max_depth=6, learning_rate=0.1, random_state=42,
        use_label_encoder=False, eval_metric="logloss",
    ),
    "LGB": LGBMClassifier(n_estimators=200, is_unbalance=True, random_state=42, verbose=-1),
}
weights = {"RF": 0.2, "XGB": 0.4, "LGB": 0.4}

for fold, (tr, te) in enumerate(skf.split(X, y)):
    fold_p = np.zeros(len(te))
    for name, model in models.items():
        model.fit(X.iloc[tr], y.iloc[tr])
        fold_p += model.predict_proba(X.iloc[te])[:, 1] * weights[name]
    sup_proba[te] = fold_p
    print(f"  Fold {fold+1} ✓")

df_feat["supervised_skor"] = sup_proba

# ══════════════════════════════════════════════════════
# MODEL 2: Isolation Forest (per tariff group)
# ══════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("🌲 MODEL 2: Isolation Forest (Unsupervised)")
print("=" * 60)

df_feat["iso_skor"] = 0.0

for tarife in df_feat["f47_tarife_grup_raw"].unique():
    mask = df_feat["f47_tarife_grup_raw"] == tarife
    X_g = X[mask]
    if len(X_g) < 10:
        continue

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_g)

    cont = min(0.15, max(0.03, y[mask].mean() * 2 if y[mask].mean() > 0 else 0.05))
    iso = IsolationForest(n_estimators=300, contamination=cont, random_state=42, n_jobs=-1)
    iso.fit(X_scaled)

    scores = iso.decision_function(X_scaled)
    df_feat.loc[mask, "iso_skor"] = 1 - (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)

    anom = (iso.predict(X_scaled) == -1).sum()
    print(f"  {str(tarife):25s}: {len(X_g):5d} subs | {anom:3d} anomalies")

# ══════════════════════════════════════════════════════
# MODEL 3: Group Time-Series Anomaly
# ══════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("📈 MODEL 3: Group Time-Series Anomaly")
print("=" * 60)

df_monthly["month"] = df_monthly["ds"].dt.month
tarife_profiles = {}

for tarife in df_monthly["tarife_grup"].unique():
    grp = df_monthly[df_monthly["tarife_grup"] == tarife]
    ms = grp.groupby("month")["y"].agg(["median", "std"]).reset_index()
    ms["std"] = ms["std"].fillna(ms["median"] * 0.3)
    ms["lower"] = (ms["median"] - 1.5 * ms["std"]).clip(lower=0)
    tarife_profiles[tarife] = ms.set_index("month")

abone_ts = {}
for tarife, profile in tarife_profiles.items():
    grp = df_monthly[df_monthly["tarife_grup"] == tarife]
    for aid in grp["ID"].unique():
        abone = grp[grp["ID"] == aid].sort_values("ds")
        if len(abone) < 3:
            continue

        residuals, below = [], 0
        for _, row in abone.iterrows():
            m = row["month"]
            if m not in profile.index:
                continue
            expected = profile.loc[m, "median"]
            if expected > 0:
                residuals.append((row["y"] - expected) / expected)
                if row["y"] < profile.loc[m, "lower"]:
                    below += 1

        if len(residuals) < 3:
            continue

        residuals = np.array(residuals)
        neg_r = residuals[residuals < 0]
        cons, max_cons = 0, 0
        for r in residuals:
            if r < -0.3:
                cons += 1; max_cons = max(max_cons, cons)
            else:
                cons = 0

        ts_score = (
            len(neg_r) / len(residuals) * 0.25
            + min(abs(np.mean(neg_r)) if len(neg_r) > 0 else 0, 1.0) * 0.25
            + min(abs(np.min(residuals)), 2.0) / 2.0 * 0.20
            + min(max_cons / 6.0, 1.0) * 0.15
            + min(below / max(len(residuals), 1), 1.0) * 0.15
        )
        abone_ts[aid] = {"ts_skor": round(ts_score, 4), "ts_ardisik_neg": max_cons}

ts_df = pd.DataFrame.from_dict(abone_ts, orient="index").reset_index()
ts_df.columns = ["ID"] + list(ts_df.columns[1:])
df_feat = df_feat.merge(ts_df, on="ID", how="left")
df_feat["ts_skor"] = df_feat["ts_skor"].fillna(0)

# ══════════════════════════════════════════════════════
# MODEL 4: Individual Time-Series (Self-Baseline)
# ══════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("📉 MODEL 4: Individual Time-Series (Self-Baseline)")
print("=" * 60)

bireysel = {}
for aid in df_monthly["ID"].unique():
    abone = df_monthly[df_monthly["ID"] == aid].sort_values("ds")
    y_ts = abone["y"].values
    n = len(y_ts)
    if n < 6:
        continue

    split = max(int(n * 0.7), 3)
    baseline, recent = y_ts[:split], y_ts[split:]
    bl_mean = np.mean(baseline)
    bl_std = np.std(baseline) + 0.01

    if bl_mean <= 0.1:
        continue

    rec_mean = np.mean(recent)
    z = (bl_mean - rec_mean) / bl_std
    son3 = np.mean(y_ts[-3:])
    son3_drop = (bl_mean - son3) / bl_mean if bl_mean > 0 else 0

    threshold = bl_mean * 0.5
    cons, max_cons = 0, 0
    for v in y_ts[split:]:
        if v < threshold:
            cons += 1; max_cons = max(max_cons, cons)
        else:
            cons = 0

    min_in_recent = 1 if np.argmin(y_ts) >= split else 0
    bl_cv = np.std(baseline) / max(bl_mean, 0.01)
    rec_cv = np.std(recent) / max(rec_mean, 0.01) if len(recent) > 1 else 0

    ind_score = (
        min(max(z, 0), 5) / 5.0 * 0.30
        + min(max(son3_drop, 0), 1.0) * 0.25
        + min(max_cons / 6.0, 1.0) * 0.20
        + min(max(rec_cv - bl_cv, 0), 2.0) / 2.0 * 0.10
        + min_in_recent * 0.15
    )

    bireysel[aid] = {
        "ind_skor": round(ind_score, 4),
        "ind_z_score": round(z, 2),
        "ind_son3_dusus_pct": round(son3_drop * 100, 1),
        "ind_ardisik_dusuk": max_cons,
        "ind_gecmis_ort": round(bl_mean, 1),
        "ind_son_donem_ort": round(rec_mean, 1),
    }

ind_df = pd.DataFrame.from_dict(bireysel, orient="index").reset_index()
ind_df.columns = ["ID"] + list(ind_df.columns[1:])
df_feat = df_feat.merge(ind_df, on="ID", how="left")
df_feat["ind_skor"] = df_feat["ind_skor"].fillna(0)

# ══════════════════════════════════════════════════════
# MEGA ENSEMBLE v2 (4 Models)
# ══════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("🏆 MEGA ENSEMBLE v2 (4 Models Combined)")
print("=" * 60)

df_feat["mega_skor_v2"] = (
    normalize_scores(df_feat["supervised_skor"].values) * 0.40
    + normalize_scores(df_feat["iso_skor"].values) * 0.20
    + normalize_scores(df_feat["ts_skor"].values) * 0.15
    + normalize_scores(df_feat["ind_skor"].values) * 0.25
)

# Performance comparison
total_kacak = int(y.sum())
print(f"\n📊 Precision@K Comparison:")
for k in [50, 100, 200, 300, 500]:
    results = {}
    for name, col in [("Supervised", "supervised_skor"), ("IsoForest", "iso_skor"),
                       ("GroupTS", "ts_skor"), ("IndivTS", "ind_skor"), ("MEGA v2", "mega_skor_v2")]:
        hits, prec = precision_at_k(df_feat, col, "target_kacak", k)
        results[name] = f"{hits:3d} ({prec*100:4.0f}%)"
    print(f"  Top {k:3d}: " + " | ".join(f"{n}: {v}" for n, v in results.items()))

# Risk categories & AI explanations
df_feat = df_feat.sort_values("mega_skor_v2", ascending=False).reset_index(drop=True)
df_feat["final_sira"] = range(1, len(df_feat) + 1)
df_feat["final_kategori"] = pd.cut(
    df_feat["mega_skor_v2"],
    bins=[-0.01, 0.15, 0.30, 0.50, 0.70, 1.01],
    labels=["DÜŞÜK", "ORTA", "YÜKSEK", "ÇOK YÜKSEK", "KRİTİK"],
)
df_feat["durum"] = np.where(df_feat["target_kacak"] == 1, "⚠ BİLİNEN", "🔍 YENİ KEŞİF")
df_feat["AI_YORUM"] = df_feat.apply(generate_ai_explanation, axis=1)

# ── Generate Excel report ──
print(f"\n📄 Generating Excel report...")

excel_path = f"{OUTPUT_DIR}FINAL_SAHA_RAPORU_v2.xlsx"
rapor_cols = [
    "final_sira", "ID", "mega_skor_v2", "final_kategori", "durum", "AI_YORUM",
    "target_kacak", "supervised_skor", "iso_skor", "ts_skor", "ind_skor",
    "ind_gecmis_ort", "ind_son_donem_ort", "ind_son3_dusus_pct",
    "f47_tarife_grup_raw", "f48_dagitim_bolgesi_raw",
    "f01_ort_tuketim", "f09_sifir_oran", "f13_trend_ratio",
    "f18_max_ardisik_dusuk", "f25_cosine_sim", "f42_negatif_geri_sarma", "f45_cos_phi",
]
rapor_cols = [c for c in rapor_cols if c in df_feat.columns]

with pd.ExcelWriter(excel_path, engine="openpyxl") as wr:
    df_feat[rapor_cols].head(500).to_excel(wr, sheet_name="Top 500 Riskli", index=False)
    df_feat[df_feat["target_kacak"] == 0][rapor_cols].head(300).to_excel(wr, sheet_name="Yeni Keşifler", index=False)
    df_feat[df_feat["target_kacak"] == 1][rapor_cols].to_excel(wr, sheet_name="Bilinen Kaçaklar", index=False)

df_feat.to_csv(f"{OUTPUT_DIR}abone_features_final_v2.csv", index=False)

print(f"\n  ✅ Report: {excel_path}")
print(f"  ✅ Top 100 Precision: {precision_at_k(df_feat, 'mega_skor_v2', 'target_kacak', 100)[1]*100:.0f}%")
