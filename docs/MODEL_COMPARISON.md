# Model Performance Comparison

## Precision@K Results

> Values approximated for confidentiality. Actual results may vary with different datasets.

| Top K | Supervised ML | Isolation Forest | Group TS | Individual TS | **MEGA v2** |
|-------|:---:|:---:|:---:|:---:|:---:|
| 50 | ~92% | ~40% | ~30% | ~35% | **~94%** |
| 100 | ~96% | ~35% | ~25% | ~30% | **~96%** |
| 200 | ~83% | ~30% | ~20% | ~25% | **~83%** |
| 300 | ~73% | ~25% | ~18% | ~22% | **~73%** |
| 500 | ~52% | ~20% | ~15% | ~18% | **~53%** |

## What Each Model Uniquely Contributes

Analyzing Top 200 subscribers per model reveals minimal overlap:

| Subset | Count | Known Theft | Insight |
|--------|-------|------------|---------|
| Only in Supervised | ~80 | High | Classic fraud patterns |
| Only in IsoForest | ~60 | Medium | Novel multi-dimensional anomalies |
| Only in Time-Series | ~50 | Medium | Recent behavioral changes |
| In all 3 | ~10 | Very High | Strongest signals — inspect first |

## Unique Value of Each Model

### Supervised Ensemble
**Strength**: Highest standalone precision. Learns directly from confirmed theft cases.
**Weakness**: Can only find fraud patterns it has seen before. "Zero-day" fraud goes undetected.

### Isolation Forest
**Strength**: Finds anomalies without any labels. Can discover completely new fraud types.
**Weakness**: Also flags legitimate anomalies (new businesses, renovations).

### Group Time-Series
**Strength**: Contextual comparison. "This commercial subscriber uses less than all other commercial subscribers in winter" is a meaningful signal.
**Weakness**: Subscribers who have always been low won't trigger this model.

### Individual Time-Series
**Strength**: Catches the exact pattern fraud consultants look for — "sudden drop from own baseline."
**Weakness**: Needs sufficient history to establish a baseline. New subscribers can't be scored.

## Feature Importance (Top 10)

From LightGBM's feature importance:

| Rank | Feature | Description | Why It Matters |
|------|---------|-------------|---------------|
| 1 | f09_sifir_oran | Zero-consumption month ratio | Active subscriber with no usage = bypass |
| 2 | f18_max_ardisik_dusuk | Max consecutive low months | Extended low period = systematic fraud |
| 3 | f01_ort_tuketim | Average consumption | Very low average = chronic theft |
| 4 | f27_peer_ort_percentile | Peer group percentile | Consistently below peers |
| 5 | f25_cosine_sim | Seasonal profile similarity | Different pattern from group |
| 6 | f13_trend_ratio | Consumption trend | Declining trend |
| 7 | f42_negatif_geri_sarma | Meter rollback count | Physical meter tampering |
| 8 | f04_cv | Coefficient of variation | Unstable consumption |
| 9 | f30_peer_anomaly_ratio | Peer anomaly frequency | Frequently at bottom of group |
| 10 | f45_cos_phi | Power factor | Abnormal reactive power |

## Efficiency Gain

| Method | Top 100 Hit Rate | Inspections Needed for 100 Catches |
|--------|-----------------|-----------------------------------|
| Random inspection | ~3.4% | ~2,940 |
| **Our model** | **~96%** | **~105** |
| **Improvement** | | **28x more efficient** |
