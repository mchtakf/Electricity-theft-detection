# System Architecture

## Overview

The pipeline processes raw meter reading data through 5 stages to produce a prioritized list of subscribers most likely to be committing electricity theft.

## Data Flow

```
[Raw Data] → [Cleaning] → [Monthly Agg.] → [Features] → [4 Models] → [Ensemble] → [Report]
  380K rows    371K rows    534K records    16.6K subs    4 scores     1 score      Excel
```

## Stage Details

### Stage 1-2: Data Ingestion & Cleaning

**Input**: Raw meter reading Excel/CSV with 32 columns per record.

**Key transformations**:
- Parse dates and compute day differences
- Derive theft labels from `EndeksTipiTanimi = 'Kaçak'`
- Calculate total active consumption (multi-tariff aware)
- Classify negative readings into 3 categories
- Filter out non-analysis records (disconnections, estimates)
- Learn seasonal profiles per tariff group
- Split multi-month readings using seasonal weights

**Output**: Clean monthly consumption time-series per subscriber.

### Stage 3: Feature Engineering

**46 leakage-free features** organized into 6 groups:

| Group | Features | Description |
|-------|----------|-------------|
| Consumption (f01-f12) | mean, median, std, CV, min, max, zero-ratio | Basic consumption statistics |
| Trend (f13-f18) | slope, sudden drops, consecutive lows | Temporal patterns |
| Seasonal (f19-f26) | winter/summer ratio, cosine similarity, peak month | Seasonal behavior |
| Peer (f27-f32) | percentile rank, anomaly count, group ratio | Comparison with similar subscribers |
| Technical (f38-f46) | reactive power, meter rollback, demand ratio | Electrical engineering signals |
| Segment (f47-f51) | tariff, region, contract power | Contextual information |

### Stage 4: Model Training

**Model 1 — Supervised Ensemble**:
- 3 gradient boosting / tree models with stratified 5-fold CV
- Class imbalance handled via `class_weight='balanced'` and `scale_pos_weight`
- Weighted averaging: RF 20%, XGBoost 40%, LightGBM 40%

**Model 2 — Isolation Forest**:
- Separate model per tariff group (7 models total)
- Dynamic contamination parameter (3-15% based on group theft rate)
- StandardScaler normalization before training

**Model 3 — Group Time-Series**:
- Monthly median + 1.5σ bands per tariff group
- Composite score from: negative residual ratio, mean deviation, max deviation, consecutive negatives, below-band count

**Model 4 — Individual Time-Series**:
- 70/30 baseline/test split per subscriber
- Z-score of recent vs baseline consumption
- Additional signals: last-3-month drop, consecutive lows, volatility change

### Stage 5: Mega Ensemble

**Score combination**:
```
mega_score = supervised × 0.40 + isolation × 0.20 + group_ts × 0.15 + individual_ts × 0.25
```

Each component score is min-max normalized to [0, 1] before combination.

**Risk categories**: 5-level system (Critical → Low) with AI-generated explanations per subscriber.

## Output Report Structure

6-sheet Excel workbook:
1. **Top 500 Riskli** — Highest risk subscribers for field inspection
2. **Yeni Keşifler** — Previously uncaught suspects (most valuable)
3. **Bilinen Kaçaklar** — Known theft cases with re-assessment
4. **Ani Düşüş Tespiti** — Sudden consumption drops (consultant-style findings)
5. **Model Karşılaştırma** — Precision@K comparison across all models
6. **Özet** — Executive summary statistics

## Performance Metrics

| Metric | Target | Achieved |
|--------|--------|----------|
| Precision@100 | >25% | **96%** (28x improvement) |
| Precision@200 | >20% | **83%** |
| Recall@500 | >30% | **46%** |
| Coverage | All tariffs | **7 tariff groups, 16,691 subscribers** |
