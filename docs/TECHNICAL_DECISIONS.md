# Technical Decisions & Rationale

This document explains the key design decisions made during the development of the electricity theft detection pipeline and the reasoning behind each choice.

## 1. Why Ensemble of 4 Models Instead of a Single Model?

Each model captures fundamentally different fraud patterns:

| Model | What It Detects | Blind Spot |
|-------|----------------|------------|
| **Supervised ML** | Subscribers that look like past theft cases | New fraud patterns never seen before |
| **Isolation Forest** | Multi-dimensional statistical outliers | Can't distinguish fraud from legitimate anomalies |
| **Group Time-Series** | Seasonal deviation from peer group | Can't detect subscribers who were always low |
| **Individual Time-Series** | Recent behavioral change from self-baseline | Can't detect long-term consistent fraud |

By combining all four, we cover each model's blind spots. The ensemble achieves +12% improvement over the best single model at Top 500.

## 2. Why Per-Tariff-Group Models?

"Normal" consumption is completely different across tariff groups:

- **Residential (Mesken)**: 200-400 kWh/month, peak in winter (heating)
- **Commercial (Ticarethane)**: 500-2000 kWh/month, peak in summer (AC)
- **Agricultural Irrigation (Tarımsal Sulama)**: 0 in winter, 5000+ in summer
- **Industrial (Sanayi)**: Relatively flat, 10,000+ kWh/month

A single model would flag every agricultural subscriber as anomalous in winter. Training per-group models ensures each group's "normal" is properly learned.

## 3. Leakage Prevention (Set A vs Set B Features)

Features f33-f37 (theft record count, disconnection count, etc.) directly leak the target variable because they are derived from the same `EndeksTipiTanimi = 'Kaçak'` field used to create labels.

- **Set A (46 features)**: Excludes these features. Used for ALL model training and evaluation.
- **Set B (51 features)**: Includes leakage features. Used only for operational scoring when you want maximum accuracy and don't care about evaluation integrity.

This distinction is critical because including leakage features would inflate metrics by 20-30 percentage points, giving a false sense of model quality.

## 4. Seasonal-Weighted Month Splitting

Problem: Some meter readings cover 90+ days (quarterly readings). Naive uniform splitting would create:
- Agricultural subscriber with 3000 kWh over Jan-Mar → 1000/1000/1000
- Reality: likely 0/0/3000 (irrigation starts in April)

Solution: Learn monthly consumption profiles per tariff group, then distribute proportionally:
- Agricultural profile: Jan=2%, Feb=2%, Mar=5%, Apr=12%, ... Aug=20%
- 3000 kWh over Jan-Mar → 222/222/556

This preserves seasonal patterns and prevents false zero-consumption signals.

## 5. Negative Consumption Classification

Negative meter readings occur for legitimate reasons (meter replacement) and suspicious reasons (meter rollback). We classify into 3 categories:

1. **Meter Change**: Previous record has different meter ID → not suspicious
2. **Rollback**: Same meter, negative reading, not a manual correction → **strong theft signal**
3. **Manual Correction**: Reading type is "Manuel" or "Tahmin" → system correction

Only category 2 (rollback) is used as a theft feature (f42). This prevents false positives from routine meter replacements.

## 6. Ensemble Weights (40/20/15/25)

Final weights were determined through:
1. Individual model precision@K analysis
2. Unique contribution analysis (what each model finds that others miss)
3. Manual tuning based on domain knowledge

| Model | Weight | Rationale |
|-------|--------|-----------|
| Supervised | 40% | Highest standalone precision, most reliable |
| Individual TS | 25% | Catches recent behavioral changes (consultant-style) |
| Isolation Forest | 20% | Unique multi-dimensional anomalies |
| Group TS | 15% | Complements individual TS with peer context |

## 7. Risk Categorization Thresholds

Categories were designed to match field operation capacity:

| Category | Score Range | Typical Count | Action |
|----------|------------|---------------|--------|
| CRITICAL | 0.70-1.00 | ~50 | Immediate inspection |
| VERY HIGH | 0.50-0.70 | ~200 | Priority inspection (this week) |
| HIGH | 0.30-0.50 | ~1,200 | Planned inspection (this month) |
| MEDIUM | 0.15-0.30 | ~6,000 | Monitoring |
| LOW | 0.00-0.15 | ~9,000 | No action needed |

A typical field team can inspect 50-100 subscribers per month, so CRITICAL + VERY HIGH (~250) represents about 2-3 months of work.

## 8. AI Explanation System

Each subscriber gets a human-readable explanation combining signals from all models. This is critical for field operators who need to know what to look for during inspection.

Design principles:
- Show only triggered signals (not all features)
- Prioritize actionable information (kWh values, percentage drops)
- Use icons for quick scanning (📈 📉 🌲 ⚠)
- Include specific numbers ("571→28 kWh") not vague descriptions

## 9. Individual Time-Series: Baseline Split at 70/30

The self-comparison model splits each subscriber's history:
- First 70%: "baseline" (assumed normal period)
- Last 30%: "test" (compared against baseline)

Why 70/30 and not 50/50?
- Longer baseline = more stable "normal" estimate
- 30% is enough to detect recent changes (typically 6-10 months)
- Fraud that started very early would be captured by other models instead

## 10. Why Not NeuralProphet?

NeuralProphet was originally planned for individual time-series forecasting. However:
1. NumPy 2.0 compatibility issues in Google Colab environment
2. Training individual models for 16,000+ subscribers is computationally expensive
3. Our statistical approach (z-score + consecutive drops) achieves similar detection with much simpler implementation

The statistical approach was validated against the consultant's NeuralProphet findings — our model independently identified the same suspicious subscribers.
