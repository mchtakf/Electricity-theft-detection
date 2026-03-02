# ⚡ Electricity Theft Detection — Multi-Model Ensemble Pipeline

> End-to-end machine learning system for detecting non-technical losses (electricity theft) in power distribution networks. Combines supervised learning, unsupervised anomaly detection, and time-series analysis into a unified scoring framework.

🇹🇷 [Türkçe README](README_TR.md)

---

## 🎯 Problem

Electricity theft causes significant revenue losses for power distribution companies. Manual field inspections are expensive and inefficient — inspecting random subscribers yields only ~3-5% hit rate. This project builds an AI system that **prioritizes which subscribers to inspect**, achieving **28x efficiency improvement** over random inspection.

## 📊 Results

| Metric | Value |
|--------|-------|
| **Precision@100** | **96%** (96 out of top 100 flagged subscribers were actual theft cases) |
| **Precision@200** | **83%** |
| **Recall@500** | **46%** |
| **Efficiency gain** | **28x** vs random inspection |
| **Subscribers analyzed** | 16,691 across 7 tariff groups |
| **Features engineered** | 46 leakage-free features |
| **Models combined** | 4 (RF + XGBoost + LightGBM + IsolationForest + TimeSeries) |

## 🏗️ Architecture

```
RAW DATA (meter readings, 380K+ records)
  │
  ├── STAGE 1: Data Cleaning
  │     ├── Date parsing & validation
  │     ├── Negative consumption handling (3 categories)
  │     └── Outlier filtering
  │
  ├── STAGE 2: Monthly Aggregation
  │     ├── Seasonal profile learning (per tariff group)
  │     └── Weighted month splitting for multi-month readings
  │
  ├── STAGE 3: Feature Engineering (46 features)
  │     ├── Consumption statistics (mean, std, CV, zero-ratio)
  │     ├── Trend analysis (slope, sudden drops, consecutive lows)
  │     ├── Seasonal patterns (winter/summer ratio, irrigation season)
  │     ├── Peer comparison (percentile rank within tariff group)
  │     ├── Reactive power analysis (cos φ anomalies)
  │     └── Technical signals (meter rollback, demand ratio)
  │
  ├── STAGE 4: Model Training
  │     ├── Supervised Ensemble (RF 20% + XGBoost 40% + LightGBM 40%)
  │     ├── Isolation Forest (unsupervised, per tariff group)
  │     ├── Group Time-Series (seasonal deviation from group median)
  │     └── Individual Time-Series (self-baseline comparison)
  │
  └── STAGE 5: Mega Ensemble
        ├── Weighted combination (Sup 40% + ISO 20% + GroupTS 15% + IndTS 25%)
        ├── Risk categorization (Critical → Low)
        ├── AI-generated explanations per subscriber
        └── Field operation Excel report
```

## 🔬 Key Technical Decisions

### Why Not a Single Model?
Each model captures different fraud patterns:
- **Supervised ML**: Learns from historically caught theft cases ("who looks like past thieves?")
- **Isolation Forest**: Finds multi-dimensional anomalies without labels ("who is statistically weird?")
- **Group Time-Series**: Detects seasonal deviations ("who consumes less than their peer group expects?")
- **Individual Time-Series**: Catches recent behavioral changes ("who suddenly dropped from their own baseline?")

### Data Leakage Prevention
Features like `kacak_kayit` (theft record count) directly leak the target. We maintain two feature sets:
- **Set A (46 features)**: Leakage-free, used for all model training
- **Set B (51 features)**: Includes historical labels, used only for operational scoring

### Negative Consumption Handling
Negative meter readings are classified into 3 categories:
1. **Meter replacement**: Different meter ID → ignore
2. **Meter rollback**: Same meter, negative reading → **theft signal**
3. **Manual correction**: Endeksör type = Manual → ignore

### Seasonal-Weighted Month Splitting
Multi-month readings (e.g., 90-day billing period) are split proportionally using learned seasonal profiles per tariff group, not uniform distribution. This prevents artificial zero-consumption months in agricultural irrigation during winter.

## 📁 Repository Structure

```
├── src/
│   ├── 01_data_exploration.py      # Data profiling & quality checks
│   ├── 02_preprocessing.py         # Cleaning, monthly aggregation
│   ├── 03_feature_engineering.py   # 46 feature extraction
│   ├── 04_model_training.py        # Supervised + Unsupervised models
│   ├── 05_ensemble_scoring.py      # Mega ensemble + report generation
│   └── utils.py                    # Helper functions
├── docs/
│   ├── ARCHITECTURE.md             # Detailed system architecture
│   ├── TECHNICAL_DECISIONS.md      # Design decisions & rationale
│   └── MODEL_COMPARISON.md         # Model performance analysis
├── diagrams/
│   └── pipeline.mermaid            # Architecture diagram
├── data/
│   └── schema.md                   # Data schema (no actual data)
└── README.md
```

## ⚙️ Tech Stack

- **Python 3.10+**
- **ML**: scikit-learn, XGBoost, LightGBM
- **Data**: pandas, NumPy, SciPy
- **Anomaly Detection**: Isolation Forest, statistical methods
- **Environment**: Google Colab (GPU not required)

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/yourusername/electricity-theft-detection.git
cd electricity-theft-detection

# Install dependencies
pip install pandas numpy scipy scikit-learn xgboost lightgbm openpyxl

# Run pipeline (requires your own data in expected schema)
python src/02_preprocessing.py --input data/your_data.xlsx --output output/
python src/03_feature_engineering.py --input output/ --output output/
python src/04_model_training.py --input output/ --output output/
python src/05_ensemble_scoring.py --input output/ --output output/
```

## 📋 Data Schema

> ⚠️ **No real data is included.** This project was developed using proprietary utility data. See `data/schema.md` for the expected input format.

Required columns: `AboneUN`, `IlkOkumaTarihi`, `SonOkumaTarihi`, `EndeksTipi`, `EndeksTipiTanimi`, `AT0Tuketim`, `AT1Tuketim`, `AT2Tuketim`, `AT3Tuketim`, `ET0Tuketim`, `KT0Tuketim`, `Demand`, `Dagitim_Bolgesi`, `Guncel_Tarife`, `SozlesmeGucu`, `KacakMi` or `EndeksTipiTanimi='Kaçak'`

## 📈 Model Comparison

| Top K | Supervised | IsoForest | Group TS | Individual TS | **MEGA v2** |
|-------|-----------|-----------|----------|---------------|-------------|
| 50 | ~92% | ~40% | ~30% | ~35% | **~94%** |
| 100 | ~96% | ~35% | ~25% | ~30% | **~96%** |
| 200 | ~83% | ~30% | ~20% | ~25% | **~83%** |
| 500 | ~52% | ~20% | ~15% | ~18% | **~53%** |

*Values approximated for confidentiality*

## 🔮 Future Work

- [ ] NeuralProphet integration for individual subscriber forecasting
- [ ] Transformer-level anomaly detection
- [ ] Real-time scoring pipeline
- [ ] Feedback loop from field inspection results
- [ ] Dashboard (Mendix or Streamlit)
- [ ] Economic impact estimation (TL/kWh loss calculation)

## 👤 Author

**Mücahit** — IT Specialist

- [LinkedIn](https://www.linkedin.com/in/m%C3%BCcahit-akfidan-665a001ba/)

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

> **Disclaimer**: This repository contains only methodology and code. No proprietary data from any utility company is included.
