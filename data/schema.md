# Data Schema

> ⚠️ **No real data is included in this repository.** This document describes the expected input format.

## Input Data

The pipeline expects a single Excel/CSV file with meter reading records. Each row represents one billing period for one subscriber.

### Required Columns

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `AboneUN` | string (UUID) | Unique subscriber identifier | `6015EFD0-26.1.14.0...` |
| `IlkOkumaTarihi` | datetime | Start date of reading period | `2025-01-17` |
| `SonOkumaTarihi` | datetime | End date of reading period | `2025-02-14` |
| `EndeksTipi` | integer | Meter type code | `2` |
| `EndeksTipiTanimi` | string | Reading type description | `Endeksör`, `Kaçak`, `Kesme` |
| `AT0Tuketim` | float | Single-tariff active consumption (kWh) | `1581.734` |
| `AT1Tuketim` | float | Peak active consumption (kWh) | `937.591` |
| `AT2Tuketim` | float | Day active consumption (kWh) | `413.452` |
| `AT3Tuketim` | float | Night active consumption (kWh) | `230.691` |
| `ET0Tuketim` | float | Inductive reactive consumption (kVArh) | `1073.385` |
| `KT0Tuketim` | float | Capacitive reactive consumption (kVArh) | `668.695` |
| `Demand` | float | Maximum demand (kW) | `0.0` |
| `Dagitim_Bolgesi` | string | Distribution region | `KIRSAL` |
| `Guncel_Tarife` | string | Current tariff group | `Mesken`, `Ticarethane` |
| `SozlesmeGucu` | float | Contract power (kW) | `5.01` |

### Optional Columns

| Column | Type | Description |
|--------|------|-------------|
| `SayacEndeksUN` | string | Meter device identifier |
| `Abone_Durum` | string | Subscriber status (NORMAL, KESME, etc.) |
| `Abone_Tipi` | string | Subscriber type (AG = low voltage) |
| `Imar_Durum` | string | Urban/rural classification |
| `IsletmeKodu` | string | Operating unit code |
| `KacakMi` | integer | Explicit theft label (0/1) — if available |

### Reading Types (EndeksTipiTanimi)

| Value | Meaning | Used in Analysis |
|-------|---------|-----------------|
| `Endeksör` | Normal automatic reading | ✅ Yes |
| `Otomatik` | Remote automatic reading | ✅ Yes |
| `Manuel` | Manual reading | ✅ Yes (flagged as correction) |
| `Kaçak` | Theft detected reading | 🏷️ Used as label |
| `Kesme` | Disconnection reading | ❌ Excluded |
| `Açma Endeksi` | Reconnection reading | ❌ Excluded |
| `Sayaç Sökme` | Meter removal reading | ❌ Excluded |
| `Tahmin Endeksi` | Estimated reading | ❌ Excluded |
| `Ters Endeks` | Reverse reading | ❌ Excluded |

### Tariff Groups

| Tariff | Typical Consumption | Seasonal Pattern |
|--------|-------------------|-----------------|
| Mesken (Residential) | 200-400 kWh/month | Peak in winter |
| Ticarethane (Commercial) | 500-2000 kWh/month | Peak in summer |
| Tarımsal Sulama (Agricultural) | 0-5000+ kWh/month | Summer only |
| Sanayi (Industrial) | 10,000+ kWh/month | Relatively flat |
| Genel Aydınlatma (Street Lighting) | Varies | Flat or winter peak |
| Aydınlatma (Lighting) | Low | Flat |

## Output Data

### Monthly Consumption (aylik_tuketim.csv)

| Column | Description |
|--------|-------------|
| `ID` | Subscriber identifier |
| `ds` | Month start date (YYYY-MM-01) |
| `y` | Monthly consumption (kWh) |
| `gun` | Days in period |
| `tarife_grup` | Tariff group |
| `kacak_mi` | Theft label (0/1) |

### Feature Dataset (abone_features_final_v2.csv)

Each row = one subscriber with 46+ features, 4 model scores, and risk classification.
