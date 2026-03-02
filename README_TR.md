# ⚡ Kaçak Elektrik Tespiti — Çoklu Model Ensemble Pipeline

> Elektrik dağıtım şebekelerinde teknik dışı kayıpları (kaçak elektrik) tespit etmek için uçtan uca makine öğrenmesi sistemi. Denetimli öğrenme, denetimsiz anomali tespiti ve zaman serisi analizini birleşik bir skorlama çerçevesinde birleştirir.

🇬🇧 [English README](README.md)

---

## 🎯 Problem

Kaçak elektrik, dağıtım şirketleri için ciddi gelir kaybına neden olur. Manuel saha denetimleri pahalı ve verimsizdir — rastgele aboneleri kontrol etmek sadece %3-5 başarı oranı verir. Bu proje, **hangi abonelerin öncelikle kontrol edilmesi gerektiğini** belirleyen bir AI sistemi oluşturur ve rastgele denetime göre **28 kat verimlilik artışı** sağlar.

## 📊 Sonuçlar

| Metrik | Değer |
|--------|-------|
| **Precision@100** | **%96** (İlk 100 şüphelinin 96'sı gerçek kaçak) |
| **Precision@200** | **%83** |
| **Recall@500** | **%46** |
| **Verimlilik artışı** | Rastgele denetime göre **28 kat** |
| **Analiz edilen abone** | 7 tarife grubunda 16.691 |
| **Üretilen feature** | 46 sızıntısız (leakage-free) özellik |
| **Birleşen model** | 4 (RF + XGBoost + LightGBM + IsolationForest + Zaman Serisi) |

## 🏗️ Mimari

```
HAM VERİ (sayaç okumaları, 380K+ kayıt)
  │
  ├── AŞAMA 1: Veri Temizleme
  │     ├── Tarih ayrıştırma ve doğrulama
  │     ├── Negatif tüketim yönetimi (3 kategori)
  │     └── Aykırı değer filtreleme
  │
  ├── AŞAMA 2: Aylık Toplama
  │     ├── Mevsimsel profil öğrenme (tarife grubu bazında)
  │     └── Çok aylık okumalarda ağırlıklı ay bölme
  │
  ├── AŞAMA 3: Feature Engineering (46 özellik)
  │     ├── Tüketim istatistikleri (ortalama, std, CV, sıfır oranı)
  │     ├── Trend analizi (eğim, ani düşüşler, ardışık düşükler)
  │     ├── Mevsimsellik (kış/yaz oranı, sulama sezonu)
  │     ├── Grup karşılaştırma (tarife grubu içi yüzdelik sırası)
  │     ├── Reaktif güç analizi (cos φ anomalileri)
  │     └── Teknik sinyaller (sayaç geri sarma, talep oranı)
  │
  ├── AŞAMA 4: Model Eğitimi
  │     ├── Denetimli Ensemble (RF %20 + XGBoost %40 + LightGBM %40)
  │     ├── Isolation Forest (denetimsiz, tarife grubu bazında)
  │     ├── Grup Zaman Serisi (grup medyanından mevsimsel sapma)
  │     └── Bireysel Zaman Serisi (kendi geçmişiyle karşılaştırma)
  │
  └── AŞAMA 5: Mega Ensemble
        ├── Ağırlıklı birleşim (Sup %40 + ISO %20 + GrupTS %15 + BirTS %25)
        ├── Risk kategorilendirme (Kritik → Düşük)
        ├── Abone bazlı AI yorum üretimi
        └── Saha operasyonu Excel raporu
```

## 🔬 Temel Teknik Kararlar

### Neden Tek Model Değil?
Her model farklı dolandırıcılık paternlerini yakalar:
- **Denetimli ML**: Geçmişte yakalanan kaçaklardan öğrenir ("geçmiş hırsızlara kim benziyor?")
- **Isolation Forest**: Etiketsiz çok boyutlu anomalileri bulur ("istatistiksel olarak kim garip?")
- **Grup Zaman Serisi**: Mevsimsel sapmaları tespit eder ("akranlarının beklentisinin altında kim tüketiyor?")
- **Bireysel Zaman Serisi**: Son dönem davranış değişikliklerini yakalar ("kendi geçmişinden kim aniden düştü?")

### Veri Sızıntısı (Leakage) Önleme
`kacak_kayit` (kaçak kayıt sayısı) gibi özellikler doğrudan hedef değişkeni sızdırır. İki feature seti kullanılır:
- **Set A (46 özellik)**: Sızıntısız, tüm model eğitimlerinde kullanılır
- **Set B (51 özellik)**: Geçmiş etiketleri içerir, sadece operasyonel skorlama için

### Negatif Tüketim Yönetimi
Negatif sayaç okumaları 3 kategoriye ayrılır:
1. **Sayaç değişimi**: Farklı sayaç ID → yoksay
2. **Sayaç geri sarma**: Aynı sayaç, negatif okuma → **kaçak sinyali**
3. **Manuel düzeltme**: EndeksTipi = Manuel → yoksay

### Mevsimsel Ağırlıklı Ay Bölme
Çok aylık okumalar (ör. 90 günlük fatura dönemi), düzgün dağılım yerine tarife grubu bazında öğrenilmiş mevsimsel profillere göre oransal olarak bölünür. Bu, tarımsal sulama abonelerinde kış aylarında yapay sıfır tüketim ayları oluşmasını engeller.

## 📁 Repo Yapısı

```
├── src/
│   ├── 01_data_exploration.py      # Veri profilleme ve kalite kontrolü
│   ├── 02_preprocessing.py         # Temizleme, aylık toplama
│   ├── 03_feature_engineering.py   # 46 özellik çıkarma
│   ├── 04_model_training.py        # Denetimli + Denetimsiz modeller
│   ├── 05_ensemble_scoring.py      # Mega ensemble + rapor üretimi
│   └── utils.py                    # Yardımcı fonksiyonlar
├── docs/
│   ├── ARCHITECTURE.md             # Detaylı sistem mimarisi
│   ├── TECHNICAL_DECISIONS.md      # Tasarım kararları ve gerekçeler
│   └── MODEL_COMPARISON.md         # Model performans analizi
├── diagrams/
│   └── pipeline.mermaid            # Mimari diyagram
├── data/
│   └── schema.md                   # Veri şeması (gerçek veri yok)
└── README.md
```

## ⚙️ Teknoloji Yığını

- **Python 3.10+**
- **ML**: scikit-learn, XGBoost, LightGBM
- **Veri**: pandas, NumPy, SciPy
- **Anomali Tespiti**: Isolation Forest, istatistiksel yöntemler
- **Ortam**: Google Colab (GPU gerekmez)

## 📈 Model Karşılaştırması

| Top K | Denetimli | IsoForest | Grup ZS | Bireysel ZS | **MEGA v2** |
|-------|-----------|-----------|---------|-------------|-------------|
| 50 | ~%92 | ~%40 | ~%30 | ~%35 | **~%94** |
| 100 | ~%96 | ~%35 | ~%25 | ~%30 | **~%96** |
| 200 | ~%83 | ~%30 | ~%20 | ~%25 | **~%83** |
| 500 | ~%52 | ~%20 | ~%15 | ~%18 | **~%53** |

*Değerler gizlilik nedeniyle yaklaşıktır*

## 👤 Geliştirici

**Mücahit** — IT Uzmanı & Veri Bilimci
- Elektrik-Elektronik Mühendisliği (Eskişehir Teknik Üniversitesi)
- SAP PM Entegrasyonu | Mendix Geliştirme | ML Mühendisliği

## 📄 Lisans

MIT Lisansı — Detaylar için [LICENSE](LICENSE) dosyasına bakınız.

> **Sorumluluk Reddi**: Bu repo yalnızca metodoloji ve kod içerir. Hiçbir dağıtım şirketine ait gerçek veri bulunmamaktadır.
