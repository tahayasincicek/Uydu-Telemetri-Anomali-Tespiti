# Veri ve Pipeline

Bu belge veri setlerini, 18 ESA özelliğinin anatomisini, feature extraction pipeline'ını ve
profil-temelli sentetik veri üretimini (3-aşamalı) açıklar.

---

## 1. Veri Setleri

| Dosya | Boyut | İçerik |
|-------|:-----:|--------|
| `data/raw/segments.csv` | 303.493 satır | **Ham telemetri** — her satır tek bir zaman noktasında tek bir sensör ölçümü (`channel, timestamp, value, label, sampling, anomaly, segment, train`) |
| `data/raw/dataset.csv` | 2123 satır | **Özellik matrisi** — her satır bir segmentin özeti (18 ESA özelliği + 5 meta) |

### segments.csv → dataset.csv ilişkisi
- 303.493 ham ölçüm, **2123 segmente** bölünmüştür (ortalama ~143 ölçüm/segment).
- Her segment **tek bir kanala** aittir (9 kanal: 3 manyetometre + 6 fotodiyot).
- `dataset.csv`, ESA'nın bu segmentlerden çıkardığı 18 handcrafted özelliği içerir.

### Kanallar
| Tip | Kanallar | Karakter |
|-----|----------|----------|
| Manyetometre | CADC0872/0873/0874 | ~1e-5 genlikli, yavaş, neredeyse monoton sinyal |
| Fotodiyot | CADC0884/0886/0888/0890/0892/0894 | 0 – π/2 arası açı; çoğunlukla sıfıra yakın (gölge) |

---

## 2. 18 ESA Handcrafted Özellik

ESA OPSSAT-AD özellikleri 3 mantıksal gruba ayrılır (Ruszczak et al., 2024, Şekil 2).
`src/feature_engineer.py::extract_esa_features` bunları üretir.

### Grup 1 — Ham segmentten (12)
`mean`, `var`, `std`, `kurtosis`, `skew`, `n_peaks` (%10 prominence ile tepe sayısı),
`duration` (saniye), `len` (nokta sayısı), `gaps_squared` (Σ Δt²), `len_weighted`
(len × sampling), `var_div_duration`, `var_div_len`.

### Grup 2 — Yumuşatılmış segmentten (2)
`smooth10_n_peaks`, `smooth20_n_peaks` — 10 ve 20 noktalı hareketli ortalama sonrası tepe
sayısı (`np.convolve`, sabit çekirdek).

### Grup 3 — Türevlerden (4)
`diff_peaks`, `diff2_peaks` (1. ve 2. türev tepe sayısı), `diff_var`, `diff2_var`
(1. ve 2. türev varyansı).

### Meta sütunlar (özellik değil)
`segment`, `anomaly`, `train`, `channel`, `sampling`.

> **Doğrulama (Notebook 10):** Otomatik `extract_esa_features`, ESA'nın orijinal
> `dataset.csv`'sini **r ≈ 0.999 korelasyonla** yeniden üretir (18 özellikten 17'si birebir;
> yalnız `smooth20_n_peaks` r=0.982 — `np.convolve` kenar etkisi).

---

## 3. Tam Yaşam Döngüsü Pipeline'ı

```
                 (Notebook 10)            (Notebook 03 — opsiyonel)
ham telemetri ──► extract_esa_features ──► 18 ESA özelliği ──► model eğitimi
(segments.csv)     (segment başına özet)    (dataset.csv)        (train_all_models.py)
```

**Ön işleme (Notebook 02):** resmi split → preprocessor (outlier-clip/scale; **eksik veri
doldurulmaz** — OPS-SAT'ta boşluklar `gaps_squared` ile korunur; yalnız T'de fit) → SMOTE
(yalnız T) → kaydet. Ayrıntı: [`metodoloji.md`](metodoloji.md#2-veri-sızıntısı-data-leakage-önleme).

---

## 4. Profil-Temelli Sentetik Veri Üretimi (3 Aşama)

`src/synthetic_generator.py` + `src/feature_engineer.py`, gerçek OPS-SAT'a sadık sentetik
telemetri üretir. Gerçek yaşam döngüsünü taklit eden **üç ayrı aşama** vardır (Notebook 09):

| Aşama | Fonksiyon | Çıktı | Gerçek-veri dayanağı |
|-------|-----------|-------|----------------------|
| 1. Ham üretim | `generate_raw_stream()` | sürekli akış (segment sınırı/etiket YOK) | sinyal stats + kampanya/boşluk yapısı |
| 2. Segmentasyon | `segment_raw_telemetry()` | `segments.csv` formatı | uzunluk dağılımı + boşluk eşikleri (hibrit) |
| 3. Özellik çıkarımı | `extract_esa_features()` | `dataset.csv` formatı | ESA 18 özellik |

### Aşama 1 — Sürekli ham akış
- **Kanal profilleri** (`data/channel_profiles.json`) gerçek OPSSAT-AD verisinden çıkarılmıştır.
- **Sinyal modelleri:** manyetometre = Ornstein-Uhlenbeck süreci (ortalamaya dönen yürüyüş,
  durağan std = profil std'si); fotodiyot = yörünge-periyodik, düşük-değer eğilimli (gamma üssü)
  sinüzoidal döngü.
- **6 anomali türü:** spike, shift, noise, gap, flat, deformation.
- **5 onboard artefakt** (anomali değil): mikro-boşluk, sıfır-zaman farkı, sıfır-değer,
  sabit-değer (sensör donması), büyük boşluk — gerçek OPS-SAT oranlarıyla.

### Aşama 2 — Hibrit segmentasyon
1. **Boşluk-bölme:** Δt eşiği `max(gap_factor·sampling, min_gap_seconds=150s)` aşılırsa yeni
   koşu (artefakt boşlukları segment içinde kalır).
2. **Uzunluk-penceresi:** her koşu, kanalın gerçek `len_mean`/`len_std` dağılımına göre bölünür.
3. **Etiket türetme:** segment, anomali örneklerini ≥%10 (veya ≥3 örnek) içeriyorsa anomali.

### Sadakat doğrulaması (Notebook 09, 13)
- Segment uzunluk dağılımı tüm kanallarda gerçeğe yakın (ör. CADC0872: 105 vs 122).
- Fotodiyot ort/max düzeltme sonrası 0.28–0.38 (gerçek 0.165–0.353).
- Sentetik vs gerçek 18 özellik ortalama **KS mesafesi = 0.29** (orta düzey uyum) — augmentasyon
  değerinin sınırlayıcısı (bkz. [`sonuclar.md`](sonuclar.md)).

---

## 5. Üretilen Dosyalar (yeniden üretilebilir, sürüm kontrolünde değil)

| Yol | Üreten | Açıklama |
|-----|--------|----------|
| `data/processed/X_*.parquet` | NB02 | leakage-free train/val/test (Ψ) |
| `data/features/dataset_reconstructed.csv` | NB10 | otomatik üretilmiş 18 özellik |
| `data/synthetic/synthetic_*.csv` | NB09 | sentetik ham/segment/özellik |
| `models/`, `reports/metrics/`, `reports/figures/` | train_all_models.py + notebooklar | model/metrik/şekil artefaktları |
