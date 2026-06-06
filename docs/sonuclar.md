# Sonuçlar ve Bulgular

Tüm sayılar resmi test seti **Ψ (529 segment)** üzerinde, 7 metrikle, AUC_PR sıralı olarak
raporlanır. Kaynak artefaktlar `reports/metrics/` altındadır ve `train_all_models.py` +
notebook'larla yeniden üretilebilir.

---

## 1. Kanonik Model Sıralaması (Faz 1)

`reports/metrics/final_comparison.json` — **42 model**, resmi Ψ'de değerlendirilmiş, AUC_PR sıralı.

### İlk 12 (en iyi)
| # | Model | AUC_PR | AUC_ROC | F1 | MCC | Precision | Recall |
|--:|-------|:------:|:-------:|:--:|:---:|:---------:|:------:|
| 1 | ExtraTrees | 0.983 | 0.994 | 0.931 | 0.914 | 0.971 | 0.894 |
| 2 | Voting Ensemble | 0.980 | 0.994 | 0.920 | 0.903 | 0.980 | 0.867 |
| 3 | MLP (≈FCNN) | 0.979 | 0.990 | **0.946** | **0.932** | 0.963 | **0.929** |
| 4 | HistGradientBoosting | 0.974 | 0.990 | 0.923 | 0.903 | 0.944 | 0.903 |
| 5 | XGBOD | 0.973 | 0.991 | 0.922 | 0.903 | 0.953 | 0.894 |
| 6 | CatBoost | 0.972 | 0.990 | 0.922 | 0.903 | 0.962 | 0.885 |
| 7 | Stacking Ensemble | 0.971 | 0.990 | 0.871 | 0.851 | 0.989 | 0.779 |
| 8 | RandomForest | 0.967 | 0.988 | 0.911 | 0.891 | 0.970 | 0.858 |
| 9 | LightGBM | 0.967 | 0.989 | 0.909 | 0.886 | 0.935 | 0.885 |
| 10 | XGBoost | 0.963 | 0.979 | 0.918 | 0.897 | 0.944 | 0.894 |
| 11 | SVM | 0.959 | 0.977 | 0.896 | 0.873 | 0.960 | 0.841 |
| 12 | Bagging | 0.958 | 0.981 | 0.921 | 0.903 | 0.971 | 0.876 |

**Çıkarımlar:** (i) Ağaç-toplulukları ve MLP, AUC_PR ≈ 0.97–0.98 ile lider. (ii) En yüksek
F1/MCC **MLP**'de (recall avantajı). (iii) Stacking ve Voting yüksek precision'a (≥0.98)
fakat görece düşük recall'a eğilimli — operasyonel olarak yanlış-alarm hassas senaryolar için
uygun.

### En düşük 4
LMDD (0.322), SOS (0.320), EllipticEnvelope (0.240), HBOS (0.213) — gözetimsiz uçtaki
dedektörler azınlık sınıfında zayıf, beklendiği gibi.

---

## 2. Benchmark Reprodüksiyonu (Faz 2)

`reports/metrics/benchmark_comparison.csv` — kanonik sonuçların Ruszczak et al. (2024) Tablo 3
ile **aynı Ψ** üzerinde karşılaştırması (22 eşleşen algoritma).

| Grup | Eşleşen | Ortalama \|ΔAUC_PR\| |
|------|:-------:|:-------------------:|
| **Gözetimli** | 7 | **0.0041** |
| Gözetimsiz | 15 | 0.093 |

**Gözetimli reprodüksiyon neredeyse birebir** (FCNN ΔAUC_PR=0.000, XGBOD −0.002, RF+ICCS +0.004).
Bu, metodolojik temelin (resmi split + leakage-free + 7 metrik) doğruluğunu kanıtlar.

**Gözetimsizde sapma daha büyük** ve iki yönlü: bazıları belirgin **iyileşme**
(KNN +0.252, PCA +0.038, IForest +0.016, COF +0.006), bazıları **gerileme** (LMDD −0.301,
LODA −0.232, OCSVM −0.211). Sapma, gözetimsiz dedektörlerin eşik/kontaminasyon seçimine ve
ön işleme detaylarına yüksek duyarlılığından kaynaklanır — makalede de bu kategori daha
değişkendir.

> İşaretler: `~` yaklaşık/paradigma eşleşmesi (FCNN≈MLP, RF+ICCS≈RandomForest, Linear+L2≈Ridge),
> `!` yöntem/uygulama farkı nedeniyle büyük sapma (ör. KNN).

---

## 3. Özgün Katkılar (Faz 3)

### 3.1 Sentetik Veri Augmentasyon Ablasyonu (NB13)
**Soru:** Profil-temelli sentetik telemetri gerçek Ψ'de tespiti iyileştirir mi?

- **Tam-veri** (`ablation_synthetic_fulldata.csv`): 0/250/500/1000/2000 sentetik segment ekleme,
  AUC_PR'ı **anlamlı değiştirmedi** (ör. RandomForest 0.967 → 0.967–0.971 bandı; ExtraTrees
  ~0.983 sabit). Zengin gerçek-etiket rejiminde sentetik katkı nötr.
- **Az-veri** (`ablation_synthetic_lowdata.csv`): gerçek verinin %15/30/50/100'ü + 1000 sentetik.
  Etki küçük ve karışık (ΔAUC_PR: −0.013, −0.004, +0.008, +0.002); MCC çoğunlukla hafif düştü.

### 3.2 Neden Nötr? — Dağılım Açığı (KS)
`synthetic_real_ks_distance.csv` — 18 özellikte sentetik vs gerçek KS mesafesi:
**ortalama 0.291**, medyan 0.288, aralık [0.116, 0.520]. En iyi uyum `diff2_peaks` (0.116),
`len` (0.122), `skew` (0.132); en kötü `n_peaks` (0.520), `kurtosis` (0.481), `diff2_var` (0.463).
Bu orta düzey açık, augmentasyonun neden büyük kazanç sağlamadığını açıklar — **dürüst negatif
bulgu** olarak raporlanır.

### 3.3 Augmentasyon Stratejisi Karşılaştırması (NB14)
`augmentation_comparison.csv` — SMOTE vs ICCS-ω vs Sentetik, aynı Ψ'de. Güçlü modellerde
AUC_PR'da büyük değişim yok; **asıl etki precision-recall dengesini kaydırmak**:

| Model | Strateji | Precision | Recall | F1 |
|-------|----------|:---------:|:------:|:--:|
| ExtraTrees | Baseline | 0.971 | 0.894 | 0.931 |
| ExtraTrees | +SMOTE | 0.928 | **0.912** | 0.920 |
| ExtraTrees | +ICCS-ω | **1.000** | 0.770 | 0.870 |

**Örüntü:** ICCS-ω → precision↑ (recall↓), SMOTE → recall↑ (precision↓). Strateji seçimi,
operasyonel önceliğe (yanlış alarm mı, kaçırma mı) göre yapılmalıdır.

### 3.4 Güç Tüketimi / Verimlilik (NB11)
`reports/power_profiles.csv` — 64 model için CPU(W), eğitim(s), çıkarım(ms), bellek(MB),
enerji(Wh), CO₂(g). En pahalı uçta derin/GAN modelleri (AnoGAN ~7.8 Wh, Transformer ~7.1 Wh);
ağaç-toplulukları yüksek F1'i çok daha düşük maliyetle sağlar. **Onboard çıkarım** için
verimlilik-doğruluk dengesinde ağaç-toplulukları öne çıkar.

### 3.5 Yorumlanabilirlik (NB07)
SHAP analizi, kararların büyük ölçüde varyans/türev temelli özellikler (`var`, `std`,
`diff_var`, türev tepe sayıları) tarafından yönlendirildiğini gösterir — fiziksel olarak
anlamlı (anomaliler tipik olarak segment değişkenliğini artırır).

---

## 4. Özet

| Faz | Çıktı | Durum |
|-----|-------|-------|
| 1 — Metodolojik temel | 42 model, leakage-free, Ψ'de 7 metrik | ✓ ExtraTrees AUC_PR 0.983 |
| 2 — Benchmark hizalama | gözetimli \|ΔAUC_PR\| = 0.004 | ✓ neredeyse birebir |
| 3 — Özgün katkılar | sentetik üretim+ablasyon, augmentasyon, güç, SHAP | ✓ (dürüst nötr/negatif bulgular dahil) |

> Araştırma bütünlüğü gereği nötr ve negatif bulgular (sentetik augmentasyonun sınırlı etkisi,
> KS açığı) açıkça raporlanmıştır.
