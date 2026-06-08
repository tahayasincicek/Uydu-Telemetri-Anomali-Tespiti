# Notebook Rehberi (01–14)

14 notebook, sıra ile çalıştırıldığında veri incelemeden nihai karşılaştırmalara kadar tüm
çalışmayı üretir. Notebook'lar üç gruba ayrılır: **temel pipeline (01–08)**, **araç/analiz
sayfaları (09–11)** ve **araştırma katkıları (12–14)**.

> Çalıştırma sırası ve bağımlılıklar için [`calistirma.md`](calistirma.md) bölümüne bakınız.
> Tüm model notebook'ları resmi split + 7 metrik kullanır (bkz. [`metodoloji.md`](metodoloji.md)).

---

## Temel Pipeline (01–08)

### 01 — Veri İnceleme (EDA)
ESA OPS-SAT telemetrisinin keşifsel analizi: kanal istatistikleri, eksik veri, dağılımlar,
korelasyon, zaman serisi ve anomali etiketi analizi. **Çıktı:** EDA görselleri.

### 02 — Veri Ön İşleme (Leakage-Free)
Eksik veri, gürültü, outlier, normalizasyon incelemesi; ardından **kanonik leakage-free akış**:
resmi split → preprocessor (yalnız T'de fit) → SMOTE (yalnız T) → validation ayırma.
**Çıktı:** `data/processed/X_{train,val,test}.parquet` (test = dokunulmamış resmi Ψ).

### 03 — Özellik Mühendisliği
ESA'nın 18 özelliği ile kendi ürettiğimiz sinyal-işleme özelliklerinin (RMS, P2P, crest, ZCR)
birleştirilmesi. **Not:** Bu genişletilmiş set ayrı bir keşif/ablasyon amaçlıdır; kanonik
baseline yalnız 18 ESA özelliğini kullanır.

### 04 — Gözetimli Öğrenme (Demo)
Resmi split + 18 özellik üzerinde temsili gözetimli modeller (RF, XGBoost, SVM, MLP), resmi
Ψ'de 7 metrikle. **Gösterim notebook'udur**; kanonik model artefaktlarını üretmez/ezmez
(tam 42-model havuzu `train_all_models.py` ile).

### 05 — Gözetimsiz Öğrenme (Demo)
Gözetimsiz protokol (yalnız nominal eğit, validation'da eşik seç): Isolation Forest,
One-Class SVM, K-Means, LOF, Autoencoder + ek PyOD dedektörleri. Resmi Ψ'de değerlendirme.
Kanonik artefaktları ezmez.

### 06 — Tüm Modellerin Karşılaştırılması
`train_all_models.py`'nin ürettiği **kanonik `final_comparison.json`'ı yükler** (resmi Ψ, 7
metrik, AUC_PR sıralı), ısı haritası + ROC/PR + confusion + çıkarım hızı grafikleri üretir.
Sonuç bölümü kanonik verilerle tutarlıdır ve diğer notebook'lara atıf yapar.

### 07 — SHAP Analizi (Yorumlanabilirlik)
RF/XGBoost/MLP modellerinin SHAP ile yorumlanması (en etkili özellikler, karar mantığı).
Kanonik artefaktları (scaler, test_data, sklearn MLP) kullanır. **Çıktı:** SHAP görselleri +
`models/shap_values.pkl`.

### 08 — Ablation Çalışması
18 ESA özelliğinin katkısının sistematik ölçümü: tekil çıkarma, kümülatif azaltma, grup ve
tip deneyleri. Leakage-free `data/processed` (NB02) verisini kullanır. **Çıktı:**
`models/ablation_results.json` + ablasyon görselleri.

---

## Araç / Analiz Sayfaları (09–11)

### 09 — Sentetik Telemetri: Ham Üretim → Segmentasyon → Özellik
Profil-temelli sentetik üretimin **3 aşaması**: sürekli ham akış → hibrit segmentasyon → 18
özellik. Sinyal modelleri, 6 anomali türü, 5 onboard artefakt ve gerçek profillerle doğrulama.
**Çıktı:** `data/synthetic/synthetic_{raw_stream,segments,dataset}.csv`.

### 10 — ESA Feature Extraction Pipeline
`extract_esa_features`'in belgelenmesi ve doğrulanması: tek segment üzerinde adım adım çıkarım,
tüm veriye uygulama ve **ESA orijinaliyle karşılaştırma (r ≈ 0.999)**. **Çıktı:**
`data/features/dataset_reconstructed.csv`.

### 11 — Güç Tüketimi ve Hesaplama Maliyeti
64 modelin enerji, karbon, bellek ve çıkarım maliyetinin modellenmesi; gerçek F1 ile
**verimlilik haritası** ve onboard dağıtım önerileri. **Çıktı:** `reports/power_profiles.csv`.

---

## Araştırma Katkıları (12–14)

### 12 — Benchmark Karşılaştırması
Kanonik sonuçların Ruszczak et al. (2024) Tablo 3 baseline'ı (30 algoritma) ile **aynı Ψ**
üzerinde karşılaştırılması. **Ana bulgu:** gözetimli reprodüksiyon ortalama |ΔAUC_PR| = 0.004.
Yaklaşık/paradigma eşleşmeleri açıkça işaretlenir. **Çıktı:** `benchmark_comparison.csv`.

### 13 — Sentetik Veri Augmentasyon Ablasyonu
"Profil-temelli sentetik veri gerçek Ψ'de tespiti iyileştirir mi?" — iki deney (tam-veri,
az-veri) + KS dağılım analizi. **Bulgu:** ~nötr; orta düzey dağılım açığıyla (KS 0.29) açıklanır.
**Çıktı:** `ablation_synthetic_{fulldata,lowdata}.csv`, `synthetic_real_ks_distance.csv`.

### 14 — Augmentasyon Stratejisi Karşılaştırması
SMOTE vs ICCS-ω vs sentetik augmentasyon, aynı Ψ'de. **Bulgu:** zengin-etiket rejiminde
augmentasyon güçlü modelleri anlamlı iyileştirmez; asıl etki **precision-recall dengesini**
kaydırmaktır (ICCS-ω → precision↑, SMOTE → recall↑). **Çıktı:** `augmentation_comparison.csv`.

---

## Bağımlılık Akışı (özet)

```
01 (EDA)
02 (ön işleme, leakage-free) ──────────────► data/processed ──► 08 (ablation)
10 (feature extraction)  ──► dataset doğrulama
train_all_models.py ──► models/ + final_comparison.json ──► 06, 07, 12
09 (sentetik) ──► data/synthetic ──► 13, 14 (augmentasyon)
11 (güç) — bağımsız;  03 (özellik genişletme) — keşif;  04/05 — demo
```
