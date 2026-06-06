# Metodoloji

Bu belge, çalışmanın **bilimsel geçerliliğini** sağlayan metodolojik temeli açıklar:
resmi train/test split, 7 zorunlu metrik, veri sızıntısı önleme ve kanonik eğitim motoru.
Bu temel, sonraki tüm karşılaştırma ve katkıların güvenilir bir zemine oturmasını sağlar.

---

## 1. Resmi Train/Test Split (Ψ)

OPS-SAT benchmark'ı (Ruszczak et al., 2024), tekrarüretilebilirlik için **sabit bir
train/test bölmesi** tanımlar. Bu bölme `dataset.csv` içindeki `train` kolonunda kodludur:

| Küme | Segment | Nominal | Anomali | Anomali oranı |
|------|:------:|:------:|:-------:|:-------------:|
| **T** (eğitim, `train=1`) | 1594 | 1273 | 321 | %20.1 |
| **Ψ** (test, `train=0`) | 529 | 416 | 113 | %21.4 |
| Toplam | 2123 | 1689 | 434 | %20.4 |

**İlke:** Tüm modeller **aynı Ψ** üzerinde değerlendirilir. Kendi rastgele bölmemizi
oluşturmak, makaleyle kıyaslanabilirliği ortadan kaldırır; bu nedenle pipeline boyunca
yalnızca resmi `train` kolonu kullanılır.

> **Önceki sürümdeki sorun (giderildi):** Notebook'lar başlangıçta kendi rastgele 80/20
> bölmelerini kullanıyordu. Tüm model notebook'ları (NB02, 04, 05, 06, 08) ve
> `train_all_models.py` resmi split'e taşındı.

---

## 2. Veri Sızıntısı (Data Leakage) Önleme

Makale, makine öğrenmesinde tekrarüretilebilirlik krizine ve veri sızıntısına açıkça atıf
yapar (Kapoor & Narayanan, 2023; Wu & Keogh, 2022). Bu çalışmada iki sızıntı kaynağı
giderilmiştir:

### (a) SMOTE split öncesi uygulanıyordu — DÜZELTİLDİ
Önceki NB02, SMOTE'u **tüm veriye** uygulayıp sonra bölüyordu; bu, test noktalarından
türeyen sentetik örneklerin eğitime sızmasına yol açar. Düzeltme:
- SMOTE **yalnızca eğitim katmanına (T)** uygulanır.
- **Ψ test seti asla dengelenmez/dokunulmaz** — gerçek %21.4 anomali oranı korunur.

### (b) Ölçekleyici tüm veride fit ediliyordu — DÜZELTİLDİ
Ölçekleyici (`StandardScaler` / `RobustScaler`) ve tüm dönüşümler **yalnızca T üzerinde fit**
edilir; Ψ'ye yalnızca `transform` uygulanır.

### Leakage-free akış (NB02)
```
dataset.csv → resmi split (T / Ψ)
            → preprocessor.fit(T)  →  transform(T), transform(Ψ)
            → SMOTE(yalnız T)      →  T'den validation ayır
            → kaydet: X_train (SMOTE'lu T), X_val, X_test (= resmi Ψ, dokunulmamış)
```

---

## 3. Yedi Zorunlu Metrik

Makale, anomali tespit sonuçlarının **7 metrikle** raporlanmasını ve tabloların **AUC_PR'a
göre** sıralanmasını zorunlu kılar. `src/metrics.py` bu standardı uygular:

| Metrik | Tanım | Aralık | Not |
|--------|-------|:------:|-----|
| Accuracy | (TP+TN)/(TP+TN+FP+FN) | [0,1] | Dengesiz veride yanıltıcı olabilir |
| Precision | TP/(TP+FP) | [0,1] | Yanlış alarm kontrolü |
| Recall | TP/(TP+FN) | [0,1] | Kaçırma kontrolü |
| F1 | 2·P·R/(P+R) | [0,1] | Precision-Recall harmonik ortalaması |
| **MCC** | Matthews Korelasyon Katsayısı | [-1,1] | **Dengesiz sınıflandırmada tercih edilen** (Chicco & Jurman, 2020) |
| **AUC_ROC** | ROC eğrisi altı alan | [0,1] | Eşikten bağımsız ayrışma |
| **AUC_PR** | Precision-Recall eğrisi altı alan | [0,1] | **Birincil sıralama ölçütü** (azınlık sınıfına duyarlı) |

Ek olarak operasyonel metrikler de sağlanır (7'ye dahil değildir): **FAR** (yanlış alarm
oranı), **FNR** (kaçırma oranı), **Inf.Time** (çıkarım süresi).

```python
from metrics import compute_metrics, metrics_table
m  = compute_metrics(y_true, y_pred, y_score)      # tek model, 7 metrik + ek
df = metrics_table(all_metrics, sort_by="AUC_PR")  # çok-model, AUC_PR sıralı
```

---

## 4. Kanonik Özellik Seti: 18 ESA Handcrafted Feature

Karşılaştırılabilirlik için **yalnızca 18 resmi ESA özelliği** kullanılır (bkz.
[`veri_ve_pipeline.md`](veri_ve_pipeline.md)). Meta sütunlar (`channel`, `sampling`) ve
`channel_id` özellik olarak **kullanılmaz**. Kendi ürettiğimiz ek özellikler (RMS, P2P,
crest, ZCR) ve kanal bilgisi, ayrı bir "özellik genişletme" deneyi olarak ele alınır
(baseline'ı kirletmemek için).

---

## 5. Kanonik Eğitim Motoru: `train_all_models.py`

Tek bir **kanonik kaynak** tüm modelleri eğitir ve değerlendirir; notebook'lar bu çıktıyı
**tüketir** (kendi split'ini/modelini üretip kanonik artefaktları ezmez):

- Girdi: `dataset.csv` → 18 ESA özelliği, resmi split, scaler yalnız T'de fit.
- Eğitim: kanonik karşılaştırmada 42 klasik/topluluk/PyOD modeli; ek olarak 15 derin sıralı ağ
  (`SupervisedAnomalyDetector`) opsiyonel olarak eğitilebilir (`.keras` olarak kaydedilir).
- Çıktı:
  - `models/*.joblib`, `models/*.keras`, `models/unsupervised/*` — eğitilmiş modeller
  - `models/scaler.joblib`, `models/test_data.joblib` — kanonik scaler + resmi Ψ
  - `reports/metrics/final_comparison.json` — 7 metrik, AUC_PR sıralı (**otoriter tablo**)

```bash
python train_all_models.py     # 42 model (~dakikalar); derin modeller etkinse +~20-40 dk
```

Notebook 06 ve 12 bu `final_comparison.json`'ı yükler; onu **yeniden hesaplamaz/ezmez**.

---

## 6. Metodolojik Tutarlılık Özeti

| Notebook | Resmi split | Leakage-free | 7 metrik | Kanonik artefakt |
|----------|:-----------:|:------------:|:--------:|:----------------:|
| NB02 (ön işleme) | ✓ | ✓ | — | üretir (data/processed) |
| NB04 (gözetimli) | ✓ | ✓ | ✓ | ezmez (demo) |
| NB05 (gözetimsiz) | ✓ | ✓ | ✓ | ezmez (demo) |
| NB06 (karşılaştırma) | ✓ (Ψ) | ✓ | ✓ | yükler |
| NB08 (ablation) | ✓ | ✓ | (AUC delta) | data/processed kullanır |
| NB12 (benchmark) | ✓ (Ψ) | ✓ | ✓ | yükler |
| `train_all_models.py` | ✓ | ✓ | ✓ | **üretir (kanonik)** |
