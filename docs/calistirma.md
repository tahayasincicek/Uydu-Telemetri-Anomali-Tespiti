# Çalıştırma Rehberi

Kurulum, yeniden üretim sırası, üretilen dosyalar ve kaynaklar.

---

## 1. Kurulum

**Gereksinim:** Python 3.11 (proje `python3.11.15` ile geliştirildi).

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` ana yığını: pandas/numpy/scipy/pyarrow, scikit-learn/imbalanced-learn,
xgboost/lightgbm/catboost/pyod, tensorflow, matplotlib/seaborn/plotly, dash, jupyter, shap.

> **Not (catch22):** `pycatch22` kurulu değildir (sistem `python3.11-devel` başlığı/derleyici
> gerektirir, sudo yok). Bu nedenle zaman-serisi augmentasyonunda catch22 yerine **ICCS-ω**
> sinyal augmentasyonu kullanılmıştır (bkz. [`notebooklar.md`](notebooklar.md) NB14).

---

## 2. Veri

Ham veri sürüm kontrolünde **değildir** (`.gitignore`: `data/raw/*`). ESA OPSSAT-AD veri setini
(`segments.csv`, `dataset.csv`) `data/raw/` altına yerleştirin. Kaynak: OPS-SAT benchmark deposu
(Ruszczak et al., 2024 — aşağıdaki kaynaklar).

---

## 3. Yeniden Üretim Sırası

### Adım 0 — Kanonik modelleri eğit (otoriter sonuç kaynağı)
```bash
python train_all_models.py
```
Üretir: `models/*.joblib|*.keras`, `models/unsupervised/*`, `models/scaler.joblib`,
`models/test_data.joblib`, `reports/metrics/final_comparison.json`.
Süre: 42 klasik model ~dakikalar (+ derin sıralı ağlar etkinse ~20–40 dk).

### Adım 1 — Notebook'ları sırayla çalıştır
Önerilen sıra ve bağımlılıklar:

| Sıra | Notebook | Girdi | Üretir |
|:----:|----------|-------|--------|
| 1 | 01 EDA | `data/raw` | EDA görselleri + HTML |
| 2 | 02 Ön işleme | `dataset.csv` | `data/processed/X_{train,val,test}.parquet` |
| 3 | 10 Feature pipeline | `segments.csv` | `data/features/dataset_reconstructed.csv` (r≈0.999 doğrulama) |
| 4 | 03 Özellik müh. (keşif) | `data/processed` | genişletilmiş özellik seti |
| 5 | 04 / 05 (demo) | resmi split | gösterim metrikleri (kanonik artefaktı ezmez) |
| 6 | 06 Karşılaştırma | `final_comparison.json` | ısı haritası, ROC/PR, confusion |
| 7 | 07 SHAP | kanonik modeller | SHAP görselleri + `models/shap_values.pkl` |
| 8 | 08 Ablation | `data/processed` | `models/ablation_results.json` |
| 9 | 09 Sentetik | profiller + `data/raw` | `data/synthetic/synthetic_*.csv` |
| 10 | 11 Güç | model meta | `reports/power_profiles.csv` |
| 11 | 12 Benchmark | `final_comparison.json` + paper baseline | `reports/metrics/benchmark_comparison.csv` |
| 12 | 13 Sentetik ablasyon | `data/synthetic` + resmi split | `ablation_synthetic_*.csv`, `synthetic_real_ks_distance.csv` |
| 13 | 14 Augmentasyon | resmi split | `reports/metrics/augmentation_comparison.csv` |

> **Önemli:** nbconvert ile toplu çalıştırmada bayat çekirdek/önbellek sorunlarını önlemek için
> her notebook'u **çekirdek yeniden başlatarak** baştan sona çalıştırın.

Komut satırından örnek:
```bash
venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/02_on_isleme.ipynb
```

### Adım 2 (opsiyonel) — Dash arayüzü
```bash
python app/app.py
```
Arayüz, eğitilmiş modelleri ve raporları interaktif keşfeder; akademik sonuçların kaynağı
değildir (bkz. [`kaynak_moduller.md`](kaynak_moduller.md#app--dash-arayüzü-ikincil-öncelik)).

---

## 4. Üretilen Dosyalar (sürüm kontrolünde değil)

`.gitignore` şunları hariç tutar (yeniden üretilebilir oldukları için):
`data/raw/*`, `data/processed/*`, `data/features/*`, `data/synthetic/`, `models/*` (joblib/keras/…),
`reports/figures/*`, `reports/metrics/*`, `reports/*.csv`, `resources/` (telifli makaleler),
`catboost_info/`.

**Sürüm kontrolünde olan:** kaynak kod (`src/`, `app/`, `train_all_models.py`), notebook'lar
(`notebooks/*.ipynb`), bu dokümantasyon (`docs/`), `requirements.txt`, kök `README.md`.

---

## 5. Kaynaklar

- Ruszczak, B., Kotowski, K., Evans, D., Nalepa, J. (2024). *The OPS-SAT benchmark for detecting
  anomalies in satellite telemetry.* arXiv:2407.04730.
- Ruszczak, B., Kotowski, K., Andrzejewski, J., et al. (2023). *Machine Learning Detects
  Anomalies in OPS-SAT Telemetry.* ICCS 2023, LNCS, doi:10.1007/978-3-031-35995-8_21.
- Chicco, D., Jurman, G. (2020). *The advantages of the Matthews correlation coefficient (MCC)
  over F1 score and accuracy in binary classification evaluation.* BMC Genomics.
- Kapoor, S., Narayanan, A. (2023). *Leakage and the reproducibility crisis in machine
  learning-based science.* Patterns.
- Wu, R., Keogh, E. (2022). *Current time series anomaly detection benchmarks are flawed and are
  creating the illusion of progress.* IEEE TKDE.

> Telif gereği makale PDF'leri depoya eklenmez (`resources/` gitignore'da). İlgili PDF'ler
> yerel `resources/` klasörüne yerleştirilebilir.
