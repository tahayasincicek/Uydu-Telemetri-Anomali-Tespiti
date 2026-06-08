# Kaynak Modüller (`src/` ve `app/`)

Bu belge, projenin yeniden kullanılabilir Python modüllerinin API'sini özetler. Notebook'lar ve
`train_all_models.py` bu modülleri içe aktarır; uygulama (`app/`) bunların üzerine bir Dash
arayüzü kurar.

---

## `src/` — Çekirdek Kütüphane

### `metrics.py` — Kanonik metrik standardı
Tüm değerlendirmelerin tek kaynağı (bkz. [`metodoloji.md`](metodoloji.md#3-yedi-zorunlu-metrik)).

| Sembol | İmza / değer | Açıklama |
|--------|--------------|----------|
| `compute_metrics` | `(y_true, y_pred, y_score=None, inf_time_ms=None) → dict` | 7 benchmark metriği + FAR/FNR (+varsa Inf.Time) |
| `metrics_table` | `(all_metrics, sort_by="AUC_PR", ascending=False) → DataFrame` | Çok-model tablosu, AUC_PR sıralı |
| `format_metrics_line` | `(name, m) → str` | Tek satır okunabilir özet |
| `BENCHMARK_METRICS` | liste | `[Accuracy, Precision, Recall, F1, MCC, AUC_ROC, AUC_PR]` |
| `PRIMARY_SORT_METRIC` | `"AUC_PR"` | Birincil sıralama ölçütü |

### `benchmark_reference.py` — Makale baseline'ı (Tablo 3)
Ruszczak et al. (2024) Tablo 3'teki 30 algoritma × 7 metrik değerlerini kodlar.

| Sembol | İçerik |
|--------|--------|
| `PAPER_BASELINE` | `{algoritma: {metrik: değer}}` |
| `PAPER_CATEGORY` | algoritma → kategori (gözetimli/gözetimsiz/derin) |
| `NAME_MAP` | makale adı → bizim model adımız |
| `APPROXIMATE_MATCHES` | yaklaşık eşleşmeler: `RF+ICCS≈RandomForest`, `Linear+L2≈Ridge`, `FCNN≈MLP` |

### `data_loader.py` — `TelemetryDataLoader`
`load_data(...)`, `get_summary()`, `validate_data(required_columns=None)`. Ham CSV/parquet
yükleme, özet istatistik ve şema doğrulama.

### `preprocessor.py` — `TelemetriPreprocessor`
`fit(data, numeric_columns)`, `transform(data)`, `fit_transform(...)`, `save_scaler/load_scaler`,
`generate_report()`. Eksik-veri impute, outlier-clip, ölçekleme. **Önemli:** `fit` yalnız
eğitim katmanı (T) üzerinde çağrılır — leakage önleme.

### `feature_engineer.py`
İki bölüm:
- **`extract_esa_features(segments_df, ...)`** — kanonik 18 ESA özelliği (segment → satır).
- **`segment_raw_telemetry(raw_df, train_ratio=0.70, gap_factor=3.0, min_gap_seconds=150.0, min_anomaly_overlap=0.10, seed=42)`** — sürekli ham akıştan hibrit segmentasyon (boşluk-bölme + uzunluk-penceresi); etiketi anomali-örtüşmesinden türetir.
- **`augment_segments_iccs(segments_df, modes=("omega1","omega2","omega3"), nominal_only=True, seed=42)`** — ICCS-ω sinyal augmentasyonu (ω1 dikey ayna, ω2 zaman tersleme, ω3 dairesel kaydırma).
- **`TelemetryFeatureEngineer`** — genişletilmiş özellik keşfi (zaman/frekans/fiziksel/çok-değişkenli/gecikme özellikleri, özellik seçimi). Kanonik baseline'da kullanılmaz; NB03 keşfi içindir.

### `synthetic_generator.py` — `SyntheticTelemetryGenerator`
Profil-temelli sentetik telemetri.

| Metot | Çıktı |
|-------|-------|
| `generate(...)` | sentetik segment seti (etiketli) |
| `generate_and_extract(...)` | segment + 18 özellik tek adımda |
| `generate_raw_stream(channels, n_segments_hint=500, anomaly_ratio=0.20, inter_campaign_gap=(300,7200))` | sürekli ham akış (segment sınırı/etiket YOK) — 3-aşamalı pipeline'ın 1. aşaması |

Sinyal modelleri: manyetometre = Ornstein-Uhlenbeck, fotodiyot = yörünge-periyodik gamma-eğilimli.

### `models/supervised.py` — `SupervisedAnomalyDetector`
**Klasik (sklearn/boosting):** `train_random_forest`, `train_extra_trees`, `train_xgboost`,
`train_xgbod`, `train_hist_gradient_boosting`, `train_gradient_boosting`, `train_adaboost`,
`train_bagging`, `train_voting`, `train_svm`, `train_lsvc`, `train_knn`,
`train_logistic_regression`, `train_ridge`, `train_sgd`, `train_decision_tree`,
`train_naive_bayes`, `train_lda`, `train_qda`, `train_mlp`.
**Derin sıralı ağlar (15):** `train_lstm`, `train_bilstm`, `train_gru`, `train_bigru`,
`train_cnn_lstm`, `train_cnn_bilstm`, `train_cnn_gru`, `train_cnn1d`, `train_tcn`,
`train_transformer`, `train_attention_bilstm`, `train_fcn`, `train_resnet1d`,
`train_inceptiontime`, `train_lstm_fcn` (hepsi `(X_train,y_train,X_val,y_val,epochs,batch_size)`).
**Değerlendirme/IO:** `evaluate_model`, `evaluate_all`, `save_model`, `save_metadata`.

### `models/unsupervised.py` — `UnsupervisedAnomalyDetector`
**Klasik:** `train_isolation_forest`, `train_one_class_svm`, `train_kmeans`, `train_lof`,
`train_gmm`, `train_elliptic_envelope`, `train_pca`, `train_dbscan`.
**PyOD ailesi:** `train_pyod` (toplu) + `train_abod/cof/sod/sos/loda/inne/lmdd`.
**Derin/GAN:** `train_autoencoder`, `train_lstm_autoencoder`, `train_vae`, `train_so_gaal`,
`train_mo_gaal`, `train_deep_svdd`, `train_lunar`, `train_dif`, `train_anogan`, `train_alad`.
**Topluluk/karar:** `compute_ensemble_score`, `detect_anomalies(threshold)`, `save_models`.
> Protokol: yalnız nominal örneklerde eğitilir; eşik validation'da seçilir.

### `models/evaluator.py` — `ModelEvaluator`
`load_models(...)`, `evaluate_all_models(X_test, y_test)`, `generate_comparison_table()`,
`plot_roc_curves`, `plot_pr_curves`, `plot_anomaly_timeline`, `export_metrics(csv, json)`.

### `utils.py`
`save_model/load_model`, `save_metrics/load_metrics`, `create_directory_structure(base)`,
`set_seed(seed=42)`.

---

## `app/` — Dash Arayüzü (ikincil öncelik)

> Arayüz, araştırma çıktılarının interaktif keşfi içindir; akademik sonuçların kaynağı
> notebook'lar + `train_all_models.py`'dir.

| Modül | Rol |
|-------|-----|
| `app.py` | Dash uygulaması giriş noktası, sayfa yönlendirme |
| `synthetic_page.py` | Sentetik telemetri üretimi sayfası (`SyntheticTelemetryGenerator` arayüzü) |
| `esa_pipeline_page.py` | ESA feature extraction pipeline sayfası |
| `power_page.py` | Güç tüketimi / verimlilik analizi sayfası |
| `ablation_page.py` | Ablation sonuçları sayfası |
| `utils/feature_extractor.py` | `extract_features_from_raw(df)` — arayüz için ham→özellik |
| `utils/model_loader.py` | `load_all()`, `predict(model, name, X, thresholds, ...)`, `load_metrics()` — eğitilmiş model yükleme/çıkarım |
