# Uydu Telemetri Anomali Tespiti — Teknik Dokümantasyon

Bu klasör, projenin **akademik ve teknik** dokümantasyonunu içerir. Çalışma, ESA OPS-SAT
uydu telemetri veri seti üzerinde anomali tespitini, **resmi benchmark protokolüne sadık**
(Ruszczak et al., 2024) ve **veri sızıntısından (data leakage) arınmış** bir metodolojiyle
ele alır. Tüm sonuçlar resmi test seti (Ψ) üzerinde, 7 zorunlu metrikle raporlanır.

> Genel/tanıtım amaçlı özet için kök dizindeki [`README.md`](../README.md) dosyasına bakınız.
> Bu klasör, makaleye temel oluşturan teknik ayrıntıları belgeler.

---

## Dokümantasyon Dizini

| Belge | İçerik |
|-------|--------|
| [`metodoloji.md`](metodoloji.md) | Resmi train/test split, 7 zorunlu metrik, veri sızıntısı önleme, kanonik pipeline |
| [`veri_ve_pipeline.md`](veri_ve_pipeline.md) | Veri setleri, 18 ESA özelliği, feature extraction, 3-aşamalı sentetik üretim |
| [`notebooklar.md`](notebooklar.md) | 14 notebook'un sıra ile rehberi (01–14) |
| [`kaynak_moduller.md`](kaynak_moduller.md) | `src/` ve `app/` modüllerinin API'si |
| [`sonuclar.md`](sonuclar.md) | Benchmark reprodüksiyonu ve özgün katkıların bulguları |
| [`calistirma.md`](calistirma.md) | Kurulum, çalıştırma sırası, üretilen dosyalar, kaynaklar |

---

## Projenin Özü (Bir Bakışta)

- **Problem:** Tek-kanallı uydu telemetri segmentlerinde (manyetometre + fotodiyot) ikili
  anomali tespiti (nominal / anomali).
- **Veri:** ESA OPSSAT-AD — 2123 segment, %20 anomali; her segment için 18 handcrafted özellik.
- **Resmi split:** T (eğitim) = 1594 segment, **Ψ (test) = 529 segment** (`dataset.csv`'deki
  `train` kolonu). Tüm değerlendirmeler bu sabit Ψ üzerinde yapılır.
- **Metrikler:** Accuracy, Precision, Recall, F1, **MCC**, **AUC_ROC**, **AUC_PR** (AUC_PR'a
  göre sıralanır — makaleyle aynı).
- **Model havuzu:** Kanonik karşılaştırmada **42 klasik/topluluk/PyOD modeli** (`train_all_models.py`
  tarafından eğitilip `final_comparison.json`'a yazılır). Ek olarak 15 derin sıralı ağ
  (`SupervisedAnomalyDetector`) opsiyonel olarak eğitilebilir.

### Kanonik Sonuçlar (resmi Ψ, AUC_PR sıralı, ilk 6)

| Model | AUC_PR | F1 | MCC | AUC_ROC |
|-------|:------:|:--:|:---:|:-------:|
| ExtraTrees | 0.983 | 0.931 | 0.914 | 0.994 |
| Voting Ensemble | 0.980 | 0.920 | 0.903 | 0.994 |
| MLP (FCNN-eşdeğeri) | 0.979 | 0.946 | 0.932 | 0.990 |
| HistGradientBoosting | 0.974 | 0.923 | 0.903 | 0.990 |
| XGBOD | 0.973 | 0.922 | 0.903 | 0.991 |
| CatBoost | 0.972 | 0.922 | 0.903 | 0.990 |

> Bu değerler, Ruszczak et al. (2024) Tablo 3 baseline'ı ile **birebir kıyaslanabilir** ve
> gözetimli tarafta ortalama |ΔAUC_PR| = 0.004 ile yeniden üretilmiştir (bkz. `sonuclar.md`).

---

## Çalışmanın Üç Fazı

1. **Metodolojik temel** — Resmi split, leakage-free ön işleme, 7 metrik, kanonik eğitim motoru.
2. **Benchmark hizalama** — Makale baseline'ının (30 algoritma) aynı Ψ üzerinde reprodüksiyonu.
3. **Özgün katkılar** — Profil-temelli sentetik telemetri üretimi/ablasyonu ve augmentasyon
   stratejisi karşılaştırması (SMOTE / ICCS-ω / sentetik); güç-tüketimi/verimlilik analizi;
   SHAP yorumlanabilirliği.

## Referanslar

- Ruszczak, B., Kotowski, K., Evans, D., Nalepa, J. (2024). *The OPS-SAT benchmark for
  detecting anomalies in satellite telemetry.* arXiv:2407.04730.
- Ruszczak, B. et al. (2023). *Machine learning detects anomalies in OPS-SAT telemetry.*
  ICCS 2023, doi:10.1007/978-3-031-35995-8_21.

Ayrıntılı kaynak listesi için [`calistirma.md`](calistirma.md#kaynaklar).
