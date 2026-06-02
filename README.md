# 🛰️ Uydu Telemetri Anomali Tespiti

> Uydu telemetri verilerinde makine öğrenmesi ve derin öğrenme yöntemleri ile anomali tespiti yapan kapsamlı bir MLOps projesi.

---

## 📋 Proje Özeti

Bu proje, uydu telemetri sinyallerindeki anormallikleri tespit etmek için **supervised** ve **unsupervised** makine öğrenmesi yöntemlerini bir arada kullanan uçtan uca bir anomali tespit pipeline'ı sunar.

### 🎯 Hedefler

- Uydu telemetri verilerinin analizi ve görselleştirilmesi
- Veri ön işleme ve feature engineering pipeline'ı oluşturulması
- Supervised modeller (Random Forest, XGBoost, SVM, MLP) ile anomali sınıflandırma
- Unsupervised modeller (Isolation Forest, One-Class SVM, Autoencoder, DBSCAN) ile anomali tespiti
- Model performanslarının karşılaştırılması ve en iyi modelin seçilmesi
- Dash (Plotly) tabanlı interaktif dashboard ile sonuçların sunulması

### 📊 Kullanılan Veri Seti

Proje, **ESA OPS-SAT** uydu misyonu veya benzeri açık kaynaklı uydu telemetri veri setlerini kullanmaktadır. Veri seti; sıcaklık, voltaj, akım, jiroskop ve güneş paneli gibi çeşitli sensör ölçümlerini içerir.

---

## 🏗️ Proje Yapısı

```
Uydu-Telemetri-Anomali-Tespiti/
├── data/
│   ├── raw/                  # Ham veri setleri (ESA OPS-SAT veya benzeri)
│   ├── processed/            # Ön işlenmiş veriler
│   └── features/             # Feature engineering çıktıları
├── notebooks/
│   ├── 01_veri_inceleme.ipynb        # Keşifsel veri analizi (EDA)
│   ├── 02_on_isleme.ipynb            # Veri temizleme ve ön işleme
│   ├── 03_feature_engineering.ipynb  # Özellik mühendisliği
│   ├── 04_model_supervised.ipynb     # Denetimli öğrenme modelleri
│   ├── 05_model_unsupervised.ipynb   # Denetimsiz öğrenme modelleri
│   ├── 06_model_karsilastirma.ipynb  # Model karşılaştırma ve değerlendirme
│   ├── 07_shap_analizi.ipynb         # SHAP ile model yorumlanabilirlik analizi
│   └── 08_ablation_study.ipynb       # Ablasyon analizi
├── src/
│   ├── __init__.py
│   ├── data_loader.py        # Veri yükleme ve bağlantı modülü
│   ├── preprocessor.py       # Veri ön işleme fonksiyonları
│   ├── feature_engineer.py   # Özellik mühendisliği modülü
│   ├── models/
│   │   ├── __init__.py
│   │   ├── supervised.py     # Denetimli öğrenme modelleri
│   │   ├── unsupervised.py   # Denetimsiz öğrenme modelleri
│   │   └── evaluator.py      # Model değerlendirme metrikleri
│   └── utils.py              # Yardımcı fonksiyonlar
├── models/                   # Eğitilmiş model dosyaları (.pkl, .h5)
├── reports/
│   ├── figures/              # Grafik ve görselleştirmeler
│   └── metrics/              # Performans metrikleri
├── app/                      # Dash arayüzü
│   ├── app.py                # Ana uygulama dosyası
│   ├── ablation_page.py      # Ablasyon analiz sayfası
│   ├── utils/                # UI yardımcı bileşenleri
│   └── assets/               # Statik dosyalar (CSS, vb.)
├── tests/                    # Birim testleri
├── requirements.txt          # Python bağımlılıkları
├── environment.yml           # Conda ortam dosyası
├── .gitignore                # Git tarafından yoksayılacak dosyalar
└── README.md                 # Bu dosya
```

---

## 🚀 Kurulum

### Ön Gereksinimler

- Python 3.9 veya üzeri
- pip veya conda paket yöneticisi
- Git

### 1. Depoyu Klonlayın

```bash
git clone https://github.com/tahayasincicek/Uydu-Telemetri-Anomali-Tespiti.git
cd Uydu-Telemetri-Anomali-Tespiti
```

### 2a. pip ile Kurulum

```bash
# Sanal ortam oluşturun
python -m venv venv

# Sanal ortamı etkinleştirin
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Bağımlılıkları yükleyin
pip install -r requirements.txt
```

### 2b. Conda ile Kurulum

```bash
conda env create -f environment.yml
conda activate uydu-anomali
```

---

## 💻 Kullanım

### Jupyter Notebook'lar

Notebook'ları sırasıyla çalıştırarak tüm pipeline'ı deneyimleyebilirsiniz:

```bash
jupyter notebook notebooks/
```

| Notebook | Açıklama |
|----------|----------|
| `01_veri_inceleme.ipynb` | Veri keşfi, istatistiksel analiz ve görselleştirme |
| `02_on_isleme.ipynb` | Eksik veri, aykırı değer ve normalizasyon işlemleri |
| `03_feature_engineering.ipynb` | İstatistiksel özellikler, zaman serisi özellikleri çıkarımı |
| `04_model_supervised.ipynb` | Random Forest, XGBoost, SVM, MLP eğitimi |
| `05_model_unsupervised.ipynb` | Isolation Forest, One-Class SVM, Autoencoder |
| `06_model_karsilastirma.ipynb` | Tüm modellerin karşılaştırılması ve raporlanması |
| `07_shap_analizi.ipynb` | SHAP ile model yorumlanabilirliği ve özellik önemi |
| `08_ablation_study.ipynb` | Modeller üzerinde ablasyon (ablation) analizi |

### Dash Dashboard (Arayüz)

```bash
python app/app.py
```

### Komut Satırından Çalıştırma

```python
from src.data_loader import TelemetryDataLoader
from src.preprocessor import TelemetryPreprocessor
from src.feature_engineer import FeatureEngineer
from src.models.supervised import SupervisedAnomalyDetector

# Veri yükleme
loader = TelemetryDataLoader(data_path="data/raw/")
data = loader.load_data()

# Ön işleme
preprocessor = TelemetryPreprocessor()
clean_data = preprocessor.fit_transform(data)

# Feature engineering
fe = FeatureEngineer()
features = fe.create_features(clean_data)

# Model eğitimi
detector = SupervisedAnomalyDetector(model_type="random_forest")
detector.fit(features, labels)
predictions = detector.predict(test_features)
```

---

## 🧠 Kullanılan Yöntemler

### Supervised (Denetimli) Modeller
| Model | Açıklama |
|-------|----------|
| **Random Forest** | Ensemble tabanlı karar ağacı sınıflandırıcı |
| **XGBoost** | Gradient boosting tabanlı güçlü sınıflandırıcı |
| **SVM** | Destek vektör makineleri ile sınıflandırma |
| **MLP** | Çok katmanlı algılayıcı sinir ağı |
| **LightGBM** | Hızlı ve performanslı gradient boosting |
| **CatBoost** | Kategorik özelliklerle başarılı gradient boosting |
| **Stacking Ensemble** | Farklı modelleri birleştiren meta-öğrenme |

### Unsupervised (Denetimsiz) Modeller
| Model | Açıklama |
|-------|----------|
| **Isolation Forest** | Anomali tespitine özel ensemble yöntem |
| **One-Class SVM** | Tek sınıf SVM ile anomali tespiti |
| **Autoencoder** | Derin öğrenme tabanlı yeniden yapılandırma hatası |
| **DBSCAN** | Yoğunluk tabanlı kümeleme ile anomali tespiti |
| **KMeans** | Uzaklık tabanlı kümeleme |

---

## 📈 Değerlendirme Metrikleri

- **Accuracy** (Doğruluk)
- **Precision** (Kesinlik)
- **Recall** (Duyarlılık)
- **F1-Score**
- **ROC-AUC**
- **Confusion Matrix**
- **PR Curve** (Precision-Recall Eğrisi)

---

## 🛠️ Teknoloji Yığını

| Kategori | Kütüphaneler |
|----------|-------------|
| **Veri İşleme** | pandas, numpy, scipy |
| **Makine Öğrenmesi** | scikit-learn, imbalanced-learn |
| **Derin Öğrenme** | TensorFlow, Keras |
| **Görselleştirme** | matplotlib, seaborn, plotly |
| **Dashboard** | Dash, Plotly |
| **Model Saklama** | joblib |

---

## 🤝 Katkıda Bulunma

1. Bu depoyu fork edin
2. Bir feature branch oluşturun (`git checkout -b feature/yeni-ozellik`)
3. Değişikliklerinizi commit edin (`git commit -m 'feat: yeni özellik eklendi'`)
4. Branch'inizi push edin (`git push origin feature/yeni-ozellik`)
5. Bir Pull Request açın

---

## 📄 Lisans

Bu proje MIT Lisansı altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakınız.

---

## 👥 Geliştiriciler

- **Taha Yasin Çiçek** ([@tahayasincicek](https://github.com/tahayasincicek))
- **Furkan Öztürk** ([@furkanozturk06](https://github.com/furkanozturk06))
- **Emirhan Keskin** ([@keskinemirhan](https://github.com/keskinemirhan))

---

<p align="center">
  <i>⭐ Bu projeyi beğendiyseniz yıldız vermeyi unutmayın!</i>
</p>