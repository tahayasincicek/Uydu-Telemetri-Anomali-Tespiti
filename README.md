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
- Streamlit tabanlı interaktif dashboard ile sonuçların sunulması

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
│   └── 06_model_karsilastirma.ipynb  # Model karşılaştırma ve değerlendirme
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
├── app/                      # Streamlit arayüzü
│   ├── main.py               # Ana uygulama dosyası
│   ├── components/           # UI bileşenleri
│   └── assets/               # Statik dosyalar (CSS, görseller)
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
| `04_model_supervised.ipynb` | 36 gözetimli model: 20 klasik (RF/ExtraTrees/Boosting'ler/SVM/KNN/LogReg/DT/NB/LDA/QDA/Bagging/Ridge/SGD/Voting/Stacking) + 16 derin (MLP, 1D-CNN, FCN, ResNet-1D, InceptionTime, LSTM-FCN, LSTM, BiLSTM, GRU, BiGRU, CNN-LSTM, CNN-BiLSTM, CNN-GRU, Attention-BiLSTM, Transformer, TCN) |
| `05_model_unsupervised.ipynb` | 14 gözetimsiz: IsolationForest, OneClassSVM, KMeans, LOF, GMM, EllipticEnvelope, PCA, DBSCAN, Autoencoder, VAE + PyOD (ECOD, COPOD, HBOS, CBLOF) |
| `06_model_karsilastirma.ipynb` | Tüm modellerin karşılaştırılması ve raporlanması |

### Streamlit Dashboard

```bash
streamlit run app/main.py
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

Proje toplam **50 model** içerir: 36 gözetimli + 14 gözetimsiz.

### Supervised (Denetimli) Modeller — 36

**Klasik / Ağaç & Boosting (20)**
| Model | Açıklama |
|-------|----------|
| **Random Forest** | Ensemble tabanlı karar ağacı sınıflandırıcı |
| **Extra Trees** | Aşırı rastgeleleştirilmiş ağaç topluluğu |
| **Decision Tree** | Tek karar ağacı (yorumlanabilir baseline) |
| **XGBoost** | Gradient boosting tabanlı güçlü sınıflandırıcı |
| **LightGBM** | Histogram tabanlı hızlı gradient boosting |
| **CatBoost** | Kategorik-dostu gradient boosting |
| **Gradient Boosting** | sklearn gradient boosting sınıflandırıcı |
| **HistGradientBoosting** | Histogram tabanlı modern/hızlı boosting |
| **AdaBoost** | Adaptif boosting topluluğu |
| **SVM** | Destek vektör makineleri ile sınıflandırma |
| **KNN** | K-en yakın komşu sınıflandırıcı |
| **Logistic Regression** | Doğrusal taban (baseline) model |
| **Gaussian Naive Bayes** | Olasılıksal baseline sınıflandırıcı |
| **LDA** | Doğrusal diskriminant analizi |
| **QDA** | Karesel diskriminant analizi |
| **Bagging** | Bootstrap aggregating topluluğu |
| **Ridge** | L2 düzenlenmiş doğrusal sınıflandırıcı |
| **SGD** | Stokastik gradyan inişli doğrusal model (log-loss) |
| **Voting Ensemble** | RF + GB + LogReg yumuşak (soft) oylama |
| **Stacking Ensemble** | RF + XGB + LGBM tabanlı meta-öğrenici |

**Derin Öğrenme & Sıralı/Hibrit Ağlar (16)**
| Model | Açıklama |
|-------|----------|
| **MLP** | Çok katmanlı algılayıcı sinir ağı |
| **1D-CNN** | Saf 1B evrişimli sinir ağı |
| **FCN** | Fully Convolutional Network (TS baseline) |
| **ResNet-1D** | Artık (residual) bloklu 1D CNN |
| **InceptionTime** | Çok ölçekli Inception modülleri (SOTA TSC) |
| **LSTM-FCN** | Paralel LSTM + FCN hibrit |
| **LSTM** | Uzun-kısa vadeli bellek ağı |
| **BiLSTM** | Çift yönlü LSTM |
| **GRU** | Geçitli tekrarlayan birim ağı |
| **BiGRU** | Çift yönlü GRU |
| **CNN-LSTM** | 1D evrişim + LSTM hibrit |
| **CNN-BiLSTM** | 1D evrişim + çift yönlü LSTM hibrit |
| **CNN-GRU** | 1D evrişim + GRU hibrit |
| **Attention-BiLSTM** | BiLSTM + self-attention havuzlama |
| **Transformer** | Self-attention tabanlı encoder |
| **TCN** | Temporal Convolutional Network (dilated causal conv) |

### Unsupervised (Denetimsiz) Modeller — 14

**sklearn / Derin (10)**
| Model | Açıklama |
|-------|----------|
| **Isolation Forest** | Anomali tespitine özel ensemble yöntem |
| **One-Class SVM** | Tek sınıf SVM ile anomali tespiti |
| **K-Means** | Küme merkezine uzaklık tabanlı anomali skoru |
| **LOF** | Local Outlier Factor (yoğunluk tabanlı) |
| **GMM** | Gaussian Mixture; düşük olabilirlik = anomali |
| **Elliptic Envelope** | Robust kovaryans / Mahalanobis |
| **PCA** | Yeniden yapılandırma hatası |
| **DBSCAN** | Çekirdek-noktaya uzaklık (novelty) |
| **Autoencoder** | Derin yeniden yapılandırma hatası |
| **VAE** | Variational Autoencoder (KL + recon) |

**PyOD Dedektörleri (4)** — *opsiyonel, `pip install pyod`*
| Model | Açıklama |
|-------|----------|
| **ECOD** | Empirik kümülatif dağılım (parametre-siz, ADBench lideri) |
| **COPOD** | Copula tabanlı outlier tespiti |
| **HBOS** | Histogram tabanlı (çok hızlı) |
| **CBLOF** | Küme tabanlı yerel outlier faktörü |

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
| **Dashboard** | Streamlit |
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

## 👤 Geliştirici

**Taha Yasin Çiçek**

- GitHub: [@tahayasincicek](https://github.com/tahayasincicek)

---

<p align="center">
  <i>⭐ Bu projeyi beğendiyseniz yıldız vermeyi unutmayın!</i>
</p>