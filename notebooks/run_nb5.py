from IPython.display import display
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import warnings

# Sklearn
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')

import sys
sys.path.insert(0, '..')
from src.models.unsupervised import UnsupervisedAnomalyDetector

print('✅ Kütüphaneler ve Gözetimsiz Model Sınıfı Yüklendi.')

# 03'te oluşturulan segment bazlı özellikleri kullanacağız
df_features = pd.read_parquet('../data/features/segment_features.parquet')
print(f"📊 Özellik Matrisi Yüklendi: {df_features.shape}")

drop_cols = ['segment', 'anomaly', 'train', 'channel']
feature_cols = [c for c in df_features.columns if c not in drop_cols]

# Sadece Normal Olaylar Eğitim Setine
normal_data = df_features[df_features['anomaly'] == 0]
anomalous_data = df_features[df_features['anomaly'] == 1]

# Basit Train/Test split
from sklearn.model_selection import train_test_split
X_normal_train, X_normal_test = train_test_split(normal_data, test_size=0.2, random_state=42)

# Test seti = Bir miktar Normal veri + Tüm Anomali Verileri
test_data = pd.concat([X_normal_test, anomalous_data])
X_train_raw = X_normal_train[feature_cols]
X_test_raw = test_data[feature_cols]
y_test = test_data['anomaly'].values

# Ölçeklendirme
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

print(f"Train (Sadece Normal): {X_train.shape}")
print(f"Test (Karışık): {X_test.shape}")

detector = UnsupervisedAnomalyDetector(random_state=42)

# Eğitim
iso_model = detector.train_isolation_forest(X_train, contamination=0.05)

# Test Skoru (Negatif skorları pozitife çevirip normalleştiriyoruz)
scores_iso = -iso_model.score_samples(X_test)

# Dağılım Görselleştirmesi
plt.figure(figsize=(10, 5))
sns.histplot(scores_iso[y_test==0], color='blue', label='Normal', kde=True, bins=50)
sns.histplot(scores_iso[y_test==1], color='red', label='Anomali', kde=True, bins=50)
plt.axvline(x=detector.thresholds['IsolationForest'], color='black', linestyle='--', label='Threshold')
plt.title('Isolation Forest Anomali Skor Dağılımı')
plt.xlabel('Anomali Skoru')
plt.legend()
plt.show()

# Hızlı eğitim için epoch düşük tutulmuştur
ae_model, history_ae = detector.train_autoencoder(X_train, X_test[y_test==0], epochs=30, batch_size=32)

plt.figure(figsize=(8, 4))
plt.plot(history_ae.history['loss'], label='Train Loss')
plt.plot(history_ae.history['val_loss'], label='Validation Loss')
plt.title('Autoencoder Loss Eğrisi')
plt.legend()
plt.show()

# Test Skoru
reconstructions = ae_model.predict(X_test, verbose=0)
scores_ae = np.mean(np.power(X_test - reconstructions, 2), axis=1)

plt.figure(figsize=(10, 5))
sns.histplot(scores_ae[y_test==0], color='blue', label='Normal', kde=True, bins=50)
sns.histplot(scores_ae[y_test==1], color='red', label='Anomali', kde=True, bins=50)
plt.axvline(x=detector.thresholds['Autoencoder'], color='black', linestyle='--', label='Threshold')
plt.title('Autoencoder Reconstruction Error Dağılımı')
plt.xlabel('MSE (Hata)')
plt.xlim([0, np.percentile(scores_ae, 95)]) # Aşırı büyük hataları kırmak için x limiti
plt.legend()
plt.show()

svm_model = detector.train_one_class_svm(X_train, nu=0.05)
scores_svm = -svm_model.decision_function(X_test)

kmeans_model = detector.train_kmeans(X_train, n_clusters=3)

distances = kmeans_model.transform(X_test)
scores_kmeans = np.min(distances, axis=1)

# 3D PCA ile Küme Görselleştirmesi
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
X_test_pca = pca.fit_transform(X_test)

fig = px.scatter_3d(
    x=X_test_pca[:, 0], y=X_test_pca[:, 1], z=X_test_pca[:, 2],
    color=y_test.astype(str),
    color_discrete_map={'0': 'blue', '1': 'red'},
    opacity=0.6,
    title="K-Means Uzayında Anomalilerin Ayrışması (3D PCA)"
)
fig.update_layout(margin=dict(l=0, r=0, b=0, t=30), template='plotly_dark')
fig.show()

lof_model = detector.train_lof(X_train, n_neighbors=20)
scores_lof = -lof_model.score_samples(X_test)

ensemble_scores = detector.compute_ensemble_score(X_test, active_models=['IsolationForest', 'Autoencoder', 'OneClassSVM', 'KMeans', 'LOF'])

plt.figure(figsize=(10, 5))
sns.histplot(ensemble_scores[y_test==0], color='blue', label='Normal', kde=True, bins=50)
sns.histplot(ensemble_scores[y_test==1], color='red', label='Anomali', kde=True, bins=50)
plt.title('Ensemble (Bütünleşik) Anomali Skoru Dağılımı')
plt.xlabel('Normalize Skoru (0-1)')
plt.legend()
plt.show()

# Ensemble için Threshold analizi ve ROC
fpr, tpr, thresholds = roc_curve(y_test, ensemble_scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Ensemble AUC = {roc_auc:.3f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.title('Gözetimsiz Ensemble ROC Eğrisi')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()

# Tüm Test segmentleri için anomali yoğunluğunu zaman ekseninde göster
df_plot = pd.DataFrame({
    'Segment_ID': test_data['segment'].values,
    'Gerçek': y_test,
    'Tahmin_Skoru': ensemble_scores
})

# Sadece ilk 200 testi görselleştirelim
fig = px.bar(df_plot.head(200), x='Segment_ID', y='Tahmin_Skoru', color='Gerçek',
             title='Zaman Ekseninde (Segment Bazlı) Anomali Skorları',
             color_continuous_scale=['blue', 'red'])
fig.update_layout(template='plotly_dark')
fig.show()

detector.save_models('../models/unsupervised/')


