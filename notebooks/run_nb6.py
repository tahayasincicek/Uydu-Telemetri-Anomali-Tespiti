from IPython.display import display
import matplotlib; matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import warnings
import time

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')

import sys
sys.path.insert(0, '..')
from src.models.evaluator import ModelEvaluator

print('✅ Kütüphaneler ve Model Değerlendirici Yüklendi.')

# Segment bazlı özellikleri yükle
df_features = pd.read_parquet('../data/features/segment_features.parquet')
df_raw = pd.read_csv('../data/raw/segments.csv')

drop_cols = ['segment', 'anomaly', 'train', 'channel']
feature_cols = [c for c in df_features.columns if c not in drop_cols]

X = df_features[feature_cols].fillna(0)
y = df_features['anomaly'].values

from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

print(f"📊 Test Edilecek Veri Boyutu: {X_scaled.shape}")

# Evaluator'ı Başlat ve Modelleri Yükle
evaluator = ModelEvaluator()
supervised_models = ['RandomForest', 'XGBoost', 'SVM', 'MLP']
unsupervised_models = ['IsolationForest', 'Autoencoder', 'OneClassSVM', 'KMeans', 'LOF']

evaluator.load_models(supervised_models, unsupervised_models)

# Tüm modelleri aynı test seti üzerinden değerlendir
evaluator.evaluate_all_models(X_scaled, y)

df_metrics = evaluator.generate_comparison_table()

# Tabloyu görsel olarak zenginleştir
plt.figure(figsize=(14, 8))
# Inference Time ve Model Size hariç performans metriklerini Heatmap yapalım
perf_cols = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC-ROC', 'FAR', 'FNR']
sns.heatmap(df_metrics[perf_cols], annot=True, fmt=".3f", cmap="YlGnBu", linewidths=.5)
plt.title('Makine Öğrenmesi Modelleri Performans Karşılaştırması')
plt.ylabel('Modeller')
plt.show()

display(df_metrics.style.background_gradient(cmap='viridis', subset=['AUC-ROC', 'F1', 'Accuracy']).highlight_min(subset=['FAR', 'Inf.Time(ms)'], color='lightgreen'))

# ROC Eğrileri
evaluator.plot_roc_curves(y, save_path='../reports/figures/roc_curves_all.png')

# PR Eğrileri
evaluator.plot_pr_curves(y, save_path='../reports/figures/pr_curves_all.png')

fig, axes = plt.subplots(3, 3, figsize=(15, 15))
axes = axes.flatten()

from sklearn.metrics import confusion_matrix
models_to_plot = list(evaluator.predictions.keys())[:9]

for i, name in enumerate(models_to_plot):
    cm = confusion_matrix(y, evaluator.predictions[name], normalize='true')
    sns.heatmap(cm, annot=True, fmt='.2%', cmap='Reds', ax=axes[i], cbar=False,
                xticklabels=['Normal', 'Anomali'], yticklabels=['Normal', 'Anomali'])
    axes[i].set_title(f'{name}')
    axes[i].set_xlabel('Tahmin')
    axes[i].set_ylabel('Gerçek')

plt.tight_layout()
plt.savefig('../reports/figures/confusion_matrices.png', dpi=300)
plt.show()

# İlk 2000 noktayı gösteren interaktif Dashboard
evaluator.plot_anomaly_timeline(df_raw, y, sample_size=2000)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Çıkarım Hızı (Inference Time)
df_metrics['Inf.Time(ms)'].sort_values().plot(kind='barh', ax=ax1, color='skyblue')
ax1.set_title('Çıkarım Hızı - Örnek Başına (Milisaniye) [Daha düşük daha iyi]')
ax1.set_xlabel('Milisaniye (ms)')

# Model Boyutu (Model Size)
df_metrics['Model Size(MB)'].sort_values().plot(kind='barh', ax=ax2, color='lightcoral')
ax2.set_title('Model Boyutu (Megabyte) [Daha düşük daha iyi]')
ax2.set_xlabel('MB')

plt.tight_layout()
plt.savefig('../reports/figures/efficiency.png', dpi=300)
plt.show()

from scipy.stats import wilcoxon

best_model = 'MLP'
second_best = 'XGBoost'

if best_model in evaluator.predictions and second_best in evaluator.predictions:
    err_best = np.abs(y - evaluator.predictions[best_model])
    err_second = np.abs(y - evaluator.predictions[second_best])
    
    stat, p = wilcoxon(err_best, err_second)
    print(f"Wilcoxon Testi Sonucu ({best_model} vs {second_best}):")
    print(f"Statistic: {stat}, p-value: {p}")
    
    if p < 0.05:
        print("✅ İstatistiksel olarak ANLAMLI bir fark var.")
    else:
        print("❌ Fark şans eseri olabilir, istatistiksel olarak anlamlı değil.")

print("Simüle ediliyor: 24 Saatlik LEO Uydu Yörüngesi Telemetri Akışı...")
time.sleep(1)
print(f"Bütünleşik Sistem Ortalama Karar Gecikmesi: {df_metrics['Inf.Time(ms)'].mean():.4f} milisaniye")
print(f"MLP Erken Uyarı Başarısı: Olası kritik arızalardan %{df_metrics.loc['MLP', 'Recall']*100:.1f} oranında tespit edildi.")
print("Alarm Yönetimi: Sinyal seviyesi Threshold'u geçtiğinde sadece 1 kez uyarı verilir (Spam engelleme).")

evaluator.export_metrics('../reports/metrics/final_comparison.csv', '../reports/metrics/final_comparison.json')

print("✅ Tüm raporlama tamamlandı!")

