import json
import os

cells = []
def md(src): cells.append({"cell_type":"markdown","metadata":{},"source":[line + "\n" for line in src.split("\n")]})
def code(src): cells.append({"cell_type":"code","metadata":{},"source":[line + "\n" for line in src.split("\n")],"execution_count":None,"outputs":[]})

md("""# ⚙️ 03 - Özellik Mühendisliği (Feature Engineering)
## Uydu Telemetri Anomali Tespiti

**Amaç:** Ön işleme adımında temizlenen ham segment verilerinden (tüm kanallar dahil) makine öğrenmesi modelleri için zaman alanı, gecikmeli ve türevsel özellikler çıkarmak.
""")

code("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import json
import sys
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')

sys.path.insert(0, '..')
from src.feature_engineer import ReactionWheelFeatureEngineer

print('✅ Kütüphaneler ve Feature Engineer yüklendi.')""")

md("""---
## 📥 Veri Yükleme""")

code("""df_segments = pd.read_csv('../data/raw/segments.csv')
df_segments['timestamp'] = pd.to_datetime(df_segments['timestamp'])
df_segments = df_segments.sort_values(by=['channel', 'timestamp']).reset_index(drop=True)

print(f'📊 Yüklenen veri boyutu: {df_segments.shape}')
display(df_segments.head(3))""")

md("""---
## 🚀 Sınıfı Başlatma ve Özellik Çıkarımı (Tüm Kanallar Üzerinde)
Eskiden sadece 1 kanalı işliyorduk, artık **tüm veri seti** işlenecek.""")

code("""telemetry_channels = ['value']

engineer = ReactionWheelFeatureEngineer(
    rolling_windows=[30, 60, 120],
    lags=[1, 5, 10, 30, 60],
    n_pca_components=3,
    corr_threshold=0.95
)

print(f"İşlenecek tüm veri boyutu: {df_segments.shape}")

# Tüm veriyi işle
df_features = engineer.transform(df_segments, columns=telemetry_channels, target_col='anomaly', fit=True)

# Kanal (Sensör) adlarını modellerin anlayabilmesi için sayısal ID'lere çevirelim
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df_features['channel_id'] = le.fit_transform(df_features['channel'])

print('\\n=== Üretilen Özelliklerin İlk 5 Satırı ===')
display(df_features.head())""")

md("""---
## 📊 1. Görselleştirme (Safe Plotting)""")

code("""time_cols = [c for c in df_features.columns if 'roll_mean' in c or 'roll_std' in c or 'rms' in c]

fig = go.Figure()
if 'value' in df_features.columns:
    fig.add_trace(go.Scatter(y=df_features['value'][:1000], mode='lines', name='Ham Sinyal', line=dict(color='gray', width=1)))
if 'value_roll_mean_60' in df_features.columns:
    fig.add_trace(go.Scatter(y=df_features['value_roll_mean_60'][:1000], mode='lines', name='Rolling Mean (w=60)', line=dict(color='red')))
if 'value_rms_60' in df_features.columns:
    fig.add_trace(go.Scatter(y=df_features['value_rms_60'][:1000], mode='lines', name='RMS (w=60)', line=dict(color='blue')))
fig.update_layout(title='Zaman Alanı Özellikleri (Mean ve RMS) - İlk 1000 Örnek', template='plotly_dark', height=500)
fig.show()""")

md("""---
## 💾 2. Özellik Matrisini ve Kataloğu Kaydetme""")

code("""# Tüm kanalların features matrisini kaydet.
df_features.to_parquet('../data/features/reaction_wheel_features.parquet')

catalog = {
    "feature_count": len(engineer.selected_features),
    "features_list": engineer.selected_features,
    "dropped_correlated": engineer.feature_metadata.get('dropped_correlated', []),
    "parameters_used": {
        "rolling_windows": engineer.rolling_windows,
        "lags": engineer.lags,
        "pca_components": engineer.n_pca_components
    }
}
with open('../data/features/feature_catalog.json', 'w', encoding='utf-8') as f:
    json.dump(catalog, f, indent=4, ensure_ascii=False)

print('✅ Özellik Matrisi ve Katalog başarıyla kaydedildi.')""")

md("### 3. HTML Rapor Export")
code("""%pip install jupyter nbconvert -q
!jupyter nbconvert --to html 03_feature_engineering.ipynb --output ../reports/03_feature_engineering_rapor.html
print("HTML Raporu kaydedildi.")""")

nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "cells": cells}
with open('c:/Users/TAHA/Desktop/Uydu-Telemetri-Anomali-Tespiti/notebooks/03_feature_engineering.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("Nb3 generated.")
