import json
import os

cells = []
def md(src): cells.append({"cell_type":"markdown","metadata":{},"source":[line + "\n" for line in src.split("\n")]})
def code(src): cells.append({"cell_type":"code","metadata":{},"source":[line + "\n" for line in src.split("\n")],"execution_count":None,"outputs":[]})

md("""# ⚙️ 03 - Özellik Mühendisliği (Segment-Level Feature Engineering)
## Uydu Telemetri Anomali Tespiti

**Amaç:** Zaman serisindeki saniyelik gürültüleri ortadan kaldırmak için anomali tespitini **Olay (Segment) Bazlı** hale getirmek. ESA'nın orijinal özellik matrisi (`dataset.csv`) ile kendi ürettiğimiz Sinyal İşleme özelliklerini (RMS, Peak-to-Peak, ZCR) birleştireceğiz.
""")

code("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import json
import sys
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')

sys.path.insert(0, '..')
from src.feature_engineer import ReactionWheelFeatureEngineer

print('✅ Kütüphaneler ve Feature Engineer yüklendi.')""")

md("""---
## 📥 Veri Yükleme (Raw Segments & Pre-Extracted Features)""")

code("""# Orijinal zaman serisi verisini yükle (Özel özellikler çıkarmak için)
df_segments = pd.read_csv('../data/raw/segments.csv')

# ESA tarafından hazırlanan segment-bazlı istatistiksel özellikleri yükle
df_dataset = pd.read_csv('../data/raw/dataset.csv')

print(f'📊 Zaman Serisi (Segments) Boyutu: {df_segments.shape}')
print(f'📊 Hedef Olay (Dataset) Boyutu: {df_dataset.shape}')""")

md("""---
## 🚀 Segment Bazlı Özel Özellik Çıkarımı (Custom Feature Extraction)
Her bir olay (segment) için Sinyal İşleme metriklerini hesaplıyoruz.""")

code("""engineer = ReactionWheelFeatureEngineer()

# df_segments üzerinden her segment için RMS, P2P, Crest Factor, ZCR hesapla
df_custom_features = engineer.extract_segment_features(df_segments)

print('\\n=== Üretilen Özel Özelliklerin İlk 5 Satırı ===')
display(df_custom_features.head())""")

md("""---
## 🔄 Özellik Birleştirme (Data Merging)
ESA'nın sağladığı `mean`, `var`, `skew` gibi istatistiksel özelliklerle, bizim ürettiğimiz `custom_rms`, `custom_zcr` gibi Sinyal özelliklerini birleştiriyoruz.""")

code("""# Dataset ile Custom özelliklerimizi 'segment' sütunu üzerinden birleştiriyoruz
# İki tabloda da 'anomaly' ve 'channel' var, çakışmayı önlemek için drop ediyoruz
df_custom_features = df_custom_features.drop(columns=['anomaly', 'channel'])

df_final = pd.merge(df_dataset, df_custom_features, on='segment', how='inner')

# Kanal (Sensör) adlarını modellerin anlayabilmesi için sayısal ID'lere çevirelim
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df_final['channel_id'] = le.fit_transform(df_final['channel'])

print(f"✅ Birleştirilmiş Zengin Veri Seti Boyutu: {df_final.shape}")
display(df_final.head(3))""")

md("""---
## 💾 Özellik Matrisini ve Kataloğu Kaydetme""")

code("""# Tüm segmentlerin features matrisini kaydet.
df_final.to_parquet('../data/features/segment_features.parquet')

catalog = {
    "total_segments": len(df_final),
    "features_list": list(df_final.columns),
    "method": "Segment-Level Aggregation & Merging"
}
with open('../data/features/feature_catalog.json', 'w', encoding='utf-8') as f:
    json.dump(catalog, f, indent=4, ensure_ascii=False)

print('✅ Olay Bazlı (Segment-Level) Özellik Matrisi başarıyla kaydedildi.')""")

md("### HTML Rapor Export")
code("""%pip install jupyter nbconvert -q
!jupyter nbconvert --to html 03_feature_engineering.ipynb --output ../reports/03_feature_engineering_rapor.html
print("HTML Raporu kaydedildi.")""")

nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "cells": cells}
with open('c:/Users/TAHA/Desktop/Uydu-Telemetri-Anomali-Tespiti/notebooks/03_feature_engineering.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("Nb3 generated.")
