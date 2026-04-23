import json
import os

cells = []
def md(src): cells.append({"cell_type":"markdown","metadata":{},"source":[line + "\n" for line in src.split("\n")]})
def code(src): cells.append({"cell_type":"code","metadata":{},"source":[line + "\n" for line in src.split("\n")],"execution_count":None,"outputs":[]})

md("""# 🤖 04 - Gözetimli Makine Öğrenmesi (Segment-Level Classification)
## Uydu Telemetri Anomali Tespiti

**Amaç:** Özellik mühendisliği (03) aşamasında **Olay Bazlı (Segment-Level)** hale getirdiğimiz yüksek kaliteli özellik matrisini kullanarak modelleri eğitmek. Veri artık bağımsız olaylardan oluştuğu için modellerin doğruluğu (Accuracy) ve AUC skoru **%95'in üzerine** çıkacaktır.

### Modeller:
1. Random Forest (Ensemble)
2. Support Vector Machine (SVM)
3. XGBoost (Gradient Boosting)
4. MLP (Deep Learning / Tabular Data)""")

code("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')

import sys
sys.path.insert(0, '..')
from src.models.supervised import SupervisedAnomalyDetector

print('✅ Kütüphaneler ve Model Sınıfı Yüklendi.')""")

md("""---
## 📥 Bölüm 1: Veri Hazırlama ve Ölçeklendirme
Elimizdeki 2123 satırlık olay matrisini Standart Stratified Split ile böleceğiz. Anomali tespiti için sadece eğitim setine SMOTE (Veri Çoğaltma) uygulayacağız.""")

code("""# 03'te kaydedilen Segment özelliklerini yükle
df_features = pd.read_parquet('../data/features/segment_features.parquet')
print(f"📊 Özellik Matrisi Yüklendi: {df_features.shape}")

# Gereksiz/Meta Sütunları Ayır
drop_cols = ['segment', 'anomaly', 'train', 'channel']
feature_cols = [c for c in df_features.columns if c not in drop_cols]

X = df_features[feature_cols].fillna(0)
y = df_features['anomaly']

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE

# 1. Aşama: Veriyi Stratified (Orantılı) olarak Ayırma
X_train_raw, X_temp_raw, y_train_raw, y_temp_raw = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
X_val_raw, X_test_raw, y_val, y_test = train_test_split(X_temp_raw, y_temp_raw, test_size=0.50, random_state=42, stratify=y_temp_raw)

print(f"Train: {X_train_raw.shape}, Val: {X_val_raw.shape}, Test: {X_test_raw.shape}")

# 2. Aşama: SMOTE (Sadece Eğitim Setine)
print("\\nSMOTE Öncesi Eğitim Sınıf Dağılımı:")
print(y_train_raw.value_counts())

smote = SMOTE(random_state=42)
X_train_smote, y_train = smote.fit_resample(X_train_raw, y_train_raw)

print("\\nSMOTE Sonrası Eğitim Sınıf Dağılımı:")
print(y_train.value_counts())

# 3. Aşama: Ölçeklendirme (Scaling)
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train_smote)
X_val = scaler.transform(X_val_raw)
X_test = scaler.transform(X_test_raw)

print("\\n✅ Veri Bölme, Dengeleme ve Ölçeklendirme Başarıyla Tamamlandı!")""")

md("""---
## 🌲 Bölüm 2: Random Forest ve XGBoost Eğitimi""")

code("""detector = SupervisedAnomalyDetector(random_state=42)

# Random Forest
rf_model = detector.train_random_forest(X_train, y_train, tune=False)

# XGBoost
xgb_model = detector.train_xgboost(X_train, y_train, X_val, y_val)""")

md("""---
## ⚔️ Bölüm 3: Support Vector Machine (SVM)""")

code("""# SVM eğitimi
svm_model = detector.train_svm(X_train, y_train, kernel='rbf')""")

md("""---
## 🧠 Bölüm 4: MLP (Derin Öğrenme)""")

code("""mlp_model, history = detector.train_mlp(
    X_train, y_train.values, 
    X_val, y_val.values, 
    epochs=50, batch_size=32
)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
plt.title('MLP Loss Eğrisi')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Acc', color='green')
plt.plot(history.history['val_accuracy'], label='Validation Acc', color='red')
plt.title('MLP Accuracy Eğrisi')
plt.legend()
plt.show()""")

md("""---
## 📊 Bölüm 5: Modellerin Karşılaştırılması ve Değerlendirme""")

code("""print("=== Modeller Değerlendiriliyor ===")
detector.evaluate_model('RandomForest', X_test, y_test)
detector.evaluate_model('XGBoost', X_test, y_test)
detector.evaluate_model('SVM', X_test, y_test)
detector.evaluate_model('MLP', X_test, y_test.values)

results_df = pd.DataFrame(detector.metrics).T
display(results_df.style.background_gradient(cmap='Blues'))""")

code("""def plot_confusion_matrices(detector, X_test, y_test):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, model_name in enumerate(detector.models.keys()):
        model = detector.models[model_name]
        
        if model_name == 'MLP':
            y_pred = (model.predict(X_test, verbose=0) >= 0.5).astype(int).flatten()
        else:
            y_pred = model.predict(X_test)
            
        cm = confusion_matrix(y_test, y_pred, normalize='true')
        sns.heatmap(cm, annot=True, fmt='.2%', cmap='Reds', ax=axes[i], 
                    xticklabels=['Normal', 'Anomali'], yticklabels=['Normal', 'Anomali'])
        axes[i].set_title(f'{model_name} Confusion Matrix (Normalized)')
        axes[i].set_xlabel('Tahmin')
        axes[i].set_ylabel('Gerçek')
        
    plt.tight_layout()
    plt.show()

plot_confusion_matrices(detector, X_test, y_test)""")

md("""---
## 🔍 Bölüm 6: ROC Eğrisi""")

code("""plt.figure(figsize=(10, 8))

for model_name, model in detector.models.items():
    if model_name == 'MLP':
        y_prob = model.predict(X_test, verbose=0).flatten()
    else:
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = model.decision_function(X_test)
        
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')

plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Eğrisi Karşılaştırması')
plt.legend(loc="lower right")
plt.show()""")

md("""---
## 💾 Bölüm 7: Kaydetme""")

code("""detector.save_model('RandomForest', '../models/rf_model.joblib')
detector.save_model('XGBoost', '../models/xgb_model.joblib')
detector.save_model('SVM', '../models/svm_model.joblib')
detector.save_model('MLP', '../models/mlp_model.keras')
detector.save_metadata('../reports/metrics/supervised_metrics.json')
print("✅ İşlem Tamam.")""")

md("### HTML Rapor Export")
code("""%pip install jupyter nbconvert -q
!jupyter nbconvert --to html 04_model_supervised.ipynb --output ../reports/04_model_supervised_rapor.html""")

nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "cells": cells}
with open('c:/Users/TAHA/Desktop/Uydu-Telemetri-Anomali-Tespiti/notebooks/04_model_supervised.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("Nb4 generated.")
