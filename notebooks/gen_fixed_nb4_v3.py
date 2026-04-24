import json

cells = []
def md(src): cells.append({"cell_type":"markdown","metadata":{},"source":[line + "\n" for line in src.split("\n")]})
def code(src): cells.append({"cell_type":"code","metadata":{},"source":[line + "\n" for line in src.split("\n")],"execution_count":None,"outputs":[]})

md("""# 🤖 04 - Gözetimli Makine Öğrenmesi (Segment-Level Classification)
## Uydu Telemetri Anomali Tespiti

**Amaç:** Segment bazlı özellik matrisini kullanarak gözetimli anomali tespiti modellerini eğitmek.

### Modeller:
1. Random Forest (Ensemble)
2. Support Vector Machine (SVM)
3. XGBoost (Gradient Boosting)
4. MLP (Deep Learning)""")

code("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
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
## 📥 Bölüm 1: Veri Hazırlama""")

code("""df_features = pd.read_parquet('../data/features/segment_features.parquet')
print(f"📊 Özellik Matrisi Yüklendi: {df_features.shape}")

drop_cols = ['segment', 'anomaly', 'train', 'channel']
feature_cols = [c for c in df_features.columns if c not in drop_cols]

X = df_features[feature_cols].fillna(0)
y = df_features['anomaly']

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE

X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.15, random_state=42, stratify=y_trainval)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

print("\\nSMOTE Öncesi Eğitim Sınıf Dağılımı:")
print(y_train.value_counts())
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print("\\nSMOTE Sonrası Eğitim Sınıf Dağılımı:")
print(y_train_sm.value_counts())

scaler = RobustScaler()
X_train_s = scaler.fit_transform(X_train_sm)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

print("\\n✅ Veri Bölme, Dengeleme ve Ölçeklendirme Tamamlandı!")""")

md("""---
## 🌲 Bölüm 2: Random Forest ve XGBoost""")

code("""detector = SupervisedAnomalyDetector(random_state=42)

rf_model = detector.train_random_forest(X_train_s, y_train_sm, tune=False)
xgb_model = detector.train_xgboost(X_train_s, y_train_sm, X_val_s, y_val)""")

md("""---
## ⚔️ Bölüm 3: SVM""")

code("""svm_model = detector.train_svm(X_train_s, y_train_sm, kernel='rbf')""")

md("""---
## 🧠 Bölüm 4: MLP (Derin Öğrenme)""")

code("""mlp_model, history = detector.train_mlp(
    X_train_s, y_train_sm.values,
    X_val_s, y_val.values,
    epochs=80, batch_size=32
)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
plt.title('MLP Loss Eğrisi'); plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Acc', color='green')
plt.plot(history.history['val_accuracy'], label='Validation Acc', color='red')
plt.title('MLP Accuracy Eğrisi'); plt.legend()
plt.show()""")

md("""---
## 📊 Bölüm 5: Model Değerlendirme""")

code("""print("=== Modeller Değerlendiriliyor ===")
detector.evaluate_model('RandomForest', X_test_s, y_test)
detector.evaluate_model('XGBoost', X_test_s, y_test)
detector.evaluate_model('SVM', X_test_s, y_test)
detector.evaluate_model('MLP', X_test_s, y_test.values)

results_df = pd.DataFrame(detector.metrics).T
display(results_df.style.background_gradient(cmap='Blues'))""")

code("""fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, model_name in enumerate(detector.models.keys()):
    model = detector.models[model_name]
    if model_name == 'MLP':
        y_pred = (model.predict(X_test_s, verbose=0) >= 0.5).astype(int).flatten()
    else:
        y_pred = model.predict(X_test_s)

    cm = confusion_matrix(y_test, y_pred, normalize='true')
    sns.heatmap(cm, annot=True, fmt='.2%', cmap='Reds', ax=axes[i],
                xticklabels=['Normal', 'Anomali'], yticklabels=['Normal', 'Anomali'])
    axes[i].set_title(f'{model_name}')
    axes[i].set_xlabel('Tahmin'); axes[i].set_ylabel('Gerçek')

plt.tight_layout()
plt.show()""")

md("""---
## 🔍 Bölüm 6: ROC Eğrisi""")

code("""plt.figure(figsize=(10, 8))
for model_name, model in detector.models.items():
    if model_name == 'MLP':
        y_prob = model.predict(X_test_s, verbose=0).flatten()
    elif hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test_s)[:, 1]
    else:
        y_prob = model.decision_function(X_test_s)

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')

plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
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
print("✅ Tüm modeller kaydedildi.")""")

md("### HTML Rapor")
code("""%pip install jupyter nbconvert -q
!jupyter nbconvert --to html 04_model_supervised.ipynb --output ../reports/04_model_supervised_rapor.html""")

nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "cells": cells}
with open('c:/Users/TAHA/Desktop/Uydu-Telemetri-Anomali-Tespiti/notebooks/04_model_supervised.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("NB4 updated.")
