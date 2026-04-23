import json
import os

cells = []
def md(src): cells.append({"cell_type":"markdown","metadata":{},"source":[line + "\n" for line in src.split("\n")]})
def code(src): cells.append({"cell_type":"code","metadata":{},"source":[line + "\n" for line in src.split("\n")],"execution_count":None,"outputs":[]})

md("""# 🤖 04 - Gözetimli Makine Öğrenmesi (Supervised Learning)
## Uydu Telemetri Anomali Tespiti

**Amaç:** Özellik mühendisliği aşamasında zenginleştirilmiş tüm veri setini (Tüm Sensörler) kullanarak gözetimli anomali tespiti modellerini eğitmek ve uzay sistemleri kriterlerine göre değerlendirmek.

### Modeller:
1. Random Forest (Ensemble)
2. Support Vector Machine (SVM)
3. XGBoost (Gradient Boosting)
4. LSTM (Derin Öğrenme / Zaman Serisi)""")

code("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
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
## 📥 Bölüm 1: Veri Hazırlama, Sızıntı Korumalı Bölme ve Ölçeklendirme

Zaman serilerinde en büyük hata, farklı olayların (segmentlerin) yarısını eğitime yarısını teste koymaktır. Bu durumu **Data Leakage (Veri Sızıntısı)** oluşturur. Bu sorunu çözmek için veriyi `Segment ID`'sine göre (GroupShuffleSplit) böleceğiz.""")

code("""# 03'te kaydedilen özellikleri yükle
df_features = pd.read_parquet('../data/features/reaction_wheel_features.parquet')
print(f"📊 Özellik Matrisi Yüklendi: {df_features.shape}")

# Gereksiz/Meta Sütunları Ayır
drop_cols = ['timestamp', 'label', 'train', 'channel', 'anomaly']
feature_cols = [c for c in df_features.columns if c not in drop_cols and c != 'segment']

X = df_features[feature_cols + ['segment']].fillna(0) # segment'i gruplama için geçici tutuyoruz
y = df_features['anomaly']

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE

# 1. Aşama: Veriyi Olaylara (Segment) Göre Ayırma
gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
train_idx, temp_idx = next(gss.split(X, y, groups=X['segment']))

X_train_raw = X.iloc[train_idx]
y_train_raw = y.iloc[train_idx]

X_temp_raw = X.iloc[temp_idx]
y_temp_raw = y.iloc[temp_idx]

# 2. Aşama: Test ve Validation Ayırma
gss_val = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
val_idx, test_idx = next(gss_val.split(X_temp_raw, y_temp_raw, groups=X_temp_raw['segment']))

X_val_raw = X_temp_raw.iloc[val_idx]
y_val = y_temp_raw.iloc[val_idx]

X_test_raw = X_temp_raw.iloc[test_idx]
y_test = y_temp_raw.iloc[test_idx]

# Gruplama için kullandığımız segment bilgisini siliyoruz
X_train_pure = X_train_raw.drop(columns=['segment'])
X_val_pure = X_val_raw.drop(columns=['segment'])
X_test_pure = X_test_raw.drop(columns=['segment'])

print(f"Train: {X_train_pure.shape}, Val: {X_val_pure.shape}, Test: {X_test_pure.shape}")""")

code("""# 3. Aşama: SMOTE (Sadece Eğitim Setine)
print("SMOTE Öncesi Eğitim Sınıf Dağılımı:")
print(y_train_raw.value_counts())

smote = SMOTE(random_state=42)
X_train_smote, y_train = smote.fit_resample(X_train_pure, y_train_raw)

print("\\nSMOTE Sonrası Eğitim Sınıf Dağılımı:")
print(y_train.value_counts())

# 4. Aşama: Ölçeklendirme (Scaling)
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train_smote)
X_val = scaler.transform(X_val_pure)
X_test = scaler.transform(X_test_pure)

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

code("""# SVM eğitimi (Hızlı olması için örneklem alıyoruz)
sample_size = min(len(X_train), 10000)
idx = np.random.choice(len(X_train), sample_size, replace=False)
X_train_svm = X_train[idx]
y_train_svm = y_train.iloc[idx]

svm_model = detector.train_svm(X_train_svm, y_train_svm, kernel='rbf')""")

md("""---
## 🧠 Bölüm 4: LSTM (Derin Öğrenme)""")

code("""seq_length = 30 

X_tr_np = X_train
y_tr_np = y_train.values
X_v_np = X_val
y_v_np = y_val.values
X_te_np = X_test
y_te_np = y_test.values

lstm_model, history = detector.train_lstm(
    X_tr_np, y_tr_np, 
    X_v_np, y_v_np, 
    seq_len=seq_length, epochs=5, batch_size=128
)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
plt.title('LSTM Loss Eğrisi')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Acc', color='green')
plt.plot(history.history['val_accuracy'], label='Validation Acc', color='red')
plt.title('LSTM Accuracy Eğrisi')
plt.legend()
plt.show()""")

md("""---
## 📊 Bölüm 5: Modellerin Karşılaştırılması ve Değerlendirme""")

code("""X_test_seq, y_test_seq = detector.prepare_lstm_data(X_te_np, y_te_np, seq_length)

print("=== Modeller Değerlendiriliyor ===")
detector.evaluate_model('RandomForest', X_test, y_test)
detector.evaluate_model('XGBoost', X_test, y_test)
detector.evaluate_model('SVM', X_test, y_test)
detector.evaluate_model('LSTM', X_test_seq, y_test_seq)

results_df = pd.DataFrame(detector.metrics).T
display(results_df.style.background_gradient(cmap='Blues'))""")

code("""def plot_confusion_matrices(detector, X_test, y_test, X_test_seq, y_test_seq):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, model_name in enumerate(detector.models.keys()):
        model = detector.models[model_name]
        
        if model_name == 'LSTM':
            y_pred = (model.predict(X_test_seq, verbose=0) >= 0.5).astype(int).flatten()
            y_true = y_test_seq
        else:
            y_pred = model.predict(X_test)
            y_true = y_test
            
        cm = confusion_matrix(y_true, y_pred, normalize='true')
        sns.heatmap(cm, annot=True, fmt='.2%', cmap='Reds', ax=axes[i], 
                    xticklabels=['Normal', 'Anomali'], yticklabels=['Normal', 'Anomali'])
        axes[i].set_title(f'{model_name} Confusion Matrix (Normalized)')
        axes[i].set_xlabel('Tahmin')
        axes[i].set_ylabel('Gerçek')
        
    plt.tight_layout()
    plt.show()

plot_confusion_matrices(detector, X_test, y_test, X_test_seq, y_test_seq)""")

md("""---
## 💾 Bölüm 6: Kaydetme""")

code("""detector.save_model('RandomForest', '../models/rf_model.joblib')
detector.save_model('XGBoost', '../models/xgb_model.joblib')
detector.save_model('SVM', '../models/svm_model.joblib')
detector.save_metadata('../reports/metrics/supervised_metrics.json')
print("✅ İşlem Tamam.")""")

md("### 6.1 HTML Rapor Export")
code("""%pip install jupyter nbconvert -q
!jupyter nbconvert --to html 04_model_supervised.ipynb --output ../reports/04_model_supervised_rapor.html""")

nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "cells": cells}
with open('c:/Users/TAHA/Desktop/Uydu-Telemetri-Anomali-Tespiti/notebooks/04_model_supervised.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("Nb4 generated.")
