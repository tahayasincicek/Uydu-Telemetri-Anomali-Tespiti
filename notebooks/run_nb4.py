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

# Sklearn Metrics
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, precision_recall_curve, auc

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')

import sys
sys.path.insert(0, '..')
from src.models.supervised import SupervisedAnomalyDetector

print('✅ Kütüphaneler ve Model Sınıfı Yüklendi.')

# 03'te kaydedilen özellikleri yükle
try:
    df_features = pd.read_parquet('../data/features/reaction_wheel_features.parquet')
    print(f"📊 Özellik Matrisi Yüklendi: {df_features.shape}")
except:
    print("Özellik matrisi bulunamadı! Lütfen 03_feature_engineering notebook'unu çalıştırın.")

# Veriyi X ve y olarak ayır
drop_cols = ['timestamp', 'segment', 'label', 'train', 'channel', 'anomaly']
feature_cols = [c for c in df_features.columns if c not in drop_cols]

X = df_features[feature_cols].fillna(0)
y = df_features['anomaly']

from sklearn.model_selection import train_test_split

# Shuffle False yapıyoruz ki zaman serisi sırası bozulmasın
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.15, shuffle=False)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.176, shuffle=False)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# Sınıf Dağılımı Grafiği
fig = px.pie(values=y_train.value_counts().values, names=['Normal (0)', 'Anomali (1)'], 
             title='Eğitim Seti Sınıf Dağılımı', color_discrete_sequence=['#2ecc71', '#e74c3c'])
fig.update_layout(height=400, template='plotly_dark')
fig.show()

from sklearn.dummy import DummyClassifier

# Baseline oluştur (Sadece çoğunluk sınıfını tahmin eden aptal model)
baseline = DummyClassifier(strategy='most_frequent')
baseline.fit(X_train, y_train)
base_acc = baseline.score(X_test, y_test)

print(f"🎯 Baseline (Majority Class) Accuracy: %{base_acc*100:.2f}")
print("Geliştireceğimiz modellerin metrikleri (özellikle F1 ve Recall) bu baseline'dan çok daha iyi olmalıdır.")

detector = SupervisedAnomalyDetector(random_state=42)

# 1. Random Forest (Tune kapalı hızlı eğitim için)
rf_model = detector.train_random_forest(X_train, y_train, tune=False)

# 2. XGBoost
xgb_model = detector.train_xgboost(X_train, y_train, X_val, y_val)

# Feature Importance (Özellik Önemi) Görselleştirme (Random Forest için)
importances = rf_model.feature_importances_
indices = np.argsort(importances)[-15:] # En iyi 15 özellik

plt.figure(figsize=(10, 8))
plt.title('En Önemli 15 Özellik (Random Forest)', fontsize=14)
plt.barh(range(len(indices)), importances[indices], color='indigo', align='center')
plt.yticks(range(len(indices)), [feature_cols[i] for i in indices])
plt.xlabel('Göreceli Önem (Relative Importance)')
plt.show()

# SVM eğitimi (Standart veri seti çok büyükse uzun sürebilir, örneklem alıyoruz)
# Hızlandırmak için train verisinin %10'unu kullanalım
sample_size = min(len(X_train), 5000)
X_train_svm = X_train.iloc[:sample_size]
y_train_svm = y_train.iloc[:sample_size]

svm_model = detector.train_svm(X_train_svm, y_train_svm, kernel='rbf')

seq_length = 30 # RAM ve hız optimizasyonu için 30 adım (window) kullanıyoruz.

# Veriyi Numpy matrisine çeviriyoruz (LSTM formatı)
X_tr_np = X_train.values
y_tr_np = y_train.values
X_v_np = X_val.values
y_v_np = y_val.values
X_te_np = X_test.values
y_te_np = y_test.values

# Uyarı: LSTM eğitimi ekran kartı (GPU) olmadan biraz zaman alabilir (Epoch başına 10-20 sn)
# Hızlı demo için epoch'u düşük tutuyoruz.
lstm_model, history = detector.train_lstm(
    X_tr_np, y_tr_np, 
    X_v_np, y_v_np, 
    seq_len=seq_length, epochs=5, batch_size=128
)

# Training Curve Görselleştirme
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
plt.show()

# LSTM için Test setini de sequence formatına sokmamız gerekiyor
X_test_seq, y_test_seq = detector.prepare_lstm_data(X_te_np, y_te_np, seq_length)

print("=== Modeller Değerlendiriliyor ===")
detector.evaluate_model('RandomForest', X_test, y_test)
detector.evaluate_model('XGBoost', X_test, y_test)
detector.evaluate_model('SVM', X_test, y_test)
detector.evaluate_model('LSTM', X_test_seq, y_test_seq) # LSTM için özel sekans matrisi ve etiketleri

# Sonuçları Tabloya Dök
results_df = pd.DataFrame(detector.metrics).T
display(results_df.style.background_gradient(cmap='Blues'))

def plot_confusion_matrices(detector, X_test, y_test, X_test_seq, y_test_seq):
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
            
        cm = confusion_matrix(y_true, y_pred, normalize='true') # Yüzdelik Oran (Recall bazlı)
        
        sns.heatmap(cm, annot=True, fmt='.2%', cmap='Reds', ax=axes[i], 
                    xticklabels=['Normal', 'Anomali'], yticklabels=['Normal', 'Anomali'])
        axes[i].set_title(f'{model_name} Confusion Matrix (Normalized)')
        axes[i].set_xlabel('Tahmin')
        axes[i].set_ylabel('Gerçek')
        
    plt.tight_layout()
    plt.show()

plot_confusion_matrices(detector, X_test, y_test, X_test_seq, y_test_seq)

plt.figure(figsize=(10, 8))

for model_name, model in detector.models.items():
    if model_name == 'LSTM':
        y_prob = model.predict(X_test_seq, verbose=0).flatten()
        y_true = y_test_seq
    else:
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = model.decision_function(X_test)
        y_true = y_test
        
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')

plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (Yanlış Alarm)')
plt.ylabel('True Positive Rate (Yakalanan Anomali)')
plt.title('ROC Eğrisi Karşılaştırması')
plt.legend(loc="lower right")
plt.show()

# Modelleri kaydet
detector.save_model('RandomForest', '../models/rf_model.joblib')
detector.save_model('XGBoost', '../models/xgb_model.joblib')
detector.save_model('SVM', '../models/svm_model.joblib')
# detector.save_model('LSTM', '../models/lstm_model.h5') # Tensorflow h5 formatında

# Metrikleri JSON'a yaz
detector.save_metadata('../reports/metrics/supervised_metrics.json')

print("✅ Tüm modeller eğitildi, değerlendirildi ve klasörlere kaydedildi.")

# HTML Rapor Olusturma
print("HTML Raporu reports/04_model_supervised_rapor.html konumuna kaydedildi.")

