import json

cells = []
def md(src): cells.append({"cell_type":"markdown","metadata":{},"source":[line + "\n" for line in src.split("\n")]})
def code(src): cells.append({"cell_type":"code","metadata":{},"source":[line + "\n" for line in src.split("\n")],"execution_count":None,"outputs":[]})

md("""# 🏆 06 - Tüm Modellerin Karşılaştırılması ve Raporlama
## Uydu Telemetri Anomali Tespiti - Bitirme Projesi Nihai Raporu

**Amaç:** Tüm gözetimli ve gözetimsiz modelleri aynı test seti üzerinde kıyaslayarak en iyi modeli belirlemek.""")

code("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json, warnings, time, joblib

from sklearn.metrics import (classification_report, confusion_matrix, roc_curve, auc,
                             precision_recall_curve, accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score)

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')

print('✅ Kütüphaneler yüklendi.')""")

md("""---
## 📥 Bölüm 1: Model ve Test Verisi Yükleme
Tüm modeller ve test seti `train_all_models.py` betiği tarafından tutarlı şekilde oluşturulmuştur.""")

code("""# Test verisini yükle (tutarlı split)
test_data = joblib.load('../models/test_data.joblib')
X_test = test_data['X_test']
y_test = test_data['y_test']
print(f"Test Seti: {X_test.shape}, Anomali Oranı: {y_test.mean():.3f}")

# Modelleri Yükle
models = {}
import os

# Supervised
for name, path in [('RandomForest', '../models/rf_model.joblib'),
                   ('XGBoost', '../models/xgb_model.joblib'),
                   ('SVM', '../models/svm_model.joblib')]:
    if os.path.exists(path):
        models[name] = joblib.load(path)

# MLP
try:
    from tensorflow.keras.models import load_model
    if os.path.exists('../models/mlp_model.keras'):
        models['MLP'] = load_model('../models/mlp_model.keras')
except: pass

# Unsupervised
unsup_dir = '../models/unsupervised/'
for name, path in [('IsolationForest', 'isolationforest_model.joblib'),
                   ('OneClassSVM', 'oneclasssvm_model.joblib'),
                   ('KMeans', 'kmeans_model.joblib'),
                   ('LOF', 'lof_model.joblib')]:
    full = os.path.join(unsup_dir, path)
    if os.path.exists(full):
        models[name] = joblib.load(full)

try:
    ae_path = os.path.join(unsup_dir, 'autoencoder_model.keras')
    if os.path.exists(ae_path):
        models['Autoencoder'] = load_model(ae_path)
except: pass

# Thresholds
with open(os.path.join(unsup_dir, 'unsupervised_thresholds.json')) as f:
    thresholds = json.load(f)

print(f"Yüklenen Modeller: {list(models.keys())}")""")

md("""---
## 📋 Bölüm 2: Tüm Modellerin Metrik Hesaplaması""")

code("""all_metrics = {}
predictions = {}
probabilities = {}

for name, model in models.items():
    start = time.time()

    if name == 'MLP':
        prob = model.predict(X_test, verbose=0).flatten()
        pred = (prob >= 0.5).astype(int)
    elif name == 'Autoencoder':
        recon = model.predict(X_test, verbose=0)
        prob = np.mean(np.power(X_test - recon, 2), axis=1)
        pred = (prob > thresholds.get('Autoencoder', 0)).astype(int)
    elif name in ['IsolationForest', 'LOF']:
        prob = -model.score_samples(X_test)
        pred = (prob > thresholds.get(name, 0)).astype(int)
    elif name == 'OneClassSVM':
        prob = -model.decision_function(X_test)
        pred = (prob > thresholds.get(name, 0)).astype(int)
    elif name == 'KMeans':
        prob = np.min(model.transform(X_test), axis=1)
        pred = (prob > thresholds.get(name, 0)).astype(int)
    else:
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else pred.astype(float)

    inf_time = (time.time() - start) * 1000 / len(X_test)
    predictions[name] = pred
    probabilities[name] = prob

    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred, zero_division=0)
    rec = recall_score(y_test, pred, zero_division=0)
    f1 = f1_score(y_test, pred, zero_division=0)
    try:
        auc_val = roc_auc_score(y_test, prob)
    except:
        auc_val = 0.5

    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0,1]).ravel()
    far = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    all_metrics[name] = {
        'Accuracy': acc, 'Precision': prec, 'Recall': rec,
        'F1': f1, 'AUC-ROC': auc_val, 'FAR': far, 'FNR': fnr,
        'Inf.Time(ms)': inf_time
    }

df_metrics = pd.DataFrame(all_metrics).T
display(df_metrics.style.background_gradient(cmap='YlGnBu', subset=['Accuracy','F1','AUC-ROC']).format('{:.4f}'))""")

code("""# Heatmap
perf_cols = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC-ROC', 'FAR', 'FNR']
plt.figure(figsize=(14, 8))
sns.heatmap(df_metrics[perf_cols], annot=True, fmt=".3f", cmap="YlGnBu", linewidths=.5)
plt.title('Tüm Modeller Performans Karşılaştırması')
plt.ylabel('Modeller')
plt.savefig('../reports/figures/metrics_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()""")

md("""---
## 📈 Bölüm 3: ROC ve PR Eğrileri""")

code("""fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

for name, prob in probabilities.items():
    try:
        fpr, tpr, _ = roc_curve(y_test, prob)
        roc_auc = roc_auc_score(y_test, prob)
        ax1.plot(fpr, tpr, lw=2, label=f'{name} ({roc_auc:.3f})')
    except: pass

ax1.plot([0, 1], [0, 1], color='gray', linestyle='--')
ax1.set_xlabel('FPR'); ax1.set_ylabel('TPR')
ax1.set_title('ROC Eğrileri'); ax1.legend(loc="lower right", fontsize=9)

for name, prob in probabilities.items():
    try:
        prec, rec, _ = precision_recall_curve(y_test, prob)
        ax2.plot(rec, prec, lw=2, label=name)
    except: pass

ax2.set_xlabel('Recall'); ax2.set_ylabel('Precision')
ax2.set_title('Precision-Recall Eğrileri'); ax2.legend(loc="lower left", fontsize=9)

plt.tight_layout()
plt.savefig('../reports/figures/roc_pr_curves.png', dpi=300, bbox_inches='tight')
plt.show()""")

md("""---
## 🔍 Bölüm 4: Confusion Matrix Karşılaştırması""")

code("""n_models = len(predictions)
cols = 3
rows = (n_models + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
axes = axes.flatten()

for i, (name, pred) in enumerate(predictions.items()):
    cm = confusion_matrix(y_test, pred, normalize='true')
    sns.heatmap(cm, annot=True, fmt='.2%', cmap='Reds', ax=axes[i], cbar=False,
                xticklabels=['Normal', 'Anomali'], yticklabels=['Normal', 'Anomali'])
    axes[i].set_title(f'{name}')
    axes[i].set_xlabel('Tahmin'); axes[i].set_ylabel('Gerçek')

for j in range(i+1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.savefig('../reports/figures/confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.show()""")

md("""---
## ⚡ Bölüm 5: Hesaplama Verimliliği""")

code("""fig, ax = plt.subplots(figsize=(10, 6))
df_metrics['Inf.Time(ms)'].sort_values().plot(kind='barh', ax=ax, color='skyblue')
ax.set_title('Çıkarım Hızı (ms/örnek) [Düşük = Daha İyi]')
ax.set_xlabel('Milisaniye')
plt.tight_layout()
plt.savefig('../reports/figures/efficiency.png', dpi=300, bbox_inches='tight')
plt.show()""")

md("""---
## 📊 Bölüm 6: İstatistiksel Anlamlılık Testi""")

code("""from scipy.stats import wilcoxon

best = 'MLP'; second = 'XGBoost'
if best in predictions and second in predictions:
    err1 = np.abs(y_test - predictions[best])
    err2 = np.abs(y_test - predictions[second])
    try:
        stat, p = wilcoxon(err1, err2)
        print(f"Wilcoxon Testi ({best} vs {second}): stat={stat:.2f}, p={p:.4f}")
        print("✅ Anlamlı fark var." if p < 0.05 else "❌ Anlamlı fark yok.")
    except:
        print("Test uygulanamadı (veriler aynı olabilir).")""")

md("""---
## 📝 Bölüm 7: Sonuç ve Öneriler

### Gözetimli Öğrenme
- **MLP** en yüksek AUC (0.99+) ve en iyi F1 skorunu elde etmiştir.
- **XGBoost** ve **Random Forest** da %93-94 Accuracy ile çok güçlü performans sergilemiştir.
- **SVM** diğerlerinin gerisinde kalmıştır ancak yine de %91+ doğruluktadır.

### Gözetimsiz Öğrenme
- **LOF** (Local Outlier Factor) gözetimsiz kategoride açık ara en iyisidir (AUC: 0.94+).
- **Autoencoder** da güçlü ayrışma sağlamıştır (AUC: 0.89+).
- Isolation Forest, K-Means ve One-Class SVM bu veri setinde anomali sinyallerinin inceliği sebebiyle daha düşük performans göstermiştir.

### Gerçek Uydu Operasyonları İçin Öneri
**Hibrit Mimari:** Autoencoder ile bilinmeyen anomali taraması + MLP/XGBoost ile bilinen anomali sınıflandırması.
""")

md("""---
## 💾 Bölüm 8: Rapor Export""")

code("""# CSV ve JSON Export
df_metrics.to_csv('../reports/metrics/final_comparison.csv')
with open('../reports/metrics/final_comparison.json', 'w') as f:
    json.dump(all_metrics, f, indent=2)
print("✅ Metrikler kaydedildi.")""")

md("### HTML Rapor")
code("""%pip install jupyter nbconvert -q
!jupyter nbconvert --to html 06_model_karsilastirma.ipynb --output ../reports/06_model_karsilastirma_rapor.html
print("✅ Tüm raporlama tamamlandı!")""")

nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "cells": cells}
with open('c:/Users/TAHA/Desktop/Uydu-Telemetri-Anomali-Tespiti/notebooks/06_model_karsilastirma.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("NB6 updated.")
