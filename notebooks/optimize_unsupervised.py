"""
Gözetimsiz Model Optimizasyon Betiği
=====================================
Her model için hiperparametre taraması yapılır, en iyi threshold
Validation seti üzerinde F1 skoru ile belirlenir.
"""
import json, os, time, joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
import warnings
warnings.filterwarnings('ignore')

os.makedirs('../models/unsupervised', exist_ok=True)

# ============================================================
# DATA — aynı split NB4/NB5/NB6 ile tutarlı
# ============================================================
df = pd.read_parquet('../data/features/segment_features.parquet')
drop_cols = ['segment', 'anomaly', 'train', 'channel']
feature_cols = [c for c in df.columns if c not in drop_cols]

X = df[feature_cols].fillna(0)
y = df['anomaly']

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.15, random_state=42, stratify=y_trainval)

# StandardScaler (SVM için zorunlu, diğerleri için de faydalı)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

# Normal veriler (eğitim için)
normal_mask = (y_train == 0).values
X_normal_train = X_train_s[normal_mask]

anomaly_ratio = float(y_train.mean())
print(f"Train: {X_train_s.shape}, Val: {X_val_s.shape}, Test: {X_test_s.shape}")
print(f"Anomali oranı (train): {anomaly_ratio:.3f}")
print(f"Normal eğitim verisi: {X_normal_train.shape}")

# ============================================================
# YARDIMCI FONKSİYONLAR
# ============================================================
def compute_metrics(y_true, y_pred, y_scores=None):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    far = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    auc_val = 0.5
    if y_scores is not None:
        try:
            auc_val = roc_auc_score(y_true, y_scores)
        except:
            pass
    return {'Accuracy': acc, 'Precision': prec, 'Recall': rec,
            'F1': f1, 'AUC-ROC': auc_val, 'FAR': far, 'FNR': fnr}

def find_best_threshold(scores_val, y_val_true, low=50, high=99):
    """Validation seti üzerinde F1'i maximize eden threshold'u bul."""
    best_f1 = 0
    best_t = np.percentile(scores_val, 90)
    for p in np.arange(low, high, 0.5):
        t = np.percentile(scores_val, p)
        preds = (scores_val > t).astype(int)
        f = f1_score(y_val_true, preds, zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_t = t
    return best_t, best_f1

all_metrics = {}

# ============================================================
# 1) ISOLATION FOREST OPTİMİZASYONU
# ============================================================
print("\n" + "="*60)
print("1) ISOLATION FOREST OPTİMİZASYONU")
print("="*60)

best_iso = {'f1': 0}
for n_est in [100, 200, 300]:
    for max_feat in [0.5, 0.75, 1.0]:
        for contam in [0.05, 0.10, 0.15, 0.20]:
            iso = IsolationForest(n_estimators=n_est, max_features=max_feat,
                                  contamination=contam, random_state=42, n_jobs=-1)
            iso.fit(X_normal_train)
            
            scores_val = -iso.score_samples(X_val_s)
            thresh, val_f1 = find_best_threshold(scores_val, y_val.values, low=40, high=99)
            
            if val_f1 > best_iso['f1']:
                best_iso = {'f1': val_f1, 'model': iso, 'thresh': thresh,
                            'params': {'n_estimators': n_est, 'max_features': max_feat,
                                       'contamination': contam}}

iso_model = best_iso['model']
iso_thresh = best_iso['thresh']
scores_test = -iso_model.score_samples(X_test_s)
y_pred = (scores_test > iso_thresh).astype(int)
all_metrics['IsolationForest'] = compute_metrics(y_test, y_pred, scores_test)
print(f"  En iyi params: {best_iso['params']}")
print(f"  Val F1: {best_iso['f1']:.4f}")
print(f"  Test -> Acc: {all_metrics['IsolationForest']['Accuracy']:.4f}, "
      f"Prec: {all_metrics['IsolationForest']['Precision']:.4f}, "
      f"F1: {all_metrics['IsolationForest']['F1']:.4f}, "
      f"AUC: {all_metrics['IsolationForest']['AUC-ROC']:.4f}, "
      f"FAR: {all_metrics['IsolationForest']['FAR']:.4f}")
joblib.dump(iso_model, '../models/unsupervised/isolationforest_model.joblib')

# ============================================================
# 2) ONE-CLASS SVM OPTİMİZASYONU
# ============================================================
print("\n" + "="*60)
print("2) ONE-CLASS SVM OPTİMİZASYONU")
print("="*60)

best_ocsvm = {'f1': 0}
for nu in [0.05, 0.10, 0.15, 0.20]:
    for gamma in ['scale', 'auto', 0.001, 0.01, 0.1]:
        ocsvm = OneClassSVM(kernel='rbf', gamma=gamma, nu=nu)
        ocsvm.fit(X_normal_train)
        
        scores_val = -ocsvm.decision_function(X_val_s)
        thresh, val_f1 = find_best_threshold(scores_val, y_val.values, low=40, high=99)
        
        if val_f1 > best_ocsvm['f1']:
            best_ocsvm = {'f1': val_f1, 'model': ocsvm, 'thresh': thresh,
                          'params': {'nu': nu, 'gamma': gamma}}

ocsvm_model = best_ocsvm['model']
ocsvm_thresh = best_ocsvm['thresh']
scores_test = -ocsvm_model.decision_function(X_test_s)
y_pred = (scores_test > ocsvm_thresh).astype(int)
all_metrics['OneClassSVM'] = compute_metrics(y_test, y_pred, scores_test)
print(f"  En iyi params: {best_ocsvm['params']}")
print(f"  Val F1: {best_ocsvm['f1']:.4f}")
print(f"  Test -> Acc: {all_metrics['OneClassSVM']['Accuracy']:.4f}, "
      f"Prec: {all_metrics['OneClassSVM']['Precision']:.4f}, "
      f"F1: {all_metrics['OneClassSVM']['F1']:.4f}, "
      f"AUC: {all_metrics['OneClassSVM']['AUC-ROC']:.4f}, "
      f"FAR: {all_metrics['OneClassSVM']['FAR']:.4f}")
joblib.dump(ocsvm_model, '../models/unsupervised/oneclasssvm_model.joblib')

# ============================================================
# 3) KMEANS OPTİMİZASYONU
# ============================================================
print("\n" + "="*60)
print("3) KMEANS OPTİMİZASYONU")
print("="*60)

best_km = {'f1': 0}
for n_clusters in [2, 3, 4, 5, 7, 10]:
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    km.fit(X_normal_train)
    
    scores_val = np.min(km.transform(X_val_s), axis=1)
    thresh, val_f1 = find_best_threshold(scores_val, y_val.values, low=40, high=99)
    
    if val_f1 > best_km['f1']:
        best_km = {'f1': val_f1, 'model': km, 'thresh': thresh,
                    'params': {'n_clusters': n_clusters}}

km_model = best_km['model']
km_thresh = best_km['thresh']
scores_test = np.min(km_model.transform(X_test_s), axis=1)
y_pred = (scores_test > km_thresh).astype(int)
all_metrics['KMeans'] = compute_metrics(y_test, y_pred, scores_test)
print(f"  En iyi params: {best_km['params']}")
print(f"  Val F1: {best_km['f1']:.4f}")
print(f"  Test -> Acc: {all_metrics['KMeans']['Accuracy']:.4f}, "
      f"Prec: {all_metrics['KMeans']['Precision']:.4f}, "
      f"F1: {all_metrics['KMeans']['F1']:.4f}, "
      f"AUC: {all_metrics['KMeans']['AUC-ROC']:.4f}, "
      f"FAR: {all_metrics['KMeans']['FAR']:.4f}")
joblib.dump(km_model, '../models/unsupervised/kmeans_model.joblib')

# ============================================================
# 4) LOF OPTİMİZASYONU
# ============================================================
print("\n" + "="*60)
print("4) LOF OPTİMİZASYONU")
print("="*60)

best_lof = {'f1': 0}
for n_neighbors in [5, 10, 15, 20, 30, 50]:
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True,
                              contamination=anomaly_ratio)
    lof.fit(X_normal_train)
    
    scores_val = -lof.score_samples(X_val_s)
    thresh, val_f1 = find_best_threshold(scores_val, y_val.values, low=40, high=99)
    
    if val_f1 > best_lof['f1']:
        best_lof = {'f1': val_f1, 'model': lof, 'thresh': thresh,
                    'params': {'n_neighbors': n_neighbors}}

lof_model = best_lof['model']
lof_thresh = best_lof['thresh']
scores_test = -lof_model.score_samples(X_test_s)
y_pred = (scores_test > lof_thresh).astype(int)
all_metrics['LOF'] = compute_metrics(y_test, y_pred, scores_test)
print(f"  En iyi params: {best_lof['params']}")
print(f"  Val F1: {best_lof['f1']:.4f}")
print(f"  Test -> Acc: {all_metrics['LOF']['Accuracy']:.4f}, "
      f"Prec: {all_metrics['LOF']['Precision']:.4f}, "
      f"F1: {all_metrics['LOF']['F1']:.4f}, "
      f"AUC: {all_metrics['LOF']['AUC-ROC']:.4f}, "
      f"FAR: {all_metrics['LOF']['FAR']:.4f}")
joblib.dump(lof_model, '../models/unsupervised/lof_model.joblib')

# ============================================================
# 5) AUTOENCODER OPTİMİZASYONU
# ============================================================
print("\n" + "="*60)
print("5) AUTOENCODER OPTİMİZASYONU")
print("="*60)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

X_normal_val = X_val_s[y_val.values == 0]
input_dim = X_normal_train.shape[1]

best_ae = {'f1': 0}
for latent_dim in [8, 16, 32]:
    for lr in [0.001, 0.0005]:
        ae = Sequential([
            Dense(64, activation='relu', input_shape=(input_dim,)),
            BatchNormalization(),
            Dropout(0.2),
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dense(latent_dim, activation='relu', name='latent'),
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dense(input_dim, activation='linear')
        ])
        ae.compile(optimizer=Adam(learning_rate=lr), loss='mse')
        ae.fit(X_normal_train, X_normal_train,
               validation_data=(X_normal_val, X_normal_val),
               epochs=150, batch_size=32,
               callbacks=[EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)],
               verbose=0)
        
        recon_val = ae.predict(X_val_s, verbose=0)
        scores_val = np.mean(np.power(X_val_s - recon_val, 2), axis=1)
        thresh, val_f1 = find_best_threshold(scores_val, y_val.values, low=40, high=99)
        
        print(f"    latent={latent_dim}, lr={lr} -> Val F1: {val_f1:.4f}")
        
        if val_f1 > best_ae['f1']:
            best_ae = {'f1': val_f1, 'model': ae, 'thresh': thresh,
                       'params': {'latent_dim': latent_dim, 'lr': lr}}

ae_model = best_ae['model']
ae_thresh = best_ae['thresh']
recon_test = ae_model.predict(X_test_s, verbose=0)
scores_test = np.mean(np.power(X_test_s - recon_test, 2), axis=1)
y_pred = (scores_test > ae_thresh).astype(int)
all_metrics['Autoencoder'] = compute_metrics(y_test, y_pred, scores_test)
print(f"  En iyi params: {best_ae['params']}")
print(f"  Val F1: {best_ae['f1']:.4f}")
print(f"  Test -> Acc: {all_metrics['Autoencoder']['Accuracy']:.4f}, "
      f"Prec: {all_metrics['Autoencoder']['Precision']:.4f}, "
      f"F1: {all_metrics['Autoencoder']['F1']:.4f}, "
      f"AUC: {all_metrics['Autoencoder']['AUC-ROC']:.4f}, "
      f"FAR: {all_metrics['Autoencoder']['FAR']:.4f}")
ae_model.save('../models/unsupervised/autoencoder_model.keras')

# ============================================================
# THRESHOLD'LARI VE SCALER'I KAYDET
# ============================================================
thresholds = {
    'IsolationForest': float(iso_thresh),
    'OneClassSVM': float(ocsvm_thresh),
    'KMeans': float(km_thresh),
    'LOF': float(lof_thresh),
    'Autoencoder': float(ae_thresh)
}
with open('../models/unsupervised/unsupervised_thresholds.json', 'w') as f:
    json.dump(thresholds, f, indent=2)

joblib.dump(scaler, '../models/scaler.joblib')
joblib.dump({'X_test': X_test_s, 'y_test': y_test.values, 'feature_cols': feature_cols},
            '../models/test_data.joblib')

# ============================================================
# SUPERVISED METRİKLERİ DE EKLE (MEVCUT DOSYADAN)
# ============================================================
sup_path = '../reports/metrics/supervised_metrics.json'
if os.path.exists(sup_path):
    with open(sup_path) as f:
        sup_data = json.load(f)
    for name, m in sup_data.get('metrics', {}).items():
        all_metrics[name] = {
            'Accuracy': m.get('Accuracy', 0),
            'Precision': m.get('Precision', 0),
            'Recall': m.get('Recall', 0),
            'F1': m.get('F1', m.get('F1_Score', 0)),
            'AUC-ROC': m.get('AUC-ROC', m.get('AUC', 0)),
            'FAR': m.get('FAR', 0),
            'FNR': m.get('FNR', 0),
        }

with open('../reports/metrics/final_comparison.json', 'w') as f:
    json.dump(all_metrics, f, indent=2)

# ============================================================
# SONUÇ TABLOSU
# ============================================================
print("\n" + "="*70)
print("✅ OPTİMİZASYON TAMAMLANDI — NİHAİ SONUÇLAR")
print("="*70)
print(f"{'Model':20s} | {'Acc':>6s} | {'Prec':>6s} | {'Rec':>6s} | {'F1':>6s} | {'AUC':>6s} | {'FAR':>6s}")
print("-"*70)
for name, m in all_metrics.items():
    print(f"{name:20s} | {m['Accuracy']:6.3f} | {m['Precision']:6.3f} | "
          f"{m['Recall']:6.3f} | {m['F1']:6.3f} | {m['AUC-ROC']:6.3f} | {m['FAR']:6.3f}")

# Hedef kontrolü
print("\n--- HEDEF KONTROL (Gözetimsiz) ---")
targets = {'Accuracy': 0.75, 'Precision': 0.60, 'F1': 0.55, 'AUC-ROC': 0.75}
unsup_names = ['IsolationForest', 'OneClassSVM', 'KMeans', 'LOF', 'Autoencoder']
for name in unsup_names:
    m = all_metrics[name]
    checks = []
    for metric, target in targets.items():
        ok = m[metric] >= target
        checks.append(f"{'✅' if ok else '❌'}{metric}={m[metric]:.3f}")
    far_ok = m['FAR'] < 0.15
    checks.append(f"{'✅' if far_ok else '❌'}FAR={m['FAR']:.3f}")
    print(f"  {name:20s}: {', '.join(checks)}")
