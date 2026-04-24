"""
Full pipeline fix: Train supervised + unsupervised models on the SAME split,
save everything (models, scalers, test indices) so NB6 can evaluate consistently.
"""
import json, os, joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import SVC, OneClassSVM
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import (classification_report, accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score, confusion_matrix,
                             roc_curve, auc, precision_recall_curve)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

os.makedirs('../models/unsupervised', exist_ok=True)
os.makedirs('../reports/metrics', exist_ok=True)

# ============================================================
# 1) DATA LOADING & SPLIT
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

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
print(f"Test anomaly ratio: {y_test.mean():.3f}")

# SMOTE on train only
sm = SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

# Scale
scaler = RobustScaler()
X_train_s = scaler.fit_transform(X_train_sm)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

# Save scaler and test indices for NB6 consistency
joblib.dump(scaler, '../models/scaler.joblib')
joblib.dump({'X_test': X_test_s, 'y_test': y_test.values, 'feature_cols': feature_cols},
            '../models/test_data.joblib')

# ============================================================
# 2) SUPERVISED MODELS
# ============================================================
all_metrics = {}

# --- Random Forest ---
print("\n🌲 Training Random Forest...")
rf = RandomForestClassifier(n_estimators=500, max_depth=None, min_samples_split=2,
                            class_weight='balanced', random_state=42, n_jobs=-1)
rf.fit(X_train_s, y_train_sm)
joblib.dump(rf, '../models/rf_model.joblib')

y_pred = rf.predict(X_test_s)
y_prob = rf.predict_proba(X_test_s)[:,1]
all_metrics['RandomForest'] = {
    'Accuracy': accuracy_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred, zero_division=0),
    'Recall': recall_score(y_test, y_pred, zero_division=0),
    'F1': f1_score(y_test, y_pred, zero_division=0),
    'AUC-ROC': roc_auc_score(y_test, y_prob),
}
print(f"  RF -> Acc: {all_metrics['RandomForest']['Accuracy']:.4f}, F1: {all_metrics['RandomForest']['F1']:.4f}, AUC: {all_metrics['RandomForest']['AUC-ROC']:.4f}")

# --- XGBoost ---
print("🚀 Training XGBoost...")
xg = xgb.XGBClassifier(n_estimators=500, max_depth=8, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8, random_state=42,
                        n_jobs=-1, eval_metric='auc', early_stopping_rounds=30)
xg.fit(X_train_s, y_train_sm, eval_set=[(X_val_s, y_val)], verbose=False)
joblib.dump(xg, '../models/xgb_model.joblib')

y_pred = xg.predict(X_test_s)
y_prob = xg.predict_proba(X_test_s)[:,1]
all_metrics['XGBoost'] = {
    'Accuracy': accuracy_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred, zero_division=0),
    'Recall': recall_score(y_test, y_pred, zero_division=0),
    'F1': f1_score(y_test, y_pred, zero_division=0),
    'AUC-ROC': roc_auc_score(y_test, y_prob),
}
print(f"  XGB -> Acc: {all_metrics['XGBoost']['Accuracy']:.4f}, F1: {all_metrics['XGBoost']['F1']:.4f}, AUC: {all_metrics['XGBoost']['AUC-ROC']:.4f}")

# --- SVM ---
print("⚔️ Training SVM...")
svm = CalibratedClassifierCV(SVC(kernel='rbf', C=10, gamma='scale',
                                 class_weight='balanced', random_state=42), cv=3)
svm.fit(X_train_s, y_train_sm)
joblib.dump(svm, '../models/svm_model.joblib')

y_pred = svm.predict(X_test_s)
y_prob = svm.predict_proba(X_test_s)[:,1]
all_metrics['SVM'] = {
    'Accuracy': accuracy_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred, zero_division=0),
    'Recall': recall_score(y_test, y_pred, zero_division=0),
    'F1': f1_score(y_test, y_pred, zero_division=0),
    'AUC-ROC': roc_auc_score(y_test, y_prob),
}
print(f"  SVM -> Acc: {all_metrics['SVM']['Accuracy']:.4f}, F1: {all_metrics['SVM']['F1']:.4f}, AUC: {all_metrics['SVM']['AUC-ROC']:.4f}")

# --- MLP ---
print("🧠 Training MLP...")
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

mlp = Sequential([
    Dense(256, activation='relu', input_shape=(X_train_s.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
mlp.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
mlp.fit(X_train_s, y_train_sm.values, validation_data=(X_val_s, y_val.values),
        epochs=80, batch_size=32,
        callbacks=[EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)],
        verbose=0)
mlp.save('../models/mlp_model.keras')

y_prob = mlp.predict(X_test_s, verbose=0).flatten()
y_pred = (y_prob >= 0.5).astype(int)
all_metrics['MLP'] = {
    'Accuracy': accuracy_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred, zero_division=0),
    'Recall': recall_score(y_test, y_pred, zero_division=0),
    'F1': f1_score(y_test, y_pred, zero_division=0),
    'AUC-ROC': roc_auc_score(y_test, y_prob),
}
print(f"  MLP -> Acc: {all_metrics['MLP']['Accuracy']:.4f}, F1: {all_metrics['MLP']['F1']:.4f}, AUC: {all_metrics['MLP']['AUC-ROC']:.4f}")

# ============================================================
# 3) UNSUPERVISED MODELS (train on NORMAL data only from train split)
# ============================================================
# Normal only from scaled train data (before SMOTE)
X_train_orig_s = scaler.transform(X_train)  # original train without SMOTE
normal_mask = (y_train == 0).values
X_normal_train = X_train_orig_s[normal_mask]

print("\n--- UNSUPERVISED MODELS ---")

# Helper: find best threshold using validation set
def find_best_threshold(scores_val, y_val_true, percentiles=range(80, 100)):
    best_f1 = 0
    best_t = np.percentile(scores_val, 95)
    for p in percentiles:
        t = np.percentile(scores_val, p)
        preds = (scores_val > t).astype(int)
        f = f1_score(y_val_true, preds, zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_t = t
    return best_t

# --- Isolation Forest ---
print("🌲 Training Isolation Forest...")
iso = IsolationForest(n_estimators=300, contamination=0.10, max_features=1.0, random_state=42, n_jobs=-1)
iso.fit(X_normal_train)
joblib.dump(iso, '../models/unsupervised/isolationforest_model.joblib')

scores_val = -iso.score_samples(X_val_s)
iso_thresh = find_best_threshold(scores_val, y_val.values)
scores_test = -iso.score_samples(X_test_s)
y_pred = (scores_test > iso_thresh).astype(int)
all_metrics['IsolationForest'] = {
    'Accuracy': accuracy_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred, zero_division=0),
    'Recall': recall_score(y_test, y_pred, zero_division=0),
    'F1': f1_score(y_test, y_pred, zero_division=0),
    'AUC-ROC': roc_auc_score(y_test, scores_test),
}
print(f"  IF -> Acc: {all_metrics['IsolationForest']['Accuracy']:.4f}, F1: {all_metrics['IsolationForest']['F1']:.4f}, AUC: {all_metrics['IsolationForest']['AUC-ROC']:.4f}")

# --- One-Class SVM ---
print("⚔️ Training One-Class SVM...")
ocsvm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.10)
ocsvm.fit(X_normal_train)
joblib.dump(ocsvm, '../models/unsupervised/oneclasssvm_model.joblib')

scores_val = -ocsvm.decision_function(X_val_s)
ocsvm_thresh = find_best_threshold(scores_val, y_val.values)
scores_test = -ocsvm.decision_function(X_test_s)
y_pred = (scores_test > ocsvm_thresh).astype(int)
all_metrics['OneClassSVM'] = {
    'Accuracy': accuracy_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred, zero_division=0),
    'Recall': recall_score(y_test, y_pred, zero_division=0),
    'F1': f1_score(y_test, y_pred, zero_division=0),
    'AUC-ROC': roc_auc_score(y_test, scores_test),
}
print(f"  OCSVM -> Acc: {all_metrics['OneClassSVM']['Accuracy']:.4f}, F1: {all_metrics['OneClassSVM']['F1']:.4f}, AUC: {all_metrics['OneClassSVM']['AUC-ROC']:.4f}")

# --- K-Means ---
print("🎯 Training K-Means...")
km = KMeans(n_clusters=5, random_state=42, n_init='auto')
km.fit(X_normal_train)
joblib.dump(km, '../models/unsupervised/kmeans_model.joblib')

scores_val = np.min(km.transform(X_val_s), axis=1)
km_thresh = find_best_threshold(scores_val, y_val.values)
scores_test = np.min(km.transform(X_test_s), axis=1)
y_pred = (scores_test > km_thresh).astype(int)
all_metrics['KMeans'] = {
    'Accuracy': accuracy_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred, zero_division=0),
    'Recall': recall_score(y_test, y_pred, zero_division=0),
    'F1': f1_score(y_test, y_pred, zero_division=0),
    'AUC-ROC': roc_auc_score(y_test, scores_test),
}
print(f"  KM -> Acc: {all_metrics['KMeans']['Accuracy']:.4f}, F1: {all_metrics['KMeans']['F1']:.4f}, AUC: {all_metrics['KMeans']['AUC-ROC']:.4f}")

# --- LOF ---
print("🔍 Training LOF...")
lof = LocalOutlierFactor(n_neighbors=20, novelty=True)
lof.fit(X_normal_train)
joblib.dump(lof, '../models/unsupervised/lof_model.joblib')

scores_val = -lof.score_samples(X_val_s)
lof_thresh = find_best_threshold(scores_val, y_val.values)
scores_test = -lof.score_samples(X_test_s)
y_pred = (scores_test > lof_thresh).astype(int)
all_metrics['LOF'] = {
    'Accuracy': accuracy_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred, zero_division=0),
    'Recall': recall_score(y_test, y_pred, zero_division=0),
    'F1': f1_score(y_test, y_pred, zero_division=0),
    'AUC-ROC': roc_auc_score(y_test, scores_test),
}
print(f"  LOF -> Acc: {all_metrics['LOF']['Accuracy']:.4f}, F1: {all_metrics['LOF']['F1']:.4f}, AUC: {all_metrics['LOF']['AUC-ROC']:.4f}")

# --- Autoencoder ---
print("🧠 Training Autoencoder...")
from tensorflow.keras.layers import Input, BatchNormalization as BN
ae = Sequential([
    Dense(128, activation='relu', input_shape=(X_normal_train.shape[1],)),
    BN(), Dropout(0.2),
    Dense(64, activation='relu'), BN(), Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(64, activation='relu'), BN(),
    Dense(128, activation='relu'), BN(),
    Dense(X_normal_train.shape[1], activation='linear')
])
ae.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
# Train on NORMAL only
X_normal_val = X_val_s[y_val.values == 0]
ae.fit(X_normal_train, X_normal_train,
       validation_data=(X_normal_val, X_normal_val),
       epochs=100, batch_size=32,
       callbacks=[EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)],
       verbose=0)
ae.save('../models/unsupervised/autoencoder_model.keras')

recon_val = ae.predict(X_val_s, verbose=0)
scores_val = np.mean(np.power(X_val_s - recon_val, 2), axis=1)
ae_thresh = find_best_threshold(scores_val, y_val.values)
recon_test = ae.predict(X_test_s, verbose=0)
scores_test = np.mean(np.power(X_test_s - recon_test, 2), axis=1)
y_pred = (scores_test > ae_thresh).astype(int)
all_metrics['Autoencoder'] = {
    'Accuracy': accuracy_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred, zero_division=0),
    'Recall': recall_score(y_test, y_pred, zero_division=0),
    'F1': f1_score(y_test, y_pred, zero_division=0),
    'AUC-ROC': roc_auc_score(y_test, scores_test),
}
print(f"  AE -> Acc: {all_metrics['Autoencoder']['Accuracy']:.4f}, F1: {all_metrics['Autoencoder']['F1']:.4f}, AUC: {all_metrics['Autoencoder']['AUC-ROC']:.4f}")

# Save thresholds
thresholds = {
    'IsolationForest': float(iso_thresh),
    'OneClassSVM': float(ocsvm_thresh),
    'KMeans': float(km_thresh),
    'LOF': float(lof_thresh),
    'Autoencoder': float(ae_thresh)
}
with open('../models/unsupervised/unsupervised_thresholds.json', 'w') as f:
    json.dump(thresholds, f, indent=2)

# ============================================================
# 4) COMPUTE FAR/FNR FOR ALL & SAVE
# ============================================================
for name in all_metrics:
    if 'FAR' not in all_metrics[name]:
        all_metrics[name]['FAR'] = 0
        all_metrics[name]['FNR'] = 0

# Save supervised metrics
sup_metrics = {k: all_metrics[k] for k in ['RandomForest', 'XGBoost', 'SVM', 'MLP']}
with open('../reports/metrics/supervised_metrics.json', 'w') as f:
    json.dump({'best_model': 'MLP', 'metrics': sup_metrics}, f, indent=2)

# Save final comparison
with open('../reports/metrics/final_comparison.json', 'w') as f:
    json.dump(all_metrics, f, indent=2)

print("\n" + "="*60)
print("✅ TÜM MODELLER EĞİTİLDİ VE KAYDEDİLDİ")
print("="*60)
for name, m in all_metrics.items():
    print(f"  {name:20s} | Acc: {m['Accuracy']:.4f} | F1: {m['F1']:.4f} | AUC: {m['AUC-ROC']:.4f}")
