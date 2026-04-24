"""
Tüm modelleri tek bir tutarlı pipeline'da eğitir.
StandardScaler + aynı split + aynı test seti.
"""
import json, os, joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import SVC, OneClassSVM
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

os.makedirs('../models/unsupervised', exist_ok=True)
os.makedirs('../reports/metrics', exist_ok=True)

# ============================================================
# DATA
# ============================================================
df = pd.read_parquet('../data/features/segment_features.parquet')
drop_cols = ['segment', 'anomaly', 'train', 'channel']
feature_cols = [c for c in df.columns if c not in drop_cols]
X = df[feature_cols].fillna(0)
y = df['anomaly']

X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.15, random_state=42, stratify=y_trainval)

sm = SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train_sm)
X_train_orig_s = scaler.transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

normal_mask = (y_train == 0).values
X_normal_train = X_train_orig_s[normal_mask]

joblib.dump(scaler, '../models/scaler.joblib')
joblib.dump({'X_test': X_test_s, 'y_test': y_test.values, 'feature_cols': feature_cols}, '../models/test_data.joblib')

print(f"Train: {X_train_s.shape}, Val: {X_val_s.shape}, Test: {X_test_s.shape}")

def metrics(y_true, y_pred, y_scores=None):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    far = fp/(fp+tn) if (fp+tn)>0 else 0
    fnr = fn/(fn+tp) if (fn+tp)>0 else 0
    auc_val = 0.5
    if y_scores is not None:
        try: auc_val = roc_auc_score(y_true, y_scores)
        except: pass
    return {'Accuracy':acc,'Precision':prec,'Recall':rec,'F1':f1,'AUC-ROC':auc_val,'FAR':far,'FNR':fnr}

def find_thresh(scores_val, y_val_true):
    best_f1, best_t = 0, np.percentile(scores_val, 90)
    for p in np.arange(40, 99.5, 0.5):
        t = np.percentile(scores_val, p)
        preds = (scores_val > t).astype(int)
        f = f1_score(y_val_true, preds, zero_division=0)
        if f > best_f1: best_f1, best_t = f, t
    return best_t

all_m = {}

# === SUPERVISED ===
print("\n--- SUPERVISED ---")
rf = RandomForestClassifier(n_estimators=500, max_depth=None, class_weight='balanced', random_state=42, n_jobs=-1)
rf.fit(X_train_s, y_train_sm); joblib.dump(rf, '../models/rf_model.joblib')
p = rf.predict(X_test_s); pr = rf.predict_proba(X_test_s)[:,1]
all_m['RandomForest'] = metrics(y_test, p, pr)
print(f"  RF  -> Acc:{all_m['RandomForest']['Accuracy']:.3f} F1:{all_m['RandomForest']['F1']:.3f} AUC:{all_m['RandomForest']['AUC-ROC']:.3f}")

xg = xgb.XGBClassifier(n_estimators=500, max_depth=8, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, eval_metric='auc', early_stopping_rounds=30)
xg.fit(X_train_s, y_train_sm, eval_set=[(X_val_s, y_val)], verbose=False); joblib.dump(xg, '../models/xgb_model.joblib')
p = xg.predict(X_test_s); pr = xg.predict_proba(X_test_s)[:,1]
all_m['XGBoost'] = metrics(y_test, p, pr)
print(f"  XGB -> Acc:{all_m['XGBoost']['Accuracy']:.3f} F1:{all_m['XGBoost']['F1']:.3f} AUC:{all_m['XGBoost']['AUC-ROC']:.3f}")

svm = CalibratedClassifierCV(SVC(kernel='rbf', C=10, gamma='scale', class_weight='balanced', random_state=42), cv=3)
svm.fit(X_train_s, y_train_sm); joblib.dump(svm, '../models/svm_model.joblib')
p = svm.predict(X_test_s); pr = svm.predict_proba(X_test_s)[:,1]
all_m['SVM'] = metrics(y_test, p, pr)
print(f"  SVM -> Acc:{all_m['SVM']['Accuracy']:.3f} F1:{all_m['SVM']['F1']:.3f} AUC:{all_m['SVM']['AUC-ROC']:.3f}")

print("  MLP eğitiliyor...")
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
mlp = Sequential([Dense(256,activation='relu',input_shape=(X_train_s.shape[1],)),BatchNormalization(),Dropout(0.3),Dense(128,activation='relu'),BatchNormalization(),Dropout(0.2),Dense(64,activation='relu'),Dropout(0.2),Dense(32,activation='relu'),Dense(1,activation='sigmoid')])
mlp.compile(optimizer=Adam(learning_rate=0.001),loss='binary_crossentropy',metrics=['accuracy'])
mlp.fit(X_train_s,y_train_sm.values,validation_data=(X_val_s,y_val.values),epochs=80,batch_size=32,callbacks=[EarlyStopping(monitor='val_loss',patience=15,restore_best_weights=True)],verbose=0)
mlp.save('../models/mlp_model.keras')
pr = mlp.predict(X_test_s,verbose=0).flatten(); p = (pr>=0.5).astype(int)
all_m['MLP'] = metrics(y_test, p, pr)
print(f"  MLP -> Acc:{all_m['MLP']['Accuracy']:.3f} F1:{all_m['MLP']['F1']:.3f} AUC:{all_m['MLP']['AUC-ROC']:.3f}")

sup_out = {'best_model':'MLP','metrics':{}}
for n in ['RandomForest','XGBoost','SVM','MLP']:
    sup_out['metrics'][n] = {k.replace('AUC-ROC','AUC').replace('F1','F1_Score'):v for k,v in all_m[n].items()}
with open('../reports/metrics/supervised_metrics.json','w') as f: json.dump(sup_out,f,indent=2)

# === UNSUPERVISED ===
print("\n--- UNSUPERVISED ---")

best_iso = {'f1':0}
for ne in [100,200,300]:
    for mf in [0.5,0.75,1.0]:
        for co in [0.05,0.10,0.15,0.20]:
            m = IsolationForest(n_estimators=ne,max_features=mf,contamination=co,random_state=42,n_jobs=-1)
            m.fit(X_normal_train)
            sv = -m.score_samples(X_val_s); t = find_thresh(sv,y_val.values)
            pv = (sv>t).astype(int); f = f1_score(y_val,pv,zero_division=0)
            if f > best_iso['f1']: best_iso = {'f1':f,'model':m,'thresh':t}
joblib.dump(best_iso['model'],'../models/unsupervised/isolationforest_model.joblib')
st = -best_iso['model'].score_samples(X_test_s); p = (st>best_iso['thresh']).astype(int)
all_m['IsolationForest'] = metrics(y_test,p,st)
print(f"  IF  -> Acc:{all_m['IsolationForest']['Accuracy']:.3f} F1:{all_m['IsolationForest']['F1']:.3f} AUC:{all_m['IsolationForest']['AUC-ROC']:.3f}")

best_oc = {'f1':0}
for nu in [0.05,0.10,0.15,0.20]:
    for ga in ['scale','auto',0.001,0.01,0.1]:
        m = OneClassSVM(kernel='rbf',gamma=ga,nu=nu); m.fit(X_normal_train)
        sv = -m.decision_function(X_val_s); t = find_thresh(sv,y_val.values)
        pv = (sv>t).astype(int); f = f1_score(y_val,pv,zero_division=0)
        if f > best_oc['f1']: best_oc = {'f1':f,'model':m,'thresh':t}
joblib.dump(best_oc['model'],'../models/unsupervised/oneclasssvm_model.joblib')
st = -best_oc['model'].decision_function(X_test_s); p = (st>best_oc['thresh']).astype(int)
all_m['OneClassSVM'] = metrics(y_test,p,st)
print(f"  OCS -> Acc:{all_m['OneClassSVM']['Accuracy']:.3f} F1:{all_m['OneClassSVM']['F1']:.3f} AUC:{all_m['OneClassSVM']['AUC-ROC']:.3f}")

best_km = {'f1':0}
for nc in [2,3,4,5,7,10]:
    m = KMeans(n_clusters=nc,random_state=42,n_init='auto'); m.fit(X_normal_train)
    sv = np.min(m.transform(X_val_s),axis=1); t = find_thresh(sv,y_val.values)
    pv = (sv>t).astype(int); f = f1_score(y_val,pv,zero_division=0)
    if f > best_km['f1']: best_km = {'f1':f,'model':m,'thresh':t}
joblib.dump(best_km['model'],'../models/unsupervised/kmeans_model.joblib')
st = np.min(best_km['model'].transform(X_test_s),axis=1); p = (st>best_km['thresh']).astype(int)
all_m['KMeans'] = metrics(y_test,p,st)
print(f"  KM  -> Acc:{all_m['KMeans']['Accuracy']:.3f} F1:{all_m['KMeans']['F1']:.3f} AUC:{all_m['KMeans']['AUC-ROC']:.3f}")

best_lof = {'f1':0}
for nn in [5,10,15,20,30,50]:
    m = LocalOutlierFactor(n_neighbors=nn,novelty=True,contamination=float(y_train.mean())); m.fit(X_normal_train)
    sv = -m.score_samples(X_val_s); t = find_thresh(sv,y_val.values)
    pv = (sv>t).astype(int); f = f1_score(y_val,pv,zero_division=0)
    if f > best_lof['f1']: best_lof = {'f1':f,'model':m,'thresh':t}
joblib.dump(best_lof['model'],'../models/unsupervised/lof_model.joblib')
st = -best_lof['model'].score_samples(X_test_s); p = (st>best_lof['thresh']).astype(int)
all_m['LOF'] = metrics(y_test,p,st)
print(f"  LOF -> Acc:{all_m['LOF']['Accuracy']:.3f} F1:{all_m['LOF']['F1']:.3f} AUC:{all_m['LOF']['AUC-ROC']:.3f}")

X_normal_val = X_val_s[y_val.values==0]
best_ae = {'f1':0}
for ld in [8,16,32]:
    for lr in [0.001,0.0005]:
        ae = Sequential([Dense(64,activation='relu',input_shape=(X_normal_train.shape[1],)),BatchNormalization(),Dropout(0.2),Dense(32,activation='relu'),BatchNormalization(),Dense(ld,activation='relu'),Dense(32,activation='relu'),BatchNormalization(),Dense(64,activation='relu'),BatchNormalization(),Dense(X_normal_train.shape[1],activation='linear')])
        ae.compile(optimizer=Adam(learning_rate=lr),loss='mse')
        ae.fit(X_normal_train,X_normal_train,validation_data=(X_normal_val,X_normal_val),epochs=150,batch_size=32,callbacks=[EarlyStopping(monitor='val_loss',patience=15,restore_best_weights=True)],verbose=0)
        rv = ae.predict(X_val_s,verbose=0); sv = np.mean(np.power(X_val_s-rv,2),axis=1)
        t = find_thresh(sv,y_val.values); pv = (sv>t).astype(int); f = f1_score(y_val,pv,zero_division=0)
        if f > best_ae['f1']: best_ae = {'f1':f,'model':ae,'thresh':t}
best_ae['model'].save('../models/unsupervised/autoencoder_model.keras')
rt = best_ae['model'].predict(X_test_s,verbose=0); st = np.mean(np.power(X_test_s-rt,2),axis=1)
p = (st>best_ae['thresh']).astype(int)
all_m['Autoencoder'] = metrics(y_test,p,st)
print(f"  AE  -> Acc:{all_m['Autoencoder']['Accuracy']:.3f} F1:{all_m['Autoencoder']['F1']:.3f} AUC:{all_m['Autoencoder']['AUC-ROC']:.3f}")

thresholds = {'IsolationForest':float(best_iso['thresh']),'OneClassSVM':float(best_oc['thresh']),'KMeans':float(best_km['thresh']),'LOF':float(best_lof['thresh']),'Autoencoder':float(best_ae['thresh'])}
with open('../models/unsupervised/unsupervised_thresholds.json','w') as f: json.dump(thresholds,f,indent=2)
with open('../reports/metrics/final_comparison.json','w') as f: json.dump(all_m,f,indent=2)

print("\n" + "="*70)
print("✅ TÜM MODELLER TUTARLI ŞEKİLDE EĞİTİLDİ")
print("="*70)
for n,m in all_m.items():
    print(f"  {n:20s} | Acc:{m['Accuracy']:.3f} Prec:{m['Precision']:.3f} F1:{m['F1']:.3f} AUC:{m['AUC-ROC']:.3f} FAR:{m['FAR']:.3f}")
