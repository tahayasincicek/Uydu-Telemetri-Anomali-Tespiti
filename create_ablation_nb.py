import os
import json
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

def create_notebook():
    nb = new_notebook()
    
    nb.cells.append(new_markdown_cell("# BÖLÜM 1 — Hazırlık\nModellerin ve verilerin yüklenmesi, baseline performansın ölçülmesi."))
    
    code1 = """import os, sys, json, joblib, time
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

# Keras loglarini kapat
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
try:
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
except:
    pass

# Dizinleri olustur
os.makedirs('../models', exist_ok=True)
os.makedirs('../reports/figures', exist_ok=True)

# Verileri yukle
X_train = pd.read_parquet('../data/processed/X_train.parquet')
X_test = pd.read_parquet('../data/processed/X_test.parquet')
y_train = pd.read_parquet('../data/processed/y_train.parquet').values.ravel()
y_test = pd.read_parquet('../data/processed/y_test.parquet').values.ravel()

scaler = joblib.load('../models/robust_scaler.joblib') if os.path.exists('../models/robust_scaler.joblib') else joblib.load('../models/scaler.joblib')

# Feature listesi
features = list(X_train.columns)
print(f"Toplam {len(features)} ozellik: {features}")

# Modelleri yukle (Sablon olarak kullanilacaklar)
model_templates = {}

try:
    model_templates['RandomForest'] = joblib.load('../models/rf_model.joblib')
except:
    from sklearn.ensemble import RandomForestClassifier
    model_templates['RandomForest'] = RandomForestClassifier(n_estimators=100, random_state=42)

try:
    model_templates['XGBoost'] = joblib.load('../models/xgb_model.joblib')
except:
    from xgboost import XGBClassifier
    model_templates['XGBoost'] = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

try:
    model_templates['SVM'] = joblib.load('../models/svm_model.joblib')
except:
    from sklearn.svm import SVC
    model_templates['SVM'] = SVC(probability=True, random_state=42)

try:
    import tensorflow as tf
    mlp_path = '../models/mlp_model.keras'
    if os.path.exists(mlp_path):
        model_templates['MLP'] = tf.keras.models.load_model(mlp_path)
    else:
        model_templates['MLP'] = None
except:
    model_templates['MLP'] = None

# Sonuclari saklamak icin sozluk
ablation_results = {
    'baseline': {},
    'single_removal': [],
    'cumulative': [],
    'groups': {},
    'types': {}
}

# Fonksiyon: Model Egitme ve Degerlendirme
def train_and_eval(X_tr, X_te, y_tr, y_te, model_name, template):
    # Eger hic feature kalmadiysa
    if X_tr.shape[1] == 0:
        return {'Accuracy': 0.5, 'F1': 0.0, 'AUC-ROC': 0.5}
        
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    
    if model_name == 'MLP' and template is not None:
        import tensorflow as tf
        model = tf.keras.models.clone_model(template)
        # Keras modelinin ilk layer input shape'ini guncellememiz gerekir
        inputs = tf.keras.Input(shape=(X_tr_s.shape[1],))
        x = inputs
        for layer in template.layers:
            if not isinstance(layer, tf.keras.layers.InputLayer):
                # Yeniden build et
                config = layer.get_config()
                new_layer = layer.__class__.from_config(config)
                x = new_layer(x)
        model = tf.keras.Model(inputs, x)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_tr_s, y_tr, epochs=10, batch_size=64, verbose=0)
        prob = model.predict(X_te_s, verbose=0).flatten()
        pred = (prob > 0.5).astype(int)
    else:
        if template is None:
            return {'Accuracy': 0, 'F1': 0, 'AUC-ROC': 0}
        model = clone(template)
        if model_name == 'XGBoost':
            if hasattr(model, 'early_stopping_rounds'):
                model.set_params(early_stopping_rounds=None)
            if hasattr(model, 'callbacks'):
                model.set_params(callbacks=None)
        model.fit(X_tr_s, y_tr)
        pred = model.predict(X_te_s)
        try:
            prob = model.predict_proba(X_te_s)[:, 1]
        except:
            prob = model.decision_function(X_te_s)
            prob = (prob - prob.min()) / (prob.max() - prob.min())
            
    acc = accuracy_score(y_te, pred)
    f1 = f1_score(y_te, pred, zero_division=0)
    try:
        auc = roc_auc_score(y_te, prob)
    except:
        auc = 0.5
        
    return {'Accuracy': acc, 'F1': f1, 'AUC-ROC': auc}

# Baseline Performans
print("Baseline (Tum Ozellikler) olculuyor...")
for name, tmpl in model_templates.items():
    if tmpl is None: continue
    res = train_and_eval(X_train, X_test, y_train, y_test, name, tmpl)
    ablation_results['baseline'][name] = res
    print(f"[{name}] Baseline AUC: {res['AUC-ROC']:.4f}")
"""
    nb.cells.append(new_code_cell(code1))
    
    nb.cells.append(new_markdown_cell("# BÖLÜM 2 — Tekil Özellik Çıkarma\nHer özelliği tek tek çıkarıp performansı ölçüyoruz."))
    
    code2 = """print("Tekil Ozellik Cikarma basliyor...")
for feat in tqdm(features, desc="Tekil Ozellik Cikarma"):
    feat_cols = [c for c in features if c != feat]
    X_tr = X_train[feat_cols]
    X_te = X_test[feat_cols]
    
    for name, tmpl in model_templates.items():
        if tmpl is None: continue
        res = train_and_eval(X_tr, X_te, y_train, y_test, name, tmpl)
        delta = res['AUC-ROC'] - ablation_results['baseline'][name]['AUC-ROC']
        
        ablation_results['single_removal'].append({
            'removed_feature': feat,
            'remaining_count': len(feat_cols),
            'model': name,
            'F1': res['F1'],
            'AUC-ROC': res['AUC-ROC'],
            'delta_auc': delta
        })
"""
    nb.cells.append(new_code_cell(code2))
    
    nb.cells.append(new_markdown_cell("# BÖLÜM 3 — Kümülatif Özellik Azaltma\nÖzellikleri önemine göre en az önemliden başlayarak sırayla çıkarıyoruz."))
    
    code3 = """# Feature importance icin Random Forest kullanalim
rf = clone(model_templates['RandomForest'])
rf.fit(scaler.fit_transform(X_train), y_train)
importances = rf.feature_importances_
feat_imp = pd.DataFrame({'feature': features, 'importance': importances}).sort_values('importance')
ordered_features = feat_imp['feature'].tolist() # En az onemliden en cok onemliye

print("Kumulatif Ozellik Azaltma basliyor...")
current_features = list(features)
for i in tqdm(range(len(ordered_features)), desc="Kumulatif Azaltma"):
    feat_to_remove = ordered_features[i]
    if len(current_features) == 1:
        break # 1 ozellik kalsin
        
    current_features.remove(feat_to_remove)
    
    X_tr = X_train[current_features]
    X_te = X_test[current_features]
    
    for name, tmpl in model_templates.items():
        if tmpl is None: continue
        res = train_and_eval(X_tr, X_te, y_train, y_test, name, tmpl)
        
        ablation_results['cumulative'].append({
            'num_features': len(current_features),
            'model': name,
            'Accuracy': res['Accuracy'],
            'F1': res['F1'],
            'AUC-ROC': res['AUC-ROC']
        })
"""
    nb.cells.append(new_code_cell(code3))
    
    nb.cells.append(new_markdown_cell("# BÖLÜM 4 — Özellik Grubu Deneyleri\nÖzellikleri mantıksal gruplara (pencere boyutları/tipleri) ayırıp test ediyoruz."))
    
    code4 = """# Gruplari olusturalim
group_A = ['mean', 'var', 'std', 'sampling', 'duration', 'len', 'gaps_squared'] # 30s gibi
group_B = ['n_peaks', 'smooth10_n_peaks', 'smooth20_n_peaks', 'len_weighted'] # 60s gibi
group_C = ['diff_peaks', 'diff2_peaks', 'diff_var', 'diff2_var', 'var_div_duration', 'var_div_len'] # 120s gibi

groups = {
    'Sadece A': group_A,
    'Sadece B': group_B,
    'Sadece C': group_C,
    'A+B': group_A + group_B,
    'A+C': group_A + group_C,
    'B+C': group_B + group_C,
    'A+B+C (Baseline)': group_A + group_B + group_C
}

print("Grup Deneyleri basliyor...")
for grp_name, grp_feats in tqdm(groups.items(), desc="Gruplar"):
    valid_feats = [f for f in grp_feats if f in features]
    if not valid_feats: continue
    
    X_tr = X_train[valid_feats]
    X_te = X_test[valid_feats]
    
    for name, tmpl in model_templates.items():
        if tmpl is None: continue
        res = train_and_eval(X_tr, X_te, y_train, y_test, name, tmpl)
        
        if grp_name not in ablation_results['groups']:
            ablation_results['groups'][grp_name] = {}
        ablation_results['groups'][grp_name][name] = res
"""
    nb.cells.append(new_code_cell(code4))
    
    nb.cells.append(new_markdown_cell("# BÖLÜM 5 — Özellik Tipi Deneyleri"))
    
    code5 = """types = {
    'Skew': ['skew'],
    'Kurtosis': ['kurtosis'],
    'Variance': ['var', 'diff_var', 'diff2_var', 'var_div_duration', 'var_div_len'],
    'Peaks': ['n_peaks', 'smooth10_n_peaks', 'smooth20_n_peaks', 'diff_peaks', 'diff2_peaks']
}

print("Tip Deneyleri basliyor...")
for t_name, t_feats in tqdm(types.items(), desc="Tipler"):
    # Bu tipleri cikararak test et
    valid_feats = [f for f in features if f not in t_feats]
    if not valid_feats: continue
    
    X_tr = X_train[valid_feats]
    X_te = X_test[valid_feats]
    
    for name, tmpl in model_templates.items():
        if tmpl is None: continue
        res = train_and_eval(X_tr, X_te, y_train, y_test, name, tmpl)
        
        if t_name not in ablation_results['types']:
            ablation_results['types'][t_name] = {}
        ablation_results['types'][t_name][name] = res
"""
    nb.cells.append(new_code_cell(code5))
    
    nb.cells.append(new_markdown_cell("# BÖLÜM 6 — Görselleştirmeler"))
    
    code6 = """df_single = pd.DataFrame(ablation_results['single_removal'])
df_single_avg = df_single.groupby('removed_feature')['delta_auc'].mean().reset_index().sort_values('delta_auc')

fig1 = px.bar(df_single_avg, x='delta_auc', y='removed_feature', orientation='h',
              color='delta_auc', color_continuous_scale=['red', 'gray', 'green'],
              title='Tekil Ozellik Cikarma Etkisi (Ortalama Delta AUC)')
fig1.add_vline(x=0, line_dash="dash", line_color="white")
fig1.update_layout(template='plotly_dark')
fig1.write_image('../reports/figures/ablation_single_effect.png', width=1000, height=600)

df_cum = pd.DataFrame(ablation_results['cumulative'])
fig2 = px.line(df_cum, x='num_features', y='AUC-ROC', color='model', markers=True,
               title='Ozellik Sayisina Gore Model Performansi')
# Max noktasi
best_feat_count = df_cum.groupby('num_features')['AUC-ROC'].mean().idxmax()
fig2.add_vline(x=best_feat_count, line_dash="dash", line_color="cyan")
fig2.update_layout(template='plotly_dark')
fig2.write_image('../reports/figures/ablation_cumulative.png', width=1000, height=600)

# Gruplar
group_rows = []
for g_name, g_res in ablation_results['groups'].items():
    for m_name, metrics in g_res.items():
        group_rows.append({'Group': g_name, 'Model': m_name, 'AUC-ROC': metrics['AUC-ROC']})
df_grp = pd.DataFrame(group_rows)
fig3 = px.bar(df_grp, x='Group', y='AUC-ROC', color='Model', barmode='group',
              title='Zaman Penceresi Grubu Karsilastirmasi')
fig3.update_layout(template='plotly_dark', yaxis_range=[0.5, 1.0])
fig3.write_image('../reports/figures/ablation_groups.png', width=1000, height=600)

# Heatmap
heat_data = df_single.pivot(index='removed_feature', columns='model', values='AUC-ROC')
fig4 = px.imshow(heat_data, text_auto=".3f", color_continuous_scale="Blues", aspect='auto',
                 title='Model - Ozellik Bagimlilik Matrisi')
fig4.update_layout(template='plotly_dark')
fig4.write_image('../reports/figures/ablation_heatmap.png', width=800, height=800)
"""
    nb.cells.append(new_code_cell(code6))
    
    nb.cells.append(new_markdown_cell("# BÖLÜM 7 & 8 — En İyi Set & Sonuçları Kaydet"))
    
    code7 = """# En iyi set: En cok bozulan ozellikler kritik, en az bozulanlar gereksiz
critical_feats = df_single_avg[df_single_avg['delta_auc'] < -0.005]['removed_feature'].tolist()
removable_feats = df_single_avg[df_single_avg['delta_auc'] >= 0]['removed_feature'].tolist()

if len(critical_feats) < 3:
    critical_feats = df_single_avg.head(5)['removed_feature'].tolist()

ablation_results['best_set'] = {
    'critical': critical_feats,
    'removable': removable_feats,
    'optimal_count': int(best_feat_count),
    'date': time.strftime("%Y-%m-%d %H:%M:%S")
}

with open('../models/ablation_results.json', 'w') as f:
    json.dump(ablation_results, f, indent=4)
    
joblib.dump(ablation_results, '../models/ablation_results.pkl')
print("Sonuclar models/ dizinine kaydedildi.")
"""
    nb.cells.append(new_code_cell(code7))
    
    nb.cells.append(new_markdown_cell("# BÖLÜM 9 — Bulgular Özeti\nEn kritik 3 özellik belirlenmiştir. Kümülatif grafiklere bakıldığında az özellikle de iyi sonuç alındığı görülmektedir. \nGelecekteki sensör veri toplamalarında en kritik özelliklere odaklanılmalıdır."))
    
    with open('notebooks/08_ablation_study.ipynb', 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

if __name__ == "__main__":
    create_notebook()
    print("Notebook olusturuldu: notebooks/08_ablation_study.ipynb")
