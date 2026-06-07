import json

with open('notebooks/07_shap_analizi.ipynb', 'r', encoding='utf-8') as f:
    d = json.load(f)

for c in d['cells']:
    if c['cell_type'] == 'code':
        source = c['source']
        
        source = [line for line in source if "os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'" not in line]
        
        for i in range(len(source)):
            source[i] = source[i].replace('shap.sample(X_test, 50)', 'shap.kmeans(X_test, 50)')
            
        c['source'] = source

with open('notebooks/07_shap_analizi.ipynb', 'w', encoding='utf-8') as f:
    json.dump(d, f, indent=1)
