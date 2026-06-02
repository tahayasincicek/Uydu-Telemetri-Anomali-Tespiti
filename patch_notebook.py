import json

with open('notebooks/07_shap_analizi.ipynb', 'r', encoding='utf-8') as f:
    d = json.load(f)

for c in d['cells']:
    if c['cell_type'] == 'code':
        source = c['source']
        
        # Insert KMP_DUPLICATE_LIB_OK to avoid OpenMP crash
        for i, line in enumerate(source):
            if 'import warnings' in line:
                source.insert(i, "os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'\n")
                break
        
        # Replace shap.kmeans with shap.sample to avoid sklearn KMeans crash
        for i in range(len(source)):
            source[i] = source[i].replace('shap.kmeans(X_test, 50)', 'shap.sample(X_test, 50)')

with open('notebooks/07_shap_analizi.ipynb', 'w', encoding='utf-8') as f:
    json.dump(d, f, indent=1)
