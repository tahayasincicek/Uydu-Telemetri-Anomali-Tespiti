"""UyduTAT — Uydu Telemetri Anomali Tespiti Dashboard"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json, os, sys, time, joblib

st.set_page_config(layout="wide", page_title="UyduTAT", page_icon="🛰️", initial_sidebar_state="expanded")

# CSS
css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

# --- Session State ---
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
    st.session_state.predictions = {}
    st.session_state.scores = {}
    st.session_state.uploaded_df = None

# --- Model Loading ---
@st.cache_resource
def load_models():
    models = {}
    model_dir = os.path.join(ROOT, "models")
    unsup_dir = os.path.join(model_dir, "unsupervised")
    for name, path in [("RandomForest","rf_model.joblib"),("XGBoost","xgb_model.joblib"),("SVM","svm_model.joblib")]:
        p = os.path.join(model_dir, path)
        if os.path.exists(p): models[name] = joblib.load(p)
    try:
        from tensorflow.keras.models import load_model
        p = os.path.join(model_dir, "mlp_model.keras")
        if os.path.exists(p): models["MLP"] = load_model(p)
        p = os.path.join(unsup_dir, "autoencoder_model.keras")
        if os.path.exists(p): models["Autoencoder"] = load_model(p)
    except: pass
    for name, path in [("IsolationForest","isolationforest_model.joblib"),("OneClassSVM","oneclasssvm_model.joblib"),("KMeans","kmeans_model.joblib"),("LOF","lof_model.joblib")]:
        p = os.path.join(unsup_dir, path)
        if os.path.exists(p): models[name] = joblib.load(p)
    thresholds = {}
    tp = os.path.join(unsup_dir, "unsupervised_thresholds.json")
    if os.path.exists(tp):
        with open(tp) as f: thresholds = json.load(f)
    scaler = None
    sp = os.path.join(model_dir, "scaler.joblib")
    if os.path.exists(sp): scaler = joblib.load(sp)
    return models, thresholds, scaler

@st.cache_data
def load_metrics():
    p = os.path.join(ROOT, "reports", "metrics", "final_comparison.json")
    if os.path.exists(p):
        with open(p) as f: return json.load(f)
    return {}

@st.cache_data
def load_demo_data():
    p = os.path.join(ROOT, "data", "features", "segment_features.parquet")
    if os.path.exists(p): return pd.read_parquet(p)
    return None

# --- KPI Helper ---
def kpi_card(icon, value, label):
    st.markdown(f"""<div class="kpi-card"><div class="kpi-icon">{icon}</div><div class="kpi-value">{value}</div><div class="kpi-label">{label}</div></div>""", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.markdown("## 🛰️ UyduTAT")
st.sidebar.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
page = st.sidebar.radio("Sayfa Seçin", ["🏠 Ana Sayfa", "📤 Veri Yükleme", "⚙️ Analiz", "📊 Sonuçlar", "📈 Model Performans", "ℹ️ Hakkında"])

models, thresholds, scaler = load_models()
all_metrics = load_metrics()

# ================================================================
# PAGE 1: ANA SAYFA
# ================================================================
if page == "🏠 Ana Sayfa":
    st.markdown("# 🛰️ Uydu Telemetri Anomali Tespiti Sistemi")
    st.markdown("##### ESA OPS-SAT Reaction Wheel Telemetri Analiz Platformu")
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    with c1: kpi_card("📡", f"{len(models)}", "Yüklü Model")
    with c2: kpi_card("🎯", "96.2%", "En İyi Doğruluk")
    with c3: kpi_card("🧠", "0.990", "En İyi AUC")
    with c4: kpi_card("⚡", "1.8%", "Min Yanlış Alarm")

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2,1])
    with col1:
        st.markdown("""<div class="welcome-card">
        <h3>🚀 Hoşgeldiniz!</h3>
        <p>Bu platform, ESA OPS-SAT uydusunun Reaction Wheel telemetri verilerinde anomali tespiti yapmak için geliştirilmiş yapay zeka destekli bir analiz sistemidir.</p>
        <h4>📋 Kullanım Adımları:</h4>
        <ol>
        <li><b>📤 Veri Yükleme:</b> CSV formatında telemetri verisi yükleyin veya demo veriyi kullanın</li>
        <li><b>⚙️ Analiz:</b> Modelleri seçin ve anomali tespitini başlatın</li>
        <li><b>📊 Sonuçlar:</b> Tespit edilen anomalileri görselleştirin ve raporlayın</li>
        </ol></div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("### 🤖 Model Durumları")
        for name in models:
            cat = "Gözetimli" if name in ["RandomForest","XGBoost","SVM","MLP"] else "Gözetimsiz"
            m = all_metrics.get(name, {})
            auc_val = m.get("AUC-ROC", 0)
            color = "🟢" if auc_val > 0.9 else ("🟡" if auc_val > 0.7 else "🔴")
            st.markdown(f"{color} **{name}** — AUC: {auc_val:.3f}")

# ================================================================
# PAGE 2: VERİ YÜKLEME
# ================================================================
elif page == "📤 Veri Yükleme":
    st.markdown("# 📤 Veri Yükleme & Ön İzleme")
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📁 Dosya Yükle", "🎮 Demo Veri"])

    with tab1:
        uploaded = st.file_uploader("CSV veya Parquet dosyası yükleyin", type=["csv","parquet"])
        if uploaded:
            try:
                if uploaded.name.endswith(".parquet"):
                    df = pd.read_parquet(uploaded)
                else:
                    df = pd.read_csv(uploaded)
                st.session_state.uploaded_df = df
                st.success(f"✅ {uploaded.name} başarıyla yüklendi! ({df.shape[0]:,} satır × {df.shape[1]} sütun)")
            except Exception as e:
                st.error(f"❌ Dosya okunamadı: {e}")

    with tab2:
        if st.button("🎮 Demo Veriyi Yükle"):
            demo = load_demo_data()
            if demo is not None:
                st.session_state.uploaded_df = demo
                st.success(f"✅ Demo veri yüklendi! ({demo.shape[0]:,} satır × {demo.shape[1]} sütun)")
            else:
                st.error("Demo veri bulunamadı.")

    df = st.session_state.uploaded_df
    if df is not None:
        st.markdown("### 📊 Veri Kalite Raporu")
        q1,q2,q3,q4 = st.columns(4)
        q1.metric("Satır Sayısı", f"{df.shape[0]:,}")
        q2.metric("Sütun Sayısı", f"{df.shape[1]}")
        q3.metric("Eksik Değer", f"{df.isnull().sum().sum()}")
        q4.metric("Bellek", f"{df.memory_usage(deep=True).sum()/1024/1024:.1f} MB")

        st.markdown("### 📋 Veri Önizleme")
        st.dataframe(df.head(100), use_container_width=True, height=350)

        st.markdown("### 📈 Temel İstatistikler")
        st.dataframe(df.describe(), use_container_width=True)

        numeric_cols = df.select_dtypes(include=[np.number]).columns[:6]
        if len(numeric_cols) > 0:
            st.markdown("### 📉 Dağılım Grafikleri")
            fig = make_subplots(rows=2, cols=3, subplot_titles=[str(c) for c in numeric_cols])
            for i, col in enumerate(numeric_cols):
                r, c = divmod(i, 3)
                fig.add_trace(go.Histogram(x=df[col], name=col, marker_color="#00D4FF", opacity=0.7), row=r+1, col=c+1)
            fig.update_layout(template="plotly_dark", height=450, showlegend=False, paper_bgcolor="#0D1117", plot_bgcolor="#161B22")
            st.plotly_chart(fig, use_container_width=True)

# ================================================================
# PAGE 3: ANALİZ
# ================================================================
elif page == "⚙️ Analiz":
    st.markdown("# ⚙️ Anomali Tespiti Analizi")
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    if st.session_state.uploaded_df is None:
        st.warning("⚠️ Önce '📤 Veri Yükleme' sayfasından veri yükleyin veya demo veriyi kullanın.")
        if st.button("🎮 Hızlı Demo Yükle"):
            demo = load_demo_data()
            if demo is not None: st.session_state.uploaded_df = demo; st.rerun()
        st.stop()

    df = st.session_state.uploaded_df
    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.markdown("### 🤖 Model Seçimi")
        sup_models = {n: st.checkbox(n, value=True) for n in ["RandomForest","XGBoost","SVM","MLP"] if n in models}
        unsup_models = {n: st.checkbox(n, value=(n in ["Autoencoder","LOF"])) for n in ["IsolationForest","OneClassSVM","KMeans","LOF","Autoencoder"] if n in models}
        selected = [n for n,v in {**sup_models, **unsup_models}.items() if v]

        with st.expander("⚙️ Gelişmiş Ayarlar"):
            threshold_adj = st.slider("Threshold Çarpanı", 0.5, 1.5, 1.0, 0.05)

    with col_right:
        st.markdown("### 🚀 Analiz Başlat")
        st.info(f"**{len(selected)} model** seçildi. Veri boyutu: {df.shape[0]:,} satır")

        if st.button("🚀 Anomali Tespiti Başlat", type="primary", use_container_width=True):
            if not selected:
                st.error("En az bir model seçin!")
            else:
                drop_cols = ['segment','anomaly','train','channel']
                feature_cols = [c for c in df.columns if c not in drop_cols]
                X = df[feature_cols].fillna(0).values

                if scaler is not None:
                    try: X_scaled = scaler.transform(X)
                    except: X_scaled = X
                else:
                    X_scaled = X

                progress = st.progress(0)
                status = st.empty()
                preds = {}
                scores = {}

                for i, name in enumerate(selected):
                    status.markdown(f"🔄 **{name}** analiz ediliyor...")
                    model = models[name]
                    try:
                        if name == "MLP":
                            sc = model.predict(X_scaled, verbose=0).flatten()
                            pr = (sc >= 0.5).astype(int)
                        elif name == "Autoencoder":
                            recon = model.predict(X_scaled, verbose=0)
                            sc = np.mean(np.power(X_scaled - recon, 2), axis=1)
                            t = thresholds.get("Autoencoder", np.percentile(sc, 90)) * threshold_adj
                            pr = (sc > t).astype(int)
                        elif name in ["IsolationForest","LOF"]:
                            sc = -model.score_samples(X_scaled)
                            t = thresholds.get(name, np.percentile(sc, 90)) * threshold_adj
                            pr = (sc > t).astype(int)
                        elif name == "OneClassSVM":
                            sc = -model.decision_function(X_scaled)
                            t = thresholds.get(name, np.percentile(sc, 90)) * threshold_adj
                            pr = (sc > t).astype(int)
                        elif name == "KMeans":
                            sc = np.min(model.transform(X_scaled), axis=1)
                            t = thresholds.get(name, np.percentile(sc, 90)) * threshold_adj
                            pr = (sc > t).astype(int)
                        else:
                            pr = model.predict(X_scaled)
                            sc = model.predict_proba(X_scaled)[:,1] if hasattr(model,"predict_proba") else pr.astype(float)

                        preds[name] = pr
                        scores[name] = sc
                    except Exception as e:
                        st.warning(f"⚠️ {name} hatası: {e}")
                    progress.progress((i+1)/len(selected))

                st.session_state.predictions = preds
                st.session_state.scores = scores
                st.session_state.analysis_done = True
                status.empty()
                progress.empty()

                total_anomalies = sum(1 for pr in preds.values() for v in pr if v == 1) // max(len(preds),1)
                st.success(f"✅ Analiz tamamlandı! {len(preds)} model çalıştırıldı. ~{total_anomalies} anomali bulundu.")

# ================================================================
# PAGE 4: SONUÇLAR
# ================================================================
elif page == "📊 Sonuçlar":
    st.markdown("# 📊 Sonuçlar & Görselleştirme")
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    if not st.session_state.analysis_done:
        st.warning("⚠️ Önce '⚙️ Analiz' sayfasından anomali tespiti çalıştırın.")
        st.stop()

    df = st.session_state.uploaded_df
    preds = st.session_state.predictions
    scores = st.session_state.scores

    # Anomali skoru overlay
    st.markdown("### 📉 Model Anomali Skorları")
    fig = go.Figure()
    colors = ["#00D4FF","#F85149","#3FB950","#D29922","#A371F7","#F778A1","#79C0FF","#FFA657","#FF7B72"]
    for i, (name, sc) in enumerate(scores.items()):
        sc_norm = (sc - sc.min()) / (sc.max() - sc.min() + 1e-10)
        fig.add_trace(go.Scatter(y=sc_norm, mode="lines", name=name, line=dict(color=colors[i % len(colors)], width=1.5), opacity=0.8))

    if "anomaly" in df.columns:
        anom_idx = df[df["anomaly"]==1].index
        for idx in anom_idx[:50]:
            fig.add_vline(x=idx, line_width=0.5, line_color="red", opacity=0.15)

    fig.update_layout(template="plotly_dark", height=400, paper_bgcolor="#0D1117", plot_bgcolor="#161B22",
                      title="Normalize Anomali Skorları (Tüm Modeller)", xaxis_title="Segment", yaxis_title="Skor (0-1)")
    st.plotly_chart(fig, use_container_width=True)

    # Confusion heatmap if labels exist
    if "anomaly" in df.columns:
        st.markdown("### 🎯 Model Karşılaştırma")
        y_true = df["anomaly"].values
        comp_data = []
        for name, pr in preds.items():
            from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
            comp_data.append({"Model": name, "Accuracy": accuracy_score(y_true, pr),
                              "Precision": precision_score(y_true, pr, zero_division=0),
                              "Recall": recall_score(y_true, pr, zero_division=0),
                              "F1": f1_score(y_true, pr, zero_division=0)})
        comp_df = pd.DataFrame(comp_data).set_index("Model")
        st.dataframe(comp_df.style.background_gradient(cmap="YlGnBu").format("{:.4f}"), use_container_width=True)

    # Anomaly list
    st.markdown("### 🔔 Tespit Edilen Anomaliler")
    ensemble = np.zeros(len(df))
    for pr in preds.values():
        ensemble += pr
    ensemble = ensemble / max(len(preds),1)

    anomaly_mask = ensemble > 0.5
    anomaly_count = anomaly_mask.sum()
    st.metric("Toplam Anomali", f"{int(anomaly_count)}", f"{anomaly_count/len(df)*100:.1f}% oran")

    if anomaly_count > 0:
        anom_df = df[anomaly_mask].copy()
        anom_df["Ensemble_Skor"] = ensemble[anomaly_mask]
        severity = np.where(anom_df["Ensemble_Skor"]>0.8, "🔴 KRİTİK", np.where(anom_df["Ensemble_Skor"]>0.5, "🟡 UYARI", "🔵 BİLGİ"))
        anom_df["Şiddet"] = severity
        st.dataframe(anom_df.head(50), use_container_width=True, height=350)

# ================================================================
# PAGE 5: MODEL PERFORMANS
# ================================================================
elif page == "📈 Model Performans":
    st.markdown("# 📈 Model Performans İzleme")
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    if not all_metrics:
        st.warning("Metrik dosyası bulunamadı.")
        st.stop()

    df_m = pd.DataFrame(all_metrics).T
    perf_cols = [c for c in ["Accuracy","Precision","Recall","F1","AUC-ROC","FAR"] if c in df_m.columns]

    st.markdown("### 📋 Metrik Tablosu")
    st.dataframe(df_m[perf_cols].style.background_gradient(cmap="YlGnBu").format("{:.4f}"), use_container_width=True)

    # Heatmap
    fig = px.imshow(df_m[perf_cols], text_auto=".3f", color_continuous_scale="YlGnBu",
                    title="Model Performans Isı Haritası", labels=dict(color="Skor"))
    fig.update_layout(template="plotly_dark", height=400, paper_bgcolor="#0D1117")
    st.plotly_chart(fig, use_container_width=True)

    # ROC from saved data
    test_path = os.path.join(ROOT, "models", "test_data.joblib")
    if os.path.exists(test_path):
        st.markdown("### 📈 ROC Eğrileri")
        test_data = joblib.load(test_path)
        X_test, y_test = test_data["X_test"], test_data["y_test"]

        fig = go.Figure()
        from sklearn.metrics import roc_curve, auc
        for name, model in models.items():
            try:
                if name == "MLP": prob = model.predict(X_test, verbose=0).flatten()
                elif name == "Autoencoder": r = model.predict(X_test, verbose=0); prob = np.mean(np.power(X_test-r,2), axis=1)
                elif name in ["IsolationForest","LOF"]: prob = -model.score_samples(X_test)
                elif name == "OneClassSVM": prob = -model.decision_function(X_test)
                elif name == "KMeans": prob = np.min(model.transform(X_test), axis=1)
                else: prob = model.predict_proba(X_test)[:,1]
                fpr, tpr, _ = roc_curve(y_test, prob)
                roc_auc = auc(fpr, tpr)
                fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} ({roc_auc:.3f})"))
            except: pass
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", line=dict(dash="dash", color="gray"), name="Rastgele"))
        fig.update_layout(template="plotly_dark", height=500, paper_bgcolor="#0D1117", plot_bgcolor="#161B22",
                          xaxis_title="FPR", yaxis_title="TPR", title="ROC Eğrileri")
        st.plotly_chart(fig, use_container_width=True)

    # Radar chart
    st.markdown("### 🎯 Radar Karşılaştırması")
    top_models = ["MLP","XGBoost","RandomForest","Autoencoder","LOF"]
    radar_models = [m for m in top_models if m in all_metrics]
    if radar_models:
        categories = ["Accuracy","Precision","Recall","F1","AUC-ROC"]
        fig = go.Figure()
        for name in radar_models:
            vals = [all_metrics[name].get(c, 0) for c in categories]
            fig.add_trace(go.Scatterpolar(r=vals + [vals[0]], theta=categories + [categories[0]], name=name, fill="toself", opacity=0.6))
        fig.update_layout(template="plotly_dark", height=450, paper_bgcolor="#0D1117",
                          polar=dict(bgcolor="#161B22", radialaxis=dict(range=[0,1])))
        st.plotly_chart(fig, use_container_width=True)

# ================================================================
# PAGE 6: HAKKINDA
# ================================================================
elif page == "ℹ️ Hakkında":
    st.markdown("# ℹ️ Sistem Bilgisi & Yardım")
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    st.markdown("""<div class="welcome-card">
    <h3>🛰️ UyduTAT — Uydu Telemetri Anomali Tespiti</h3>
    <p>ESA OPS-SAT uydu misyonu Reaction Wheel telemetri verileri üzerinde makine öğrenmesi ve derin öğrenme tabanlı anomali tespit sistemi.</p>
    <p><b>Geliştirici:</b> Taha Yasin Çiçek</p>
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🤖 Yüklü Modeller")
        for name in models:
            cat = "Gözetimli" if name in ["RandomForest","XGBoost","SVM","MLP"] else "Gözetimsiz"
            st.markdown(f"- **{name}** ({cat})")

    with col2:
        st.markdown("### 📚 Anomali Tipleri Sözlüğü")
        st.markdown("""
        - **Spike:** Ani değer sıçraması (tek nokta)
        - **Drift:** Yavaş kayma (uzun süreli trend değişimi)
        - **Step Change:** Ani seviye değişikliği
        - **Sensor Failure:** Sabit değer veya NaN serisi
        - **Oscillation:** Anormal titreşim deseni
        """)

    st.markdown("### ❓ Sıkça Sorulan Sorular")
    with st.expander("Hangi model en iyisidir?"):
        st.write("Gözetimli: MLP (%96.2 Accuracy, 0.990 AUC). Gözetimsiz: Autoencoder (0.950 AUC).")
    with st.expander("Yeni veri ile nasıl analiz yapılır?"):
        st.write("1) Veri Yükleme sayfasından CSV yükleyin. 2) Analiz sayfasından modelleri seçip çalıştırın. 3) Sonuçlar sayfasından inceleyin.")
    with st.expander("Threshold nedir?"):
        st.write("Modelin normal/anomali kararını verdiği eşik değeridir. Düşük threshold = daha fazla anomali (daha hassas), yüksek threshold = daha az anomali (daha az yanlış alarm).")
