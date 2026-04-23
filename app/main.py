"""
Uydu Telemetri Anomali Tespiti - Streamlit Dashboard
=====================================================
Eğitilmiş modelleri kullanarak interaktif anomali tespiti arayüzü.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

# Proje kök dizinini path'e ekle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ─── Sayfa Yapılandırması ───
st.set_page_config(
    page_title="🛰️ Uydu Telemetri Anomali Tespiti",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Başlık ───
st.title("🛰️ Uydu Telemetri Anomali Tespiti")
st.markdown("*Makine öğrenmesi ile uydu telemetri verilerinde anomali tespiti*")
st.markdown("---")

# ─── Sidebar ───
with st.sidebar:
    st.header("⚙️ Ayarlar")
    
    uploaded_file = st.file_uploader(
        "Telemetri Verisi Yükle", type=["csv", "parquet"]
    )
    
    st.markdown("---")
    
    model_type = st.selectbox(
        "Model Seçimi",
        ["Random Forest", "Isolation Forest", "One-Class SVM", "Autoencoder"],
    )
    
    contamination = st.slider(
        "Anomali Oranı (%)", min_value=1, max_value=20, value=5
    )
    
    st.markdown("---")
    st.markdown("**Geliştirici:** Taha Yasin Çiçek")

# ─── Ana İçerik ───
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("📡 Toplam Veri Noktası", "—")
with col2:
    st.metric("🔴 Tespit Edilen Anomali", "—")
with col3:
    st.metric("✅ Normal Veri", "—")
with col4:
    st.metric("📊 Anomali Oranı", "—%")

st.markdown("---")

if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_parquet(uploaded_file)

    st.subheader("📋 Veri Önizleme")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("📈 Zaman Serisi Görselleştirme")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        selected_col = st.selectbox("Sütun Seçin", numeric_cols)
        fig = px.line(df, y=selected_col, title=f"{selected_col} Zaman Serisi")
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 İstatistiksel Özet")
    st.dataframe(df.describe(), use_container_width=True)
else:
    st.info("👈 Lütfen sol panelden bir telemetri veri dosyası yükleyin.")

# ─── Footer ───
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>"
    "🛰️ Uydu Telemetri Anomali Tespiti Dashboard v0.1.0"
    "</p>",
    unsafe_allow_html=True,
)
