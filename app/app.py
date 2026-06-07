"""Uydu Telemetri Anomali Tespit Arayüzü"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import sys, json, time, base64, io, datetime
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dash import Dash, html, dcc, dash_table, callback, Input, Output, State, no_update, ctx
import dash_bootstrap_components as dbc
from dash_iconify import DashIconify

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))
from utils.model_loader import load_all, predict, load_metrics
from utils.feature_extractor import extract_features_from_raw
from utils.ui import PLT_LAYOUT, icon, metric_card
import joblib

MODELS, THRESHOLDS, SCALER, TEST_DATA = load_all()
ALL_METRICS = load_metrics()
DEMO_PATH = os.path.join(ROOT, "data", "features", "segment_features.parquet")

LIVE_DATA_PATH = os.path.join(ROOT, "data", "raw", "segments.csv")
try:
    LIVE_DATA = pd.read_csv(LIVE_DATA_PATH)
except Exception:
    LIVE_DATA = pd.DataFrame()


SHAP_PKL = os.path.join(ROOT, "models", "shap_values.pkl")
try:
    if os.path.exists(SHAP_PKL):
        SHAP_DATA = joblib.load(SHAP_PKL)
    else:
        SHAP_DATA = None
except Exception:
    SHAP_DATA = None

FEATURE_NAMES_TR = {
    'sampling': 'Ornekleme Frekansi',
    'duration': 'Segment Suresi',
    'len': 'Segment Uzunlugu',
    'mean': 'Ortalama Deger',
    'var': 'Varyans',
    'std': 'Standart Sapma',
    'kurtosis': 'Basiklik (Kurtosis)',
    'skew': 'Carpiklik (Skewness)',
    'n_peaks': 'Tepe Sayisi',
    'smooth10_n_peaks': 'Yumusatilmis Tepe (w=10)',
    'smooth20_n_peaks': 'Yumusatilmis Tepe (w=20)',
    'diff_peaks': 'Fark Tepe Sayisi',
    'diff2_peaks': '2. Fark Tepe Sayisi',
    'diff_var': 'Fark Varyansi',
    'diff2_var': '2. Fark Varyansi',
    'gaps_squared': 'Bosluk Karesi',
    'len_weighted': 'Agirlikli Uzunluk',
    'var_div_duration': 'Varyans/Sure',
    'var_div_len': 'Varyans/Uzunluk',
}

DROP_COLS = ['segment', 'anomaly', 'train', 'channel']
FEATURE_COLS = TEST_DATA.get("feature_cols", None) if TEST_DATA else None

BENCHMARK_METRICS = ["Accuracy", "Precision", "Recall", "F1", "MCC", "AUC_ROC", "AUC_PR"]
PRIMARY_METRIC = "AUC_PR"

SUP_MODEL_NAMES = ["RandomForest","XGBoost","SVM","MLP","LightGBM","CatBoost","Stacking Ensemble",
                   "ExtraTrees","GradientBoosting","HistGradientBoosting","AdaBoost","KNN",
                   "LogisticRegression","DecisionTree","NaiveBayes","Voting Ensemble",
                   "LDA","QDA","Bagging","Ridge","SGD","LSVC","XGBOD",
                   "LSTM","BiLSTM","GRU","BiGRU","CNN1D","CNN_LSTM","CNN_BiLSTM","CNN_GRU",
                   "Transformer","TCN","Attention_BiLSTM","FCN","ResNet1D","InceptionTime","LSTM_FCN"]
UNSUP_MODEL_NAMES = ["IsolationForest","OneClassSVM","KMeans","LOF","Autoencoder",
                     "GMM","EllipticEnvelope","PCA","DBSCAN","VAE",
                     "ECOD","COPOD","HBOS","CBLOF",
                     "ABOD","COF","SOD","SOS","LODA","INNE","LMDD",
                     "SO_GAAL","MO_GAAL","DeepSVDD","LUNAR","DIF","AnoGAN","ALAD"]

# ── Operatör tespit profilleri (preset) — gözetimli, güvenilir modeller ──
ANALYSIS_PRESETS = {
    "hizli": {"title": "Hızlı Tarama", "icon": "mdi:flash",
              "desc": "Tek hafif model ile düşük maliyetli ön tarama.",
              "sup": ["HistGradientBoosting"], "unsup": [], "thr": 1.0},
    "dogru": {"title": "Yüksek Doğruluk", "icon": "mdi:bullseye-arrow",
              "desc": "En iyi modellerin topluluğu — en güvenilir tespit.",
              "sup": ["ExtraTrees", "Voting Ensemble", "MLP"], "unsup": [], "thr": 1.0},
    "dusuk_alarm": {"title": "Düşük Yanlış Alarm", "icon": "mdi:shield-check-outline",
                    "desc": "Yüksek kesinlikli modeller + sıkı eşik; yanlış alarmı en aza indirir.",
                    "sup": ["Stacking Ensemble", "Voting Ensemble"], "unsup": [], "thr": 1.15},
}

_SHAP_EXPLAINERS = {}
def get_tree_explainer(model):
    """TreeExplainer'ı model başına bir kez kurup önbelleğe alır (her tıklamada
    yeniden kurmamak için — detay sayfasında belirgin hızlanma sağlar)."""
    key = id(model)
    if key not in _SHAP_EXPLAINERS:
        import shap
        _SHAP_EXPLAINERS[key] = shap.TreeExplainer(model)
    return _SHAP_EXPLAINERS[key]

def best_model(metric=PRIMARY_METRIC, among=None):
    """ALL_METRICS içinde verilen metriğe göre en iyi modeli (ad, metrik_dict) döndürür."""
    pool = {n: v for n, v in ALL_METRICS.items() if among is None or n in among}
    if not pool:
        return None, {}
    name = max(pool, key=lambda n: pool[n].get(metric, 0))
    return name, pool[name]

def nav_item(ic, text, page_id):
    return html.Button(id={"type": "nav", "page": page_id}, n_clicks=0,
                       className="nav-item", children=[icon(ic, 18), html.Span(text)])

topbar = html.Div(className="topbar", children=[
    html.Div(className="topbar-left", children=[
        html.Span("Uydu Telemetri Anomali Tespiti", className="topbar-title"),
    ]),
    html.Div(id="utc-clock", className="topbar-center"),
    html.Div(className="topbar-right", children=[
        html.Div(className="topbar-status", children=[
            html.Span(id="global-live-dot", className="topbar-dot"), html.Span("VERİ AKIŞI")]),
        html.Div(className="topbar-status", children=[
            html.Span(className="topbar-dot"), html.Span("MODEL")]),
        html.Div(className="topbar-status", children=[
            html.Span(className="topbar-dot"), html.Span("SİSTEM")]),
    ]),
])

sidebar = html.Div(className="sidebar", children=[
    html.Div(className="sidebar-logo", children=[
        html.Div([icon("mdi:satellite-variant", 26, "#06B6D4")], className="logo-icon"),
        html.Div([html.Div("Uydu Telemetri", className="logo-text"),
                  html.Div("Anomali Tespit Sistemi", className="logo-sub")])
    ]),
    html.Div(className="sidebar-nav", children=[
        html.Div("OPERASYON", className="nav-section-label",
                 style={"fontSize": "10px", "letterSpacing": "2px", "color": "#3A5068",
                        "fontWeight": "600", "padding": "8px 16px 4px"}),
        nav_item("mdi:view-dashboard", "Operasyon Paneli", "dashboard"),
        nav_item("mdi:satellite-variant", "Canlı İzleme", "live"),
        nav_item("mdi:upload", "Veri Yükle", "upload"),
        nav_item("mdi:chart-timeline-variant", "Analiz", "analysis"),
        nav_item("mdi:chart-scatter-plot", "Sonuçlar", "results"),
        nav_item("mdi:magnify-expand", "Anomali Detay", "detail"),
        html.Details(open=False, className="nav-group", children=[
            html.Summary("GELİŞTİRİCİ / ARAŞTIRMA", className="nav-group-header",
                         style={"fontSize": "10px", "letterSpacing": "2px", "color": "#3A5068",
                                "fontWeight": "600", "padding": "12px 16px 4px", "cursor": "pointer",
                                "userSelect": "none", "outline": "none"}),
            nav_item("mdi:gauge", "Model Performans", "performance"),
            nav_item("mdi:brain", "SHAP Analiz", "shap"),
            nav_item("mdi:test-tube", "Ablasyon Analizi", "ablation"),
            nav_item("mdi:lightning-bolt", "Güç Tüketimi", "power"),
            nav_item("mdi:flask-outline", "Sentetik Lab", "synthetic"),
            nav_item("mdi:rocket-launch-outline", "ESA Pipeline", "esa_pipeline"),
        ]),
    ]),
    html.Div(className="sidebar-footer", children=[
        html.Div(className="status-indicator", children=[
            html.Span(className="status-dot"), html.Span(f"Sistem Aktif  -  {len(MODELS)} model")]),
        html.Div(className="sidebar-version", children=[
            icon("mdi:cpu-64-bit", 14), html.Span("v1.0.0")]),
        html.Div(className="sidebar-version", children=[
            icon("mdi:clock-outline", 14), html.Span(time.strftime("%d.%m.%Y %H:%M"))]),
        html.Div(f"VER 2.0.0 / MDL {len(MODELS)} / ENV PROD", className="sidebar-sys-info"),
    ])
])

app = Dash(__name__, suppress_callback_exceptions=True,
           external_stylesheets=[dbc.themes.BOOTSTRAP],
           title="Uydu Telemetri", update_title=None)

app.layout = html.Div(id="app-root", children=[
    dcc.Store(id="current-page", data="dashboard"),
    dcc.Store(id="uploaded-data"),
    dcc.Store(id="prediction-results"),
    dcc.Store(id="selected-anomaly"),
    dcc.Store(id="anomaly-list"),
    dcc.Download(id="download-pdf-report"),
    dcc.Store(id="live-sim-state", data={"index": 0, "is_running": False, "anomalies": []}),
    dcc.Interval(id="clock-interval", interval=1000, n_intervals=0),
    dcc.Interval(id="live-interval", interval=500, n_intervals=0, disabled=True),
    dcc.Download(id="download-csv"),
    topbar,
    sidebar,
    html.Div(id="page-content", className="main-content"),
    html.Div(id="results-overlay", className="main-content",
             style={"display": "none"}, children=[
        html.Div(className="page-header", children=[
            html.Div("Sonuçlar", className="page-title"),
            html.Div("Anomali tespit sonuçları ve görselleştirme", className="page-subtitle")]),
        dcc.Loading(
            id="loading-results",
            type="circle",
            color="#3B82F6",
            children=[
                html.Div(id="results-content", children=[
                    html.Div(className="info-box", children=[
                        icon("mdi:chart-scatter-plot", 32, "#3B82F6"), html.Br(), html.Br(),
                        "Henüz analiz yapılmadı. Önce Analiz sayfasından işlem başlatınız."])
                ])
            ]
        )
    ]),
    html.Div(id="detail-overlay", className="main-content",
             style={"display": "none"}, children=[
        html.Div(id="detail-page-content", className="detail-page-container", children=[
            html.Div("Henüz anomali seçilmedi.", className="info-box")
        ])
    ]),
])

def page_dashboard():
    """Operasyon paneli — anomali tespit ekibi için telemetri sağlık durumu ve alarm özeti.
    (Model performans/benchmark görünümü Geliştirici > Model Performans sayfasındadır.)"""
    df_seg = pd.read_parquet(DEMO_PATH) if os.path.exists(DEMO_PATH) else pd.DataFrame()
    n_seg = len(df_seg)
    has_anom = 'anomaly' in df_seg.columns and n_seg > 0
    has_ch = 'channel' in df_seg.columns and n_seg > 0
    n_anomaly = int(df_seg['anomaly'].sum()) if has_anom else 0
    anom_ratio = f"%{df_seg['anomaly'].mean()*100:.1f}" if has_anom else "N/A"
    n_channels = int(df_seg['channel'].nunique()) if has_ch else 0
    n_raw = len(LIVE_DATA)

    # ── Kanal sağlığı: kanal başına anomali oranı (hangi sensör sorunlu) ──
    fig_health = go.Figure()
    if has_anom and has_ch:
        ch = df_seg.groupby('channel')['anomaly'].agg(['sum', 'count'])
        ch['rate'] = 100 * ch['sum'] / ch['count']
        ch = ch.sort_values('rate')
        bar_clr = ["#10B981" if r < 15 else "#F59E0B" if r < 30 else "#EF4444" for r in ch['rate']]
        fig_health = go.Figure(go.Bar(
            y=ch.index.tolist(), x=ch['rate'].tolist(), orientation='h', marker_color=bar_clr,
            text=[f"{r:.0f}%  ({int(s)}/{int(c)})" for r, s, c in zip(ch['rate'], ch['sum'], ch['count'])],
            textposition='outside', textfont=dict(size=10, color="#94A3B8")))
        fig_health.update_layout(**PLT_LAYOUT, height=360, title="Kanal Sağlığı — Anomali Oranı (%)",
                                 xaxis_title="Anomali oranı (%)", xaxis_range=[0, max(ch['rate']) * 1.25 + 5])

    # ── Son alarmlar: anomalik segmentler, değişkenliğe (var) göre şiddet ──
    alarm_rows = []
    if has_anom:
        anoms = df_seg[df_seg['anomaly'] == 1].copy()
        if 'var' in df_seg.columns and len(anoms):
            p50, p85 = df_seg['var'].quantile(0.50), df_seg['var'].quantile(0.85)
            anoms = anoms.sort_values('var', ascending=False)
            for _, r in anoms.head(12).iterrows():
                v = r['var']
                sev = ("Kritik", "badge-error") if v >= p85 else (("Uyarı", "badge-warning") if v >= p50 else ("Düşük", "badge-success"))
                seg_id = int(r['segment']) if 'segment' in r else "-"
                chn = r['channel'] if 'channel' in r else "-"
                alarm_rows.append((seg_id, chn, f"{v:.3g}", sev))

    def sev_badge(sev):
        return html.Span(sev[0], className=sev[1])

    now = time.strftime("%d.%m.%Y %H:%M")

    return html.Div([
        html.Div(className="page-header", children=[
            html.Div("Operasyon Paneli", className="page-title"),
            html.Div("Telemetri sağlık durumu ve anomali alarm özeti", className="page-subtitle")]),

        dbc.Row([
            dbc.Col(metric_card("mdi:database-outline", f"{n_seg:,}", "İzlenen Segment", "blue",
                                f"{n_raw:,} ham ölçüm"), md=3),
            dbc.Col(metric_card("mdi:alert-circle-outline", n_anomaly, "Aktif Anomali", "red",
                                f"{anom_ratio} oran"), md=3),
            dbc.Col(metric_card("mdi:satellite-uplink", n_channels, "İzlenen Kanal", "cyan",
                                "Manyetometre + Fotodiyot"), md=3),
            dbc.Col(metric_card("mdi:check-network-outline", "Aktif", "Sistem Durumu", "green",
                                f"{len(MODELS)} model hazır"), md=3),
        ], className="mb-4 g-3"),

        dbc.Row([
            dbc.Col(html.Div(className="panel", children=[
                dcc.Graph(figure=fig_health, config={"displayModeBar": False})]), md=7),
            dbc.Col(html.Div(className="panel", style={"height": "100%"}, children=[
                html.Div(className="panel-title", children=[icon("mdi:information-outline", 16), "Sistem Durumu"]),
                html.Table(className="log-table", children=[html.Tbody([
                    html.Tr([html.Td("Veri akışı"), html.Td(html.Span("AKTİF", className="badge-success"))]),
                    html.Tr([html.Td("Ham telemetri"), html.Td(f"{n_raw:,} ölçüm")]),
                    html.Tr([html.Td("İzlenen kanal"), html.Td(f"{n_channels} kanal")]),
                    html.Tr([html.Td("Tespit motoru"), html.Td(f"{len(MODELS)} model yüklü")]),
                    html.Tr([html.Td("Son güncelleme"), html.Td(now)]),
                ])]),
                html.Div(style={"marginTop": "12px", "fontSize": "11px", "color": "#64748B"},
                         children="Model geliştirme ve benchmark için sol menüde Geliştirici / Araştırma bölümüne bakınız."),
            ]), md=5),
        ], className="mb-4 g-3"),

        html.Div(className="panel", children=[
            html.Div(className="panel-title", children=[
                icon("mdi:bell-alert-outline", 16), f"Son Alarmlar ({len(alarm_rows)} anomali)"]),
            html.Table(className="log-table", children=[
                html.Thead(html.Tr([html.Th(c) for c in ["Segment", "Kanal", "Değişkenlik (var)", "Şiddet"]])),
                html.Tbody([html.Tr([html.Td(seg), html.Td(chn), html.Td(val), html.Td(sev_badge(sev))])
                            for seg, chn, val, sev in alarm_rows]
                           or [html.Tr([html.Td("Anomali kaydı yok", colSpan=4)])])
            ])
        ])
    ])


def page_upload():
    return html.Div([
        html.Div(className="page-header", children=[
            html.Div("Veri Yükle", className="page-title"),
            html.Div("CSV veya Parquet formatında telemetri verisi yükleyin", className="page-subtitle")]),
        dbc.Row([
            dbc.Col([
                dcc.Upload(id="file-upload", children=html.Div(className="upload-area", children=[
                    html.Div(icon("mdi:cloud-upload-outline", 56, "#3B82F6"), className="upload-icon"),
                    html.Div("Dosyanızı sürükleyin veya tıklayın", className="upload-text"),
                    html.Div("CSV, Parquet desteklenir", className="upload-hint"),
                ]), multiple=False),
                html.Div(style={"textAlign": "center", "marginTop": "12px"}, children=[
                    html.Button("Demo Veri Kullan", id="btn-demo", n_clicks=0, className="btn-outline")])
            ], md=12)
        ], className="mb-4"),
        dcc.Loading(
            id="loading-upload",
            type="circle",
            color="#3B82F6",
            children=[html.Div(id="upload-preview")]
        )
    ])


def page_analysis():
    sup = [n for n in SUP_MODEL_NAMES if n in MODELS]
    unsup = [n for n in UNSUP_MODEL_NAMES if n in MODELS]
    def model_option(name):
        f1 = ALL_METRICS.get(name, {}).get("F1", 0)
        return html.Span([name, html.Span(f"F1: {f1:.3f}", className="model-f1-badge")])

    def preset_option(key, p):
        return {"value": key, "label": html.Span([
            html.Span([icon(p["icon"], 15, "#06B6D4"), html.Span(p["title"],
                       style={"fontWeight": "600", "marginLeft": "6px"})]),
            html.Div(p["desc"], style={"fontSize": "11px", "color": "#64748B",
                                       "marginLeft": "21px", "lineHeight": "1.4"}),
        ])}
    # Varsayılan profil: Yüksek Doğruluk
    default = ANALYSIS_PRESETS["dogru"]
    def_sup = [m for m in default["sup"] if m in MODELS]

    return html.Div([
        html.Div(className="page-header", children=[
            html.Div("Anomali Analizi", className="page-title"),
            html.Div("Tespit profili seçin ve analizi başlatın", className="page-subtitle")]),
        dbc.Row([
            dbc.Col([html.Div(className="panel", children=[
                html.Div(className="panel-title", children=[icon("mdi:tune-vertical", 16), "Tespit Profili"]),
                dcc.RadioItems(
                    id="preset-select",
                    options=[preset_option(k, p) for k, p in ANALYSIS_PRESETS.items()],
                    value="dogru", className="preset-radio",
                    inputStyle={"marginRight": "8px"},
                    labelStyle={"display": "block", "padding": "8px 0", "cursor": "pointer"}),
                html.Button("Analizi Başlat", id="btn-analyze", n_clicks=0, className="btn-primary",
                            style={"width": "100%", "marginTop": "12px"}),
                html.Div(id="selection-counter", className="selection-counter"),

                html.Details(open=False, style={"marginTop": "16px"}, children=[
                    html.Summary("Gelişmiş — model seçimi", style={
                        "fontSize": "11px", "letterSpacing": "1px", "color": "#3A5068",
                        "fontWeight": "600", "cursor": "pointer", "userSelect": "none",
                        "outline": "none", "padding": "4px 0"}),
                    html.Div("GÖZETİMLİ", className="section-label"),
                    dcc.Checklist(id="sup-models", options=[{"label": model_option(n), "value": n} for n in sup],
                                  value=def_sup, className="model-checklist", inputStyle={"marginRight": "8px"}),
                    html.Div("GÖZETİMSİZ", className="section-label"),
                    dcc.Checklist(id="unsup-models", options=[{"label": model_option(n), "value": n} for n in unsup],
                                  value=[], className="model-checklist", inputStyle={"marginRight": "8px"}),
                    html.Div("EŞİK ÇARPANI", className="section-label"),
                    dcc.Slider(id="threshold-slider", min=0.5, max=1.5, step=0.05, value=default["thr"],
                               marks={0.5: "0.5", 1.0: "1.0", 1.5: "1.5"},
                               tooltip={"placement": "bottom", "always_visible": False}),
                    html.Div("Düşük değer: hassas tespit, yüksek yanlış alarm. Yüksek değer: güvenilir ama az tespit.",
                             style={"fontSize": "11px", "color": "#64748B", "marginTop": "8px", "lineHeight": "1.5"}),
                ]),
            ])], md=3),
            dbc.Col([
                dcc.Loading(
                    id="loading-analysis",
                    type="circle",
                    color="#3B82F6",
                    children=[
                        html.Div(id="analysis-output", className="panel", children=[
                            html.Div(className="info-box", children=[
                                icon("mdi:information-outline", 32, "#3B82F6"), html.Br(), html.Br(),
                                "Sol panelden model seçip analizi başlatınız."])
                        ])
                    ]
                )
            ], md=9)
        ], className="g-3")
    ])


def page_results():
    return html.Div([
        html.Div(className="page-header", children=[
            html.Div("Sonuçlar", className="page-title"),
            html.Div("Anomali tespit sonuçları ve görselleştirme", className="page-subtitle")]),
        html.Div(id="results-content")
    ])


def _performance_recommendation():
    """ALL_METRICS'ten veri-güdümlü öneri kutusu üretir (AUC_PR birincil ölçüt)."""
    best_name, m = best_model(PRIMARY_METRIC)
    if not best_name:
        return html.Div()
    u_name, um = best_model(PRIMARY_METRIC, among=set(UNSUP_MODEL_NAMES))
    parts = [
        f"{best_name} modeli {m.get('AUC_PR',0):.3f} AUC-PR, {m.get('F1',0):.3f} F1 ve "
        f"{m.get('MCC',0):.3f} MCC ile resmi test setinde (Ψ) en yüksek performansı göstermiştir. ",
    ]
    if "FAR" in m:
        parts.append(f"Yanlış alarm oranı (FAR) {m.get('FAR',0):.3f}. ")
    if u_name:
        parts.append(f"Gözetimsiz modeller arasında {u_name} {um.get('AUC_PR',0):.3f} AUC-PR ile öne çıkmaktadır.")
    return html.Div(className="recommendation-box", children=[
        html.Div(className="rec-title", children=[icon("mdi:trophy-outline", 18, "#3B82F6"), f"Önerimiz: {best_name}"]),
        html.Div(className="rec-body", children=parts),
    ])


def page_performance():
    if not ALL_METRICS:
        return html.Div("Metrik verisi bulunamadi.")

    mdf = pd.DataFrame(ALL_METRICS).T
    cols = [c for c in BENCHMARK_METRICS + ["FAR"] if c in mdf.columns]
    ranked = sorted(ALL_METRICS, key=lambda n: ALL_METRICS[n].get(PRIMARY_METRIC, 0), reverse=True)

    fig_roc = go.Figure()
    if TEST_DATA:
        from sklearn.metrics import roc_curve, auc
        X_t, y_t = TEST_DATA["X_test"], TEST_DATA["y_test"]
        clrs = ["#3B82F6","#10B981","#F59E0B","#EF4444","#8B5CF6","#06B6D4","#F778A1","#A78BFA","#FB923C"]
        for i, (name, model) in enumerate(MODELS.items()):
            try:
                _, prob = predict(model, name, X_t, THRESHOLDS, 1.0)
                fpr, tpr, _ = roc_curve(y_t, prob)
                a = auc(fpr, tpr)
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} ({a:.3f})",
                                             line=dict(color=clrs[i % len(clrs)], width=2)))
            except Exception as e:
                print(f"ROC çizilemedi ({name}):", e)
    fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", line=dict(dash="dash", color="#4A5568"), showlegend=False))
    fig_roc.update_layout(**PLT_LAYOUT, height=400, title="ROC Egrileri", xaxis_title="FPR", yaxis_title="TPR")

    top = ranked[:6]
    cats = ["Accuracy","Precision","Recall","F1","MCC","AUC_ROC","AUC_PR"]
    fig_radar = go.Figure()
    for n in top:
        vals = [ALL_METRICS[n].get(c, 0) for c in cats]
        fig_radar.add_trace(go.Scatterpolar(r=vals + [vals[0]], theta=cats + [cats[0]], name=n, fill="toself", opacity=0.5))
    fig_radar.update_layout(**PLT_LAYOUT, height=400, polar=dict(bgcolor="#080C14", radialaxis=dict(range=[0,1], showticklabels=True, tickfont=dict(size=10))))

    return html.Div([
        html.Div(className="page-header", children=[
            html.Div("Model Performans", className="page-title"),
            html.Div("Tüm modellerin karşılaştırmalı analizi", className="page-subtitle")]),
        html.Div(className="panel mb-4", children=[
            html.Div(className="panel-title", children=[icon("mdi:table", 16), "Metrik Tablosu"]),
            dash_table.DataTable(
                columns=[{"name": "Model", "id": "Model"}] + [{"name": c, "id": c} for c in cols],
                data=[{"Model": n, **{c: f"{ALL_METRICS[n].get(c,0):.4f}" for c in cols}} for n in ranked],
                style_header={"backgroundColor": "#0D1117", "color": "#64748B", "fontWeight": "600",
                               "border": "1px solid #1E2A3A", "textTransform": "uppercase", "fontSize": "11px"},
                style_cell={"backgroundColor": "#151C28", "color": "#F1F5F9", "border": "1px solid #1E2A3A",
                             "fontFamily": "IBM Plex Sans", "fontSize": "12.5px", "padding": "10px"},
                style_data_conditional=[
                    {"if": {"row_index": "odd"}, "backgroundColor": "#111827"},
                ] + [
                    cond
                    for col in ["AUC_PR", "AUC_ROC", "F1", "MCC"] if col in cols
                    for cond in [
                        {"if": {"filter_query": f'{{{col}}} > 0.95', "column_id": col}, "color": "#00FF9C", "fontWeight": "600"},
                        {"if": {"filter_query": f'{{{col}}} > 0.80 && {{{col}}} <= 0.95', "column_id": col}, "color": "#FFB300"},
                        {"if": {"filter_query": f'{{{col}}} <= 0.80', "column_id": col}, "color": "#FF3B5C"},
                    ]
                ],
            )
        ]),
        _performance_recommendation(),
        html.Br(),
        dbc.Row([
            dbc.Col(html.Div(className="panel", children=[dcc.Graph(figure=fig_roc, config={"displayModeBar": False})]), md=7),
            dbc.Col(html.Div(className="panel", children=[dcc.Graph(figure=fig_radar, config={"displayModeBar": False})]), md=5),
        ], className="g-3")
    ])


def page_shap():
    if SHAP_DATA is None:
        return html.Div([
            html.Div(className="page-header", children=[
                html.Div("SHAP Analiz", className="page-title"),
                html.Div("Model yorumlanabilirlik analizi", className="page-subtitle")]),
            html.Div(className="warning-box", children=[
                icon("mdi:alert-outline", 40, "#F59E0B"),
                html.Div("SHAP Verileri Bulunamadi", className="warning-title"),
                html.Div([
                    "SHAP analiz verileri henuz hesaplanmamis. ",
                    "Lutfen once ", html.Code("notebooks/07_shap_analizi.ipynb"),
                    " notebook'unu calistirin.",
                    html.Br(), html.Br(),
                    "Notebook calistirildiktan sonra ",
                    html.Code("models/shap_values.pkl"),
                    " dosyasi olusturulacak ve bu sayfa aktif hale gelecektir."
                ], className="warning-body")
            ])
        ])

    feature_labels = SHAP_DATA.get('feature_labels', SHAP_DATA.get('feature_cols', []))
    y_test = SHAP_DATA['y_test']
    anomaly_indices = np.where(y_test == 1)[0]
    anomaly_options = [{"label": f"Segment #{i} (index {idx})", "value": int(idx)}
                       for i, idx in enumerate(anomaly_indices, 1)]

    return html.Div([
        html.Div(className="page-header", children=[
            html.Div("SHAP Analiz", className="page-title"),
            html.Div("Model yorumlanabilirlik ve ozellik onemi analizi", className="page-subtitle")]),
        dcc.Tabs(id="shap-tabs", value="tab-importance", className="custom-tabs", children=[
            dcc.Tab(label="Ozellik Onemi", value="tab-importance", className="tab", selected_className="tab--selected"),
            dcc.Tab(label="Anomali Aciklama", value="tab-explain", className="tab", selected_className="tab--selected"),
            dcc.Tab(label="Model Karsilastirma", value="tab-compare", className="tab", selected_className="tab--selected"),
        ]),
        html.Div(id="shap-tab-content", style={"marginTop": "20px"}),
        dcc.Store(id="shap-anomaly-options", data=[o["value"] for o in anomaly_options]),
    ])


def page_live():
    channels = LIVE_DATA['channel'].unique().tolist() if not LIVE_DATA.empty and 'channel' in LIVE_DATA.columns else []
    fast_models = [n for n in ["IsolationForest", "LOF", "OneClassSVM", "KMeans"] if n in MODELS]
    
    fig_sig = go.Figure()
    fig_sig.update_layout(**PLT_LAYOUT, height=300,
                          xaxis=dict(showgrid=True, gridcolor="#1C2A3A"), yaxis=dict(showgrid=True, gridcolor="#1C2A3A"))
    fig_score = go.Figure()
    fig_score.update_layout(**PLT_LAYOUT, height=150,
                            xaxis=dict(showgrid=True, gridcolor="#1C2A3A"), yaxis=dict(range=[0, 1.05]))
                            
    return html.Div(className="live-page-container", children=[
        html.Div(className="page-header", children=[
            html.Div("Canlı İzleme", className="page-title"),
            html.Div("Gerçek zamanlı telemetri akışı ve anında anomali tespiti", className="page-subtitle")
        ]),
        
        html.Div(className="panel live-control-panel", children=[
            html.Div(className="live-controls-left", children=[
                html.Div([
                    html.Label("Kanal:"),
                    dcc.Dropdown(id="live-channel", options=[{"label": c, "value": c} for c in channels],
                                 value=channels[0] if channels else None, className="custom-dropdown", clearable=False)
                ], className="control-group"),
                html.Div([
                    html.Label("Hızlı Model:"),
                    dcc.Dropdown(id="live-model", options=[{"label": m, "value": m} for m in fast_models],
                                 value=fast_models[0] if fast_models else None, className="custom-dropdown", clearable=False)
                ], className="control-group"),
                html.Div([
                    html.Label("Hız:"),
                    dcc.Dropdown(id="live-speed", options=[
                        {"label": "Yavaş (1x)", "value": 1},
                        {"label": "Normal (5x)", "value": 5},
                        {"label": "Hızlı (20x)", "value": 20}
                    ], value=5, className="custom-dropdown", clearable=False)
                ], className="control-group"),
            ]),
            html.Div(className="live-controls-right", children=[
                html.Button([icon("mdi:play", 18), " Başlat"], id="live-start", n_clicks=0, className="btn-primary"),
                html.Button([icon("mdi:stop", 18), " Durdur"], id="live-stop", n_clicks=0, className="btn-error", disabled=True),
                html.Button([icon("mdi:refresh", 18), " Sıfırla"], id="live-reset", n_clicks=0, className="btn-outline"),
            ])
        ]),
        
        html.Div(className="live-status-bar", children=[
            html.Span(id="live-stat-read", children="OKUNAN: 0"), html.Span("|", className="stat-divider"),
            html.Span(id="live-stat-total", children=f"TOPLAM: {len(LIVE_DATA)}"), html.Span("|", className="stat-divider"),
            html.Span(id="live-stat-prog", children="%0.0"), html.Span("|", className="stat-divider"),
            html.Span(id="live-stat-anom", children="ANOMALİ: 0"), html.Span("|", className="stat-divider"),
            html.Span(id="live-stat-last", children="SON ALARM: Yok"), html.Span("|", className="stat-divider"),
            html.Span(id="live-stat-model", children="MODEL: -"),
        ]),
        
        html.Div(className="live-main-area", children=[
            html.Div(className="live-charts-area", children=[
                html.Div(className="panel live-chart-panel", children=[
                    html.Div(className="panel-title", style={"display": "flex", "justifyContent": "space-between"}, children=[
                        html.Span([icon("mdi:chart-timeline-variant", 16), " Telemetri Sinyali"]),
                        html.Span(id="live-indicator", children=[html.Span(className="status-dot"), "DURDURULDU"], className="live-indicator-badge")
                    ]),
                    dcc.Graph(id="live-signal-graph", figure=fig_sig, config={"displayModeBar": False}, style={"height": "300px"})
                ]),
                html.Div(className="panel live-chart-panel", style={"marginTop": "16px"}, children=[
                    html.Div(className="panel-title", children=[icon("mdi:chart-bell-curve", 16), " Anomali Skoru"]),
                    dcc.Graph(id="live-score-graph", figure=fig_score, config={"displayModeBar": False}, style={"height": "150px"})
                ]),
            ]),
            
            html.Div(className="live-alarm-panel", children=[
                html.Div("ALARM KAYITLARI", className="alarm-panel-title"),
                html.Div(id="live-alarm-list", className="alarm-list-container", children=[
                    html.Div("Anomali Yok", className="no-alarm-msg")
                ]),
                html.Div(id="live-alarm-count", className="alarm-count-footer", children="0 Alarm")
            ])
        ])
    ])


def page_detail():
    return html.Div()

from ablation_page import get_ablation_layout, register_ablation_callbacks
from power_page import get_power_layout, register_power_callbacks
from synthetic_page import get_synthetic_layout, register_synthetic_callbacks
from esa_pipeline_page import get_esa_pipeline_layout, register_esa_pipeline_callbacks
PAGES = {"dashboard": page_dashboard, "upload": page_upload, "analysis": page_analysis,
         "results": page_results, "shap": page_shap, "performance": page_performance, "live": page_live, "detail": page_detail, "ablation": get_ablation_layout,
         "power": lambda: get_power_layout(ALL_METRICS),
         "synthetic": get_synthetic_layout, "esa_pipeline": get_esa_pipeline_layout}

@callback(Output("current-page", "data"),
          [Input({"type": "nav", "page": p}, "n_clicks") for p in PAGES],
          prevent_initial_call=True)
def navigate(*clicks):
    if not ctx.triggered_id: return "dashboard"
    return ctx.triggered_id["page"]

app.clientside_callback(
    """
    function(n_intervals) {
        var d = new Date();
        return "UTC  " + d.toISOString().replace('T', '  ').substring(0, 19);
    }
    """,
    Output("utc-clock", "children"),
    Input("clock-interval", "n_intervals")
)

@callback(Output("page-content", "children"), Output("page-content", "style"),
          Output("results-overlay", "style"),
          Output("detail-overlay", "style"),
          Input("current-page", "data"))
def render_page(page_id):
    if page_id == "results":
        return html.Div(), {"display": "none"}, {"display": "block"}, {"display": "none"}
    if page_id == "detail":
        return html.Div(), {"display": "none"}, {"display": "none"}, {"display": "block"}
    return PAGES.get(page_id, page_dashboard)(), {"display": "block"}, {"display": "none"}, {"display": "none"}

@callback(Output("uploaded-data", "data"), Output("upload-preview", "children"),
          Input("file-upload", "contents"), Input("btn-demo", "n_clicks"),
          State("file-upload", "filename"), prevent_initial_call=True)
def handle_upload(contents, demo_clicks, filename):
    trigger = ctx.triggered_id
    if trigger == "btn-demo":
        if os.path.exists(DEMO_PATH):
            df = pd.read_parquet(DEMO_PATH)
            return df.to_json(date_format='iso', orient='split'), build_preview(df, "demo_data.parquet")
        return no_update, html.Div("Demo veri bulunamadi.", style={"color": "#EF4444"})
    if contents:
        _, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        try:
            if filename.endswith('.parquet'):
                df = pd.read_parquet(io.BytesIO(decoded))
            else:
                df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
                
            ESA_CORE = {'mean', 'var', 'std', 'n_peaks', 'diff_var'}
            is_featurized = ESA_CORE.issubset(set(df.columns))
            if not is_featurized:
                if 'value' not in df.columns:
                    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    if num_cols:
                        df = df.rename(columns={num_cols[0]: 'value'})

                if 'segment' not in df.columns:
                    df['segment'] = np.repeat(np.arange(len(df) // 250 + 1), 250)[:len(df)]

                if 'channel' not in df.columns: df['channel'] = 'AUTO_SENSOR'
                if 'anomaly' not in df.columns: df['anomaly'] = 0
                if 'train' not in df.columns: df['train'] = 0
                if 'sampling' not in df.columns: df['sampling'] = 1

                df = extract_features_from_raw(df)

            return df.to_json(date_format='iso', orient='split'), build_preview(df, filename)
        except Exception as e:
            import traceback
            err_msg = str(e) + "\\n" + traceback.format_exc()
            return no_update, html.Div(f"Hata: {err_msg}", style={"color": "#EF4444", "whiteSpace": "pre-wrap", "fontSize": "11px"})
    return no_update, no_update

def build_preview(df, filename):
    n_miss = df.isnull().sum().sum()
    anom = f"{df['anomaly'].mean()*100:.1f}%" if 'anomaly' in df.columns else "N/A"

    col_items = []
    for c in df.columns:
        dtype_str = str(df[c].dtype)
        col_items.append(html.Div(className="col-grid-item", children=[
            html.Span(c, className="col-name"),
            html.Span(dtype_str, className="col-dtype"),
        ]))

    ts_chart = None
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    plot_col = [c for c in num_cols if c not in ['segment','anomaly','train','sampling']]
    if plot_col:
        sample = df.head(500).copy()
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(y=sample[plot_col[0]], mode="lines", name=plot_col[0],
                                     line=dict(color="#3B82F6", width=1.5)))
        if 'anomaly' in sample.columns:
            anom_pts = sample[sample['anomaly'] == 1]
            if len(anom_pts) > 0:
                fig_ts.add_trace(go.Scatter(x=anom_pts.index, y=anom_pts[plot_col[0]],
                                             mode="markers", name="Anomali",
                                             marker=dict(color="#EF4444", size=6)))
        fig_ts.update_layout(**PLT_LAYOUT, height=300, title=f"Zaman Serisi: {plot_col[0]}")
        ts_chart = html.Div(className="panel mb-4", children=[
            dcc.Graph(figure=fig_ts, config={"displayModeBar": False})])

    children = [
        dbc.Row([
            dbc.Col(metric_card("mdi:file-document-outline", filename[:25], "Dosya", "blue"), md=3),
            dbc.Col(metric_card("mdi:table-row", f"{df.shape[0]:,}", "Satır", "green"), md=3),
            dbc.Col(metric_card("mdi:table-column", df.shape[1], "Sütun", "cyan"), md=3),
            dbc.Col(metric_card("mdi:alert-circle-outline", anom, "Anomali Oranı", "red"), md=3),
        ], className="mb-4 g-3"),
        html.Div(className="panel mb-4", children=[
            html.Div(className="panel-title", children=[icon("mdi:format-list-bulleted", 16), "Sütun Listesi"]),
            html.Div(className="col-grid", children=col_items),
        ]),
    ]
    if ts_chart:
        children.append(ts_chart)
    children.append(html.Div(className="panel", children=[
        html.Div(className="panel-title", children=[icon("mdi:table-large", 16), "Veri Önizleme"]),
        dash_table.DataTable(
            columns=[{"name": c, "id": c} for c in df.columns],
            data=df.head(50).to_dict('records'), page_size=15,
            style_header={"backgroundColor": "#0D1117", "color": "#64748B", "fontWeight": "600",
                           "border": "1px solid #1E2A3A", "fontSize": "11px"},
            style_cell={"backgroundColor": "#151C28", "color": "#F1F5F9", "border": "1px solid #1E2A3A",
                         "fontFamily": "IBM Plex Sans", "fontSize": "12px", "padding": "8px", "maxWidth": "150px",
                         "overflow": "hidden", "textOverflow": "ellipsis"},
            style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": "#111827"}],
            sort_action="native", filter_action="native",
        )
    ]))
    return html.Div(children)

@callback(Output("analysis-output", "children"), Output("prediction-results", "data"),
          Input("btn-analyze", "n_clicks"),
          State("sup-models", "value"), State("unsup-models", "value"),
          State("threshold-slider", "value"), State("uploaded-data", "data"),
          prevent_initial_call=True)
def run_analysis(n, sup_sel, unsup_sel, thresh_mult, data_json):
    if not n: return no_update, no_update
    selected = (sup_sel or []) + (unsup_sel or [])
    if not selected:
        return html.Div("En az bir model seçiniz.", style={"color": "#EF4444"}), no_update

    if data_json:
        df = pd.read_json(io.StringIO(data_json), orient='split')
    elif os.path.exists(DEMO_PATH):
        df = pd.read_parquet(DEMO_PATH)
    else:
        return html.Div("Veri bulunamadı.", style={"color": "#EF4444"}), no_update

    if FEATURE_COLS:
        for c in FEATURE_COLS:
            if c not in df.columns:
                df[c] = 0
        X = df[FEATURE_COLS].fillna(0).values
    else:
        feature_cols = [c for c in df.columns if c not in DROP_COLS]
        X = df[feature_cols].fillna(0).values

    if SCALER:
        try: X = SCALER.transform(X)
        except Exception as e:
            return html.Div(f"Scaler hatası: {e}", style={"color": "#EF4444"}), no_update

    results = {}
    rows = []
    for name in selected:
        if name not in MODELS: continue
        try:
            pr, sc = predict(MODELS[name], name, X, THRESHOLDS, thresh_mult)
            n_anom = int(pr.sum())
            results[name] = {"preds": pr.tolist(), "scores": sc.tolist(), "n_anomaly": n_anom}
            rows.append(html.Div(className="progress-row", children=[
                html.Span(name, className="progress-model-name"),
                dbc.Progress(value=100, color="success", style={"flex": 1, "height": "6px"}),
                html.Span(f"{n_anom} anomali", style={"fontSize": "12px", "color": "#10B981", "width": "100px", "textAlign": "right"}),
            ]))
        except Exception as e:
            rows.append(html.Div(className="progress-row", children=[
                html.Span(name, className="progress-model-name"),
                html.Span(f"Hata: {str(e)[:40]}", style={"fontSize": "12px", "color": "#EF4444"}),
            ]))

    total = sum(r["n_anomaly"] for r in results.values())
    summary = dbc.Row([
        dbc.Col(metric_card("mdi:check-circle-outline", len(results), "Başarılı Model", "green"), md=4),
        dbc.Col(metric_card("mdi:alert-outline", total, "Toplam Anomali", "red"), md=4),
        dbc.Col(metric_card("mdi:timer-outline", f"{len(selected)}", "Çalışan Model", "blue"), md=4),
    ], className="mb-3 g-3")

    return html.Div([summary, html.Div(className="panel-title", children=[icon("mdi:format-list-bulleted",16), "Model Sonuçları"]), *rows]), json.dumps(results)

@callback(Output("sup-models", "value"), Output("unsup-models", "value"),
          Output("threshold-slider", "value"),
          Input("preset-select", "value"), prevent_initial_call=True)
def apply_preset(preset):
    """Seçilen operatör profilini gelişmiş kontrollere (model seçimi + eşik) uygular."""
    p = ANALYSIS_PRESETS.get(preset)
    if not p:
        return no_update, no_update, no_update
    sup = [m for m in p["sup"] if m in MODELS]
    unsup = [m for m in p["unsup"] if m in MODELS]
    return sup, unsup, p["thr"]

@callback(Output("selection-counter", "children"),
          Input("sup-models", "value"), Input("unsup-models", "value"))
def update_counter(sup_sel, unsup_sel):
    ns = len(sup_sel or [])
    nu = len(unsup_sel or [])
    return f"{ns} gözetimli + {nu} gözetimsiz model etkin"

@callback(Output("results-content", "children"),
          Input("prediction-results", "data"),
          State("uploaded-data", "data"), prevent_initial_call=True)
def update_results(pred_json, data_json):
    if not pred_json:
        return html.Div(className="info-box", children=["Henüz analiz yapılmadı."])
    results = json.loads(pred_json)
    if not results:
        return html.Div(className="info-box", children=["Sonuç bulunamadı."])

    if data_json:
        df = pd.read_json(io.StringIO(data_json), orient='split')
    elif os.path.exists(DEMO_PATH):
        df = pd.read_parquet(DEMO_PATH)
    else:
        return html.Div("Veri yok.")

    ensemble_binary = np.zeros(len(df))
    for r in results.values():
        ensemble_binary += np.array(r["preds"])
    ensemble_binary /= max(len(results), 1)
    anom_mask = ensemble_binary > 0
    n_anom = int(anom_mask.sum())
    agreement = sum(1 for r in results.values() for p in r["preds"] if p == 1) / max(len(results) * len(df), 1)

    score_ensemble = np.zeros(len(df))
    n_score_models = 0
    for name, r in results.items():
        sc = np.array(r["scores"])
        sc_min, sc_max = sc.min(), sc.max()
        if sc_max - sc_min > 1e-10:
            sc_n = (sc - sc_min) / (sc_max - sc_min)
        else:
            sc_n = np.zeros_like(sc)
        score_ensemble += sc_n
        n_score_models += 1
    if n_score_models > 0:
        score_ensemble /= n_score_models

    avg_score = float(np.mean(score_ensemble[anom_mask])) if n_anom > 0 else 0

    fig_scores = go.Figure()
    clrs = ["#3B82F6","#10B981","#EF4444","#F59E0B","#8B5CF6","#06B6D4","#F778A1","#A78BFA","#FB923C"]
    for i, (name, r) in enumerate(results.items()):
        sc = np.array(r["scores"])
        sc_n = (sc - sc.min()) / (sc.max() - sc.min() + 1e-10)
        fig_scores.add_trace(go.Scatter(y=sc_n, mode="lines", name=name, line=dict(color=clrs[i%len(clrs)], width=1.5)))
    in_region = False; start = 0
    for i in range(len(score_ensemble)):
        if anom_mask[i] and not in_region: start = i; in_region = True
        elif (not anom_mask[i] or i == len(score_ensemble)-1) and in_region:
            fig_scores.add_vrect(x0=start, x1=i, fillcolor="rgba(239,68,68,0.08)", line_width=0, layer="below")
            in_region = False
    fig_scores.update_layout(**PLT_LAYOUT, height=400, title="Anomali Skorları (Normalize)",
                              yaxis_title="Normalize Anomali Skoru", xaxis_title="Segment")

    anom_indices = np.where(anom_mask)[0]
    
    n_crit = n_warn = n_low = 0
    for idx in anom_indices:
        if score_ensemble[idx] > 0.8: n_crit += 1
        elif score_ensemble[idx] > 0.5: n_warn += 1
        else: n_low += 1
        
    table_data = []
    for row_no, idx in enumerate(anom_indices[:100], 1):
        sev = "Kritik" if score_ensemble[idx] > 0.8 else "Uyarı" if score_ensemble[idx] > 0.5 else "Düşük"
        ch = df.iloc[idx].get("channel", "N/A") if "channel" in df.columns else "N/A"
        table_data.append({"NO": row_no, "Segment": int(df.iloc[idx].get("segment", idx)),
                           "Kanal": ch, "Skor": f"{score_ensemble[idx]:.2f}", "Şiddet": sev, "Detay": "İncele", "_idx": int(idx)})

    return html.Div([
        dbc.Row([
            dbc.Col(metric_card("mdi:file-document-check-outline", len(df), "Analiz Edilen", "blue"), md=3),
            dbc.Col(metric_card("mdi:alert-circle-outline", n_anom, "Tespit Edilen", "red"), md=3),
            dbc.Col(metric_card("mdi:chart-line", f"{avg_score:.3f}", "Ortalama Skor", "yellow"), md=3),
            dbc.Col(metric_card("mdi:handshake-outline", f"%{agreement*100:.1f}", "Model Uzlaşması", "green"), md=3),
        ], className="mb-4 g-3"),
        html.Div(className="panel mb-4", children=[dcc.Graph(figure=fig_scores, config={"displayModeBar": False})]),
        dbc.Row([
            dbc.Col(html.Div(className="metric-card red", style={"padding":"12px"}, children=[
                html.Span(f"{n_crit}", style={"fontSize":"20px","fontWeight":"700","fontFamily":"IBM Plex Mono"}),
                html.Span(" Kritik", style={"color":"#FCA5A5","fontSize":"12px","marginLeft":"6px"})]), md=4),
            dbc.Col(html.Div(className="metric-card yellow", style={"padding":"12px"}, children=[
                html.Span(f"{n_warn}", style={"fontSize":"20px","fontWeight":"700","fontFamily":"IBM Plex Mono"}),
                html.Span(" Uyarı", style={"color":"#FCD34D","fontSize":"12px","marginLeft":"6px"})]), md=4),
            dbc.Col(html.Div(className="metric-card green", style={"padding":"12px"}, children=[
                html.Span(f"{n_low}", style={"fontSize":"20px","fontWeight":"700","fontFamily":"IBM Plex Mono"}),
                html.Span(" Düşük", style={"color":"#86EFAC","fontSize":"12px","marginLeft":"6px"})]), md=4),
        ], className="mb-3 g-3"),
        html.Div(className="panel", children=[
            html.Div(className="panel-title", children=[icon("mdi:format-list-bulleted", 16), f"Anomali Listesi ({len(table_data)} kayıt)"]),
            dash_table.DataTable(
                id="results-table",
                columns=[{"name": c, "id": c} for c in ["NO","Segment","Kanal","Skor","Şiddet", "Detay"]],
                data=table_data, page_size=12, row_selectable="single",
                style_header={"backgroundColor":"#0D1117","color":"#64748B","fontWeight":"600","border":"1px solid #1E2A3A","fontSize":"11px"},
                style_cell={"backgroundColor":"#151C28","color":"#F1F5F9","border":"1px solid #1E2A3A","fontFamily":"IBM Plex Sans","fontSize":"12px","padding":"8px"},
                style_data_conditional=[
                    {"if":{"filter_query":'{Şiddet} = "Kritik"'},"backgroundColor":"rgba(239,68,68,0.08)","color":"#FCA5A5"},
                    {"if":{"filter_query":'{Şiddet} = "Uyarı"'},"backgroundColor":"rgba(245,158,11,0.08)","color":"#FCD34D"},
                    {"if":{"filter_query":'{Şiddet} = "Düşük"'},"backgroundColor":"rgba(16,185,129,0.08)","color":"#86EFAC"},
                    {"if":{"column_id":"Detay"}, "color":"#00C8FF", "cursor":"pointer", "textDecoration":"underline", "fontWeight":"bold"},
                    {"if":{"row_index":"odd"},"backgroundColor":"#111827"},
                ],
                sort_action="native", filter_action="native",
            ),
            html.Div(id="detail-info-msg", className="info-box", style={"marginTop":"15px", "textAlign":"center"},
                     children="Detay görüntülemek için tabloda bir anomali satırına tıklayın."),
            html.Div(style={"marginTop": "12px", "textAlign": "right"}, children=[
                html.Button("CSV Olarak İndir", id="btn-csv-download", n_clicks=0, className="btn-download"),
            ]),
            dcc.Store(id="csv-store", data=table_data),
            html.Div(id="shap-mini-waterfall-container", style={"marginTop": "20px"}),
        ]),
    ])

@callback(Output("selected-anomaly", "data"), Output("anomaly-list", "data"),
          Output("current-page", "data", allow_duplicate=True), Output("detail-info-msg", "children"),
          Input("results-table", "active_cell"), State("results-table", "data"), prevent_initial_call=True)
def select_anomaly(active_cell, data):
    if not active_cell or not data: return no_update, no_update, no_update, "Detay görüntülemek için tabloda bir anomali satırına tıklayın."
    row_idx = active_cell["row"]
    col_id = active_cell["column_id"]
    if col_id != "Detay": return no_update, no_update, no_update, no_update
    
    selected = data[row_idx]
    return selected, data, "detail", no_update

@callback(Output("download-csv", "data"),
          Input("btn-csv-download", "n_clicks"),
          State("csv-store", "data"),
          prevent_initial_call=True)
def download_csv(n, data):
    if not n or not data: return no_update
    df_out = pd.DataFrame(data)
    return dcc.send_data_frame(df_out.to_csv, "anomali_sonuclari.csv", index=False)

@callback(Output("shap-tab-content", "children"),
          Input("shap-tabs", "value"),
          prevent_initial_call=False)
def render_shap_tab(tab):
    if SHAP_DATA is None:
        return html.Div()

    feature_labels = SHAP_DATA.get('feature_labels', SHAP_DATA.get('feature_cols', []))
    feature_cols = SHAP_DATA.get('feature_cols', [])
    y_test = SHAP_DATA['y_test']
    anomaly_indices = np.where(y_test == 1)[0]

    if tab == "tab-importance":
        return html.Div([
            html.Div(className="panel", children=[
                html.Div(className="panel-title", children=[icon("mdi:chart-bar", 16), "Model Secimi"]),
                dcc.Dropdown(
                    id="shap-model-select",
                    options=[
                        {"label": "Random Forest", "value": "rf"},
                        {"label": "XGBoost", "value": "xgb"},
                    ],
                    value="rf",
                    className="shap-dropdown"
                ),
            ]),
            html.Div(id="shap-importance-chart", className="panel", style={"marginTop": "16px"}),
            html.Div(id="shap-importance-text", className="panel", style={"marginTop": "16px"}),
        ])

    elif tab == "tab-explain":
        anomaly_options = [{"label": f"Segment #{i} (index {idx})", "value": int(idx)}
                           for i, idx in enumerate(anomaly_indices, 1)]
        return html.Div([
            html.Div(className="panel", children=[
                html.Div(className="panel-title", children=[icon("mdi:magnify", 16), "Anomali Secimi"]),
                dcc.Dropdown(
                    id="shap-anomaly-select",
                    options=anomaly_options,
                    value=anomaly_options[0]["value"] if anomaly_options else None,
                    placeholder="Bir anomali segmenti secin...",
                    className="shap-dropdown"
                ),
            ]),
            html.Div(id="shap-waterfall-chart", className="panel", style={"marginTop": "16px"}),
            html.Div(id="shap-waterfall-text", className="panel", style={"marginTop": "16px"}),
        ])

    elif tab == "tab-compare":
        rf_shap = SHAP_DATA['rf_shap_values']
        xgb_shap = SHAP_DATA['xgb_shap_values']
        rf_imp = np.abs(rf_shap).mean(axis=0)
        xgb_imp = np.abs(xgb_shap).mean(axis=0)

        combined = rf_imp + xgb_imp
        top_idx = np.argsort(combined)[-10:][::-1]
        top_labels = [feature_labels[i] for i in top_idx]

        fig = go.Figure()
        fig.add_trace(go.Bar(name="Random Forest", y=top_labels, x=rf_imp[top_idx],
                             orientation='h', marker_color='#3B82F6', opacity=0.85))
        fig.add_trace(go.Bar(name="XGBoost", y=top_labels, x=xgb_imp[top_idx],
                             orientation='h', marker_color='#10B981', opacity=0.85))
        fig.update_layout(**PLT_LAYOUT, height=500, barmode='group',
                          title="RF vs XGBoost - SHAP Ozellik Onemi Karsilastirmasi",
                          xaxis_title="Ortalama |SHAP Degeri|",
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        fig.update_yaxes(autorange="reversed")

        diff = np.abs(rf_imp - xgb_imp)
        diff_idx = np.argsort(diff)[-3:][::-1]
        diff_text_parts = []
        for idx in diff_idx:
            lbl = feature_labels[idx]
            rf_v = rf_imp[idx]
            xgb_v = xgb_imp[idx]
            dominant = "Random Forest" if rf_v > xgb_v else "XGBoost"
            diff_text_parts.append(f"{lbl}: {dominant} modeli bu ozelligi daha onemli buluyor (RF: {rf_v:.4f}, XGB: {xgb_v:.4f})")

        return html.Div([
            html.Div(className="panel", children=[
                dcc.Graph(figure=fig, config={"displayModeBar": False})
            ]),
            html.Div(className="panel", style={"marginTop": "16px"}, children=[
                html.Div(className="panel-title", children=[icon("mdi:compare-horizontal", 16), "Karsilastirma Notlari"]),
                html.Div([
                    html.P("Iki model arasinda en buyuk fark gosteren ozellikler:", style={"color": "#94A3B8", "marginBottom": "12px"}),
                    *[html.Div(className="shap-note-item", children=[
                        icon("mdi:circle-small", 16, "#F59E0B"),
                        html.Span(t, style={"color": "#CBD5E1", "fontSize": "13px"})
                    ]) for t in diff_text_parts]
                ])
            ])
        ])

    return html.Div()


@callback(Output("shap-importance-chart", "children"),
          Output("shap-importance-text", "children"),
          Input("shap-model-select", "value"),
          prevent_initial_call=False)
def update_shap_importance(model):
    if SHAP_DATA is None or model is None:
        return html.Div(), html.Div()

    feature_labels = SHAP_DATA.get('feature_labels', SHAP_DATA.get('feature_cols', []))
    feature_cols = SHAP_DATA.get('feature_cols', [])

    if model == "rf":
        shap_vals = SHAP_DATA['rf_shap_values']
        model_name = "Random Forest"
    else:
        shap_vals = SHAP_DATA['xgb_shap_values']
        model_name = "XGBoost"

    importance = np.abs(shap_vals).mean(axis=0)
    top_idx = np.argsort(importance)[-10:][::-1]
    top_labels = [feature_labels[i] for i in top_idx]
    top_values = importance[top_idx]

    colors = []
    max_val = top_values[0] if len(top_values) > 0 else 1
    for v in top_values:
        ratio = v / max_val
        if ratio > 0.7:
            colors.append("#3B82F6")
        elif ratio > 0.4:
            colors.append("#06B6D4")
        else:
            colors.append("#64748B")

    fig = go.Figure(go.Bar(
        y=top_labels, x=top_values, orientation='h',
        marker_color=colors, text=[f"{v:.4f}" for v in top_values],
        textposition='outside', textfont=dict(size=11, color='#94A3B8')
    ))
    fig.update_layout(**PLT_LAYOUT, height=450,
                      title=f"{model_name} - En Onemli 10 Ozellik (SHAP)",
                      xaxis_title="Ortalama |SHAP Degeri|")
    fig.update_yaxes(autorange="reversed")

    chart = dcc.Graph(figure=fig, config={"displayModeBar": False})

    feature_explanations = {
        'Varyans': 'Sinyal varyansindaki ani degisimler, uydu alt sistemlerindeki beklenmeyen davranislari yansitir.',
        'Standart Sapma': 'Sinyal dagiliminin genisligi; yuksek sapma operasyonel anomaliye isaret eder.',
        'Fark Varyansi': 'Sinyalin turevindeki degiskenlik, ani gecisleri ve bozulmalari yakalar.',
        '2. Fark Varyansi': 'Sinyalin ikinci turevindeki degiskenlik, ivmelenme anomalilerini gosterir.',
        'Tepe Sayisi': 'Sinyaldeki tepe noktasi sayisi; normalden sapma mekanik sorunlara isaret edebilir.',
        'Ortalama Deger': 'Sinyal ortalamasi; kaymalar kalibrasyon sorunlarini gosterir.',
        'RMS Degeri': 'Karekok ortalama sinyal gucu; enerji seviyesindeki anomalileri tespit eder.',
        'Tepeden Tepeye': 'Sinyal genliginin tam araligi; asiri dalgalanmalar anomalidir.',
        'Tepe Faktoru': 'Tepe-RMS orani; impulsif bozulmalari tespit eder.',
        'Sifir Gecis Orani': 'Sinyalin sifir cizgisini gecme sikligi; frekans anomalilerini gosterir.',
        'Basiklik (Kurtosis)': 'Dagilimin sivriligi; yuksek kurtosis ani sapmalara isaret eder.',
        'Carpiklik (Skewness)': 'Dagilimin asimetrisi; tek yonlu sapmalar anomali belirtisidir.',
        'Segment Suresi': 'Veri segmentinin suresi; beklenmeyen sure anomali gostergesidir.',
        'Segment Uzunlugu': 'Veri noktasi sayisi; eksik veya fazla veri anomalidir.',
        'Ornekleme Frekansi': 'Veri toplama hizi; sapma sensor sorunlarini gosterir.',
        'Yumusatilmis Tepe (w=10)': 'Kisa pencere ile yumusatilmis tepe sayisi.',
        'Yumusatilmis Tepe (w=20)': 'Genis pencere ile yumusatilmis tepe sayisi.',
        'Fark Tepe Sayisi': 'Turev sinyalindeki tepe sayisi.',
        '2. Fark Tepe Sayisi': 'Ikinci turev sinyalindeki tepe sayisi.',
        'Bosluk Karesi': 'Veri bosluk karelerinin toplami; veri kaybi gostergesi.',
        'Agirlikli Uzunluk': 'Sure ile agirliklandirilmis segment uzunlugu.',
        'Varyans/Sure': 'Birim zamandaki varyans; normalize edilmis oynaklik.',
        'Varyans/Uzunluk': 'Veri noktasi basina varyans.',
        'Kanal Numarasi': 'Telemetri kanal kimlik numarasi.',
    }

    text_items = []
    for rank, idx in enumerate(top_idx[:3], 1):
        lbl = feature_labels[idx]
        exp = feature_explanations.get(lbl, f"{lbl} ozelligi anomali tespitinde onemli bir rol oynamaktadir.")
        text_items.append(
            html.Div(className="shap-explanation-item", children=[
                html.Div(f"{rank}. {lbl}", className="shap-exp-title"),
                html.Div(f"SHAP Degeri: {importance[idx]:.4f}", className="shap-exp-value"),
                html.Div(exp, className="shap-exp-desc")
            ])
        )

    text_block = html.Div([
        html.Div(className="panel-title", children=[icon("mdi:text-box-outline", 16), "En Onemli Uc Ozellik Aciklamasi"]),
        *text_items
    ])

    return chart, text_block


@callback(Output("shap-waterfall-chart", "children"),
          Output("shap-waterfall-text", "children"),
          Input("shap-anomaly-select", "value"),
          prevent_initial_call=False)
def update_shap_waterfall(selected_idx):
    if SHAP_DATA is None or selected_idx is None:
        return html.Div(), html.Div()

    feature_labels = SHAP_DATA.get('feature_labels', SHAP_DATA.get('feature_cols', []))
    shap_vals = SHAP_DATA['rf_shap_values']
    expected = SHAP_DATA['rf_expected_value']
    X_test = SHAP_DATA['X_test']

    idx = int(selected_idx)
    if idx >= len(shap_vals):
        return html.Div("Gecersiz index."), html.Div()

    vals = shap_vals[idx]
    data_row = X_test[idx]

    abs_vals = np.abs(vals)
    top_idx = np.argsort(abs_vals)[-15:][::-1]
    sorted_labels = [feature_labels[i] for i in top_idx]
    sorted_vals = vals[top_idx]
    sorted_data = data_row[top_idx]

    colors = ['#EF4444' if v > 0 else '#10B981' for v in sorted_vals]

    fig = go.Figure(go.Bar(
        y=sorted_labels, x=sorted_vals, orientation='h',
        marker_color=colors,
        text=[f"{v:+.4f}" for v in sorted_vals],
        textposition='outside', textfont=dict(size=10, color='#94A3B8')
    ))
    fig.update_layout(**PLT_LAYOUT, height=500,
                      title=f"Anomali Aciklamasi - Segment Index: {idx} (Random Forest)",
                      xaxis_title="SHAP Degeri",
                      annotations=[dict(text="Kirmizi: Anomaliye iter | Yesil: Normale iter",
                                        xref="paper", yref="paper", x=0.5, y=-0.08,
                                        showarrow=False, font=dict(size=11, color="#64748B"))])
    fig.update_yaxes(autorange="reversed")
    fig.add_vline(x=0, line_dash="dash", line_color="#4A5568", line_width=1)

    chart = dcc.Graph(figure=fig, config={"displayModeBar": False})

    top3_items = []
    for rank, i in enumerate(top_idx[:3], 1):
        lbl = feature_labels[i]
        val = vals[i]
        direction = "anomaliye dogru itiyor" if val > 0 else "normale dogru itiyor"
        color = "#FCA5A5" if val > 0 else "#86EFAC"
        top3_items.append(
            html.Div(className="shap-explanation-item", children=[
                html.Div(f"{rank}. {lbl}", className="shap-exp-title"),
                html.Div([
                    html.Span(f"SHAP: {val:+.4f}", style={"color": color, "fontFamily": "IBM Plex Mono, monospace", "fontSize": "13px", "fontWeight": "600"}),
                    html.Span(f" - {direction}", style={"color": "#94A3B8", "fontSize": "13px"}),
                ], className="shap-exp-value"),
                html.Div(f"Ozellik degeri: {data_row[i]:.4f}", className="shap-exp-desc")
            ])
        )

    text_block = html.Div([
        html.Div(className="panel-title", children=[icon("mdi:text-box-outline", 16), "En Cok Katkida Bulunan Uc Ozellik"]),
        *top3_items
    ])

    return chart, text_block

@callback(Output("shap-mini-waterfall-container", "children"),
          Input("results-table", "selected_rows"),
          State("results-table", "data"),
          prevent_initial_call=True)
def update_mini_waterfall(selected_rows, table_data):
    if not selected_rows or not table_data or SHAP_DATA is None:
        return html.Div()

    row = table_data[selected_rows[0]]
    segment_no = row.get("Segment", 0)

    feature_labels = SHAP_DATA.get('feature_labels', SHAP_DATA.get('feature_cols', []))
    shap_vals = SHAP_DATA['rf_shap_values']
    X_test = SHAP_DATA['X_test']
    y_test = SHAP_DATA['y_test']

    anomaly_indices = np.where(y_test == 1)[0]
    row_no = row.get("NO", 1) - 1
    if row_no >= len(anomaly_indices):
        return html.Div("Bu segment icin SHAP verisi bulunamadi.", style={"color": "#F59E0B", "padding": "12px"})

    idx = anomaly_indices[row_no]
    if idx >= len(shap_vals):
        return html.Div("SHAP index araligi disinda.", style={"color": "#F59E0B", "padding": "12px"})

    vals = shap_vals[idx]
    abs_vals = np.abs(vals)
    top_idx = np.argsort(abs_vals)[-10:][::-1]
    sorted_labels = [feature_labels[i] for i in top_idx]
    sorted_vals = vals[top_idx]

    colors = ['#EF4444' if v > 0 else '#10B981' for v in sorted_vals]

    fig = go.Figure(go.Bar(
        y=sorted_labels, x=sorted_vals, orientation='h',
        marker_color=colors,
        text=[f"{v:+.4f}" for v in sorted_vals],
        textposition='outside', textfont=dict(size=10, color='#94A3B8')
    ))
    fig.update_layout(**PLT_LAYOUT, height=350,
                      title=f"SHAP Aciklamasi - Segment {segment_no}",
                      xaxis_title="SHAP Degeri")
    fig.update_yaxes(autorange="reversed")
    fig.add_vline(x=0, line_dash="dash", line_color="#4A5568", line_width=1)

    return html.Div(className="panel", children=[
        html.Div(className="panel-title", children=[icon("mdi:brain", 16), f"SHAP Anomali Aciklamasi - Segment {segment_no}"]),
        dcc.Graph(figure=fig, config={"displayModeBar": False}),
        html.Div("Kirmizi: Anomaliye iter | Yesil: Normale iter",
                 style={"color": "#64748B", "fontSize": "11px", "textAlign": "center", "marginTop": "8px"})
    ])


@callback(
    Output("live-interval", "disabled"),
    Output("live-sim-state", "data"),
    Output("live-start", "disabled"),
    Output("live-stop", "disabled"),
    Output("live-indicator", "children"),
    Output("live-indicator", "className"),
    Output("global-live-dot", "className"),
    Output("live-signal-graph", "figure"),
    Output("live-score-graph", "figure"),
    Output("live-alarm-list", "children"),
    Input("live-start", "n_clicks"),
    Input("live-stop", "n_clicks"),
    Input("live-reset", "n_clicks"),
    State("live-sim-state", "data"),
    prevent_initial_call=True
)
def control_live_sim(start_n, stop_n, reset_n, state):
    ctx_id = ctx.triggered_id
    if ctx_id == "live-start":
        state["is_running"] = True
        ind = [html.Span(className="status-dot"), " CANLI"]
        return False, state, True, False, ind, "live-indicator-badge live-active", "topbar-dot blink", no_update, no_update, no_update
    elif ctx_id == "live-stop":
        state["is_running"] = False
        ind = [html.Span(className="status-dot"), " DURDURULDU"]
        return True, state, False, True, ind, "live-indicator-badge", "topbar-dot slow-blink", no_update, no_update, no_update
    elif ctx_id == "live-reset":
        state["index"] = 0
        state["is_running"] = False
        state["anomalies"] = []
        ind = [html.Span(className="status-dot"), " DURDURULDU"]
        
        fig_sig = go.Figure()
        fig_sig.update_layout(**PLT_LAYOUT, height=300,
                              xaxis=dict(showgrid=True, gridcolor="#1C2A3A"), yaxis=dict(showgrid=True, gridcolor="#1C2A3A"))
        fig_sig.add_trace(go.Scatter(x=[], y=[], mode="lines", line=dict(color="#6A8099", width=1.5), name="Sinyal"))
        fig_sig.add_trace(go.Scatter(x=[], y=[], mode="markers", marker=dict(color="#FF3B5C", size=8), name="Anomali"))
        
        fig_score = go.Figure()
        fig_score.update_layout(**PLT_LAYOUT, height=150,
                                xaxis=dict(showgrid=True, gridcolor="#1C2A3A"), yaxis=dict(range=[0, 1.05]))
        fig_score.add_trace(go.Scatter(x=[], y=[], mode="lines", line=dict(color="#00C8FF", width=2), fill='tozeroy', fillcolor='rgba(0,200,255,0.1)', name="Skor"))
        fig_score.add_hline(y=0.5, line_dash="dash", line_color="#FF3B5C")
        
        alarm_msg = html.Div("Anomali Yok", className="no-alarm-msg")
        return True, state, False, True, ind, "live-indicator-badge", "topbar-dot", fig_sig, fig_score, [alarm_msg]
    return no_update

@callback(
    Output("live-signal-graph", "extendData"),
    Output("live-score-graph", "extendData"),
    Output("live-sim-state", "data", allow_duplicate=True),
    Output("live-alarm-list", "children", allow_duplicate=True),
    Output("live-stat-read", "children"),
    Output("live-stat-prog", "children"),
    Output("live-stat-anom", "children"),
    Output("live-stat-last", "children"),
    Output("live-stat-model", "children"),
    Output("live-alarm-count", "children"),
    Input("live-interval", "n_intervals"),
    State("live-sim-state", "data"),
    State("live-channel", "value"),
    State("live-model", "value"),
    State("live-speed", "value"),
    State("live-alarm-list", "children"),
    prevent_initial_call=True
)
def update_live_sim(n_int, state, channel, model_name, speed, current_alarms):
    if not state.get("is_running", False) or LIVE_DATA.empty or not channel or not model_name:
        return no_update
        
    idx = state["index"]
    df_slice = LIVE_DATA[LIVE_DATA['channel'] == channel]
    
    if idx >= len(df_slice):
        state["is_running"] = False
        return no_update
        
    end_idx = min(idx + speed, len(df_slice))
    chunk = df_slice.iloc[idx:end_idx]
    
    state["index"] = end_idx
    
    times = chunk['timestamp'].tolist()
    vals = chunk['value'].tolist()
    
    start_win = max(0, end_idx - 30)
    win_data = df_slice.iloc[start_win:end_idx]['value'].values

    if FEATURE_COLS and len(win_data) >= 3:
        samp = df_slice['sampling'].iloc[0] if 'sampling' in df_slice.columns else 1
        win_df = pd.DataFrame({'value': win_data, 'segment': 0,
                               'channel': channel, 'sampling': samp})
        feats = extract_features_from_raw(win_df)
        X = feats.reindex(columns=FEATURE_COLS, fill_value=0).fillna(0).values
    elif FEATURE_COLS:
        X = np.zeros((1, len(FEATURE_COLS)))
    else:
        X = np.array([[np.mean(win_data) if len(win_data) else 0,
                       np.std(win_data) if len(win_data) else 0, 0, 0]])

    if SCALER:
        try: X = SCALER.transform(X)
        except Exception as e: print("Live scaler hatası:", e)
        
    model = MODELS.get(model_name)
    if not model: return no_update
    
    try:
        pr, sc = predict(model, model_name, X, THRESHOLDS, 1.0)
        score = sc[0]
        t = THRESHOLDS.get(model_name, 0)
        if t == 0: t = 0.5
        norm_score = max(0, min(1, 0.5 + (score - t)/ (abs(t) + 1e-6)))
        if pr[0] == 1: norm_score = max(norm_score, 0.6)
        is_anom = int(pr[0]) == 1
    except Exception as e:
        print("Prediction error:", e)
        norm_score = 0
        is_anom = False
        
    sig_x = [times]
    sig_y = [vals]
    
    anom_x = [[times[-1]]] if is_anom else [[]]
    anom_y = [[vals[-1]]] if is_anom else [[]]
    
    sig_update = (dict(x=sig_x + anom_x, y=sig_y + anom_y), [0, 1], 200)
    score_update = (dict(x=[[times[-1]]], y=[[norm_score]]), [0], 200)
    
    alarms = current_alarms if isinstance(current_alarms, list) and not getattr(current_alarms[0], 'props', {}).get('className', '') == 'no-alarm-msg' else []
    
    if is_anom:
        state["anomalies"].append({"time": times[-1], "score": norm_score})
        sev_class = "critical" if norm_score > 0.8 else "warning"
        sev_text = "KRİTİK" if norm_score > 0.8 else "UYARI"
        
        new_alarm = html.Div(className=f"alarm-card {sev_class}", children=[
            html.Div(className="alarm-card-top", children=[
                html.Span(times[-1].split("T")[-1][:8], className="alarm-time"),
                html.Span(sev_text, className="alarm-badge")
            ]),
            html.Div(className="alarm-card-bottom", children=[
                html.Span(channel, className="alarm-channel"),
                html.Span(f"Skor: {norm_score:.2f}", className="alarm-score")
            ])
        ])
        alarms.insert(0, new_alarm)
        alarms = alarms[:20]
        
    if not alarms:
        alarms = [html.Div("Anomali Yok", className="no-alarm-msg")]
        
    prog = (end_idx / len(df_slice)) * 100 if len(df_slice) > 0 else 0
    n_anom = len(state["anomalies"])
    last_anom = state["anomalies"][-1]["time"].split("T")[-1][:8] if n_anom > 0 else "Yok"
    
    return (
        sig_update, score_update, state, alarms,
        f"OKUNAN: {end_idx}", f"%{prog:.1f}", f"ANOMALİ: {n_anom}", f"SON ALARM: {last_anom}", f"MODEL: {model_name}",
        f"{n_anom} Alarm"
    )


@callback(Output("detail-page-content", "children"),
          Input("selected-anomaly", "data"),
          Input("current-page", "data"),
          State("anomaly-list", "data"),
          State("uploaded-data", "data"),
          prevent_initial_call=False)
def render_anomaly_detail(selected, current_page, all_anomalies, data_json):
    if current_page != "detail":
        return no_update
    if not selected:
        return html.Div(className="info-box", style={"marginTop":"50px"}, children=["Detaylarını görmek istediğiniz anomaliyi Sonuçlar sayfasındaki tablodan seçiniz."])
    
    seg = selected.get("Segment", 0)
    ch = selected.get("Kanal", "N/A")
    score = selected.get("Skor", 0)
    sev = selected.get("Şiddet", "Bilinmiyor")
    idx = selected.get("_idx", 0)
    row_no = selected.get("NO", 0)
    
    badge_color = "#FF3B5C" if sev == "Kritik" else "#FFB300" if sev == "Uyarı" else "#86EFAC"
    
    header = html.Div(className="anomaly-detail-header", children=[
        html.Div(children=[
            html.Div("ANOMALİ DETAY", className="detail-header-title"),
            html.Div(f"SEGMENT  #{seg}", className="detail-header-segment")
        ]),
        html.Div(className="detail-header-right", children=[
            html.Div(sev.upper(), className="severity-badge", style={"borderColor": badge_color, "color": badge_color}),
            html.Div(f"SKOR  {score}", className="score-display"),
            html.Div("MODEL  Topluluk", className="model-display")
        ])
    ])
    
    metrics = dbc.Row([
        dbc.Col(metric_card("mdi:numeric", seg, "Segment Numarası", "blue"), md=2),
        dbc.Col(metric_card("mdi:satellite-uplink", ch, "Kanal Adı", "blue"), md=3),
        dbc.Col(metric_card("mdi:chart-bell-curve", score, "Anomali Skoru", "blue"), md=3),
        dbc.Col(metric_card("mdi:alert-circle", sev, "Şiddet Seviyesi", "red" if sev=="Kritik" else "yellow"), md=2),
        dbc.Col(metric_card("mdi:brain", "1+", "Tespit Eden", "green"), md=2),
    ], className="mb-4 g-3", style={"marginTop": "20px"})
    
    context_fig = go.Figure()
    context_fig.update_layout(**PLT_LAYOUT, height=350, title="Anomali Bağlamı (±100 Segment)", xaxis_title="Segment", yaxis_title="Sinyal")
    
    stats_table_content = html.Div("Veri yüklenemedi.", className="info-box")
    
    if not LIVE_DATA.empty and ch != "N/A":
        ch_data = LIVE_DATA[LIVE_DATA['channel'] == ch].reset_index(drop=True)
        if not ch_data.empty:
            start_idx = max(0, seg - 100)
            end_idx = min(len(ch_data) - 1, seg + 100)
            ctx_df = ch_data.iloc[start_idx:end_idx+1]
            
            context_fig.add_trace(go.Scatter(x=ctx_df['segment'], y=ctx_df['value'], mode='lines', line=dict(color='#6A8099', width=1.5), name='Sinyal'))
            
            anom_df = ch_data[ch_data['segment'] == seg]
            if not anom_df.empty:
                val = anom_df['value'].values[0]
                context_fig.add_trace(go.Scatter(x=[seg], y=[val], mode='markers', marker=dict(color='#FF3B5C', size=10), name='Anomali'))
                context_fig.add_vrect(x0=seg-1, x1=seg+1, fillcolor="rgba(239,68,68,0.08)", line_width=1, line_dash="dash", line_color="#FF3B5C")
    
    df = None
    if data_json:
        df = pd.read_json(io.StringIO(data_json), orient='split')
    elif os.path.exists(DEMO_PATH):
        df = pd.read_parquet(DEMO_PATH)
        
    row_feats = {}
    shap_vals = []
    shap_feats = []
    
    if df is not None and idx < len(df):
        row_data = df.iloc[idx]
        if FEATURE_COLS:
            table_rows = []
            for feat in FEATURE_COLS:
                if feat in df.columns:
                    val = row_data[feat]
                    mean_val = df[feat].mean()
                    std_val = df[feat].std()
                    diff = val - mean_val
                    z_score = diff / (std_val + 1e-9)
                    
                    if abs(z_score) > 2:
                        color = "#FF3B5C" if z_score > 0 else "#86EFAC"
                        sign = "+" if z_score > 0 else ""
                        diff_str = f"{sign}{z_score:.1f}σ"
                    else:
                        color = "#6A8099"
                        diff_str = "Normal"
                        
                    table_rows.append(html.Tr([
                        html.Td(feat), html.Td(f"{mean_val:.2f}"), html.Td(f"{val:.2f}"), html.Td(diff_str, style={"color": color, "fontWeight": "bold" if abs(z_score)>2 else "normal"})
                    ]))
            
            stats_table_content = html.Table(className="custom-table", children=[
                html.Thead(html.Tr([html.Th("Özellik"), html.Th("Normal Ort."), html.Th("Bu Segment"), html.Th("Fark")])),
                html.Tbody(table_rows)
            ])
            
            if "XGBoost" in MODELS:
                try:
                    xgb_model = MODELS["XGBoost"]
                    X_row = row_data[FEATURE_COLS].to_frame().T
                    explainer = get_tree_explainer(xgb_model)
                    sv = explainer.shap_values(X_row)[0]
                    
                    if len(sv.shape) > 1:
                        sv = sv[:, 1]
                    
                    for f, s, v in zip(FEATURE_COLS, sv, X_row.values[0]):
                        shap_vals.append(float(s))
                        shap_feats.append(f)
                        row_feats[f] = v
                except Exception as e:
                    print("SHAP Error:", e)

    signal_analysis = dbc.Row([
        dbc.Col(html.Div(className="panel", children=[
            dcc.Graph(figure=context_fig, config={"displayModeBar": False})
        ]), md=7),
        dbc.Col(html.Div(className="panel", children=[
            html.Div(className="panel-title", children=[icon("mdi:table-compare", 16), " İstatistik Karşılaştırma"]),
            html.Div(stats_table_content, style={"maxHeight": "300px", "overflowY": "auto"})
        ]), md=5)
    ], className="mb-4 g-3")
    
    shap_section = html.Div("Bu model için SHAP değerleri hesaplanamadı.", className="info-box")
    
    if shap_vals and len(shap_vals) > 0:
        sorted_idx = np.argsort(np.abs(shap_vals))[::-1][:10]
        top_feats = [shap_feats[i] for i in sorted_idx]
        top_shaps = [shap_vals[i] for i in sorted_idx]
        
        colors = ["#FF3B5C" if s > 0 else "#86EFAC" for s in top_shaps]
        
        shap_fig = go.Figure()
        shap_fig.add_trace(go.Bar(
            y=top_feats[::-1], x=top_shaps[::-1], orientation='h',
            marker_color=colors[::-1]
        ))
        shap_fig.update_layout(**PLT_LAYOUT, height=350, title="Bu Anomaliye Katkıda Bulunan Özellikler", 
                               margin=dict(l=10, r=20, t=50, b=30), yaxis=dict(tickmode="linear"))
        
        top_positive = [f for f, s in zip(top_feats, top_shaps) if s > 0]
        if len(top_positive) > 0:
            f1 = top_positive[0]
            desc_text = f"Bu segment anomali olarak tespit edildi. Tespitin birincil nedeni '{f1}' değerindeki anormal sapmadır. "
            if len(top_positive) > 1:
                desc_text += f"Buna ek olarak '{top_positive[1]}' özelliği de anomali kararını desteklemiştir."
        else:
            desc_text = "Bu segmentteki anomali kararı birçok özelliğin küçük sapmalarının birleşimiyle alınmıştır."
            
        shap_section = dbc.Row([
            dbc.Col(html.Div(className="panel", children=[dcc.Graph(figure=shap_fig, config={"displayModeBar": False})]), md=6),
            dbc.Col(html.Div(className="panel", style={"height": "100%"}, children=[
                html.Div("NEDEN ANOMALİ?", style={"fontSize": "11px", "letterSpacing": "2px", "color": "#3A5068", "fontWeight": "bold", "marginBottom": "15px"}),
                html.P(desc_text, style={"fontSize": "14px", "lineHeight": "1.6", "color": "#E8F0F8"}),
                html.Div(style={"marginTop": "20px"}, children=[
                    html.Div(className="shap-feat-card", style={"borderLeft": "4px solid #FF3B5C" if top_shaps[0]>0 else "4px solid #86EFAC"}, children=[
                        html.Div(top_feats[0], style={"fontWeight": "bold"}),
                        html.Div(f"SHAP: {top_shaps[0]:.3f}", style={"fontFamily": "IBM Plex Mono", "color": "#FF3B5C" if top_shaps[0]>0 else "#86EFAC"})
                    ])
                ])
            ]), md=6)
        ], className="mb-4 g-3")
        
    action_panel = html.Div(className="panel anomaly-action-panel", children=[
        html.Div(className="nav-buttons", children=[
            html.Button([icon("mdi:chevron-left"), " Önceki Anomali"], id="btn-prev-anomaly", className="btn-nav"),
            html.Button(["Sonraki Anomali ", icon("mdi:chevron-right")], id="btn-next-anomaly", className="btn-nav")
        ]),
        html.Div(f"{row_no} / {len(all_anomalies) if all_anomalies else '?'} Anomali", className="nav-counter"),
        html.Div(className="action-buttons", children=[
            html.Button("Sonuçlara Dön", id="btn-back-results", className="btn-nav"),
            html.Button([icon("mdi:file-pdf-box"), " PDF Rapor"], id="btn-pdf-report", className="btn-action-primary")
        ])
    ])
    
    return html.Div([header, metrics, signal_analysis, shap_section, action_panel])

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib.colors import HexColor

@callback(Output("selected-anomaly", "data", allow_duplicate=True),
          Input("btn-prev-anomaly", "n_clicks"),
          Input("btn-next-anomaly", "n_clicks"),
          State("selected-anomaly", "data"),
          State("anomaly-list", "data"),
          prevent_initial_call=True)
def navigate_anomaly(n_prev, n_next, current, anomaly_list):
    if not current or not anomaly_list: return no_update
    trig = ctx.triggered_id
    
    current_idx = -1
    for i, a in enumerate(anomaly_list):
        if a.get("_idx") == current.get("_idx"):
            current_idx = i
            break
            
    if trig == "btn-prev-anomaly" and current_idx > 0:
        return anomaly_list[current_idx - 1]
    elif trig == "btn-next-anomaly" and current_idx < len(anomaly_list) - 1 and current_idx != -1:
        return anomaly_list[current_idx + 1]
    
    return no_update

@callback(Output("current-page", "data", allow_duplicate=True),
          Input("btn-back-results", "n_clicks"), prevent_initial_call=True)
def back_to_results(n):
    if n: return "results"
    return no_update

@callback(Output("download-pdf-report", "data"),
          Input("btn-pdf-report", "n_clicks"),
          State("selected-anomaly", "data"),
          prevent_initial_call=True)
def generate_pdf_report(n, selected):
    if not n or not selected: return no_update
    
    seg = selected.get("Segment", 0)
    ch = selected.get("Kanal", "N/A")
    score = selected.get("Skor", 0)
    sev = selected.get("Şiddet", "Bilinmiyor")
    
    def create_pdf(file_path):
        c = canvas.Canvas(file_path, pagesize=A4)
        width, height = A4
        
        c.setFillColor(HexColor("#080C14"))
        c.rect(0, height-80, width, 80, stroke=0, fill=1)
        
        c.setFillColor(HexColor("#FFFFFF"))
        c.setFont("Helvetica-Bold", 20)
        c.drawString(40, height-45, f"Anomali Raporu - Segment #{seg}")
        
        c.setFillColor(HexColor("#000000"))
        c.setFont("Helvetica", 12)
        
        y = height - 120
        c.drawString(40, y, f"Kanal: {ch}")
        c.drawString(40, y-25, f"Anomali Skoru: {score}")
        
        sev_color = "#FF3B5C" if sev == "Kritik" else "#FFB300" if sev == "Uyarı" else "#86EFAC"
        c.setFillColor(HexColor(sev_color))
        c.drawString(40, y-50, f"Siddet: {sev.upper()}")
        
        c.setFillColor(HexColor("#000000"))
        c.drawString(40, y-90, "Bu rapor otomatik olarak olusturulmustur.")
        
        c.showPage()
        c.save()
        
    return dcc.send_bytes(
        lambda f: create_pdf(f), 
        f"anomali_raporu_seg_{seg}.pdf"
    )

register_ablation_callbacks(app)
register_power_callbacks(app, ALL_METRICS)
register_synthetic_callbacks(app)
register_esa_pipeline_callbacks(app)

if __name__ == "__main__":
    app.run(
        debug=os.environ.get("DASH_DEBUG", "0") == "1",
        host=os.environ.get("DASH_HOST", "0.0.0.0"),
        port=int(os.environ.get("DASH_PORT", "8050")),
    )
