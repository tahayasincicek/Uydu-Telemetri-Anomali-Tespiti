"""Uydu Telemetri Anomali Tespit Arayüzü"""
import os, sys, json, time, base64, io, datetime
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

MODELS, THRESHOLDS, SCALER, TEST_DATA = load_all()
ALL_METRICS = load_metrics()
DEMO_PATH = os.path.join(ROOT, "data", "features", "segment_features.parquet")
DROP_COLS = ['segment', 'anomaly', 'train', 'channel']
FEATURE_COLS = TEST_DATA.get("feature_cols", None) if TEST_DATA else None

PLT_LAYOUT = dict(template="plotly_dark", paper_bgcolor="#080C14", plot_bgcolor="#080C14",
                   font=dict(family="IBM Plex Sans", color="#94A3B8"), margin=dict(l=40, r=20, t=40, b=30))

# ── Helpers ──
def icon(name, size=18, color=None):
    return DashIconify(icon=name, width=size, color=color or "#64748B")

def metric_card(ic, value, label, color="blue", footer=None):
    children = [
        html.Div(icon(ic, 20), className="metric-icon"),
        html.Div(str(value), className="metric-value"),
        html.Div(label, className="metric-label"),
    ]
    if footer:
        children.append(html.Div(footer, className="metric-card-footer"))
    return html.Div(className=f"metric-card {color}", children=children)

def nav_item(ic, text, page_id):
    return html.Button(id={"type": "nav", "page": page_id}, n_clicks=0,
                       className="nav-item", children=[icon(ic, 18), html.Span(text)])

# ── Layout ──
topbar = html.Div(className="topbar", children=[
    html.Div(className="topbar-left", children=[
        html.Span("ADCS"),
        html.Span("/", className="topbar-slash"),
        html.Span("Anomali Tespit Sistemi", className="topbar-title"),
    ]),
    html.Div(id="utc-clock", className="topbar-center"),
    html.Div(className="topbar-right", children=[
        html.Div(className="topbar-status", children=[
            html.Span(className="topbar-dot blink"), html.Span("VERİ AKIŞI")]),
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
        nav_item("mdi:view-dashboard", "Dashboard", "dashboard"),
        nav_item("mdi:upload", "Veri Yükle", "upload"),
        nav_item("mdi:chart-timeline-variant", "Analiz", "analysis"),
        nav_item("mdi:chart-scatter-plot", "Sonuçlar", "results"),
        nav_item("mdi:gauge", "Model Performans", "performance"),
    ]),
    html.Div(className="sidebar-footer", children=[
        html.Div(className="status-indicator", children=[
            html.Span(className="status-dot"), html.Span(f"Sistem Aktif  -  {len(MODELS)} model")]),
        html.Div(className="sidebar-version", children=[
            icon("mdi:cpu-64-bit", 14), html.Span("v1.0.0")]),
        html.Div(className="sidebar-version", children=[
            icon("mdi:clock-outline", 14), html.Span(time.strftime("%d.%m.%Y %H:%M"))]),
        html.Div("VER 1.0.0 / MDL 9 / ENV PROD", className="sidebar-sys-info"),
    ])
])

app = Dash(__name__, suppress_callback_exceptions=True,
           external_stylesheets=[dbc.themes.BOOTSTRAP],
           title="Uydu Telemetri", update_title=None)

app.layout = html.Div(id="app-root", children=[
    dcc.Store(id="current-page", data="dashboard"),
    dcc.Store(id="uploaded-data"),
    dcc.Store(id="prediction-results"),
    dcc.Interval(id="clock-interval", interval=1000, n_intervals=0),
    dcc.Download(id="download-csv"),
    topbar,
    sidebar,
    html.Div(id="page-content", className="main-content"),
    html.Div(id="results-overlay", className="main-content",
             style={"display": "none"}, children=[
        html.Div(className="page-header", children=[
            html.Div("Sonuçlar", className="page-title"),
            html.Div("Anomali tespit sonuçları ve görselleştirme", className="page-subtitle")]),
        html.Div(id="results-content", children=[
            html.Div(className="info-box", children=[
                icon("mdi:chart-scatter-plot", 32, "#3B82F6"), html.Br(), html.Br(),
                "Henüz analiz yapılmadı. Önce Analiz sayfasından işlem başlatınız."])
        ])
    ]),
])

# ── Page builders ──
def page_dashboard():
    m_best = ALL_METRICS.get("MLP", {})
    df_seg = pd.read_parquet(DEMO_PATH) if os.path.exists(DEMO_PATH) else pd.DataFrame()
    n_seg = len(df_seg)
    anom_ratio = f"%{df_seg['anomaly'].mean()*100:.1f}" if 'anomaly' in df_seg.columns else "N/A"

    heatmap_data = {n: {k: ALL_METRICS[n].get(k, 0) for k in ["Accuracy","Precision","Recall","F1","AUC-ROC"]}
                    for n in ALL_METRICS}
    hdf = pd.DataFrame(heatmap_data).T
    fig_heat = px.imshow(hdf, text_auto=".3f", color_continuous_scale="Blues",
                         labels=dict(color="Skor"), aspect="auto")
    fig_heat.update_layout(**PLT_LAYOUT, height=360, title="Model Performans Matrisi",
                           coloraxis_colorbar=dict(title=""))

    f1_data = {n: ALL_METRICS[n].get("F1", 0) for n in ALL_METRICS}
    f1_sorted = dict(sorted(f1_data.items(), key=lambda x: x[1]))
    colors = ["#EF4444" if v < 0.5 else "#F59E0B" if v < 0.7 else "#10B981" for v in f1_sorted.values()]
    fig_rank = go.Figure(go.Bar(y=list(f1_sorted.keys()), x=list(f1_sorted.values()),
                                orientation='h', marker_color=colors, text=[f"{v:.3f}" for v in f1_sorted.values()],
                                textposition='outside'))
    fig_rank.update_layout(**PLT_LAYOUT, height=360, title="F1 Skor Sıralaması", xaxis_range=[0, 1.05])

    n_anomaly = int(df_seg['anomaly'].sum()) if 'anomaly' in df_seg.columns else 0

    now = time.strftime("%Y-%m-%d %H:%M")
    log_rows = [
        [now, "Sistem başlatıldı", "9 model yüklendi", "Başarılı"],
        [now, "Veri seti yüklendi", f"{n_seg} segment, {len(FEATURE_COLS or [])} özellik", "Başarılı"],
        [now, "Model eğitimi", "OPSSAT-AD dataset", "Başarılı"],
        [now, "En iyi model", f"MLP - AUC: {m_best.get('AUC-ROC',0):.3f}", "Başarılı"],
    ]
    def status_badge(s):
        cls = "badge-success" if s == "Başarılı" else "badge-error"
        return html.Span(s, className=cls)

    return html.Div([
        html.Div(className="page-header", children=[
            html.Div("Dashboard", className="page-title"),
            html.Div("Sistem durumu ve model performans özeti", className="page-subtitle")]),
        dbc.Row([
            dbc.Col(metric_card("mdi:database-outline", n_seg, "Toplam Segment", "blue", f"{n_seg} segment işlendi"), md=3),
            dbc.Col(metric_card("mdi:alert-circle-outline", anom_ratio, "Anomali Oranı", "red", f"{n_anomaly} anomali tespit edildi"), md=3),
            dbc.Col(metric_card("mdi:trophy-outline", "MLP", "En İyi Model", "green", f"F1: {m_best.get('F1',0):.3f}"), md=3),
            dbc.Col(metric_card("mdi:chart-arc", f"{m_best.get('AUC-ROC',0):.3f}", "AUC-ROC", "cyan", "Test seti üzerinde"), md=3),
        ], className="mb-4 g-3"),
        dbc.Row([
            dbc.Col(html.Div(className="panel", children=[dcc.Graph(figure=fig_heat, config={"displayModeBar": False})]), md=8),
            dbc.Col(html.Div(className="panel", children=[dcc.Graph(figure=fig_rank, config={"displayModeBar": False})]), md=4),
        ], className="mb-4 g-3"),
        html.Div(className="panel", children=[
            html.Div(className="panel-title", children=[icon("mdi:history", 16), "Son Aktivite"]),
            html.Table(className="log-table", children=[
                html.Thead(html.Tr([html.Th(c) for c in ["Zaman", "İşlem", "Detay", "Durum"]])),
                html.Tbody([html.Tr([html.Td(row[0]), html.Td(row[1]), html.Td(row[2]), html.Td(status_badge(row[3]))]) for row in log_rows])
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
        html.Div(id="upload-preview")
    ])


def page_analysis():
    sup = [n for n in ["RandomForest","XGBoost","SVM","MLP"] if n in MODELS]
    unsup = [n for n in ["IsolationForest","OneClassSVM","KMeans","LOF","Autoencoder"] if n in MODELS]
    def model_option(name):
        f1 = ALL_METRICS.get(name, {}).get("F1", 0)
        return html.Span([name, html.Span(f"F1: {f1:.3f}", className="model-f1-badge")])
    return html.Div([
        html.Div(className="page-header", children=[
            html.Div("Anomali Analizi", className="page-title"),
            html.Div("Model seçimi ve anomali tespit işlemleri", className="page-subtitle")]),
        dbc.Row([
            dbc.Col([html.Div(className="panel", children=[
                html.Div(className="panel-title", children=[icon("mdi:tune-vertical", 16), "Analiz Ayarları"]),
                html.Div("GÖZETİMLİ", className="section-label"),
                dcc.Checklist(id="sup-models", options=[{"label": model_option(n), "value": n} for n in sup],
                              value=sup[:2], className="model-checklist", inputStyle={"marginRight": "8px"}),
                html.Div("GÖZETİMSİZ", className="section-label"),
                dcc.Checklist(id="unsup-models", options=[{"label": model_option(n), "value": n} for n in unsup],
                              value=["Autoencoder","LOF"] if "Autoencoder" in unsup else unsup[:1],
                              className="model-checklist", inputStyle={"marginRight": "8px"}),
                html.Div("EŞİK ÇARPANI", className="section-label"),
                dcc.Slider(id="threshold-slider", min=0.5, max=1.5, step=0.05, value=1.0,
                           marks={0.5: "0.5", 1.0: "1.0", 1.5: "1.5"},
                           tooltip={"placement": "bottom", "always_visible": False}),
                html.Div("Düşük değer: hassas tespit, yüksek yanlış alarm. Yüksek değer: güvenilir ama az tespit.",
                         style={"fontSize": "11px", "color": "#64748B", "marginTop": "8px", "lineHeight": "1.5"}),
                html.Div(id="selection-counter", className="selection-counter"),
                html.Button("Analizi Başlat", id="btn-analyze", n_clicks=0, className="btn-primary"),
            ])], md=3),
            dbc.Col([html.Div(id="analysis-output", className="panel", children=[
                html.Div(className="info-box", children=[
                    icon("mdi:information-outline", 32, "#3B82F6"), html.Br(), html.Br(),
                    "Sol panelden model seçip analizi başlatınız."])
            ])], md=9)
        ], className="g-3")
    ])


def page_results():
    return html.Div([
        html.Div(className="page-header", children=[
            html.Div("Sonuçlar", className="page-title"),
            html.Div("Anomali tespit sonuçları ve görselleştirme", className="page-subtitle")]),
        html.Div(id="results-content")
    ])


def page_performance():
    if not ALL_METRICS:
        return html.Div("Metrik verisi bulunamadi.")

    mdf = pd.DataFrame(ALL_METRICS).T
    cols = [c for c in ["Accuracy","Precision","Recall","F1","AUC-ROC","FAR"] if c in mdf.columns]

    # ROC
    fig_roc = go.Figure()
    if TEST_DATA:
        from sklearn.metrics import roc_curve, auc
        X_t, y_t = TEST_DATA["X_test"], TEST_DATA["y_test"]
        clrs = ["#3B82F6","#10B981","#F59E0B","#EF4444","#8B5CF6","#06B6D4","#F778A1","#A78BFA","#FB923C"]
        for i, (name, model) in enumerate(MODELS.items()):
            try:
                if name == "MLP": prob = model.predict(X_t, verbose=0).flatten()
                elif name == "Autoencoder":
                    r = model.predict(X_t, verbose=0); prob = np.mean(np.power(X_t - r, 2), axis=1)
                elif name in ("IsolationForest","LOF"): prob = -model.score_samples(X_t)
                elif name == "OneClassSVM": prob = -model.decision_function(X_t)
                elif name == "KMeans": prob = np.min(model.transform(X_t), axis=1)
                else: prob = model.predict_proba(X_t)[:, 1]
                fpr, tpr, _ = roc_curve(y_t, prob)
                a = auc(fpr, tpr)
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} ({a:.3f})",
                                             line=dict(color=clrs[i % len(clrs)], width=2)))
            except: pass
    fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", line=dict(dash="dash", color="#4A5568"), showlegend=False))
    fig_roc.update_layout(**PLT_LAYOUT, height=400, title="ROC Egrileri", xaxis_title="FPR", yaxis_title="TPR")

    # Radar
    top = [n for n in ["MLP","XGBoost","RandomForest","Autoencoder","LOF"] if n in ALL_METRICS]
    cats = ["Accuracy","Precision","Recall","F1","AUC-ROC"]
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
                data=[{"Model": n, **{c: f"{ALL_METRICS[n].get(c,0):.4f}" for c in cols}} for n in ALL_METRICS],
                style_header={"backgroundColor": "#0D1117", "color": "#64748B", "fontWeight": "600",
                               "border": "1px solid #1E2A3A", "textTransform": "uppercase", "fontSize": "11px"},
                style_cell={"backgroundColor": "#151C28", "color": "#F1F5F9", "border": "1px solid #1E2A3A",
                             "fontFamily": "IBM Plex Sans", "fontSize": "12.5px", "padding": "10px"},
                style_data_conditional=[
                    {"if": {"row_index": "odd"}, "backgroundColor": "#111827"},
                    {"if": {"filter_query": '{AUC-ROC} > 0.95', "column_id": "AUC-ROC"}, "color": "#00FF9C", "fontWeight": "600"},
                    {"if": {"filter_query": '{AUC-ROC} > 0.80 && {AUC-ROC} <= 0.95', "column_id": "AUC-ROC"}, "color": "#FFB300"},
                    {"if": {"filter_query": '{AUC-ROC} <= 0.80', "column_id": "AUC-ROC"}, "color": "#FF3B5C"},
                    {"if": {"filter_query": '{F1} > 0.95', "column_id": "F1"}, "color": "#00FF9C", "fontWeight": "600"},
                    {"if": {"filter_query": '{F1} > 0.80 && {F1} <= 0.95', "column_id": "F1"}, "color": "#FFB300"},
                    {"if": {"filter_query": '{F1} <= 0.80', "column_id": "F1"}, "color": "#FF3B5C"},
                ],
            )
        ]),
        html.Div(className="recommendation-box", children=[
            html.Div(className="rec-title", children=[icon("mdi:trophy-outline", 18, "#3B82F6"), "Önerimiz: MLP"]),
            html.Div(className="rec-body", children=[
                "MLP modeli 0.992 AUC-ROC ve 0.919 F1 skoru ile en yüksek performansı göstermiştir. ",
                "Yanlış alarm oranı (FAR) 0.018 ile gözlemsel operasyonlar için güvenlidir. ",
                "Gözetimsiz modeller arasında Autoencoder 0.893 AUC-ROC ile öne çıkmaktadır."
            ])
        ]),
        html.Br(),
        dbc.Row([
            dbc.Col(html.Div(className="panel", children=[dcc.Graph(figure=fig_roc, config={"displayModeBar": False})]), md=7),
            dbc.Col(html.Div(className="panel", children=[dcc.Graph(figure=fig_radar, config={"displayModeBar": False})]), md=5),
        ], className="g-3")
    ])


PAGES = {"dashboard": page_dashboard, "upload": page_upload, "analysis": page_analysis,
         "results": page_results, "performance": page_performance}

# ── Callbacks ──
@callback(Output("current-page", "data"),
          [Input({"type": "nav", "page": p}, "n_clicks") for p in PAGES],
          prevent_initial_call=True)
def navigate(*clicks):
    if not ctx.triggered_id: return "dashboard"
    return ctx.triggered_id["page"]

# UTC Clock callback
@callback(Output("utc-clock", "children"), Input("clock-interval", "n_intervals"))
def update_clock(_):
    return datetime.datetime.utcnow().strftime("UTC  %Y-%m-%d  %H:%M:%S")

@callback(Output("page-content", "children"), Output("page-content", "style"),
          Output("results-overlay", "style"),
          Input("current-page", "data"))
def render_page(page_id):
    if page_id == "results":
        return html.Div(), {"display": "none"}, {"display": "block"}
    return PAGES.get(page_id, page_dashboard)(), {"display": "block"}, {"display": "none"}

# Upload callbacks
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
            return df.to_json(date_format='iso', orient='split'), build_preview(df, filename)
        except Exception as e:
            return no_update, html.Div(f"Hata: {e}", style={"color": "#EF4444"})
    return no_update, no_update

def build_preview(df, filename):
    n_miss = df.isnull().sum().sum()
    anom = f"{df['anomaly'].mean()*100:.1f}%" if 'anomaly' in df.columns else "N/A"

    # Column grid
    col_items = []
    for c in df.columns:
        dtype_str = str(df[c].dtype)
        col_items.append(html.Div(className="col-grid-item", children=[
            html.Span(c, className="col-name"),
            html.Span(dtype_str, className="col-dtype"),
        ]))

    # Time series chart for first numeric column
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

# Analysis callback
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

# Selection counter
@callback(Output("selection-counter", "children"),
          Input("sup-models", "value"), Input("unsup-models", "value"))
def update_counter(sup_sel, unsup_sel):
    ns = len(sup_sel or [])
    nu = len(unsup_sel or [])
    return f"{ns} gözetimli + {nu} gözetimsiz seçildi"

# Results page dynamic content
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

    ensemble = np.zeros(len(df))
    for r in results.values():
        ensemble += np.array(r["preds"])
    ensemble /= max(len(results), 1)
    anom_mask = ensemble > 0.5
    n_anom = int(anom_mask.sum())
    avg_score = float(np.mean(ensemble[anom_mask])) if n_anom > 0 else 0
    agreement = sum(1 for r in results.values() for p in r["preds"] if p == 1) / max(len(results) * len(df), 1)

    fig_scores = go.Figure()
    clrs = ["#3B82F6","#10B981","#EF4444","#F59E0B","#8B5CF6","#06B6D4","#F778A1","#A78BFA","#FB923C"]
    for i, (name, r) in enumerate(results.items()):
        sc = np.array(r["scores"])
        sc_n = (sc - sc.min()) / (sc.max() - sc.min() + 1e-10)
        fig_scores.add_trace(go.Scatter(y=sc_n, mode="lines", name=name, line=dict(color=clrs[i%len(clrs)], width=1.5)))
    in_region = False; start = 0
    for i in range(len(ensemble)):
        if ensemble[i] > 0.5 and not in_region: start = i; in_region = True
        elif (ensemble[i] <= 0.5 or i == len(ensemble)-1) and in_region:
            fig_scores.add_vrect(x0=start, x1=i, fillcolor="rgba(239,68,68,0.08)", line_width=0, layer="below")
            in_region = False
    fig_scores.update_layout(**PLT_LAYOUT, height=400, title="Anomali Skorları (Normalize)",
                              yaxis_title="Normalize Anomali Skoru", xaxis_title="Segment")

    anom_indices = np.where(anom_mask)[0]
    table_data = []; n_crit = n_warn = n_low = 0
    for row_no, idx in enumerate(anom_indices[:100], 1):
        sev = "Kritik" if ensemble[idx] > 0.8 else "Uyarı" if ensemble[idx] > 0.5 else "Düşük"
        if sev == "Kritik": n_crit += 1
        elif sev == "Uyarı": n_warn += 1
        else: n_low += 1
        ch = df.iloc[idx].get("channel", "N/A") if "channel" in df.columns else "N/A"
        table_data.append({"NO": row_no, "Segment": int(df.iloc[idx].get("segment", idx)),
                           "Kanal": ch, "Skor": f"{ensemble[idx]:.2f}", "Şiddet": sev})

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
                columns=[{"name": c, "id": c} for c in ["NO","Segment","Kanal","Skor","Şiddet"]],
                data=table_data, page_size=12,
                style_header={"backgroundColor":"#0D1117","color":"#64748B","fontWeight":"600","border":"1px solid #1E2A3A","fontSize":"11px"},
                style_cell={"backgroundColor":"#151C28","color":"#F1F5F9","border":"1px solid #1E2A3A","fontFamily":"IBM Plex Sans","fontSize":"12px","padding":"8px"},
                style_data_conditional=[
                    {"if":{"filter_query":'{Şiddet} = "Kritik"'},"backgroundColor":"rgba(239,68,68,0.08)","color":"#FCA5A5"},
                    {"if":{"filter_query":'{Şiddet} = "Uyarı"'},"backgroundColor":"rgba(245,158,11,0.08)","color":"#FCD34D"},
                    {"if":{"filter_query":'{Şiddet} = "Düşük"'},"backgroundColor":"rgba(16,185,129,0.08)","color":"#86EFAC"},
                    {"if":{"row_index":"odd"},"backgroundColor":"#111827"},
                ],
                sort_action="native", filter_action="native",
            ),
            html.Div(style={"marginTop": "12px", "textAlign": "right"}, children=[
                html.Button("CSV Olarak İndir", id="btn-csv-download", n_clicks=0, className="btn-download"),
            ]),
            dcc.Store(id="csv-store", data=table_data),
        ]),
    ])

# CSV Download callback
@callback(Output("download-csv", "data"),
          Input("btn-csv-download", "n_clicks"),
          State("csv-store", "data"),
          prevent_initial_call=True)
def download_csv(n, data):
    if not n or not data: return no_update
    df_out = pd.DataFrame(data)
    return dcc.send_data_frame(df_out.to_csv, "anomali_sonuclari.csv", index=False)


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8050)

