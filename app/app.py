"""ADCS Monitor - Uydu Telemetri Anomali Tespit Arayuzu"""
import os, sys, json, time, base64, io
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

PLT_LAYOUT = dict(template="plotly_dark", paper_bgcolor="#0A0E1A", plot_bgcolor="#111827",
                   font=dict(family="Inter", color="#94A3B8"), margin=dict(l=40, r=20, t=40, b=30))

# ── Helpers ──
def icon(name, size=18, color=None):
    return DashIconify(icon=name, width=size, color=color or "#64748B")

def metric_card(ic, value, label, color="blue"):
    return html.Div(className=f"metric-card {color}", children=[
        html.Div(icon(ic, 20), className="metric-icon"),
        html.Div(str(value), className="metric-value"),
        html.Div(label, className="metric-label"),
    ])

def nav_item(ic, text, page_id):
    return html.Button(id={"type": "nav", "page": page_id}, n_clicks=0,
                       className="nav-item", children=[icon(ic, 18), html.Span(text)])

# ── Layout ──
sidebar = html.Div(className="sidebar", children=[
    html.Div(className="sidebar-logo", children=[
        html.Div([icon("mdi:satellite-variant", 26, "#06B6D4")], className="logo-icon"),
        html.Div([html.Div("ADCS Monitor", className="logo-text"),
                  html.Div("Anomali Tespit Sistemi", className="logo-sub")])
    ]),
    html.Div(className="sidebar-nav", children=[
        nav_item("mdi:view-dashboard", "Dashboard", "dashboard"),
        nav_item("mdi:upload", "Veri Yukle", "upload"),
        nav_item("mdi:chart-timeline-variant", "Analiz", "analysis"),
        nav_item("mdi:chart-scatter-plot", "Sonuclar", "results"),
        nav_item("mdi:gauge", "Model Performans", "performance"),
    ]),
    html.Div(className="sidebar-footer", children=[
        html.Div(className="status-indicator", children=[
            html.Span(className="status-dot"), html.Span(f"Sistem Aktif  -  {len(MODELS)} model")])
    ])
])

app = Dash(__name__, suppress_callback_exceptions=True,
           external_stylesheets=[dbc.themes.BOOTSTRAP],
           title="ADCS Monitor", update_title=None)

app.layout = html.Div([
    dcc.Store(id="current-page", data="dashboard"),
    dcc.Store(id="uploaded-data"),
    dcc.Store(id="prediction-results"),
    sidebar,
    html.Div(id="page-content", className="main-content"),
    # Results content lives outside page routing to persist
    html.Div(id="results-overlay", className="main-content",
             style={"display": "none"}, children=[
        html.Div(className="page-header", children=[
            html.Div("Sonuclar", className="page-title"),
            html.Div("Anomali tespit sonuclari ve gorsellestime", className="page-subtitle")]),
        html.Div(id="results-content", children=[
            html.Div(className="info-box", children=[
                icon("mdi:chart-scatter-plot", 32, "#3B82F6"), html.Br(), html.Br(),
                "Henuz analiz yapilmadi. Once Analiz sayfasindan islem baslatiniz."])
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
    fig_rank.update_layout(**PLT_LAYOUT, height=360, title="F1 Skor Siralamasi", xaxis_range=[0, 1.05])

    now = time.strftime("%Y-%m-%d %H:%M")
    log_rows = [
        [now, "Sistem baslatildi", "-", "Basarili"],
        [now, "Modeller yuklendi", f"{len(MODELS)} model", "Basarili"],
        [now, "Metrikler yuklendi", f"{len(ALL_METRICS)} kayit", "Basarili"],
    ]

    return html.Div([
        html.Div(className="page-header", children=[
            html.Div("Dashboard", className="page-title"),
            html.Div("Sistem durumu ve model performans ozeti", className="page-subtitle")]),
        dbc.Row([
            dbc.Col(metric_card("mdi:database-outline", n_seg, "Toplam Segment", "blue"), md=3),
            dbc.Col(metric_card("mdi:alert-circle-outline", anom_ratio, "Anomali Orani", "red"), md=3),
            dbc.Col(metric_card("mdi:trophy-outline", "MLP", "En Iyi Model", "green"), md=3),
            dbc.Col(metric_card("mdi:chart-arc", f"{m_best.get('AUC-ROC',0):.3f}", "AUC-ROC", "cyan"), md=3),
        ], className="mb-4 g-3"),
        dbc.Row([
            dbc.Col(html.Div(className="panel", children=[dcc.Graph(figure=fig_heat, config={"displayModeBar": False})]), md=8),
            dbc.Col(html.Div(className="panel", children=[dcc.Graph(figure=fig_rank, config={"displayModeBar": False})]), md=4),
        ], className="mb-4 g-3"),
        html.Div(className="panel", children=[
            html.Div(className="panel-title", children=[icon("mdi:history", 16), "Son Aktivite"]),
            html.Table(className="log-table", children=[
                html.Tbody([html.Tr([html.Td(c) for c in row]) for row in log_rows])
            ])
        ])
    ])


def page_upload():
    return html.Div([
        html.Div(className="page-header", children=[
            html.Div("Veri Yukle", className="page-title"),
            html.Div("CSV veya Parquet formatinda telemetri verisi yukleyin", className="page-subtitle")]),
        dbc.Row([
            dbc.Col([
                dcc.Upload(id="file-upload", children=html.Div(className="upload-area", children=[
                    html.Div(icon("mdi:cloud-upload-outline", 48, "#64748B"), className="upload-icon"),
                    html.Div("Dosyanizi surukleyin veya tiklayin", className="upload-text"),
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
    return html.Div([
        html.Div(className="page-header", children=[
            html.Div("Anomali Analizi", className="page-title"),
            html.Div("Model secimi ve anomali tespit islemleri", className="page-subtitle")]),
        dbc.Row([
            dbc.Col([html.Div(className="panel", children=[
                html.Div(className="panel-title", children=[icon("mdi:tune-vertical", 16), "Analiz Ayarlari"]),
                html.Div("GOZETIMLI", className="section-label"),
                dcc.Checklist(id="sup-models", options=[{"label": n, "value": n} for n in sup],
                              value=sup[:2], className="model-checklist", inputStyle={"marginRight": "8px"}),
                html.Div("GOZETIMSIZ", className="section-label"),
                dcc.Checklist(id="unsup-models", options=[{"label": n, "value": n} for n in unsup],
                              value=["Autoencoder","LOF"] if "Autoencoder" in unsup else unsup[:1],
                              className="model-checklist", inputStyle={"marginRight": "8px"}),
                html.Div("ESIK CARPANI", className="section-label"),
                dcc.Slider(id="threshold-slider", min=0.5, max=1.5, step=0.05, value=1.0,
                           marks={0.5: "0.5", 1.0: "1.0", 1.5: "1.5"},
                           tooltip={"placement": "bottom", "always_visible": True}),
                html.Br(),
                html.Button("Analizi Baslat", id="btn-analyze", n_clicks=0, className="btn-primary"),
            ])], md=3),
            dbc.Col([html.Div(id="analysis-output", className="panel", children=[
                html.Div(className="info-box", children=[
                    icon("mdi:information-outline", 32, "#3B82F6"), html.Br(), html.Br(),
                    "Sol panelden model secip analizi baslatiniz."])
            ])], md=9)
        ], className="g-3")
    ])


def page_results():
    return html.Div([
        html.Div(className="page-header", children=[
            html.Div("Sonuclar", className="page-title"),
            html.Div("Anomali tespit sonuclari ve gorsellestime", className="page-subtitle")]),
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
    fig_radar.update_layout(**PLT_LAYOUT, height=400, polar=dict(bgcolor="#111827", radialaxis=dict(range=[0,1], showticklabels=True, tickfont=dict(size=10))))

    return html.Div([
        html.Div(className="page-header", children=[
            html.Div("Model Performans", className="page-title"),
            html.Div("Tum modellerin karsilastirmali analizi", className="page-subtitle")]),
        html.Div(className="panel mb-4", children=[
            html.Div(className="panel-title", children=[icon("mdi:table", 16), "Metrik Tablosu"]),
            dash_table.DataTable(
                columns=[{"name": "Model", "id": "Model"}] + [{"name": c, "id": c} for c in cols],
                data=[{"Model": n, **{c: f"{ALL_METRICS[n].get(c,0):.4f}" for c in cols}} for n in ALL_METRICS],
                style_header={"backgroundColor": "#111827", "color": "#64748B", "fontWeight": "600",
                               "border": "1px solid #2D3748", "textTransform": "uppercase", "fontSize": "11px"},
                style_cell={"backgroundColor": "#1A2235", "color": "#F1F5F9", "border": "1px solid #2D3748",
                             "fontFamily": "Inter", "fontSize": "12.5px", "padding": "10px"},
                style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": "#161D2F"}],
            )
        ]),
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
    return html.Div([
        dbc.Row([
            dbc.Col(metric_card("mdi:file-document-outline", filename[:25], "Dosya", "blue"), md=3),
            dbc.Col(metric_card("mdi:table-row", f"{df.shape[0]:,}", "Satir", "green"), md=3),
            dbc.Col(metric_card("mdi:table-column", df.shape[1], "Sutun", "cyan"), md=3),
            dbc.Col(metric_card("mdi:alert-circle-outline", anom, "Anomali Orani", "red"), md=3),
        ], className="mb-4 g-3"),
        html.Div(className="panel", children=[
            html.Div(className="panel-title", children=[icon("mdi:table-large", 16), "Veri Onizleme"]),
            dash_table.DataTable(
                columns=[{"name": c, "id": c} for c in df.columns],
                data=df.head(50).to_dict('records'), page_size=15,
                style_header={"backgroundColor": "#111827", "color": "#64748B", "fontWeight": "600",
                               "border": "1px solid #2D3748", "fontSize": "11px"},
                style_cell={"backgroundColor": "#1A2235", "color": "#F1F5F9", "border": "1px solid #2D3748",
                             "fontFamily": "Inter", "fontSize": "12px", "padding": "8px", "maxWidth": "150px",
                             "overflow": "hidden", "textOverflow": "ellipsis"},
                style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": "#161D2F"}],
                sort_action="native", filter_action="native",
            )
        ])
    ])

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
        return html.Div("En az bir model seciniz.", style={"color": "#EF4444"}), no_update

    if data_json:
        df = pd.read_json(io.StringIO(data_json), orient='split')
    elif os.path.exists(DEMO_PATH):
        df = pd.read_parquet(DEMO_PATH)
    else:
        return html.Div("Veri bulunamadi.", style={"color": "#EF4444"}), no_update

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
            return html.Div(f"Scaler hatasi: {e}", style={"color": "#EF4444"}), no_update

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
        dbc.Col(metric_card("mdi:check-circle-outline", len(results), "Basarili Model", "green"), md=4),
        dbc.Col(metric_card("mdi:alert-outline", total, "Toplam Anomali", "red"), md=4),
        dbc.Col(metric_card("mdi:timer-outline", f"{len(selected)}", "Calisan Model", "blue"), md=4),
    ], className="mb-3 g-3")

    return html.Div([summary, html.Div(className="panel-title", children=[icon("mdi:format-list-bulleted",16), "Model Sonuclari"]), *rows]), json.dumps(results)

# Results page dynamic content - triggers on prediction update
@callback(Output("results-content", "children"),
          Input("prediction-results", "data"),
          State("uploaded-data", "data"), prevent_initial_call=True)
def update_results(pred_json, data_json):
    if not pred_json:
        return html.Div(className="info-box", children=["Henuz analiz yapilmadi."])
    results = json.loads(pred_json)
    if not results:
        return html.Div(className="info-box", children=["Sonuc bulunamadi."])

    if data_json:
        df = pd.read_json(io.StringIO(data_json), orient='split')
    elif os.path.exists(DEMO_PATH):
        df = pd.read_parquet(DEMO_PATH)
    else:
        return html.Div("Veri yok.")

    # Score overlay
    fig_scores = go.Figure()
    clrs = ["#3B82F6","#10B981","#EF4444","#F59E0B","#8B5CF6","#06B6D4","#F778A1","#A78BFA","#FB923C"]
    for i, (name, r) in enumerate(results.items()):
        sc = np.array(r["scores"])
        sc_n = (sc - sc.min()) / (sc.max() - sc.min() + 1e-10)
        fig_scores.add_trace(go.Scatter(y=sc_n, mode="lines", name=name, line=dict(color=clrs[i%len(clrs)], width=1.5)))
    fig_scores.update_layout(**PLT_LAYOUT, height=350, title="Anomali Skorlari (Normalize)")

    # Ensemble bar
    counts = {n: r["n_anomaly"] for n, r in results.items()}
    fig_bar = go.Figure(go.Bar(x=list(counts.keys()), y=list(counts.values()),
                                marker_color=[clrs[i%len(clrs)] for i in range(len(counts))]))
    fig_bar.update_layout(**PLT_LAYOUT, height=300, title="Model Basina Anomali Sayisi")

    # Anomaly table
    ensemble = np.zeros(len(df))
    for r in results.values():
        ensemble += np.array(r["preds"])
    ensemble /= max(len(results), 1)
    anom_mask = ensemble > 0.5
    anom_indices = np.where(anom_mask)[0]

    table_data = []
    for idx in anom_indices[:100]:
        sev = "Kritik" if ensemble[idx] > 0.8 else "Uyari" if ensemble[idx] > 0.5 else "Dusuk"
        ch = df.iloc[idx].get("channel", "N/A") if "channel" in df.columns else "N/A"
        table_data.append({"Segment": int(df.iloc[idx].get("segment", idx)),
                           "Kanal": ch, "Skor": f"{ensemble[idx]:.2f}", "Siddet": sev})

    return html.Div([
        html.Div(className="panel mb-4", children=[dcc.Graph(figure=fig_scores, config={"displayModeBar": False})]),
        dbc.Row([
            dbc.Col(html.Div(className="panel", children=[dcc.Graph(figure=fig_bar, config={"displayModeBar": False})]), md=5),
            dbc.Col(html.Div(className="panel", children=[
                html.Div(className="panel-title", children=[icon("mdi:format-list-bulleted", 16), f"Anomali Listesi ({len(table_data)} kayit)"]),
                dash_table.DataTable(
                    columns=[{"name": c, "id": c} for c in ["Segment","Kanal","Skor","Siddet"]],
                    data=table_data, page_size=12,
                    style_header={"backgroundColor": "#111827", "color": "#64748B", "fontWeight": "600", "border": "1px solid #2D3748", "fontSize": "11px"},
                    style_cell={"backgroundColor": "#1A2235", "color": "#F1F5F9", "border": "1px solid #2D3748", "fontFamily": "Inter", "fontSize": "12px", "padding": "8px"},
                    style_data_conditional=[
                        {"if": {"filter_query": '{Siddet} = "Kritik"'}, "backgroundColor": "rgba(239,68,68,0.08)", "color": "#FCA5A5"},
                        {"if": {"filter_query": '{Siddet} = "Uyari"'}, "backgroundColor": "rgba(245,158,11,0.08)", "color": "#FCD34D"},
                        {"if": {"row_index": "odd"}, "backgroundColor": "#161D2F"},
                    ],
                    sort_action="native", filter_action="native",
                )
            ]), md=7),
        ], className="g-3")
    ])


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8050)
