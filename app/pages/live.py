"""Canli Izleme sayfasi: layout + callback'ler."""
import os
import io
import json
import time
import base64
import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import html, dcc, dash_table, callback, Input, Output, State, no_update, ctx
import dash_bootstrap_components as dbc

from utils.model_loader import predict
from utils.feature_extractor import extract_features_from_raw
from utils.ui import PLT_LAYOUT, icon, metric_card
from core.constants import (DEMO_PATH, LIVE_DATA_PATH, SHAP_PKL, BENCHMARK_METRICS,
                            PRIMARY_METRIC, DROP_COLS, SUP_MODEL_NAMES, UNSUP_MODEL_NAMES,
                            ANALYSIS_PRESETS, channel_label)
from core.state import (MODELS, THRESHOLDS, SCALER, TEST_DATA, ALL_METRICS, FEATURE_COLS,
                        LIVE_DATA, SHAP_DATA, get_tree_explainer, best_model)


def page_live():
    channels = LIVE_DATA['channel'].unique().tolist() if not LIVE_DATA.empty and 'channel' in LIVE_DATA.columns else []
    fast_models = [n for n in ["IsolationForest", "LOF", "OneClassSVM", "KMeans"] if n in MODELS]
    
    # Grafikleri başlangıçta boş iz'lerle (trace) kur — extendData yalnızca var olan
    # iz'leri uzatabilir; aksi halde "Başlat"ta (Sıfırla'ya basılmadan) hiçbir şey çizilmez.
    fig_sig = go.Figure()
    fig_sig.update_layout(**PLT_LAYOUT, height=300,
                          xaxis=dict(showgrid=True, gridcolor="#E2E8F0"), yaxis=dict(showgrid=True, gridcolor="#E2E8F0"))
    fig_sig.add_trace(go.Scatter(x=[], y=[], mode="lines", line=dict(color="#64748B", width=1.5), name="Sinyal"))
    fig_sig.add_trace(go.Scatter(x=[], y=[], mode="markers", marker=dict(color="#FF3B5C", size=8), name="Anomali"))
    fig_score = go.Figure()
    fig_score.update_layout(**PLT_LAYOUT, height=150,
                            xaxis=dict(showgrid=True, gridcolor="#E2E8F0"), yaxis=dict(range=[0, 1.05]))
    fig_score.add_trace(go.Scatter(x=[], y=[], mode="lines", line=dict(color="#0284C7", width=2),
                                   fill='tozeroy', fillcolor='rgba(0,200,255,0.1)', name="Skor"))
    fig_score.add_hline(y=0.5, line_dash="dash", line_color="#FF3B5C")

    return html.Div(className="live-page-container", children=[
        html.Div(className="page-header", children=[
            html.Div("Canlı İzleme", className="page-title"),
            html.Div("Gerçek zamanlı telemetri akışı ve anında anomali tespiti", className="page-subtitle")
        ]),
        
        html.Div(className="panel live-control-panel", children=[
            html.Div(className="live-controls-left", children=[
                html.Div([
                    html.Label("Kanal:"),
                    dcc.Dropdown(id="live-channel", options=[{"label": channel_label(c), "value": c} for c in channels],
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
                              xaxis=dict(showgrid=True, gridcolor="#E2E8F0"), yaxis=dict(showgrid=True, gridcolor="#E2E8F0"))
        fig_sig.add_trace(go.Scatter(x=[], y=[], mode="lines", line=dict(color="#64748B", width=1.5), name="Sinyal"))
        fig_sig.add_trace(go.Scatter(x=[], y=[], mode="markers", marker=dict(color="#FF3B5C", size=8), name="Anomali"))
        
        fig_score = go.Figure()
        fig_score.update_layout(**PLT_LAYOUT, height=150,
                                xaxis=dict(showgrid=True, gridcolor="#E2E8F0"), yaxis=dict(range=[0, 1.05]))
        fig_score.add_trace(go.Scatter(x=[], y=[], mode="lines", line=dict(color="#0284C7", width=2), fill='tozeroy', fillcolor='rgba(0,200,255,0.1)', name="Skor"))
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

    # Anomali iz'ini sinyalle AYNI uzunlukta besle (ayni x'ler, yalniz anomali
    # noktasinda y dolu, gerisi None). Boylece iki iz maxpoints=200 ile ayni
    # pencerede birlikte kayar; eski anomali noktalari cizgiyle birlikte soldan
    # dusulur (aksi halde stale anomali isaretleri x-eksenini gererek cizgiyi
    # "soldan kayboluyor" gibi gosteriyordu).
    anom_marks = [None] * len(times)
    if is_anom and times:
        anom_marks[-1] = vals[-1]
    anom_x = [times]
    anom_y = [anom_marks]

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
                html.Span(channel_label(channel), className="alarm-channel"),
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
