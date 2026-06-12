"""Veri Yükle sayfasi: layout + callback'ler."""
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
from utils.ui import PLT_LAYOUT, icon, metric_card, stat_strip
from core.constants import (DEMO_PATH, LIVE_DATA_PATH, SHAP_PKL, BENCHMARK_METRICS,
                            PRIMARY_METRIC, DROP_COLS, SUP_MODEL_NAMES, UNSUP_MODEL_NAMES,
                            ANALYSIS_PRESETS)
from core.state import (MODELS, THRESHOLDS, SCALER, TEST_DATA, ALL_METRICS, FEATURE_COLS,
                        LIVE_DATA, SHAP_DATA, get_tree_explainer, best_model)


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
        stat_strip([
            ("Dosya", filename[:25], None, "blue"),
            ("Satır", f"{df.shape[0]:,}", None, "green"),
            ("Sütun", df.shape[1], None, "cyan"),
            ("Anomali Oranı", anom, None, "red"),
        ]),
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
            style_header={"backgroundColor": "#EEF2F8", "color": "#64748B", "fontWeight": "600",
                           "border": "1px solid #E2E8F0", "fontSize": "11px"},
            style_cell={"backgroundColor": "#FFFFFF", "color": "#1E293B", "border": "1px solid #E2E8F0",
                         "fontFamily": "IBM Plex Sans", "fontSize": "12px", "padding": "8px", "maxWidth": "150px",
                         "overflow": "hidden", "textOverflow": "ellipsis"},
            style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": "#F4F6FB"}],
            sort_action="native", filter_action="native",
        )
    ]))
    return html.Div(children)
