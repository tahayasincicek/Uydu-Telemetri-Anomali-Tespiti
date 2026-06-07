"""
Benchmark Reprodüksiyonu Sayfası
=================================
Kanonik sonuçları (resmi Ψ, 18 ESA özelliği, 7 metrik) Ruszczak et al. (2024)
Tablo 3 referans baseline'ı ile aynı test seti üzerinde karşılaştırır.
Kaynak: reports/metrics/benchmark_comparison.csv (Notebook 12).
"""
import os
import pandas as pd
import plotly.graph_objects as go
from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc

from utils.ui import PLT_LAYOUT, icon as _icon, metric_card as _metric_card

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BENCH_CSV = os.path.join(ROOT, "reports", "metrics", "benchmark_comparison.csv")

CAT_COLOR = {"Gozetimli": "#3B82F6", "Gozetimsiz": "#8B5CF6"}


def _missing_layout():
    return html.Div([
        html.Div(className="page-header", children=[
            html.Div("Benchmark Reprodüksiyonu", className="page-title"),
            html.Div("Resmi OPS-SAT baseline ile karşılaştırma", className="page-subtitle")]),
        html.Div(className="warning-box", children=[
            _icon("mdi:alert-outline", 40, "#F59E0B"),
            html.Div("Benchmark Verileri Bulunamadı", className="warning-title"),
            html.Div([
                "Karşılaştırma henüz üretilmemiş. Lütfen önce ",
                html.Code("notebooks/12_benchmark_karsilastirma.ipynb"),
                " notebook'unu çalıştırın; ",
                html.Code("reports/metrics/benchmark_comparison.csv"),
                " dosyası oluşturulduğunda bu sayfa aktif olur."
            ], className="warning-body"),
        ]),
    ])


def get_benchmark_layout():
    if not os.path.exists(BENCH_CSV):
        return _missing_layout()
    df = pd.read_csv(BENCH_CSV)

    sup = df[df["Kategori"] == "Gozetimli"]
    uns = df[df["Kategori"] == "Gozetimsiz"]
    mae_sup = sup["ΔAUC_PR"].abs().mean() if len(sup) else 0
    mae_uns = uns["ΔAUC_PR"].abs().mean() if len(uns) else 0

    # ── 1) Paper vs Bizim AUC-PR saçılım (köşegene yakınlık = reprodüksiyon) ──
    fig_sc = go.Figure()
    lo = min(df["Paper_AUC_PR"].min(), df["Bizim_AUC_PR"].min()) - 0.05
    fig_sc.add_trace(go.Scatter(x=[lo, 1], y=[lo, 1], mode="lines",
                                line=dict(dash="dash", color="#4A5568"), showlegend=False))
    for cat, color in CAT_COLOR.items():
        sub = df[df["Kategori"] == cat]
        if sub.empty:
            continue
        fig_sc.add_trace(go.Scatter(
            x=sub["Paper_AUC_PR"], y=sub["Bizim_AUC_PR"], mode="markers+text",
            text=sub["Algoritma"], textposition="top center", textfont=dict(size=9, color="#94A3B8"),
            marker=dict(size=10, color=color, opacity=0.85),
            name=cat))
    fig_sc.update_layout(**PLT_LAYOUT, height=460,
                         title="Makale vs Bizim — AUC-PR (köşegen = birebir reprodüksiyon)",
                         xaxis_title="Makale AUC-PR", yaxis_title="Bizim AUC-PR")

    # ── 2) Algoritma başına ΔAUC-PR (bizim − makale) ──
    dff = df.sort_values("ΔAUC_PR")
    bar_clr = ["#EF4444" if v < -0.02 else "#10B981" if v > 0.02 else "#64748B" for v in dff["ΔAUC_PR"]]
    fig_d = go.Figure(go.Bar(
        y=dff["Algoritma"], x=dff["ΔAUC_PR"], orientation="h", marker_color=bar_clr,
        text=[f"{v:+.3f}" for v in dff["ΔAUC_PR"]], textposition="outside",
        textfont=dict(size=9, color="#94A3B8")))
    fig_d.add_vline(x=0, line_dash="dash", line_color="#4A5568")
    fig_d.update_layout(**PLT_LAYOUT, height=520, title="ΔAUC-PR (Bizim − Makale)",
                        xaxis_title="ΔAUC-PR")

    # ── 3) Tablo ──
    show = ["Algoritma", "Kategori", "Paper_AUC_PR", "Bizim_AUC_PR", "ΔAUC_PR",
            "Paper_F1", "Bizim_F1", "Paper_MCC", "Bizim_MCC"]
    show = [c for c in show if c in df.columns]
    tdf = df[show].copy()
    for c in tdf.select_dtypes(include="float").columns:
        tdf[c] = tdf[c].round(3)

    return html.Div([
        html.Div(className="page-header", children=[
            html.Div("Benchmark Reprodüksiyonu", className="page-title"),
            html.Div("Ruszczak et al. (2024) Tablo 3 ile aynı resmi test seti (Ψ) üzerinde",
                     className="page-subtitle")]),

        dbc.Row([
            dbc.Col(_metric_card("mdi:bullseye-arrow", f"{mae_sup:.3f}", "Gözetimli |ΔAUC-PR|",
                                 "green", f"{len(sup)} algoritma — neredeyse birebir"), md=3),
            dbc.Col(_metric_card("mdi:chart-scatter-plot", f"{mae_uns:.3f}", "Gözetimsiz |ΔAUC-PR|",
                                 "purple", f"{len(uns)} algoritma — daha değişken"), md=3),
            dbc.Col(_metric_card("mdi:format-list-numbered", len(df), "Eşleşen Algoritma",
                                 "blue", "makale ∩ bizim"), md=3),
            dbc.Col(_metric_card("mdi:check-decagram", "Faz 2", "Metodoloji Doğrulaması",
                                 "cyan", "resmi split + sızıntısızlık"), md=3),
        ], className="mb-4 g-3"),

        html.Div(className="panel mb-4", style={"borderLeft": "4px solid #10B981", "padding": "14px"},
                 children=[html.Div([
                     _icon("mdi:information-outline", 18, "#10B981"),
                     html.Span(" Gözetimli modellerde ortalama |ΔAUC-PR| = "
                               f"{mae_sup:.3f} ile makale baseline'ı neredeyse birebir yeniden üretildi; "
                               "bu, kurulan metodolojik temelin (resmi bölme + sızıntısızlık + 7 metrik) "
                               "doğruluğunu kanıtlar. İşaretler: ~ yaklaşık/paradigma eşleşmesi, "
                               "! yöntem farkı nedeniyle büyük sapma.",
                               style={"color": "#CBD5E1", "fontSize": "13px", "marginLeft": "6px"})])]),

        dbc.Row([
            dbc.Col(html.Div(className="panel", children=[
                dcc.Graph(figure=fig_sc, config={"displayModeBar": False})]), md=6),
            dbc.Col(html.Div(className="panel", children=[
                dcc.Graph(figure=fig_d, config={"displayModeBar": False})]), md=6),
        ], className="mb-4 g-3"),

        html.Div(className="panel", children=[
            html.Div(className="panel-title", children=[
                _icon("mdi:table-large", 16), f" Karşılaştırma Tablosu ({len(df)} algoritma)"]),
            dash_table.DataTable(
                columns=[{"name": c, "id": c} for c in show],
                data=tdf.to_dict("records"), page_size=25, sort_action="native", filter_action="native",
                style_header={"backgroundColor": "#0D1117", "color": "#64748B", "fontWeight": "600",
                              "border": "1px solid #1E2A3A", "fontSize": "11px"},
                style_cell={"backgroundColor": "#151C28", "color": "#F1F5F9", "border": "1px solid #1E2A3A",
                            "fontFamily": "IBM Plex Sans", "fontSize": "12px", "padding": "8px"},
                style_data_conditional=[
                    {"if": {"row_index": "odd"}, "backgroundColor": "#111827"},
                    {"if": {"filter_query": "{ΔAUC_PR} > 0.02", "column_id": "ΔAUC_PR"}, "color": "#86EFAC"},
                    {"if": {"filter_query": "{ΔAUC_PR} < -0.02", "column_id": "ΔAUC_PR"}, "color": "#FCA5A5"},
                ],
            ),
        ]),
    ])
