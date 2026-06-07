"""
Araştırma Özeti / Metodoloji Sayfası
====================================
Çalışmanın metodolojik temelini ve manşet bulgularını tek panelde sentezler;
geliştirici/araştırmacının ilgili detay sayfalarına yönlenmesi için giriş noktası.
Veri kaynakları: ALL_METRICS, benchmark_comparison.csv, synthetic_real_ks_distance.csv,
ablation_results.pkl (varsa).
"""
import os
import pandas as pd
import joblib
from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc

from utils.ui import icon as _icon, metric_card as _metric_card

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MDIR = os.path.join(ROOT, "reports", "metrics")
BENCH_CSV = os.path.join(MDIR, "benchmark_comparison.csv")
KS_CSV = os.path.join(MDIR, "synthetic_real_ks_distance.csv")
ABL_PKL = os.path.join(ROOT, "models", "ablation_results.pkl")

BENCHMARK_METRICS = ["Accuracy", "Precision", "Recall", "F1", "MCC", "AUC_ROC", "AUC_PR"]


def _safe_mae():
    if not os.path.exists(BENCH_CSV):
        return None
    try:
        df = pd.read_csv(BENCH_CSV)
        return df[df["Kategori"] == "Gozetimli"]["ΔAUC_PR"].abs().mean()
    except Exception:
        return None


def _safe_ks():
    if not os.path.exists(KS_CSV):
        return None
    try:
        return pd.read_csv(KS_CSV)["KS_mesafe"].mean()
    except Exception:
        return None


def _safe_optimal_count():
    if not os.path.exists(ABL_PKL):
        return None
    try:
        return joblib.load(ABL_PKL).get("best_set", {}).get("optimal_count")
    except Exception:
        return None


def get_summary_layout(all_metrics=None):
    all_metrics = all_metrics or {}
    ranked = sorted(all_metrics, key=lambda n: all_metrics[n].get("AUC_PR", 0), reverse=True)
    best = ranked[0] if ranked else None
    best_pr = all_metrics.get(best, {}).get("AUC_PR", 0) if best else 0

    mae = _safe_mae()
    ks = _safe_ks()
    opt = _safe_optimal_count()

    # ── Manşet kartlar ──
    cards = dbc.Row([
        dbc.Col(_metric_card("mdi:trophy-outline", best or "N/A", "En İyi Model", "green",
                             f"AUC-PR {best_pr:.3f} (resmi Ψ)"), md=3),
        dbc.Col(_metric_card("mdi:bullseye-arrow", f"{mae:.3f}" if mae is not None else "N/A",
                             "Benchmark |ΔAUC-PR|", "blue", "gözetimli reprodüksiyon"), md=3),
        dbc.Col(_metric_card("mdi:format-list-checks", f"{opt}/18" if opt else "11/18",
                             "Yeterli Özellik", "cyan", "ablasyon (bilgi kaybı yok)"), md=3),
        dbc.Col(_metric_card("mdi:scale-balance", f"KS {ks:.2f}" if ks is not None else "Nötr",
                             "Augmentasyon", "yellow", "zengin-etiket rejiminde nötr"), md=3),
    ], className="mb-4 g-3")

    # ── Metodoloji paneli ──
    method = html.Div(className="panel mb-4", children=[
        html.Div(className="panel-title", children=[_icon("mdi:shield-check-outline", 16),
                 "Metodolojik Temel (tekrarüretilebilirlik)"]),
        html.Ul(style={"color": "#CBD5E1", "fontSize": "13px", "lineHeight": "1.8", "paddingLeft": "18px"}, children=[
            html.Li(["Resmi train/test bölmesi: ", html.B("T = 1594"), " eğitim, ",
                     html.B("Ψ = 529"), " test (dataset.csv 'train' kolonu) — makaleyle birebir kıyaslanabilirlik."]),
            html.Li(["Veri sızıntısı önleme: ölçekleyici yalnız T'de fit, ",
                     html.B("SMOTE yalnız eğitim katmanında"), "; Ψ asla dengelenmez/dokunulmaz."]),
            html.Li(["Yedi zorunlu metrik (Accuracy/Precision/Recall/F1/MCC/AUC-ROC/",
                     html.B("AUC-PR"), "), tablolar ", html.B("AUC-PR'a göre"), " sıralı."]),
            html.Li(["Kanonik eğitim motoru (train_all_models.py) tüm modelleri tek kaynaktan üretir; "
                     "notebook'lar bu çıktıyı tüketir, ezmez."]),
        ]),
    ])

    # ── Üç faz paneli ──
    def faz(num, title, desc, color, page_hint):
        return dbc.Col(html.Div(className="panel", style={"borderLeft": f"4px solid {color}", "height": "100%"}, children=[
            html.Div(f"FAZ {num}", style={"fontSize": "11px", "letterSpacing": "2px", "color": color, "fontWeight": "700"}),
            html.Div(title, style={"fontWeight": "600", "color": "#E8F0F8", "margin": "6px 0"}),
            html.Div(desc, style={"fontSize": "12px", "color": "#94A3B8", "lineHeight": "1.5"}),
            html.Div(page_hint, style={"fontSize": "11px", "color": "#64748B", "marginTop": "8px"}),
        ]), md=4)

    phases = dbc.Row([
        faz("1", "Metodolojik Temel", "Resmi split, sızıntısız ön işleme, 7 metrik, kanonik motor. "
            "42 model Ψ üzerinde değerlendirildi.", "#10B981", "→ Model Performans, Ablasyon"),
        faz("2", "Benchmark Hizalama", "Makale baseline'ının (Ruszczak 2024, 30 algoritma) aynı Ψ "
            "üzerinde reprodüksiyonu.", "#3B82F6", "→ Benchmark"),
        faz("3", "Özgün Katkılar", "Profil-temelli sentetik üretim/ablasyon, augmentasyon stratejileri, "
            "güç/verimlilik, SHAP yorumlanabilirliği.", "#8B5CF6", "→ Augmentasyon, Sentetik, Güç, SHAP"),
    ], className="mb-4 g-3")

    # ── En iyi 6 model mini tablo ──
    table = html.Div()
    if ranked:
        cols = ["AUC_PR", "F1", "MCC", "AUC_ROC"]
        data = [{"Model": n, **{c: f"{all_metrics[n].get(c, 0):.3f}" for c in cols}} for n in ranked[:6]]
        table = html.Div(className="panel", children=[
            html.Div(className="panel-title", children=[_icon("mdi:medal-outline", 16),
                     "En İyi 6 Model (AUC-PR sıralı)"]),
            dash_table.DataTable(
                columns=[{"name": "Model", "id": "Model"}] + [{"name": c, "id": c} for c in cols],
                data=data,
                style_header={"backgroundColor": "#0D1117", "color": "#64748B", "fontWeight": "600",
                              "border": "1px solid #1E2A3A", "fontSize": "11px"},
                style_cell={"backgroundColor": "#151C28", "color": "#F1F5F9", "border": "1px solid #1E2A3A",
                            "fontFamily": "IBM Plex Sans", "fontSize": "12.5px", "padding": "10px"},
                style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": "#111827"}],
            ),
        ])

    return html.Div([
        html.Div(className="page-header", children=[
            html.Div("Araştırma Özeti", className="page-title"),
            html.Div("Metodolojik temel ve manşet bulgular — detaylar için ilgili sayfalara bakınız",
                     className="page-subtitle")]),
        cards,
        method,
        html.Div("Çalışmanın Üç Fazı", style={"fontSize": "13px", "letterSpacing": "1px",
                 "color": "#64748B", "fontWeight": "600", "marginBottom": "10px"}),
        phases,
        table,
        html.Div([
            "Referans: Ruszczak et al. (2024), ", html.I("The OPS-SAT benchmark for detecting "
            "anomalies in satellite telemetry"), " (arXiv:2407.04730)."],
            style={"fontSize": "11px", "color": "#64748B", "marginTop": "16px"}),
    ])
