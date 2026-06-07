"""Ortak arayüz yardımcıları — sayfalar arası tekrar eden Plotly düzeni ve UI
bileşenleri tek kaynakta toplanır (app.py, power/synthetic/esa/ablation sayfaları)."""
from dash import html
from dash_iconify import DashIconify

PLT_LAYOUT = dict(
    template="plotly_dark", paper_bgcolor="#080C14", plot_bgcolor="#080C14",
    font=dict(family="IBM Plex Sans", color="#94A3B8"),
    margin=dict(l=40, r=20, t=40, b=30),
)


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
