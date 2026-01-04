from __future__ import annotations
import pandas as pd
import plotly.express as px

def line_monthly(df: pd.DataFrame, x: str, y: str, title: str):
    if df.empty or x not in df.columns or y not in df.columns:
        return None
    fig = px.line(df, x=x, y=y, markers=True, title=title)
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=320)
    return fig

def bar_top(df: pd.DataFrame, x: str, y: str, title: str, topn: int = 10, orientation="h"):
    if df.empty or x not in df.columns or y not in df.columns:
        return None
    d = df[[x, y]].dropna().sort_values(y, ascending=False).head(topn)
    if orientation == "h":
        fig = px.bar(d.sort_values(y), x=y, y=x, title=title, orientation="h")
    else:
        fig = px.bar(d, x=x, y=y, title=title)
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=320)
    return fig

def funnel_simple_from_month(df_funnel: pd.DataFrame, title: str):
    # expects columns: view_sessions, cart_sessions, purchase_sessions
    if df_funnel.empty:
        return None
    cols = ["view_sessions", "cart_sessions", "purchase_sessions"]
    if not all(c in df_funnel.columns for c in cols):
        return None
    totals = df_funnel[cols].sum(numeric_only=True)
    dd = pd.DataFrame({"step": ["View", "Cart", "Purchase"], "sessions": [totals[cols[0]], totals[cols[1]], totals[cols[2]]]})
    fig = px.bar(dd, x="step", y="sessions", title=title)
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=320)
    return fig
