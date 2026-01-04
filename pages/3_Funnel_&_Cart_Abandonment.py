from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.io import (
    load_sessions,
    load_session_dim,
)
from src.filters import (
    month_filter_ui,
    apply_month_filter,
    nice_number,
    nice_percent,
)

# =========================
# Page config
# =========================
st.set_page_config(page_title="Funnel & Cart Abandonment", layout="wide")

st.markdown(
    """
    <style>
      .block-container {
        padding-top: 1.0rem;
        padding-bottom: 1.0rem;
        padding-left: 2.0rem;
        padding-right: 2.0rem;
        max-width: 1500px;
      }
      h1 { margin-bottom: 0.25rem; }
      [data-testid="stMetric"] { padding: 0.35rem 0.55rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Funnel Analysis & Cart Abandonment")

# =========================
# Load data
# =========================
sessions = load_sessions()           # sessions.parquet
session_dim = load_session_dim()     # session_dim.parquet

# =========================
# Helpers
# =========================
def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def treat_empty_as_all(selected: list[str], all_options: list[str]) -> list[str]:
    return all_options if (selected is None or len(selected) == 0) else selected

def safe_metric(val, fmt_fn, dash="-"):
    return dash if val is None else fmt_fn(val)

ses_key = find_col(sessions, ["user_session", "session_id", "Session ID"])
dim_key = find_col(session_dim, ["user_session", "session_id", "Session ID"])

# =========================
# Sidebar filters (Month, Brand, Category)
# =========================
st.sidebar.header("Filters")

base_for_month = sessions if not sessions.empty else session_dim
month_range = month_filter_ui("Month Range", base_for_month, col="month")
if month_range is not None:
    start_m, end_m = month_range
else:
    start_m, end_m = None, None

brand_options: list[str] = []
cat_options: list[str] = []
if not session_dim.empty:
    if "brand" in session_dim.columns:
        brand_options = sorted(session_dim["brand"].dropna().astype(str).unique().tolist())
    if "category_main" in session_dim.columns:
        cat_options = sorted(session_dim["category_main"].dropna().astype(str).unique().tolist())

with st.sidebar.expander("Brand", expanded=False):
    brand_all = st.checkbox("Select all brands", value=True, key="funnel_brand_all")
    selected_brands = st.multiselect(
        "Brand",
        options=brand_options,
        default=(brand_options if brand_all else []),
        placeholder="Search brand...",
        key="funnel_brand_ms",
    )
    if brand_all:
        selected_brands = brand_options

with st.sidebar.expander("Category", expanded=False):
    cat_all = st.checkbox("Select all categories", value=True, key="funnel_cat_all")
    selected_cats = st.multiselect(
        "Category",
        options=cat_options,
        default=(cat_options if cat_all else []),
        placeholder="Search category...",
        key="funnel_cat_ms",
    )
    if cat_all:
        selected_cats = cat_options

selected_brands = treat_empty_as_all(selected_brands, brand_options)
selected_cats = treat_empty_as_all(selected_cats, cat_options)

# =========================
# Apply filters consistently
# 1) Filter session_dim -> dim_f
# 2) allowed_sessions -> sessions_f
# =========================
dim_f = session_dim.copy()

if start_m is not None and end_m is not None and (not dim_f.empty) and ("month" in dim_f.columns):
    dim_f = apply_month_filter(dim_f, start_m, end_m, col="month")

if (not dim_f.empty) and selected_brands and ("brand" in dim_f.columns):
    dim_f = dim_f[dim_f["brand"].astype(str).isin([str(x) for x in selected_brands])]

if (not dim_f.empty) and selected_cats and ("category_main" in dim_f.columns):
    dim_f = dim_f[dim_f["category_main"].astype(str).isin([str(x) for x in selected_cats])]

allowed_sessions = None
if (not dim_f.empty) and dim_key:
    allowed_sessions = dim_f[dim_key].dropna().astype(str).unique()

sessions_f = sessions.copy()

if start_m is not None and end_m is not None and (not sessions_f.empty) and ("month" in sessions_f.columns):
    sessions_f = apply_month_filter(sessions_f, start_m, end_m, col="month")

if (allowed_sessions is not None) and (not sessions_f.empty) and ses_key:
    sessions_f = sessions_f[sessions_f[ses_key].astype(str).isin(allowed_sessions)].copy()

# =========================
# Validate required columns
# =========================
need_cols = {"has_view", "has_cart", "has_purchase"}
if sessions_f.empty or not need_cols.issubset(set(sessions_f.columns)):
    st.error("Data sessions belum punya kolom has_view/has_cart/has_purchase. Cek output preprocess sessions.parquet.")
    st.stop()

# normalize flags
for c in ["has_view", "has_cart", "has_purchase"]:
    sessions_f[c] = pd.to_numeric(sessions_f[c], errors="coerce").fillna(0).astype(int)

# define abandoned flag (prefer existing, else compute)
if "cart_abandoned" in sessions_f.columns:
    sessions_f["cart_abandoned"] = pd.to_numeric(sessions_f["cart_abandoned"], errors="coerce").fillna(0).astype(int)
else:
    sessions_f["cart_abandoned"] = ((sessions_f["has_cart"] == 1) & (sessions_f["has_purchase"] == 0)).astype(int)

# =========================
# KPI (overall within filters)
# =========================
view_sessions = int(sessions_f["has_view"].sum())
cart_sessions = int(sessions_f["has_cart"].sum())
purchase_sessions_raw = int(sessions_f["has_purchase"].sum())

# guard: purchase should not exceed cart in strict funnel
purchase_sessions = min(purchase_sessions_raw, cart_sessions)

view_to_purchase = (purchase_sessions / view_sessions) if view_sessions > 0 else None
view_to_cart = (cart_sessions / view_sessions) if view_sessions > 0 else None
cart_to_purchase = (purchase_sessions / cart_sessions) if cart_sessions > 0 else None

total_abandoned = int(sessions_f["cart_abandoned"].sum())
cart_abandon_rate = ((cart_sessions - purchase_sessions) / cart_sessions) if cart_sessions > 0 else None

# =========================
# KPI row
# =========================
k1, k2, k3, k4, k5 = st.columns(5)
with k1: st.metric("View → Purchase", safe_metric(view_to_purchase, nice_percent))
with k2: st.metric("View → Cart", safe_metric(view_to_cart, nice_percent))
with k3: st.metric("Cart → Purchase", safe_metric(cart_to_purchase, nice_percent))
with k4: st.metric("Abandoned Sessions", safe_metric(total_abandoned, lambda v: nice_number(v, sig=4)))
with k5: st.metric("Cart Abandonment Rate", safe_metric(cart_abandon_rate, nice_percent))

st.divider()

# =========================
# Visuals row: Funnel | Top Abandoned Category | Median Duration
# =========================
c1, c2, c3 = st.columns([1.05, 1.05, 0.9])

# ---------- Funnel ----------
with c1:
    st.subheader("Funnel")

    steps = ["View", "Cart", "Purchase"]
    values = [view_sessions, cart_sessions, purchase_sessions]

    fig_f = go.Figure(
        go.Funnel(
            y=steps,
            x=values,
            textinfo="value+percent initial",
        )
    )
    fig_f.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_f, use_container_width=True)

    # Caption insight (PPT Funnel)
    v2c_txt = safe_metric(view_to_cart, nice_percent)
    st.caption(
        f"Bottleneck utama ada di **View → Cart** (indikator saat filter aktif: **{v2c_txt}**) — fokus optimasi sebaiknya di tahap add-to-cart, bukan checkout."
    )

# ---------- Top Abandoned Category ----------
with c2:
    st.subheader("Top Abandoned Category")

    if dim_f.empty or ("category_main" not in dim_f.columns) or ses_key is None or dim_key is None:
        st.info("session_dim belum siap (butuh category_main + session key).")
        st.caption("Abandonment paling tinggi terjadi di **apparel** dan **sport** — prioritaskan intervensi per kategori untuk dampak konversi tercepat.")
    else:
        # join sessions_f with dim_f to get category_main per session
        dfj = sessions_f[[ses_key, "has_cart", "has_purchase"]].copy()
        dfj[ses_key] = dfj[ses_key].astype(str)

        dd = dim_f[[dim_key, "category_main"]].copy()
        dd[dim_key] = dd[dim_key].astype(str)

        dfj = dfj.merge(dd, left_on=ses_key, right_on=dim_key, how="left")

        dfj["has_cart"] = pd.to_numeric(dfj["has_cart"], errors="coerce").fillna(0).astype(int)
        dfj["has_purchase"] = pd.to_numeric(dfj["has_purchase"], errors="coerce").fillna(0).astype(int)

        # definition-first: abandoned = cart but no purchase
        dfj["is_abandoned"] = (dfj["has_cart"] == 1) & (dfj["has_purchase"] == 0)

        agg = dfj.groupby("category_main", as_index=False).agg(
            cart_sessions=("has_cart", "sum"),
            abandoned=("is_abandoned", "sum"),
        )
        agg = agg.loc[agg["cart_sessions"] > 0].copy()
        agg["abandonment_rate"] = agg["abandoned"] / agg["cart_sessions"]
        agg = agg.sort_values("abandonment_rate", ascending=False).head(12)

        # horizontal bar
        plot_df = agg.sort_values("abandonment_rate", ascending=True).copy()
        fig_ab = px.bar(
            plot_df,
            y="category_main",
            x="abandonment_rate",
            orientation="h",
            text=plot_df["abandonment_rate"].map(lambda v: f"{v:.1%}"),
        )
        fig_ab.update_traces(textposition="outside", cliponaxis=False)
        fig_ab.update_layout(
            height=420,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_tickformat=".0%",
            xaxis_title="Cart Abandonment Rate",
            yaxis_title="Category",
        )
        st.plotly_chart(fig_ab, use_container_width=True)

        # Caption insight (PPT Funnel)
        st.caption(
            "Abandonment paling tinggi terjadi di **apparel** dan **sport** — intervensi per kategori berpotensi memberi dampak konversi paling cepat."
        )

# ---------- Median Session Duration by Status ----------
with c3:
    st.subheader("Median Session Duration by Abandonment Status")

    start_col = find_col(sessions_f, ["session_start_time", "start_time", "session_start"])
    end_col = find_col(sessions_f, ["session_end_time", "end_time", "session_end"])

    if start_col is None or end_col is None:
        st.info("Kolom duration belum ada (session_start_time / session_end_time).")
        st.caption("Tambahan analisis perilaku: bandingkan durasi sesi antara Purchased vs Abandoned untuk indikasi friksi pada tahap cart.")
    else:
        tmp = sessions_f.copy()
        tmp[start_col] = pd.to_datetime(tmp[start_col], errors="coerce")
        tmp[end_col] = pd.to_datetime(tmp[end_col], errors="coerce")
        tmp = tmp.dropna(subset=[start_col, end_col]).copy()

        tmp["duration_min"] = (tmp[end_col] - tmp[start_col]).dt.total_seconds() / 60.0
        tmp["duration_min"] = tmp["duration_min"].clip(lower=0)

        # only sessions that reached cart
        tmp = tmp[tmp["has_cart"] == 1].copy()

        tmp["status"] = np.where(tmp["has_purchase"] == 1, "Purchased", "Abandoned")

        med = tmp.groupby("status", as_index=False)["duration_min"].median()
        order = ["Purchased", "Abandoned"]
        med["status"] = pd.Categorical(med["status"], categories=order, ordered=True)
        med = med.sort_values("status")

        fig_d = px.bar(med, x="status", y="duration_min", text=med["duration_min"].round(0).astype(int))
        fig_d.update_traces(textposition="outside", cliponaxis=False)
        fig_d.update_layout(
            height=420,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Abandonment Status",
            yaxis_title="Median Session Duration (min)",
        )
        st.plotly_chart(fig_d, use_container_width=True)

        st.caption(
            "Perbandingan durasi sesi Purchased vs Abandoned dapat mengindikasikan friksi (mis. kebingungan/pertimbangan lebih lama) sebelum keputusan checkout."
        )

# =========================
# Quick Analysis (Key Takeaways) — aligned to PPT
# =========================
st.divider()
st.subheader("Quick Analysis (Key Takeaways)")

v2p_txt = safe_metric(view_to_purchase, nice_percent)
v2c_txt = safe_metric(view_to_cart, nice_percent)
cab_txt = safe_metric(cart_abandon_rate, nice_percent)
abn_txt = safe_metric(total_abandoned, lambda v: nice_number(v, sig=4))

takeaways = [
    f"- Dari funnel, bottleneck utama ada di **View → Cart** (saat filter aktif: **{v2c_txt}**) — optimasi harus difokuskan pada **add-to-cart stage**, bukan checkout.",
    f"- **Cart abandonment tinggi** (rate: **{cab_txt}**, abandoned sessions: **{abn_txt}**) → *quick win* lewat cart recovery (reminder bertahap + retargeting).",
    f"- Prioritaskan kategori dengan abandonment tertinggi (sering muncul **apparel** & **sport**) untuk intervensi yang paling cepat memberi dampak.",
    f"- Konversi total (View → Purchase) saat filter aktif: **{v2p_txt}** — gunakan metrik ini sebagai baseline untuk A/B test perbaikan product page & incentive.",
]

st.markdown("\n".join(takeaways))
