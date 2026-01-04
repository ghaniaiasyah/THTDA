from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.io import load_purchases, load_rfm
from src.filters import month_filter_ui, apply_month_filter, nice_number, nice_currency, nice_percent

# =========================
# Page config
# =========================
st.set_page_config(page_title="RFM Segmentation", layout="wide")

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

st.title("RFM Customer Segmentation")

# =========================
# Load data
# =========================
purchases = load_purchases()
rfm = load_rfm()

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

# Detect columns robustly
uid_p = find_col(purchases, ["user_id", "user", "UserID", "User ID"])
uid_r = find_col(rfm, ["user_id", "user", "UserID", "User ID"])

seg_col = find_col(rfm, ["Segment", "rfm_segment", "RFM_Segment", "segment"])
recency_bucket_col = find_col(rfm, ["Recency_Bucket", "recency_bucket", "Recency Bucket", "recency_bucket_label"])

# raw numeric columns (if present)
recency_col = find_col(rfm, ["Recency", "recency"])
freq_col = find_col(rfm, ["Frequency", "frequency"])
monetary_col = find_col(rfm, ["Monetary", "monetary"])

# score columns (if present)
r_score_col = find_col(rfm, ["R_score", "r_score", "R_Score", "Rscore"])
f_score_col = find_col(rfm, ["F_score", "f_score", "F_Score", "Fscore"])
m_score_col = find_col(rfm, ["M_score", "m_score", "M_Score", "Mscore"])

# =========================
# Guardrails
# =========================
if purchases.empty:
    st.error("purchases.parquet kosong / belum ada.")
    st.stop()

if rfm.empty:
    st.error("rfm.parquet kosong / belum ada.")
    st.stop()

if uid_p is None or uid_r is None:
    st.error("Kolom user_id tidak ketemu di purchases/rfm.")
    st.stop()

if seg_col is None:
    st.error("Kolom Segment (RFM segment) tidak ketemu di rfm.")
    st.stop()

# =========================
# Sidebar Filters
# =========================
st.sidebar.header("Filters")

month_range = month_filter_ui("Month Range", purchases, col="month")
if month_range is not None:
    start_m, end_m = month_range
else:
    start_m, end_m = None, None

segment_options = sorted(rfm[seg_col].dropna().astype(str).unique().tolist())

rec_bucket_options = []
if recency_bucket_col and recency_bucket_col in rfm.columns:
    rec_bucket_options = sorted(rfm[recency_bucket_col].dropna().astype(str).unique().tolist())

with st.sidebar.expander("Recency Bucket", expanded=False):
    rb_all = st.checkbox("Select all recency buckets", value=True, key="rfm_rb_all")
    selected_rb = st.multiselect(
        "Recency Bucket",
        options=rec_bucket_options,
        default=(rec_bucket_options if rb_all else []),
        placeholder="Search bucket...",
        key="rfm_rb_ms",
    )
    if rb_all:
        selected_rb = rec_bucket_options

with st.sidebar.expander("Segment", expanded=False):
    seg_all = st.checkbox("Select all segments", value=True, key="rfm_seg_all")
    selected_segments = st.multiselect(
        "Segment",
        options=segment_options,
        default=(segment_options if seg_all else []),
        placeholder="Search segment...",
        key="rfm_seg_ms",
    )
    if seg_all:
        selected_segments = segment_options

selected_rb = treat_empty_as_all(selected_rb, rec_bucket_options) if rec_bucket_options else []
selected_segments = treat_empty_as_all(selected_segments, segment_options)

# =========================
# Apply filters consistently
# Strategy:
# 1) Purchases filtered by month (time scope)
# 2) Active users from purchases_f
# 3) Filter rfm by segment + recency bucket + active users
# 4) Restrict purchases to those users (so revenue/charts move)
# =========================
purchases_f = purchases.copy()

if start_m is not None and end_m is not None and "month" in purchases_f.columns:
    purchases_f = apply_month_filter(purchases_f, start_m, end_m, col="month")

if "month" in purchases_f.columns:
    purchases_f["month"] = pd.to_datetime(purchases_f["month"], errors="coerce")

purchases_f[uid_p] = purchases_f[uid_p].astype(str)

active_users = purchases_f[uid_p].dropna().unique()
if len(active_users) == 0:
    st.warning("Tidak ada purchase pada rentang bulan yang dipilih. Coba perluas Month Range.")
    # tetap lanjut supaya UI tidak crash, tapi chart akan kosong

rfm_f = rfm.copy()
rfm_f[uid_r] = rfm_f[uid_r].astype(str)

# month affects RFM by active users
if len(active_users) > 0:
    rfm_f = rfm_f[rfm_f[uid_r].isin(active_users)].copy()
else:
    rfm_f = rfm_f.iloc[0:0].copy()

# segment filter
rfm_f = rfm_f[rfm_f[seg_col].astype(str).isin([str(x) for x in selected_segments])].copy()

# recency bucket filter (optional)
if recency_bucket_col and rec_bucket_options and selected_rb:
    rfm_f = rfm_f[rfm_f[recency_bucket_col].astype(str).isin([str(x) for x in selected_rb])].copy()

# restrict purchases to filtered users
allowed_users = rfm_f[uid_r].dropna().unique()
purchases_f = purchases_f[purchases_f[uid_p].isin(allowed_users)].copy()

# =========================
# KPIs
# =========================
total_purchase_customers = int(purchases_f[uid_p].nunique()) if not purchases_f.empty else 0

total_revenue = None
if "price" in purchases_f.columns:
    total_revenue = float(pd.to_numeric(purchases_f["price"], errors="coerce").fillna(0).sum())

# Avg Monetary/Frequency/Recency
avg_monetary = None
if monetary_col and monetary_col in rfm_f.columns:
    avg_monetary = float(pd.to_numeric(rfm_f[monetary_col], errors="coerce").dropna().mean())
else:
    if (not purchases_f.empty) and ("price" in purchases_f.columns):
        urev = (
            purchases_f.assign(price=pd.to_numeric(purchases_f["price"], errors="coerce").fillna(0))
            .groupby(uid_p, as_index=False)["price"].sum()
        )
        avg_monetary = float(urev["price"].mean()) if not urev.empty else None

avg_frequency = None
if freq_col and freq_col in rfm_f.columns:
    avg_frequency = float(pd.to_numeric(rfm_f[freq_col], errors="coerce").dropna().mean())
else:
    if not purchases_f.empty:
        freq_df = purchases_f.groupby(uid_p, as_index=False).size().rename(columns={"size": "freq"})
        avg_frequency = float(freq_df["freq"].mean()) if not freq_df.empty else None

avg_recency = None
if recency_col and recency_col in rfm_f.columns:
    avg_recency = float(pd.to_numeric(rfm_f[recency_col], errors="coerce").dropna().mean())

# KPI row
k1, k2, k3, k4, k5 = st.columns(5)
with k1: st.metric("Total Purchase Customers", safe_metric(total_purchase_customers, lambda v: nice_number(v, sig=4)))
with k2: st.metric("Total Revenue", safe_metric(total_revenue, lambda v: nice_currency(v, sig=4, symbol="$")))
with k3: st.metric("AVG Monetary", safe_metric(avg_monetary, lambda v: nice_currency(v, sig=4, symbol="$")))
with k4: st.metric("AVG Frequency", safe_metric(avg_frequency, lambda v: nice_number(v, sig=4)))
with k5: st.metric("AVG Recency", safe_metric(avg_recency, lambda v: nice_number(v, sig=4)))

st.divider()

# =========================
# 1) Total Customers & Revenue by Segment (combo)
# =========================
c1, c2 = st.columns([1.15, 0.85])

with c1:
    st.subheader("Total Customers and Revenue by Segment")

    if rfm_f.empty or purchases_f.empty:
        st.info("Data kosong setelah filter.")
        st.caption("Segmen bernilai tinggi biasanya jumlahnya kecil namun kontribusi revenue besar — retensi/win-back lebih berdampak daripada akuisisi massal.")
    else:
        cust = rfm_f.groupby(seg_col, as_index=False)[uid_r].nunique().rename(columns={uid_r: "customers"})

        rev_join = purchases_f[[uid_p, "price"]].copy() if "price" in purchases_f.columns else purchases_f[[uid_p]].copy()
        rev_join[uid_p] = rev_join[uid_p].astype(str)

        seg_map = rfm_f[[uid_r, seg_col]].drop_duplicates().rename(columns={uid_r: uid_p})
        comb = rev_join.merge(seg_map, on=uid_p, how="left")

        if "price" in comb.columns:
            comb["price"] = pd.to_numeric(comb["price"], errors="coerce").fillna(0)
            rev = comb.groupby(seg_col, as_index=False)["price"].sum().rename(columns={"price": "revenue"})
        else:
            rev = comb.groupby(seg_col, as_index=False).size().rename(columns={"size": "revenue"})

        out = cust.merge(rev, on=seg_col, how="left").fillna(0)
        out = out.sort_values("customers", ascending=False)

        fig = go.Figure()
        fig.add_trace(go.Bar(x=out[seg_col], y=out["customers"], name="Total Customers"))
        fig.add_trace(go.Scatter(x=out[seg_col], y=out["revenue"], name="Total Revenue", mode="lines+markers", yaxis="y2"))

        fig.update_layout(
            height=420,
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis=dict(title="Total Customers"),
            yaxis2=dict(title="Total Revenue", overlaying="y", side="right"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption("Segmen bernilai tinggi menyumbang revenue besar meski jumlahnya kecil — prioritaskan retensi/win-back segmen high value.")

with c2:
    st.subheader("Revenue by Category")

    if purchases_f.empty or "category_main" not in purchases_f.columns or "price" not in purchases_f.columns:
        st.info("Kategori/price tidak tersedia.")
        st.caption("Kategori membantu konteks strategi retensi: fokuskan rekomendasi/cross-sell pada kategori yang paling berkontribusi terhadap revenue.")
    else:
        tmp = purchases_f.copy()
        tmp["price"] = pd.to_numeric(tmp["price"], errors="coerce").fillna(0)
        cat = tmp.groupby("category_main", as_index=False)["price"].sum().rename(columns={"price": "revenue"})
        cat = cat.sort_values("revenue", ascending=False).head(12)

        fig2 = px.bar(cat.sort_values("revenue", ascending=True), y="category_main", x="revenue", orientation="h")
        fig2.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Total Revenue", yaxis_title="Category")
        st.plotly_chart(fig2, use_container_width=True)

        st.caption("Gunakan kategori teratas sebagai basis personalisasi rekomendasi & bundling untuk meningkatkan repeat purchase pada segmen prioritas.")

st.divider()

# =========================
# 2) Revenue trend by key segments
# =========================
st.subheader("Revenue Trend of Key Customer Segments")

if purchases_f.empty or "month" not in purchases_f.columns or "price" not in purchases_f.columns or rfm_f.empty:
    st.info("Trend revenue tidak tersedia.")
    st.caption("Performa segmen sering paling dibedakan oleh recency — churn menjadi isu utama dibanding nilai transaksi.")
else:
    tmp = purchases_f.copy()
    tmp["month"] = pd.to_datetime(tmp["month"], errors="coerce")
    tmp["price"] = pd.to_numeric(tmp["price"], errors="coerce").fillna(0)

    seg_map = rfm_f[[uid_r, seg_col]].drop_duplicates().rename(columns={uid_r: uid_p})
    tmp = tmp.merge(seg_map, on=uid_p, how="left")

    seg_rev = tmp.groupby(seg_col, as_index=False)["price"].sum().sort_values("price", ascending=False)
    key_segments = seg_rev.head(4)[seg_col].astype(str).tolist()

    tmp2 = tmp[tmp[seg_col].astype(str).isin(key_segments)].copy()
    trend = tmp2.groupby(["month", seg_col], as_index=False)["price"].sum()

    fig3 = px.line(trend, x="month", y="price", color=seg_col)
    fig3.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="Total Revenue")
    st.plotly_chart(fig3, use_container_width=True)

    st.caption("Performa segmen sering paling dibedakan oleh **recency** — indikasi bahwa churn/reaktivasi lebih krusial dibanding sekadar menaikkan AOV.")

st.divider()

# =========================
# 3) Avg R/F/M score by segment
# =========================
st.subheader("AVG R/F/M Score by Segment")

if rfm_f.empty:
    st.info("RFM data kosong.")
    st.caption("Recency adalah pembeda utama performa segmen — gunakan untuk prioritas re-engagement dan strategi win-back.")
else:
    df = rfm_f.copy()

    # Determine score columns
    if r_score_col is None or f_score_col is None or m_score_col is None:
        if recency_col and freq_col and monetary_col:
            df["_r"] = pd.to_numeric(df[recency_col], errors="coerce")
            df["_f"] = pd.to_numeric(df[freq_col], errors="coerce")
            df["_m"] = pd.to_numeric(df[monetary_col], errors="coerce")

            # recency lower better => reverse rank
            df["R_score"] = pd.qcut(-df["_r"], 5, labels=[1, 2, 3, 4, 5]).astype(int)
            df["F_score"] = pd.qcut(df["_f"], 5, labels=[1, 2, 3, 4, 5]).astype(int)
            df["M_score"] = pd.qcut(df["_m"], 5, labels=[1, 2, 3, 4, 5]).astype(int)

            r_score_col_use, f_score_col_use, m_score_col_use = "R_score", "F_score", "M_score"
        else:
            st.info("Tidak ada kolom score maupun numeric R/F/M untuk chart score.")
            r_score_col_use = f_score_col_use = m_score_col_use = None
    else:
        r_score_col_use, f_score_col_use, m_score_col_use = r_score_col, f_score_col, m_score_col

    if r_score_col_use and f_score_col_use and m_score_col_use:
        df[r_score_col_use] = pd.to_numeric(df[r_score_col_use], errors="coerce")
        df[f_score_col_use] = pd.to_numeric(df[f_score_col_use], errors="coerce")
        df[m_score_col_use] = pd.to_numeric(df[m_score_col_use], errors="coerce")

        scores = df.groupby(seg_col, as_index=False).agg(
            avg_r=(r_score_col_use, "mean"),
            avg_f=(f_score_col_use, "mean"),
            avg_m=(m_score_col_use, "mean"),
        )

        long = scores.melt(
            id_vars=[seg_col],
            value_vars=["avg_r", "avg_f", "avg_m"],
            var_name="metric",
            value_name="value"
        )
        long["metric"] = long["metric"].map({"avg_r": "AVG R Score", "avg_f": "AVG F Score", "avg_m": "AVG M Score"})

        fig4 = px.bar(long, x=seg_col, y="value", color="metric", barmode="group")
        fig4.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="Score")
        st.plotly_chart(fig4, use_container_width=True)

        st.caption("Recency cenderung jadi pembeda paling kuat antar segmen — fokuskan program retensi pada segmen bernilai tinggi yang mulai tidak aktif (win-back).")

# =========================
# Quick Analysis (Key Takeaways) — aligned to PPT RFM insight
# =========================
st.divider()
st.subheader("Quick Analysis (Key Takeaways)")

rev_txt = safe_metric(total_revenue, lambda v: nice_currency(v, sig=4, symbol="$"))
cust_txt = safe_metric(total_purchase_customers, lambda v: nice_number(v, sig=4))
rec_txt = safe_metric(avg_recency, lambda v: nice_number(v, sig=4))
freq_txt = safe_metric(avg_frequency, lambda v: nice_number(v, sig=4))

takeaways = [
    "- Segmen bernilai tinggi menyumbang revenue besar meski jumlahnya kecil → **retensi & win-back** lebih berdampak daripada akuisisi massal.",
    f"- Indikasi repeat purchase masih lemah (AVG Frequency saat filter aktif: **{freq_txt}**) → butuh strategi dorong pembelian ke-2 lebih cepat.",
    f"- Performa segmen paling dibedakan oleh **recency** (AVG Recency: **{rec_txt}**) → churn menjadi isu utama.",
    f"- Context (filtered): Purchase customers **{cust_txt}**, Revenue **{rev_txt}**.",
]

st.markdown("\n".join(takeaways))
