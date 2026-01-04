from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from src.io import load_purchases, load_rfm, read_parquet_safe
from src.filters import (
    month_filter_ui,
    apply_month_filter,
    nice_number,
    nice_percent,
    nice_currency,
)

# =========================
# Page config
# =========================
st.set_page_config(page_title="Cohort Retention & CLV", layout="wide")

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
      .smallnote { color:#6b7280; font-size: 12px; margin-top: -0.25rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Cohort Retention & CLV Analysis")

# =========================
# Load data
# =========================
purchases = load_purchases()
rfm = load_rfm()
clv = read_parquet_safe("clv.parquet")

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

# Required columns
uid_p = find_col(purchases, ["user_id", "user", "UserID", "User ID"])
uid_r = find_col(rfm, ["user_id", "user", "UserID", "User ID"])
uid_c = find_col(clv, ["user_id", "user", "UserID", "User ID"])

seg_rfm_col = find_col(rfm, ["Segment", "rfm_segment", "RFM_Segment", "segment"])
clv_value_col = find_col(clv, ["predicted_clv_3m", "clv_3m", "clv", "CLV_3M", "Predicted_CLV_3M"])
clv_seg_col = find_col(clv, ["clv_segment", "CLV_Segment", "segment_clv", "Segment"])

# Basic guardrails
if purchases.empty:
    st.error("purchases.parquet kosong / belum ada.")
    st.stop()
if rfm.empty or uid_r is None or seg_rfm_col is None:
    st.error("rfm.parquet belum siap (butuh user_id + Segment).")
    st.stop()
if clv.empty or uid_c is None or clv_value_col is None:
    st.error("clv.parquet belum siap (butuh user_id + predicted_clv_3m).")
    st.stop()
if uid_p is None:
    st.error("Kolom user_id tidak ketemu di purchases.")
    st.stop()

# Normalize types
purchases = purchases.copy()
rfm = rfm.copy()
clv = clv.copy()

purchases[uid_p] = purchases[uid_p].astype(str)
rfm[uid_r] = rfm[uid_r].astype(str)
clv[uid_c] = clv[uid_c].astype(str)

# =========================
# Sidebar Filters
# =========================
st.sidebar.header("Filters")

month_range = month_filter_ui("Month Range", purchases, col="month")
if month_range is not None:
    start_m, end_m = month_range
else:
    start_m, end_m = None, None

# Segment filters
rfm_segments = sorted(rfm[seg_rfm_col].dropna().astype(str).unique().tolist())

with st.sidebar.expander("Customer Segment", expanded=False):
    seg_all = st.checkbox("Select all segments", value=True, key="cohort_seg_all")
    selected_segments = st.multiselect(
        "Customer Segment",
        options=rfm_segments,
        default=(rfm_segments if seg_all else []),
        placeholder="Search segment...",
        key="cohort_seg_ms",
    )
    if seg_all:
        selected_segments = rfm_segments

selected_segments = treat_empty_as_all(selected_segments, rfm_segments)

# CLV segment options
if clv_seg_col and clv_seg_col in clv.columns:
    clv_segments = sorted(clv[clv_seg_col].dropna().astype(str).unique().tolist())
else:
    clv_segments = []

with st.sidebar.expander("CLV Segment", expanded=False):
    clv_all = st.checkbox("Select all CLV segments", value=True, key="cohort_clv_all")
    selected_clv_segments = st.multiselect(
        "CLV Segment",
        options=clv_segments,
        default=(clv_segments if clv_all else []),
        placeholder="Search CLV segment...",
        key="cohort_clv_ms",
    )
    if clv_all and clv_segments:
        selected_clv_segments = clv_segments

# =========================
# Filter pipeline (connect everything)
# =========================
purchases_f = purchases.copy()
if start_m is not None and end_m is not None and "month" in purchases_f.columns:
    purchases_f = apply_month_filter(purchases_f, start_m, end_m, col="month")

purchases_f["month"] = pd.to_datetime(purchases_f["month"], errors="coerce")
purchases_f = purchases_f.dropna(subset=["month"]).copy()

active_users = purchases_f[uid_p].dropna().unique()

rfm_f = rfm[rfm[uid_r].isin(active_users)].copy()
rfm_f = rfm_f[rfm_f[seg_rfm_col].astype(str).isin([str(x) for x in selected_segments])].copy()
allowed_users = rfm_f[uid_r].dropna().unique()

clv_f = clv[clv[uid_c].isin(allowed_users)].copy()
clv_f[clv_value_col] = pd.to_numeric(clv_f[clv_value_col], errors="coerce")
clv_f = clv_f.dropna(subset=[clv_value_col]).copy()

# If CLV segment column missing, create default quantile-based segment (within filtered users)
if clv_seg_col is None or clv_seg_col not in clv_f.columns:
    q1 = clv_f[clv_value_col].quantile(0.33)
    q2 = clv_f[clv_value_col].quantile(0.67)
    clv_f["clv_segment"] = np.where(
        clv_f[clv_value_col] >= q2, "High CLV",
        np.where(clv_f[clv_value_col] >= q1, "Mid CLV", "Low CLV")
    )
    clv_seg_col_use = "clv_segment"
else:
    clv_seg_col_use = clv_seg_col

# Apply CLV segment selection
if clv_segments:
    selected_clv_segments = treat_empty_as_all(selected_clv_segments, clv_segments)
    clv_f = clv_f[clv_f[clv_seg_col_use].astype(str).isin([str(x) for x in selected_clv_segments])].copy()
else:
    gen_segments = sorted(clv_f[clv_seg_col_use].dropna().astype(str).unique().tolist())
    if not selected_clv_segments:
        selected_clv_segments = gen_segments
    clv_f = clv_f[clv_f[clv_seg_col_use].astype(str).isin([str(x) for x in selected_clv_segments])].copy()

allowed_users2 = clv_f[uid_c].dropna().unique()
purchases_f = purchases_f[purchases_f[uid_p].isin(allowed_users2)].copy()

# =========================
# Cohort Month selector
# =========================
if purchases_f.empty:
    st.info("Data purchase kosong setelah filter. Coba longgarkan filter.")
    st.stop()

first_purchase = (
    purchases_f.groupby(uid_p, as_index=False)["month"]
              .min()
              .rename(columns={"month": "cohort_month"})
)
first_purchase["cohort_month"] = first_purchase["cohort_month"].dt.to_period("M").dt.to_timestamp()

cohort_month_options = ["All"] + sorted(first_purchase["cohort_month"].dropna().dt.strftime("%Y-%m").unique().tolist())

st.sidebar.header("Cohort")
selected_cohort_month = st.sidebar.selectbox("Cohort Month", cohort_month_options, index=0)

# =========================
# Build cohort table
# =========================
pum = purchases_f[[uid_p, "month"]].drop_duplicates().copy()
pum["month"] = pum["month"].dt.to_period("M").dt.to_timestamp()

dfc = pum.merge(first_purchase, on=uid_p, how="left").dropna(subset=["cohort_month"]).copy()

if selected_cohort_month != "All":
    target = pd.to_datetime(selected_cohort_month + "-01", errors="coerce")
    dfc = dfc[dfc["cohort_month"] == target].copy()

dfc["cohort_index"] = (
    (dfc["month"].dt.year - dfc["cohort_month"].dt.year) * 12
    + (dfc["month"].dt.month - dfc["cohort_month"].dt.month)
).astype(int)

dfc = dfc[dfc["cohort_index"] >= 0].copy()

cohort_size = (
    dfc[dfc["cohort_index"] == 0]
    .groupby("cohort_month", as_index=False)[uid_p]
    .nunique()
    .rename(columns={uid_p: "cohort_size"})
)

retained = (
    dfc.groupby(["cohort_month", "cohort_index"], as_index=False)[uid_p]
       .nunique()
       .rename(columns={uid_p: "retained_users"})
)

ret = retained.merge(cohort_size, on="cohort_month", how="left")
ret["retention_rate"] = (ret["retained_users"] / ret["cohort_size"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)

# show horizon 0..5 by default (M0–M5)
max_horizon = int(min(5, ret["cohort_index"].max() if not ret.empty else 0))
ret_small = ret[ret["cohort_index"].between(0, max_horizon)].copy()

matrix = (
    ret_small.pivot(index="cohort_month", columns="cohort_index", values="retention_rate")
    .fillna(0.0)
)

# =========================
# KPI cards
# =========================
m1 = ret_small[ret_small["cohort_index"] == 1].copy()
avg_m1 = float(m1["retention_rate"].mean()) if not m1.empty else None

avg_clv = float(clv_f[clv_value_col].mean()) if not clv_f.empty else None
cap_p99 = float(clv_f[clv_value_col].quantile(0.99)) if not clv_f.empty else None
pct_at_cap = None
if (cap_p99 is not None) and (not clv_f.empty):
    pct_at_cap = float((clv_f[clv_value_col] >= cap_p99).mean())

k1, k2, k3, k4 = st.columns(4)
with k1: st.metric("AVG M1 Retention Rate", safe_metric(avg_m1, nice_percent))
with k2: st.metric("AVG Predicted CLV (3M)", safe_metric(avg_clv, lambda v: nice_currency(v, sig=4, symbol="$")))
with k3: st.metric("CLV Cap (P99)", safe_metric(cap_p99, lambda v: nice_currency(v, sig=4, symbol="$")))
with k4: st.metric("% Users at/above P99", safe_metric(pct_at_cap, nice_percent))

st.divider()

# =========================
# Top row: Cohort Matrix + M1 Trend
# =========================
c1, c2 = st.columns([1.25, 1.0])

with c1:
    st.subheader("Cohort Matrix")
    st.markdown('<div class="smallnote">M1 retention rendah konsisten → indikasi churn terjadi sangat awal</div>', unsafe_allow_html=True)

    if matrix.empty:
        st.info("Cohort matrix kosong setelah filter.")
        st.caption("Retention bulan pertama (M1) tetap rendah di berbagai cohort — churn terjadi sangat awal, sehingga onboarding & re-engagement dini perlu diprioritaskan.")
    else:
        mat = matrix.copy()
        mat.index = mat.index.strftime("%b %Y")
        mat.columns = [f"M{c}" for c in mat.columns]  # clearer labels

        fig_h = px.imshow(
            mat,
            text_auto=".1%",
            aspect="auto",
            labels=dict(x="Month Index", y="Cohort Month", color="Retention"),
        )
        fig_h.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_h, use_container_width=True)

        st.caption(
            "Retention bulan pertama (M1) tetap rendah di berbagai cohort — churn terjadi sangat awal, sehingga onboarding pasca pembelian & re-engagement dini perlu diprioritaskan."
        )

with c2:
    st.subheader("M1 Retention Trend")

    m1_trend = ret[ret["cohort_index"] == 1].copy()
    if m1_trend.empty:
        st.info("Tidak ada data M1 retention.")
        st.caption("M1 retention yang rendah menunjukkan kebutuhan program win-back lebih cepat (jam/hari pertama setelah add-to-cart atau pembelian).")
    else:
        m1_trend = m1_trend.sort_values("cohort_month")
        fig_line = px.line(m1_trend, x="cohort_month", y="retention_rate", markers=True)
        fig_line.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
        fig_line.update_yaxes(tickformat=".0%", title="M1 Retention Rate")
        fig_line.update_xaxes(title="Month")
        st.plotly_chart(fig_line, use_container_width=True)

        st.caption(
            "M1 retention cenderung stabil rendah dari waktu ke waktu — perbaikan terbesar kemungkinan datang dari intervensi awal (post-purchase onboarding & second purchase trigger)."
        )

st.divider()

# =========================
# Bottom row: Avg CLV by segment | CLV segment distribution | Top user list
# =========================
b1, b2, b3 = st.columns([1.0, 1.0, 1.2])

rfm_map = rfm_f[[uid_r, seg_rfm_col]].drop_duplicates().rename(columns={uid_r: uid_c})
clv_join = clv_f.merge(rfm_map, on=uid_c, how="left")

with b1:
    st.subheader("AVG Predicted CLV by RFM Segment")

    if clv_join.empty or seg_rfm_col not in clv_join.columns:
        st.info("CLV by segment kosong.")
        st.caption("CLV biasanya terkonsentrasi di segmen high-value (mis. Champion / Cannot Lose Them) — retensi segmen ini paling berdampak ke revenue.")
    else:
        seg_clv = (
            clv_join.groupby(seg_rfm_col, as_index=False)[clv_value_col]
                    .mean()
                    .rename(columns={clv_value_col: "avg_clv"})
                    .sort_values("avg_clv", ascending=False)
        )
        plot_df = seg_clv.sort_values("avg_clv", ascending=True)
        fig_seg = px.bar(
            plot_df,
            y=seg_rfm_col,
            x="avg_clv",
            orientation="h",
            text=plot_df["avg_clv"].map(lambda v: f"${v:,.0f}"),
        )
        fig_seg.update_traces(textposition="outside", cliponaxis=False)
        fig_seg.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Avg CLV")
        st.plotly_chart(fig_seg, use_container_width=True)

        st.caption(
            "CLV terkonsentrasi pada segmen bernilai tinggi — prioritaskan retensi (VIP/perks) dan win-back selektif untuk segmen tersebut."
        )

with b2:
    st.subheader("CLV Segment Distribution")

    if clv_f.empty or clv_seg_col_use not in clv_f.columns:
        st.info("CLV segment distribution kosong.")
        st.caption("Distribusi CLV membantu menentukan fokus program: mayoritas Low/Mid butuh nudges, sementara High butuh perlakuan premium.")
    else:
        dist = (
            clv_f[clv_seg_col_use].astype(str)
                .value_counts()
                .reset_index()
        )
        dist.columns = ["clv_segment", "users"]
        fig_dist = px.bar(dist, x="clv_segment", y="users")
        fig_dist.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="CLV Segment", yaxis_title="Total Users")
        st.plotly_chart(fig_dist, use_container_width=True)

        st.caption(
            "Distribusi CLV membantu prioritisasi: segmen High CLV paling worth untuk retensi premium, sedangkan Low/Mid fokus pada peningkatan repeat purchase."
        )

with b3:
    st.subheader("Top CLV User List")

    show = clv_join.copy()
    show = show.rename(columns={uid_c: "user_id", clv_value_col: "clv_3m", seg_rfm_col: "customer_segment"})
    if clv_seg_col_use in show.columns:
        show = show.rename(columns={clv_seg_col_use: "clv_segment"})
    else:
        show["clv_segment"] = "NA"

    show["user_id"] = show["user_id"].astype(str)
    show["customer_segment"] = show["customer_segment"].astype(str)
    show["clv_segment"] = show["clv_segment"].astype(str)

    show["clv_3m"] = pd.to_numeric(show["clv_3m"], errors="coerce")
    show = show.replace([np.inf, -np.inf], np.nan).dropna(subset=["clv_3m"]).copy()
    show = show.sort_values("clv_3m", ascending=False).head(50)

    show_display = pd.DataFrame({
        "User ID": show["user_id"].values,
        "CLV 3 Months": show["clv_3m"].map(lambda v: f"${v:,.0f}").values,
        "CLV Segment": show["clv_segment"].values,
        "Customer Segment": show["customer_segment"].values,
    }).fillna("").astype(str)

    st.dataframe(show_display, use_container_width=True, height=380)

    st.caption(
        "Daftar top CLV users mendukung eksekusi retensi yang lebih presisi (VIP perks, early access, dan win-back selektif)."
    )

# =========================
# Quick Analysis (Key Takeaways) — aligned to PPT cohort+CLV insight
# =========================
st.divider()
st.subheader("Quick Analysis (Key Takeaways)")

m1_txt = safe_metric(avg_m1, nice_percent)
clv_txt = safe_metric(avg_clv, lambda v: nice_currency(v, sig=4, symbol="$"))
p99_txt = safe_metric(cap_p99, lambda v: nice_currency(v, sig=4, symbol="$"))
cap_pct_txt = safe_metric(pct_at_cap, nice_percent)
ncohort_txt = nice_number(len(matrix.index) if not matrix.empty else 0, sig=4)

takeaways = [
    f"- **M1 retention rendah** (avg saat filter aktif: **{m1_txt}**) → churn terjadi sangat awal; fokus pada onboarding pasca pembelian & trigger pembelian ke-2.",
    "- Retention antar cohort relatif konsisten rendah → masalah lebih ke **behavior/experience** daripada cohort tertentu.",
    f"- **CLV 3M** memberikan prioritas retensi (avg: **{clv_txt}**) — invest lebih besar pada segmen high-value.",
    f"- Ada indikasi **outlier CLV** (P99: **{p99_txt}**, users at/above P99: **{cap_pct_txt}**) → cocok untuk strategi VIP yang sangat selektif.",
    f"- Context: cohorts shown **{ncohort_txt}** (sesuai filter).",
]
st.markdown("\n".join(takeaways))
