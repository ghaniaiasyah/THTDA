from __future__ import annotations

import streamlit as st
import pandas as pd

from src.io import (
    load_sessions,
    load_purchases,
    load_rfm,
    read_parquet_safe,
)
from src.filters import nice_number, nice_currency

# =========================
# Page config
# =========================
st.set_page_config(page_title="E-Commerce Events Dashboard", layout="wide")

st.markdown(
    """
    <style>
      .block-container {
        padding-top: 1.2rem;
        padding-bottom: 1.2rem;
        padding-left: 2.0rem;
        padding-right: 2.0rem;
        max-width: 1500px;
      }
      .hero {
        padding: 1.25rem 1.5rem;
        border: 1px solid rgba(49, 51, 63, 0.2);
        border-radius: 14px;
        background: rgba(240, 242, 246, 0.55);
      }
      .muted { color:#6b7280; font-size: 13px; }
      .card {
        border: 1px solid rgba(49, 51, 63, 0.2);
        border-radius: 14px;
        padding: 1rem 1.25rem;
        background: white;
      }
      h1 { margin-bottom: 0.25rem; }
      h2 { margin-top: 0.5rem; }
      [data-testid="stMetric"] { padding: 0.35rem 0.55rem; }
      ul { margin-top: 0.25rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# Title / Hero
# =========================
st.markdown(
    """
    <div class="hero">
      <h1>E-Commerce Events Dashboard</h1>
      <div class="muted">
        Streamlit deployment version of your Power BI dashboard (EDA → Funnel & Abandonment → RFM → Cohort & CLV).
      </div>
      <div class="muted" style="margin-top:0.35rem;">
        👉 Use the <b>Pages</b> menu in the left sidebar to navigate.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

# =========================
# Optional quick health check (lightweight)
# =========================
sessions = load_sessions()
purchases = load_purchases()
rfm = load_rfm()
clv = read_parquet_safe("clv.parquet")

total_sessions = len(sessions) if not sessions.empty else None
total_users = sessions["user_id"].nunique() if (not sessions.empty and "user_id" in sessions.columns) else None

total_revenue = None
if not purchases.empty and "price" in purchases.columns:
    total_revenue = float(pd.to_numeric(purchases["price"], errors="coerce").fillna(0).sum())

rfm_users = rfm["user_id"].nunique() if (not rfm.empty and "user_id" in rfm.columns) else None
avg_clv = None
if not clv.empty and "predicted_clv_3m" in clv.columns:
    avg_clv = float(pd.to_numeric(clv["predicted_clv_3m"], errors="coerce").dropna().mean())

k1, k2, k3, k4, k5 = st.columns(5)
with k1: st.metric("Sessions", nice_number(total_sessions, sig=4))
with k2: st.metric("Users", nice_number(total_users, sig=4))
with k3: st.metric("Revenue", nice_currency(total_revenue, sig=4, symbol="$"))
with k4: st.metric("RFM Users", nice_number(rfm_users, sig=4))
with k5: st.metric("Avg CLV (3M)", nice_currency(avg_clv, sig=4, symbol="$"))

st.divider()

# =========================
# What’s inside (page guide)
# =========================
c1, c2 = st.columns([1.05, 0.95])

with c1:
    st.markdown(
        """
        <div class="card">
          <h2>Pages</h2>
          <ul>
            <li><b>Executive Overview</b>: KPIs, revenue trend, funnel, top abandoned category, M1 retention, CLV snapshots.</li>
            <li><b>EDA</b>: event trend & mix, top categories (stacked), top brands (purchase events), price distribution, day/hour heatmap.</li>
            <li><b>Funnel & Cart Abandonment</b>: funnel ratios, abandonment KPIs, top abandoned categories, duration analysis.</li>
            <li><b>RFM Segmentation</b>: segment KPIs, customers & revenue by segment, revenue trend, score distribution.</li>
            <li><b>Cohort & CLV</b>: cohort matrix, M1 trend, CLV by segment, CLV distribution, top CLV users.</li>
          </ul>
          <div class="muted">All pages share the same filtering principles (Month / Brand / Category / Segment) and are designed to move together.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        """
        <div class="card">
          <h2>How to use</h2>
          <ul>
            <li>Start from <b>Executive Overview</b> to get the big picture.</li>
            <li>Use <b>EDA</b> to validate behavior patterns (events, categories, brands).</li>
            <li>Use <b>Funnel</b> to find drop-offs and abandonment drivers.</li>
            <li>Use <b>RFM</b> to understand customer value & segments.</li>
            <li>Use <b>Cohort & CLV</b> for retention dynamics and future value.</li>
          </ul>
          <div class="muted">Tip: Filters live in the left sidebar. Most dropdowns support multi-select + select-all.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")
st.info("Open a page from the left sidebar to explore the dashboard.")
