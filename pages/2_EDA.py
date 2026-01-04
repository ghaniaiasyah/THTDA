from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.express as px

from src.io import (
    load_sessions,
    load_session_dim,
    load_purchases,
    read_parquet_safe,
)
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
st.set_page_config(page_title="EDA", layout="wide")

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

st.title("EDA")

# =========================
# Load data
# =========================
sessions = load_sessions()
session_dim = load_session_dim()
purchases = load_purchases()
events = read_parquet_safe("events.parquet")  # from preprocess

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
evt_ses_key = find_col(events, ["user_session", "session_id", "Session ID"])

event_type_col = find_col(events, ["event_type", "Event Type", "type"])
time_col = find_col(events, ["event_time", "Event Time", "timestamp", "time"])

# =========================
# Sidebar filters (month/brand/category ONLY)
# =========================
st.sidebar.header("Filters")

base_for_month = purchases if not purchases.empty else (sessions if not sessions.empty else session_dim)
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
    brand_all = st.checkbox("Select all brands", value=True, key="eda_brand_all")
    selected_brands = st.multiselect(
        "Brand",
        options=brand_options,
        default=(brand_options if brand_all else []),
        placeholder="Search brand...",
        key="eda_brand_ms",
    )
    if brand_all:
        selected_brands = brand_options

with st.sidebar.expander("Category", expanded=False):
    cat_all = st.checkbox("Select all categories", value=True, key="eda_cat_all")
    selected_cats = st.multiselect(
        "Category",
        options=cat_options,
        default=(cat_options if cat_all else []),
        placeholder="Search category...",
        key="eda_cat_ms",
    )
    if cat_all:
        selected_cats = cat_options

selected_brands = treat_empty_as_all(selected_brands, brand_options)
selected_cats = treat_empty_as_all(selected_cats, cat_options)

# =========================
# Apply filters CONSISTENTLY
# 1) Filter session_dim => allowed_sessions
# 2) Filter events by month + allowed_sessions (+ optional strict brand/category)
# 3) purchases filter kept for revenue/price
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

# events filtered (main charts)
events_f = events.copy()

if start_m is not None and end_m is not None and (not events_f.empty) and ("month" in events_f.columns):
    events_f = apply_month_filter(events_f, start_m, end_m, col="month")

if (allowed_sessions is not None) and (not events_f.empty) and evt_ses_key:
    events_f = events_f[events_f[evt_ses_key].astype(str).isin(allowed_sessions)].copy()

# Optional strict filter (keep - consistent with your earlier approach)
if (not events_f.empty) and selected_brands and ("brand" in events_f.columns):
    events_f = events_f[events_f["brand"].astype(str).isin([str(x) for x in selected_brands])]

if (not events_f.empty) and selected_cats and ("category_main" in events_f.columns):
    events_f = events_f[events_f["category_main"].astype(str).isin([str(x) for x in selected_cats])]

# purchases filtered (for revenue KPI + price dist)
purchases_f = purchases.copy()

if start_m is not None and end_m is not None and (not purchases_f.empty) and ("month" in purchases_f.columns):
    purchases_f = apply_month_filter(purchases_f, start_m, end_m, col="month")

if (not purchases_f.empty) and selected_brands and ("brand" in purchases_f.columns):
    purchases_f = purchases_f[purchases_f["brand"].astype(str).isin([str(x) for x in selected_brands])]

if (not purchases_f.empty) and selected_cats and ("category_main" in purchases_f.columns):
    purchases_f = purchases_f[purchases_f["category_main"].astype(str).isin([str(x) for x in selected_cats])]

# =========================
# KPIs (events_f for consistency)
# =========================
total_events = len(events_f) if not events_f.empty else None

purchase_events = None
if (not events_f.empty) and event_type_col:
    purchase_events = int((events_f[event_type_col].astype(str).str.lower() == "purchase").sum())

revenue = None
if (not purchases_f.empty) and ("price" in purchases_f.columns):
    revenue = float(pd.to_numeric(purchases_f["price"], errors="coerce").fillna(0).sum())

total_sessions = None
unique_users = None
if not events_f.empty and evt_ses_key:
    total_sessions = events_f[evt_ses_key].astype(str).nunique()

uid_col = find_col(events_f, ["user_id", "user", "User ID", "UserID"])
if not events_f.empty and uid_col:
    unique_users = events_f[uid_col].astype(str).nunique()

k1, k2, k3, k4, k5, k6 = st.columns(6)
with k1: st.metric("Total Events", safe_metric(total_events, lambda v: nice_number(v, sig=4)))
with k2: st.metric("Purchase Events", safe_metric(purchase_events, lambda v: nice_number(v, sig=4)))
with k3: st.metric("Total Sessions", safe_metric(total_sessions, lambda v: nice_number(v, sig=4)))
with k4: st.metric("Unique Users", safe_metric(unique_users, lambda v: nice_number(v, sig=4)))
with k5: st.metric("Revenue", safe_metric(revenue, lambda v: nice_currency(v, sig=4, symbol="$")))
with k6:
    rate = (purchase_events / total_events) if (purchase_events is not None and total_events and total_events > 0) else None
    st.metric("Purchase / Event", safe_metric(rate, nice_percent))

st.divider()

# =========================
# 1) Monthly trend: Total Events & Purchase Events
# =========================
c1, c2 = st.columns([1.25, 1.0])

with c1:
    st.subheader("Monthly Trend: Total Events vs Purchase Events")
    if events_f.empty or "month" not in events_f.columns or event_type_col is None:
        st.info("Trend data kosong setelah filter.")
        st.caption("Meski volume event berfluktuasi, purchase relatif stagnan (±7–8K/bulan) — perlu optimasi funnel dan kualitas engagement.")
    else:
        tmp = events_f.copy()
        tmp["event_type_l"] = tmp[event_type_col].astype(str).str.lower()
        tmp["month"] = pd.to_datetime(tmp["month"], errors="coerce")

        trend = (
            tmp.groupby("month", as_index=False).agg(
                total_events=("event_type_l", "count"),
                purchase_events=("event_type_l", lambda x: (x == "purchase").sum()),
            )
            .dropna(subset=["month"])
            .sort_values("month")
        )

        fig = px.line(trend, x="month", y=["total_events", "purchase_events"])
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

        # caption (PPT EDA)
        st.caption(
            "Meski volume event berfluktuasi, purchase relatif stagnan (±7–8K/bulan) — perlu optimasi funnel dan kualitas engagement."
        )

with c2:
    st.subheader("Event Type Mix")
    if events_f.empty or (event_type_col is None):
        st.info("Event type kosong setelah filter.")
        st.caption("Proporsi purchase kecil dibanding view/cart — peluang utama ada pada peningkatan kualitas journey dan konversi, bukan sekadar traffic.")
    else:
        mix = (
            events_f[event_type_col].astype(str).str.lower()
            .value_counts(dropna=False)
            .reset_index()
        )
        mix.columns = ["event_type", "count"]

        COLOR_MAP = {
            "purchase": "#2ca02c",  # green
            "view": "#1f77b4",      # blue
            "cart": "#ff7f0e",      # orange
        }
        EVENT_ORDER = ["view", "cart", "purchase"]

        mix["event_type"] = mix["event_type"].astype(str).str.lower()
        fig2 = px.pie(
            mix,
            names="event_type",
            values="count",
            hole=0.55,
            color="event_type",
            color_discrete_map=COLOR_MAP,
            category_orders={"event_type": EVENT_ORDER},
        )
        fig2.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)

        st.caption(
            "Proporsi purchase relatif kecil dibanding view/cart — peningkatan kualitas journey lebih berdampak daripada menambah traffic."
        )

st.divider()

# =========================
# 2) Top Category by Total Event (stacked) + Top Brands by Purchase
# =========================
c3, c4 = st.columns([1.15, 0.85])

with c3:
    st.subheader("Top Category by Total Event")
    if events_f.empty or (event_type_col is None) or ("category_main" not in events_f.columns):
        st.info("Category event mix kosong setelah filter.")
        st.caption("Computers menjadi kategori dengan interaksi tertinggi, namun tingginya unknown category mengindikasikan isu data quality yang dapat memengaruhi prioritas kategori.")
    else:
        tmp = events_f.copy()
        tmp["event_type_l"] = tmp[event_type_col].astype(str).str.lower()
        tmp["category_main"] = tmp["category_main"].astype(str)

        cat_mix = (
            tmp.groupby(["category_main", "event_type_l"], as_index=False)
               .size()
               .rename(columns={"size": "events"})
        )

        # rank by total events
        totals = (
            cat_mix.groupby("category_main", as_index=False)["events"]
            .sum()
            .rename(columns={"events": "total"})
            .sort_values("total", ascending=False)
        )
        top_cats = totals.head(12)["category_main"].tolist()
        cat_mix2 = cat_mix[cat_mix["category_main"].isin(top_cats)].copy()

        # ensure category sorted by total DESC (top appears on top)
        cat_order = totals.head(12)["category_main"].tolist()
        cat_mix2["category_main"] = pd.Categorical(cat_mix2["category_main"], categories=cat_order[::-1], ordered=True)
        cat_mix2["event_type_l"] = cat_mix2["event_type_l"].astype(str).str.lower()

        # Consistent colors
        COLOR_MAP = {
            "purchase": "#2ca02c",
            "view": "#1f77b4",
            "cart": "#ff7f0e",
        }
        EVENT_ORDER = ["view", "cart", "purchase"]

        fig3 = px.bar(
            cat_mix2,
            y="category_main",
            x="events",
            color="event_type_l",
            orientation="h",
            barmode="stack",
            color_discrete_map=COLOR_MAP,
            category_orders={
                "event_type_l": EVENT_ORDER,
                "category_main": cat_order[::-1],
            },
        )
        fig3.update_layout(
            height=460,
            margin=dict(l=10, r=10, t=10, b=10),
            legend_title_text="Event Type",
        )
        st.plotly_chart(fig3, use_container_width=True)

        st.caption(
            "Computers menjadi kategori dengan interaksi tertinggi, namun tingginya unknown category mengindikasikan isu data quality yang dapat memengaruhi prioritas kategori."
        )

with c4:
    st.subheader("Top Brands by Purchase Events")
    if events_f.empty or ("brand" not in events_f.columns) or (event_type_col is None):
        st.info("Brand purchase events kosong setelah filter.")
        st.caption("Dominasi unknown brand berpotensi menimbulkan bias analisis — perbaikan mapping brand diperlukan sebelum strategi berbasis brand dijalankan.")
    else:
        tmp = events_f.copy()
        tmp["event_type_l"] = tmp[event_type_col].astype(str).str.lower()

        brand_p = (
            tmp[tmp["event_type_l"] == "purchase"]
            .groupby("brand", as_index=False)
            .size()
            .rename(columns={"size": "purchase_events"})
            .sort_values("purchase_events", ascending=False)
            .head(15)
        )

        fig4 = px.bar(brand_p, x="brand", y="purchase_events")
        fig4.update_layout(height=460, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig4, use_container_width=True)

        st.caption(
            "Dominasi unknown brand berpotensi menimbulkan bias analisis — perbaikan mapping brand diperlukan sebelum strategi berbasis brand dijalankan."
        )

st.divider()

# =========================
# 3) Price distribution + Activity heatmap
# =========================
c5, c6 = st.columns([1.0, 1.0])

with c5:
    st.subheader("Purchase Price Distribution")
    if purchases_f.empty or ("price" not in purchases_f.columns):
        st.info("Price distribution kosong setelah filter.")
        st.caption("Purchase terkonsentrasi di harga rendah (0–50); harga tinggi (500+) cenderung butuh insentif tambahan untuk mendorong konversi.")
    else:
        pr = purchases_f.copy()
        pr["price"] = pd.to_numeric(pr["price"], errors="coerce")
        pr = pr.dropna(subset=["price"])

        fig5 = px.histogram(pr, x="price", nbins=40)
        fig5.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig5, use_container_width=True)

        st.caption(
            "Purchase terkonsentrasi di harga rendah (0–50); harga tinggi (500+) cenderung butuh insentif tambahan untuk mendorong konversi."
        )

with c6:
    st.subheader("Activity by Day & Hour")
    if events_f.empty or (time_col is None):
        st.info("Tidak ada kolom waktu event untuk heatmap (event_time/timestamp).")
        st.caption("Aktivitas user tertinggi terjadi di jam kerja–sore (±09–19) sehingga promo dan push notif paling efektif dijalankan pada jam ini.")
    else:
        tmp = events_f.copy()
        tmp[time_col] = pd.to_datetime(tmp[time_col], errors="coerce")
        tmp = tmp.dropna(subset=[time_col])

        tmp["hour"] = tmp[time_col].dt.hour

        # robust day mapping (avoid locale issues)
        tmp["dow"] = tmp[time_col].dt.dayofweek
        map_dow = {0:"Monday",1:"Tuesday",2:"Wednesday",3:"Thursday",4:"Friday",5:"Saturday",6:"Sunday"}
        tmp["dow"] = tmp["dow"].map(map_dow)

        heat = tmp.groupby(["dow", "hour"], as_index=False).size().rename(columns={"size": "events"})
        dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        heat["dow"] = pd.Categorical(heat["dow"], categories=dow_order, ordered=True)
        heat = heat.sort_values(["dow", "hour"])

        fig6 = px.density_heatmap(heat, x="hour", y="dow", z="events")
        fig6.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig6, use_container_width=True)

        st.caption(
            "Aktivitas user tertinggi terjadi di jam kerja–sore (±09–19) sehingga promo dan push notif paling efektif dijalankan pada jam ini."
        )

# =========================
# Quick analysis (Key Takeaways) — aligned to PPT EDA
# =========================
st.divider()
st.subheader("Quick Analysis (Key Takeaways)")

pe_txt = safe_metric(purchase_events, lambda v: nice_number(v, sig=4))
te_txt = safe_metric(total_events, lambda v: nice_number(v, sig=4))
rev_txt = safe_metric(revenue, lambda v: nice_currency(v, sig=4, symbol="$"))
p_over_e = safe_metric(
    (purchase_events / total_events) if (purchase_events is not None and total_events and total_events > 0) else None,
    nice_percent
)

takeaways = [
    f"- **Purchase relatif kecil & cenderung stagnan** dibanding volume event → fokus perbaikan pada kualitas journey & konversi (Purchase/Event saat filter aktif: **{p_over_e}**).",
    "- **Peak activity** terjadi di jam kerja–sore (±09–19) → jadwalkan promo/push notif di jam ini untuk efisiensi.",
    "- **Computers** termasuk kategori dengan engagement tinggi, namun tingginya **unknown category** menandakan isu data quality yang bisa menggeser prioritas kategori.",
    "- **Unknown brand** dominan → perbaikan mapping brand diperlukan sebelum strategi berbasis brand dijalankan.",
    "- **Price sensitivity**: purchase terkonsentrasi di **harga rendah (0–50)**, sedangkan **500+** butuh insentif tambahan untuk mendorong konversi.",
    f"- Context (filtered): Events **{te_txt}**, Purchase events **{pe_txt}**, Revenue **{rev_txt}**.",
]

st.markdown("\n".join(takeaways))
