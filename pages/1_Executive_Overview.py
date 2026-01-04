from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from src.io import (
    load_sessions,
    load_session_dim,
    load_purchases,
    load_rfm,
    read_parquet_safe,
)

from src.filters import (
    month_filter_ui,
    apply_month_filter,
    nice_number,
    nice_percent,
    nice_currency,
)

from src.charts import (
    line_monthly,
    bar_top,
    funnel_simple_from_month,
)

# =========================
# Page setup
# =========================
st.set_page_config(page_title="Executive Overview", layout="wide")

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
      [data-testid="stMetric"] { padding: 0.4rem 0.6rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Executive Overview")

# =========================
# Load data
# =========================
sessions = load_sessions()                 # session-level flags
session_dim = load_session_dim()           # session -> brand/category + month
purchases = load_purchases()               # purchase events only
rfm = load_rfm()                           # user-level RFM
clv = read_parquet_safe("clv.parquet")     # user-level predicted_clv_3m

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

sessions_uid = find_col(sessions, ["user_id", "user", "User ID", "UserID"])
rfm_uid = find_col(rfm, ["user_id", "user", "User ID", "UserID"])
clv_uid = find_col(clv, ["user_id", "user", "User ID", "UserID"])

ses_key = find_col(sessions, ["user_session", "session_id", "Session ID"])
dim_key = find_col(session_dim, ["user_session", "session_id", "Session ID"])

rfm_seg_col = find_col(rfm, ["Segment", "rfm_segment", "RFM_Segment", "segment"])

# =========================
# Sidebar filters (clean & stable keys)
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

rfm_segment_options: list[str] = []
if (not rfm.empty) and (rfm_seg_col is not None):
    rfm_segment_options = sorted(rfm[rfm_seg_col].dropna().astype(str).unique().tolist())

# ---- Brand expander
with st.sidebar.expander("Brand", expanded=False):
    brand_all = st.checkbox("Select all brands", value=True, key="brand_all")
    selected_brands = st.multiselect(
        "Brand",
        options=brand_options,
        default=brand_options if brand_all else [],
        placeholder="Search brand...",
        key="brand_ms",
    )
    if brand_all:
        selected_brands = brand_options

# ---- Category expander
with st.sidebar.expander("Category", expanded=False):
    cat_all = st.checkbox("Select all categories", value=True, key="cat_all")
    selected_cats = st.multiselect(
        "Category",
        options=cat_options,
        default=cat_options if cat_all else [],
        placeholder="Search category...",
        key="cat_ms",
    )
    if cat_all:
        selected_cats = cat_options

# ---- RFM segment expander
with st.sidebar.expander("RFM Segment", expanded=False):
    rfm_all = st.checkbox("Select all RFM segments", value=True, key="rfm_all")
    selected_rfm_segments = st.multiselect(
        "RFM Segment",
        options=rfm_segment_options,
        default=rfm_segment_options if rfm_all else [],
        placeholder="Search segment...",
        key="rfm_ms",
    )
    if rfm_all:
        selected_rfm_segments = rfm_segment_options

# Treat empty selection as "All"
selected_brands = treat_empty_as_all(selected_brands, brand_options)
selected_cats = treat_empty_as_all(selected_cats, cat_options)
selected_rfm_segments = treat_empty_as_all(selected_rfm_segments, rfm_segment_options)

# =========================
# Apply filters consistently
# =========================

# 1) Filter session_dim by month + brand + category
dim_f = session_dim.copy()

if start_m is not None and end_m is not None and (not dim_f.empty) and ("month" in dim_f.columns):
    dim_f = apply_month_filter(dim_f, start_m, end_m, col="month")

if not dim_f.empty and selected_brands and "brand" in dim_f.columns:
    dim_f = dim_f[dim_f["brand"].astype(str).isin([str(x) for x in selected_brands])]

if not dim_f.empty and selected_cats and "category_main" in dim_f.columns:
    dim_f = dim_f[dim_f["category_main"].astype(str).isin([str(x) for x in selected_cats])]

# 2) Filter sessions by month + allowed sessions from dim_f
sessions_f = sessions.copy()

if start_m is not None and end_m is not None and (not sessions_f.empty) and ("month" in sessions_f.columns):
    sessions_f = apply_month_filter(sessions_f, start_m, end_m, col="month")

if (not sessions_f.empty) and (not dim_f.empty) and ses_key and dim_key:
    allowed_sessions = dim_f[dim_key].dropna().astype(str).unique()
    sessions_f = sessions_f[sessions_f[ses_key].astype(str).isin(allowed_sessions)].copy()

# 3) RFM filter builds allowed users
allowed_users = None
if (
    selected_rfm_segments
    and rfm_seg_col
    and (not rfm.empty)
    and rfm_uid
):
    allowed_users = (
        rfm.loc[
            rfm[rfm_seg_col].astype(str).isin([str(x) for x in selected_rfm_segments]),
            rfm_uid
        ]
        .dropna()
        .astype(str)
        .unique()
    )

# Apply allowed_users to sessions
if allowed_users is not None and (not sessions_f.empty) and sessions_uid:
    sessions_f = sessions_f[sessions_f[sessions_uid].astype(str).isin(allowed_users)].copy()

# 4) Filter purchases by month + brand + category + allowed_users
purchases_f = purchases.copy()

if start_m is not None and end_m is not None and (not purchases_f.empty) and ("month" in purchases_f.columns):
    purchases_f = apply_month_filter(purchases_f, start_m, end_m, col="month")

if not purchases_f.empty and selected_brands and "brand" in purchases_f.columns:
    purchases_f = purchases_f[purchases_f["brand"].astype(str).isin([str(x) for x in selected_brands])]

if not purchases_f.empty and selected_cats and "category_main" in purchases_f.columns:
    purchases_f = purchases_f[purchases_f["category_main"].astype(str).isin([str(x) for x in selected_cats])]

if allowed_users is not None and (not purchases_f.empty) and ("user_id" in purchases_f.columns):
    purchases_f = purchases_f[purchases_f["user_id"].astype(str).isin(allowed_users)].copy()

# User pool to force brand/category/month filters to affect CLV visuals too
purchase_user_pool = None
if not purchases_f.empty and "user_id" in purchases_f.columns:
    purchase_user_pool = purchases_f["user_id"].dropna().astype(str).unique()

# =========================
# KPI calculations
# =========================
total_sessions = len(sessions_f) if not sessions_f.empty else None

total_users = None
if not sessions_f.empty and sessions_uid:
    total_users = sessions_f[sessions_uid].nunique(dropna=True)

# Funnel from sessions_f
funnel_f = pd.DataFrame()
need_cols = {"month", "has_view", "has_cart", "has_purchase"}
if not sessions_f.empty and need_cols.issubset(sessions_f.columns):
    tmp = sessions_f.copy()
    for c in ["has_view", "has_cart", "has_purchase"]:
        tmp[c] = pd.to_numeric(tmp[c], errors="coerce").fillna(0).astype(int)

    funnel_f = (
        tmp.groupby("month", as_index=False)[["has_view", "has_cart", "has_purchase"]].sum()
        .rename(columns={"has_view": "view_sessions", "has_cart": "cart_sessions", "has_purchase": "purchase_sessions"})
        .sort_values("month")
    )

view_to_purchase = cart_abandon_rate = None
view_sessions = cart_sessions = purchase_sessions = 0.0
if not funnel_f.empty:
    view_sessions = float(funnel_f["view_sessions"].sum())
    cart_sessions = float(funnel_f["cart_sessions"].sum())
    purchase_sessions = float(funnel_f["purchase_sessions"].sum())
    if view_sessions > 0:
        view_to_purchase = purchase_sessions / view_sessions
    if cart_sessions > 0:
        # guard: purchase shouldn't exceed cart in a strict funnel
        purchase_sessions = min(purchase_sessions, cart_sessions)
        cart_abandon_rate = (cart_sessions - purchase_sessions) / cart_sessions

total_abandoned_carts = None
if not sessions_f.empty:
    for cand in ["cart_abandoned", "is_abandoned", "abandoned", "Cart_Abandoned"]:
        if cand in sessions_f.columns:
            total_abandoned_carts = int(pd.to_numeric(sessions_f[cand], errors="coerce").fillna(0).sum())
            break

# Revenue from purchases_f
total_revenue = None
if not purchases_f.empty and "price" in purchases_f.columns:
    total_revenue = float(pd.to_numeric(purchases_f["price"], errors="coerce").fillna(0).sum())

# =========================
# KPI Cards
# =========================
k1, k2, k3, k4, k5, k6 = st.columns(6)
with k1:
    st.metric("Total Sessions", safe_metric(total_sessions, lambda v: nice_number(v, sig=4)))
with k2:
    st.metric("Total Users", safe_metric(total_users, lambda v: nice_number(v, sig=4)))
with k3:
    st.metric("View → Purchase", safe_metric(view_to_purchase, nice_percent))
with k4:
    st.metric("Cart Abandonment", safe_metric(cart_abandon_rate, nice_percent))
with k5:
    st.metric("Total Abandoned", safe_metric(total_abandoned_carts, lambda v: nice_number(v, sig=4)))
with k6:
    st.metric("Revenue", safe_metric(total_revenue, lambda v: nice_currency(v, sig=4, symbol="$")))

st.divider()

# =========================
# Row 1: Revenue + Funnel
# =========================
c1, c2 = st.columns([1.25, 1.0])

with c1:
    st.subheader("Trend Line Revenue")
    if purchases_f.empty or "month" not in purchases_f.columns or "price" not in purchases_f.columns:
        st.info("Revenue data kosong setelah filter.")
    else:
        rev_f = (
            purchases_f.assign(price=pd.to_numeric(purchases_f["price"], errors="coerce").fillna(0))
            .groupby("month", as_index=False)["price"].sum()
            .rename(columns={"price": "revenue"})
            .sort_values("month")
        )
        fig = line_monthly(rev_f, x="month", y="revenue", title="")
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

    # --- Caption insight (from PPT)
    st.caption(
        "Revenue meningkat hingga puncak di Jan 2021 lalu sedikit menurun — indikasi *seasonal peak* yang perlu dimaksimalkan lewat campaign dan kesiapan inventory."
    )

with c2:
    st.subheader("Funnel Overview")

    if sessions_f.empty or not {"has_view", "has_cart", "has_purchase"}.issubset(set(sessions_f.columns)):
        st.info("Funnel data kosong setelah filter.")
        # still show caption for consistency
        st.caption("Bottleneck utama ada di View → Cart, sehingga fokus optimasi sebaiknya di tahap add-to-cart, bukan checkout.")
    else:
        tmp = sessions_f.copy()
        for c in ["has_view", "has_cart", "has_purchase"]:
            tmp[c] = pd.to_numeric(tmp[c], errors="coerce").fillna(0).astype(int)

        v = int(tmp["has_view"].sum())
        c_ = int(tmp["has_cart"].sum())
        p = int(min(tmp["has_purchase"].sum(), c_))

        steps = ["View", "Cart", "Purchase"]
        values = [v, c_, p]

        fig2 = go.Figure(
            go.Funnel(
                y=steps,
                x=values,
                textinfo="value+percent initial",
            )
        )
        fig2.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)

        # --- Caption insight (from PPT) + optionally inject filtered rate
        rate_txt = f" (filtered View→Cart: {nice_percent((c_/v) if v else None)})" if v else ""
        st.caption(
            f"Bottleneck utama ada di View → Cart, sehingga fokus optimasi sebaiknya di tahap add-to-cart, bukan checkout.{rate_txt}"
        )

st.divider()

# =========================
# Row 2: Abandoned Category + M1 Retention
# =========================
c3, c4 = st.columns([1.25, 1.0])

with c3:
    st.subheader("Top Abandoned Category")
    if sessions_f.empty or dim_f.empty or (ses_key is None) or (dim_key is None) or ("category_main" not in dim_f.columns):
        st.info("Abandonment data belum siap (butuh session_dim + sessions).")
        st.caption("Abandonment tertinggi terjadi di kategori apparel dan sport — prioritas intervensi per kategori berpotensi memberi dampak tercepat.")
    else:
        dfj = sessions_f[[ses_key, "has_cart", "has_purchase"]].copy()
        dfj[ses_key] = dfj[ses_key].astype(str)

        dd = dim_f[[dim_key, "category_main"]].copy()
        dd[dim_key] = dd[dim_key].astype(str)

        dfj = dfj.merge(dd, left_on=ses_key, right_on=dim_key, how="left")

        dfj["has_cart"] = pd.to_numeric(dfj["has_cart"], errors="coerce").fillna(0).astype(int)
        dfj["has_purchase"] = pd.to_numeric(dfj["has_purchase"], errors="coerce").fillna(0).astype(int)

        # clearer definition: abandoned = cart but no purchase
        dfj["is_abandoned"] = (dfj["has_cart"] == 1) & (dfj["has_purchase"] == 0)

        agg = dfj.groupby("category_main", as_index=False).agg(
            cart_sessions=("has_cart", "sum"),
            abandoned=("is_abandoned", "sum"),
        )
        agg = agg.loc[agg["cart_sessions"] > 0].copy()
        agg["abandonment_rate"] = agg["abandoned"] / agg["cart_sessions"]
        agg = agg.sort_values("abandonment_rate", ascending=False)

        fig3 = bar_top(agg, x="category_main", y="abandonment_rate", title="", topn=10, orientation="h")
        fig3.update_layout(height=420)
        st.plotly_chart(fig3, use_container_width=True)

        # --- Caption insight (from PPT)
        st.caption(
            "Abandonment tertinggi terjadi di kategori apparel dan sport — prioritaskan intervensi per kategori untuk dampak konversi paling cepat."
        )

with c4:
    st.subheader("M1 Retention Trend")

    if purchases_f.empty or "user_id" not in purchases_f.columns or "month" not in purchases_f.columns:
        st.info("Retention data kosong setelah filter.")
        st.caption("M1 retention sangat rendah (±1–4%) yang menandakan churn terjadi sangat awal — onboarding dan re-engagement dini menjadi prioritas.")
    else:
        pu = purchases_f[["user_id", "month"]].dropna().copy()
        pu["user_id"] = pu["user_id"].astype(str)
        pu["month"] = pd.to_datetime(pu["month"], errors="coerce")
        pu = pu.dropna(subset=["month"]).drop_duplicates()

        first = pu.groupby("user_id", as_index=False)["month"].min().rename(columns={"month": "cohort_month"})
        first["cohort_month"] = first["cohort_month"].dt.to_period("M").dt.to_timestamp()

        user_month = set(zip(pu["user_id"], pu["month"].dt.to_period("M").astype(str)))

        retained_flag = []
        for u, cm in zip(first["user_id"], first["cohort_month"]):
            next_m = (pd.Timestamp(cm) + pd.offsets.MonthBegin(1)).to_period("M")
            retained_flag.append(1 if (u, str(next_m)) in user_month else 0)

        first["retained_m1"] = retained_flag

        m1_line = (
            first.groupby("cohort_month", as_index=False)
                .agg(cohort_size=("user_id", "count"), retained=("retained_m1", "sum"))
        )
        m1_line["retention_rate"] = (m1_line["retained"] / m1_line["cohort_size"]).fillna(0.0)
        m1_line = m1_line.sort_values("cohort_month")

        fig4 = line_monthly(m1_line, x="cohort_month", y="retention_rate", title="")
        fig4.update_layout(height=420)
        st.plotly_chart(fig4, use_container_width=True)

        # --- Caption insight (from PPT)
        st.caption(
            "M1 retention sangat rendah (±1–4%) di semua cohort — churn terjadi sangat awal setelah pembelian pertama, sehingga onboarding dan re-engagement dini perlu diprioritaskan."
        )

st.divider()

# =========================
# Row 3: Avg CLV by RFM Segment + Top CLV Users
# =========================
c5, c6 = st.columns([1.0, 1.0])

with c5:
    st.subheader("Avg CLV by RFM Segment")
    if clv.empty or rfm.empty or (clv_uid is None) or (rfm_uid is None) or (rfm_seg_col is None) or ("predicted_clv_3m" not in clv.columns):
        st.info("CLV/RFM data belum siap (butuh clv.parquet + rfm.parquet).")
        st.caption("CLV terkonsentrasi pada segmen Champion dan Cannot Lose Them — retensi segmen ini paling berdampak terhadap revenue.")
    else:
        clv2 = clv[[clv_uid, "predicted_clv_3m"]].copy().rename(columns={clv_uid: "user_id"})
        clv2["user_id"] = clv2["user_id"].astype(str)
        clv2["predicted_clv_3m"] = pd.to_numeric(clv2["predicted_clv_3m"], errors="coerce")

        rfm2 = rfm[[rfm_uid, rfm_seg_col]].copy().rename(columns={rfm_uid: "user_id", rfm_seg_col: "rfm_segment"})
        rfm2["user_id"] = rfm2["user_id"].astype(str)

        dfc = clv2.merge(rfm2, on="user_id", how="left").dropna(subset=["predicted_clv_3m", "rfm_segment"])

        # Apply user pool from filtered purchases (so Brand/Category/Month affect CLV)
        if purchase_user_pool is not None:
            dfc = dfc[dfc["user_id"].isin(purchase_user_pool)]

        # Apply selected RFM segments
        if selected_rfm_segments:
            dfc = dfc[dfc["rfm_segment"].astype(str).isin([str(x) for x in selected_rfm_segments])]

        if dfc.empty:
            st.info("CLV by RFM kosong setelah filter.")
            st.caption("CLV terkonsentrasi pada segmen Champion dan Cannot Lose Them — retensi segmen ini paling berdampak terhadap revenue.")
        else:
            avg_by_seg = (
                dfc.groupby("rfm_segment", as_index=False)["predicted_clv_3m"].mean()
                .rename(columns={"predicted_clv_3m": "avg_clv_3m"})
                .sort_values("avg_clv_3m", ascending=False)
            )
            fig5 = bar_top(avg_by_seg, x="rfm_segment", y="avg_clv_3m", title="", topn=20, orientation="v")
            fig5.update_layout(height=420)
            st.plotly_chart(fig5, use_container_width=True)

            # --- Caption insight (from PPT)
            st.caption(
                "CLV terkonsentrasi pada segmen Champion dan Cannot Lose Them — retensi segmen ini paling berdampak terhadap revenue."
            )

with c6:
    st.subheader("Top CLV Users")
    if clv.empty or ("predicted_clv_3m" not in clv.columns) or (clv_uid is None):
        st.info("CLV data belum siap.")
        st.caption("Daftar user dengan CLV tertinggi membantu prioritisasi VIP/retensi dan strategi win-back yang lebih personal.")
    else:
        top = clv[[clv_uid, "predicted_clv_3m"]].copy().rename(columns={clv_uid: "user_id"})
        top["user_id"] = top["user_id"].astype(str)
        top["predicted_clv_3m"] = pd.to_numeric(top["predicted_clv_3m"], errors="coerce")

        # Apply user pool so Brand/Category/Month affect this table too
        if purchase_user_pool is not None:
            top = top[top["user_id"].isin(purchase_user_pool)]

        # Apply RFM filter if possible
        if (not rfm.empty) and (rfm_uid is not None) and (rfm_seg_col is not None) and selected_rfm_segments:
            rfm_map = rfm[[rfm_uid, rfm_seg_col]].copy().rename(columns={rfm_uid: "user_id", rfm_seg_col: "rfm_segment"})
            rfm_map["user_id"] = rfm_map["user_id"].astype(str)
            top = top.merge(rfm_map, on="user_id", how="left")
            top = top[top["rfm_segment"].astype(str).isin([str(x) for x in selected_rfm_segments])]

        top = top.dropna(subset=["predicted_clv_3m"]).sort_values("predicted_clv_3m", ascending=False).head(50)
        st.dataframe(top, use_container_width=True, height=420)

        # --- Caption insight (aligned with PPT CLV prioritization)
        st.caption(
            "Daftar user dengan CLV tertinggi untuk target retensi (VIP/perks) dan win-back selektif—selaras dengan prioritas segmen bernilai tinggi."
        )

# =========================
# Quick analysis (Insight inti halaman — from PPT)
# =========================
st.divider()
st.subheader("Quick Analysis (Key Takeaways)")

# Use filtered metrics as context (optional), but the core message follows PPT
v2p_txt = safe_metric(view_to_purchase, nice_percent)
cab_txt = safe_metric(cart_abandon_rate, nice_percent)

qa = [
    f"- Dari total sessions, hanya sebagian kecil yang berujung purchase → peluang terbesar ada pada **peningkatan konversi** (View→Purchase saat filter aktif: **{v2p_txt}**).",
    "- **Bottleneck funnel** ada di **View → Cart**, sehingga optimasi sebaiknya difokuskan pada kualitas product page & dorongan add-to-cart (bukan checkout).",
    f"- **Cart abandonment tinggi** → peluang konversi tercepat lewat cart recovery (Cart Abandonment saat filter aktif: **{cab_txt}**).",
    "- **M1 retention sangat rendah (±1–4%)** → churn terjadi sangat awal; onboarding pasca pembelian & re-engagement dini jadi prioritas.",
    "- **CLV terkonsentrasi pada Champion & Cannot Lose Them** → retensi/win-back segmen ini paling berdampak terhadap revenue.",
]

st.markdown("\n".join(qa))
