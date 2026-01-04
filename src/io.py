from __future__ import annotations
from pathlib import Path
import pandas as pd
import streamlit as st

DATA_DIR = Path("data/processed")

@st.cache_data(show_spinner=False)
def read_parquet_safe(filename: str) -> pd.DataFrame:
    path = DATA_DIR / filename
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)

@st.cache_data(show_spinner=False)
def load_sessions() -> pd.DataFrame:
    return read_parquet_safe("sessions.parquet")

@st.cache_data(show_spinner=False)
def load_monthly_revenue() -> pd.DataFrame:
    df = read_parquet_safe("monthly_revenue.parquet")
    if not df.empty and "month" in df.columns:
        df["month"] = pd.to_datetime(df["month"], errors="coerce")
        df = df.dropna(subset=["month"]).sort_values("month")
    return df

@st.cache_data(show_spinner=False)
def load_funnel_summary() -> pd.DataFrame:
    df = read_parquet_safe("funnel_summary.parquet")
    if not df.empty and "month" in df.columns:
        df["month"] = pd.to_datetime(df["month"], errors="coerce")
        df = df.dropna(subset=["month"]).sort_values("month")
    return df

@st.cache_data(show_spinner=False)
def load_abandoned_category() -> pd.DataFrame:
    return read_parquet_safe("abandoned_category.parquet")

@st.cache_data(show_spinner=False)
def load_m1_retention_trend() -> pd.DataFrame:
    df = read_parquet_safe("m1_retention_trend.parquet")
    if not df.empty and "cohort_month" in df.columns:
        df["cohort_month"] = pd.to_datetime(df["cohort_month"], errors="coerce")
        df = df.dropna(subset=["cohort_month"]).sort_values("cohort_month")
    return df

@st.cache_data(show_spinner=False)
def load_avg_clv_by_segment() -> pd.DataFrame:
    return read_parquet_safe("avg_clv_by_segment.parquet")

@st.cache_data(show_spinner=False)
def load_top_clv_users() -> pd.DataFrame:
    return read_parquet_safe("top_clv_users.parquet")

@st.cache_data(show_spinner=False)
def load_monthly_revenue_dim() -> pd.DataFrame:
    df = read_parquet_safe("monthly_revenue_dim.parquet")
    if not df.empty and "month" in df.columns:
        df["month"] = pd.to_datetime(df["month"], errors="coerce")
        df = df.dropna(subset=["month"]).sort_values("month")
    return df

@st.cache_data(show_spinner=False)
def load_abandoned_category_by_month() -> pd.DataFrame:
    df = read_parquet_safe("abandoned_category_by_month.parquet")
    if not df.empty and "month" in df.columns:
        df["month"] = pd.to_datetime(df["month"], errors="coerce")
        df = df.dropna(subset=["month"]).sort_values("month")
    return df

@st.cache_data(show_spinner=False)
def load_rfm() -> pd.DataFrame:
    return read_parquet_safe("rfm.parquet")


@st.cache_data(show_spinner=False)
def load_session_dim() -> pd.DataFrame:
    df = read_parquet_safe("session_dim.parquet")
    if not df.empty and "month" in df.columns:
        df["month"] = pd.to_datetime(df["month"], errors="coerce")
        df = df.dropna(subset=["month"])
    return df

@st.cache_data(show_spinner=False)
def load_purchases() -> pd.DataFrame:
    df = read_parquet_safe("purchases.parquet")
    if not df.empty and "month" in df.columns:
        df["month"] = pd.to_datetime(df["month"], errors="coerce")
        df = df.dropna(subset=["month"]).sort_values("month")
    return df
