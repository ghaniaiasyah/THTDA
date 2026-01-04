from __future__ import annotations
import pandas as pd
import streamlit as st

def month_range_from_df(df: pd.DataFrame, col: str = "month"):
    if df.empty or col not in df.columns:
        return None
    s = pd.to_datetime(df[col], errors="coerce").dropna()
    if s.empty:
        return None
    return (s.min(), s.max())

def month_filter_ui(label: str, df: pd.DataFrame, col: str = "month"):
    rng = month_range_from_df(df, col)
    if rng is None:
        st.sidebar.info("Filter bulan tidak tersedia (kolom month tidak ditemukan).")
        return None
    start, end = rng
    picked = st.sidebar.slider(
        label,
        min_value=start.to_pydatetime(),
        max_value=end.to_pydatetime(),
        value=(start.to_pydatetime(), end.to_pydatetime()),
        format="YYYY-MM",
    )
    return pd.to_datetime(picked[0]), pd.to_datetime(picked[1])

def apply_month_filter(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, col: str = "month"):
    if df.empty or col not in df.columns:
        return df
    m = pd.to_datetime(df[col], errors="coerce")
    return df.loc[(m >= start) & (m <= end)].copy()

def _compact_number(x: float | int | None, sig: int = 4) -> str:
    """3–4 angka penting, compact K/M/B."""
    try:
        if x is None:
            return "N/A"
        x = float(x)
        neg = x < 0
        x = abs(x)

        if x < 1_000:
            s = f"{x:.0f}"
        elif x < 1_000_000:
            s = f"{x/1_000:.{sig-1}g}K"
        elif x < 1_000_000_000:
            s = f"{x/1_000_000:.{sig-1}g}M"
        else:
            s = f"{x/1_000_000_000:.{sig-1}g}B"

        return f"-{s}" if neg else s
    except Exception:
        return "N/A"

def nice_number(x, sig: int = 4) -> str:
    return _compact_number(x, sig=sig)

def nice_percent(x, digits=1):
    try:
        if x is None:
            return "N/A"
        return f"{float(x)*100:.{digits}f}%"
    except Exception:
        return "N/A"

def nice_currency(x, sig: int = 4, symbol: str = "$") -> str:
    try:
        if x is None:
            return "N/A"
        # pakai compact number tapi prefix $
        s = _compact_number(float(x), sig=sig)
        return f"{symbol}{s}"
    except Exception:
        return "N/A"
