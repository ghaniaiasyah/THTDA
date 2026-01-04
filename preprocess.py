from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import numpy as np

# =========================
# CONFIG
# =========================
RAW_XLSX = Path("data/raw/tht da.xlsx")  # ubah kalau nama file beda
OUT_DIR = Path("data/processed")


# =========================
# Helpers
# =========================
def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def pick_sheet(sheet_names: list[str], keywords: list[str]) -> str | None:
    for sh in sheet_names:
        nsh = norm(sh)
        if any(k in nsh for k in keywords):
            return sh
    return None


def mode_or_unknown(s: pd.Series) -> str:
    s = s.dropna().astype(str)
    if s.empty:
        return "Unknown"
    m = s.mode()
    return m.iloc[0] if not m.empty else "Unknown"


def ensure_datetime(df: pd.DataFrame, col_candidates: list[str]) -> str | None:
    for c in col_candidates:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
            return c
    return None


def ensure_user_id(df: pd.DataFrame) -> str | None:
    for c in ["user_id", "User ID", "user", "userid", "UserID"]:
        if c in df.columns:
            if c != "user_id":
                df.rename(columns={c: "user_id"}, inplace=True)
            return "user_id"
    return None


def ensure_session_id(df: pd.DataFrame) -> str | None:
    for c in ["user_session", "session_id", "Session ID", "session"]:
        if c in df.columns:
            if c != "user_session":
                df.rename(columns={c: "user_session"}, inplace=True)
            return "user_session"
    return None


def ensure_event_type(df: pd.DataFrame) -> str | None:
    for c in ["event_type", "Event Type", "type"]:
        if c in df.columns:
            if c != "event_type":
                df.rename(columns={c: "event_type"}, inplace=True)
            return "event_type"
    return None


def ensure_price(df: pd.DataFrame) -> str | None:
    for c in ["price", "Price", "revenue", "Revenue"]:
        if c in df.columns:
            if c != "price":
                df.rename(columns={c: "price"}, inplace=True)
            df["price"] = pd.to_numeric(df["price"], errors="coerce")
            return "price"
    return None


def ensure_brand(df: pd.DataFrame) -> None:
    if "brand" not in df.columns:
        for c in ["Brand", "brand_name"]:
            if c in df.columns:
                df.rename(columns={c: "brand"}, inplace=True)
                break
    if "brand" not in df.columns:
        df["brand"] = "Unknown"
    df["brand"] = df["brand"].fillna("Unknown").astype(str)


def ensure_category_main(df: pd.DataFrame) -> None:
    if "category_main" in df.columns:
        df["category_main"] = df["category_main"].fillna("Unknown").astype(str)
        return

    cat_col = None
    for c in ["category", "category_code", "Category", "Category Code"]:
        if c in df.columns:
            cat_col = c
            break

    if cat_col is None:
        df["category_main"] = "Unknown"
        return

    s = df[cat_col].astype(str)
    if s.str.contains(r"\.").any():
        df["category_main"] = s.str.split(".").str[0]
    else:
        df["category_main"] = s

    df["category_main"] = (
        df["category_main"]
        .replace({"nan": "Unknown", "None": "Unknown"})
        .fillna("Unknown")
        .astype(str)
    )


def add_month_from_time(df: pd.DataFrame, time_col: str) -> None:
    df["month"] = pd.to_datetime(df[time_col], errors="coerce").dt.to_period("M").dt.to_timestamp()


# =========================
# Main
# =========================
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not RAW_XLSX.exists():
        raise FileNotFoundError(
            f"File Excel tidak ketemu: {RAW_XLSX}\n"
            "Taruh dataset kamu di: data/raw/ lalu rename sesuai config RAW_XLSX."
        )

    xl = pd.ExcelFile(RAW_XLSX)
    sheets = xl.sheet_names

    events_sheet = pick_sheet(sheets, ["event", "events"]) or sheets[0]
    rfm_sheet = pick_sheet(sheets, ["rfm"])
    clv_sheet = pick_sheet(sheets, ["clv"])

    print("📄 Sheets detected:")
    print(f" - events_sheet: {events_sheet}")
    print(f" - rfm_sheet: {rfm_sheet}")
    print(f" - clv_sheet: {clv_sheet}")

    # =========================
    # Load events
    # =========================
    events = pd.read_excel(xl, sheet_name=events_sheet)

    ensure_user_id(events)
    ensure_session_id(events)
    ensure_event_type(events)
    ensure_price(events)
    ensure_brand(events)
    ensure_category_main(events)

    time_col = ensure_datetime(events, ["event_time", "Event Time", "timestamp", "time"])
    if time_col is None:
        raise ValueError("Kolom waktu event tidak ketemu. Harus ada event_time/timestamp/time.")

    add_month_from_time(events, time_col)

    events["event_type"] = events["event_type"].astype(str).str.lower()
    if "price" not in events.columns:
        events["price"] = 0.0
    events["price"] = pd.to_numeric(events["price"], errors="coerce").fillna(0.0)

    # save events
    events_out = OUT_DIR / "events.parquet"
    events.to_parquet(events_out, index=False)
    print(f"✅ Saved: {events_out}")

    # =========================
    # Build sessions
    # =========================
    if "user_session" not in events.columns:
        raise ValueError("Tidak ada kolom user_session/session_id di events. Needed for sessions table.")

    sess = events.groupby("user_session", as_index=False).agg(
        session_start_time=(time_col, "min"),
        session_end_time=(time_col, "max"),
        user_id=("user_id", lambda x: x.dropna().astype(str).iloc[0] if len(x.dropna()) else None),
        num_events=("event_type", "count"),
    )
    sess["month"] = pd.to_datetime(sess["session_start_time"], errors="coerce").dt.to_period("M").dt.to_timestamp()

    et = events[["user_session", "event_type"]].copy()
    et["has_view"] = (et["event_type"] == "view").astype(int)
    et["has_cart"] = (et["event_type"] == "cart").astype(int)
    et["has_purchase"] = (et["event_type"] == "purchase").astype(int)

    flags = et.groupby("user_session", as_index=False).agg(
        has_view=("has_view", "max"),
        has_cart=("has_cart", "max"),
        has_purchase=("has_purchase", "max"),
    )

    sess = sess.merge(flags, on="user_session", how="left").fillna(0)
    sess["has_view"] = sess["has_view"].astype(int)
    sess["has_cart"] = sess["has_cart"].astype(int)
    sess["has_purchase"] = sess["has_purchase"].astype(int)
    sess["cart_abandoned"] = ((sess["has_cart"] == 1) & (sess["has_purchase"] == 0)).astype(int)

    sessions_out = OUT_DIR / "sessions.parquet"
    sess.to_parquet(sessions_out, index=False)
    print(f"✅ Saved: {sessions_out}")

    # =========================
    # Build session_dim
    # =========================
    session_dim = (
        events.groupby(["month", "user_session"], as_index=False)
             .agg(
                 brand=("brand", mode_or_unknown),
                 category_main=("category_main", mode_or_unknown),
             )
    )

    session_dim_out = OUT_DIR / "session_dim.parquet"
    session_dim.to_parquet(session_dim_out, index=False)
    print(f"✅ Saved: {session_dim_out}")

    # =========================
    # Build purchases
    # =========================
    purchases = events.loc[events["event_type"].eq("purchase")].copy()
    for col in ["user_id", "month", "brand", "category_main", "price"]:
        if col not in purchases.columns:
            purchases[col] = "Unknown" if col in ["brand", "category_main"] else 0

    purchases["user_id"] = purchases["user_id"].astype(str)
    purchases["price"] = pd.to_numeric(purchases["price"], errors="coerce").fillna(0.0)
    purchases = purchases[["user_id", "month", "brand", "category_main", "price"]].dropna(subset=["month"])

    purchases_out = OUT_DIR / "purchases.parquet"
    purchases.to_parquet(purchases_out, index=False)
    print(f"✅ Saved: {purchases_out}")

    # =========================
    # Monthly revenue dim (optional)
    # =========================
    monthly_revenue_dim = (
        purchases.groupby(["month", "category_main", "brand"], as_index=False)["price"].sum()
                 .rename(columns={"price": "revenue"})
                 .sort_values(["month", "revenue"], ascending=[True, False])
    )
    mr_out = OUT_DIR / "monthly_revenue_dim.parquet"
    monthly_revenue_dim.to_parquet(mr_out, index=False)
    print(f"✅ Saved: {mr_out}")

    # =========================
    # Abandoned category by month (optional)
    # =========================
    ses_cat = (
        events.groupby(["month", "user_session"])["category_main"]
              .agg(mode_or_unknown)
              .reset_index()
    )

    cart_ses = events.loc[events["event_type"].eq("cart"), ["month", "user_session"]].dropna().drop_duplicates()
    pur_ses = events.loc[events["event_type"].eq("purchase"), ["month", "user_session"]].dropna().drop_duplicates()

    cart_ses = cart_ses.merge(ses_cat, on=["month", "user_session"], how="left")
    pur_ses = pur_ses.merge(ses_cat, on=["month", "user_session"], how="left")

    cart_cat = cart_ses.groupby(["month", "category_main"], as_index=False).size().rename(columns={"size": "cart_sessions"})
    pur_cat = pur_ses.groupby(["month", "category_main"], as_index=False).size().rename(columns={"size": "purchase_sessions"})

    aban_m = cart_cat.merge(pur_cat, on=["month", "category_main"], how="left").fillna({"purchase_sessions": 0})
    aban_m["abandonment_rate"] = (aban_m["cart_sessions"] - aban_m["purchase_sessions"]) / aban_m["cart_sessions"]

    aban_out = OUT_DIR / "abandoned_category_by_month.parquet"
    aban_m.to_parquet(aban_out, index=False)
    print(f"✅ Saved: {aban_out}")

    # =========================
    # M1 retention trend (optional) - CLEAN, no cohort_month bug
    # =========================
    if not purchases.empty:
        pu = purchases[["user_id", "month"]].dropna().drop_duplicates().copy()
        pu["user_id"] = pu["user_id"].astype(str)
        pu["month"] = pd.to_datetime(pu["month"], errors="coerce")
        pu = pu.dropna(subset=["month"])

        first = pu.groupby("user_id", as_index=False)["month"].min().rename(columns={"month": "cohort_month"})
        first["cohort_month"] = first["cohort_month"].dt.to_period("M").dt.to_timestamp()

        user_month = set(zip(pu["user_id"], pu["month"].dt.to_period("M").astype(str)))

        retained_flag = []
        for u, cm in zip(first["user_id"], first["cohort_month"]):
            next_p = (pd.Timestamp(cm) + pd.offsets.MonthBegin(1)).to_period("M")
            retained_flag.append(1 if (u, str(next_p)) in user_month else 0)

        first["retained_m1"] = retained_flag

        m1_line = (
            first.groupby("cohort_month", as_index=False)
                 .agg(cohort_size=("user_id", "count"), retained=("retained_m1", "sum"))
        )
        m1_line["retention_rate"] = (m1_line["retained"] / m1_line["cohort_size"]).fillna(0.0)
        m1_line = m1_line.sort_values("cohort_month")

        m1_out = OUT_DIR / "m1_retention_trend.parquet"
        m1_line.to_parquet(m1_out, index=False)
        print(f"✅ Saved: {m1_out}")

    # =========================
    # RFM: load from sheet (preferred) else compute
    # =========================
    if rfm_sheet is not None:
        rfm_df = pd.read_excel(xl, sheet_name=rfm_sheet)
        ensure_user_id(rfm_df)
        if "Segment" not in rfm_df.columns and "rfm_segment" in rfm_df.columns:
            rfm_df.rename(columns={"rfm_segment": "Segment"}, inplace=True)
        rfm_out = OUT_DIR / "rfm.parquet"
        rfm_df.to_parquet(rfm_out, index=False)
        print(f"✅ Saved: {rfm_out} (from Excel sheet)")
    else:
        if purchases.empty:
            print("⚠️ purchases kosong. Tidak bisa hitung RFM.")
        else:
            max_month = purchases["month"].max()
            snapshot = pd.Timestamp(max_month) + pd.offsets.MonthEnd(0)

            rfm_calc = purchases.copy()
            rfm_calc["month"] = pd.to_datetime(rfm_calc["month"], errors="coerce")
            rfm_calc = rfm_calc.dropna(subset=["month"])

            user_last = rfm_calc.groupby("user_id", as_index=False)["month"].max().rename(columns={"month": "last_purchase_month"})
            user_freq = rfm_calc.groupby("user_id", as_index=False).size().rename(columns={"size": "frequency"})
            user_mon = rfm_calc.groupby("user_id", as_index=False)["price"].sum().rename(columns={"price": "monetary"})

            rfm_df = user_last.merge(user_freq, on="user_id").merge(user_mon, on="user_id")
            rfm_df["recency_days"] = (snapshot - (rfm_df["last_purchase_month"] + pd.offsets.MonthEnd(0))).dt.days

            # simple 3-bin scores
            rfm_df["R_score"] = pd.qcut(rfm_df["recency_days"].rank(method="first"), 3, labels=[3, 2, 1]).astype(int)
            rfm_df["F_score"] = pd.qcut(rfm_df["frequency"].rank(method="first"), 3, labels=[1, 2, 3]).astype(int)
            rfm_df["M_score"] = pd.qcut(rfm_df["monetary"].rank(method="first"), 3, labels=[1, 2, 3]).astype(int)

            def seg(row):
                if row["R_score"] == 3 and row["F_score"] == 3:
                    return "Champions"
                if row["R_score"] >= 2 and row["F_score"] >= 2:
                    return "Loyal"
                if row["R_score"] == 1 and row["F_score"] >= 2:
                    return "At Risk"
                if row["R_score"] == 1 and row["F_score"] == 1:
                    return "Hibernating"
                return "Potential"

            rfm_df["Segment"] = rfm_df.apply(seg, axis=1)

            rfm_out = OUT_DIR / "rfm.parquet"
            rfm_df.to_parquet(rfm_out, index=False)
            print(f"✅ Saved: {rfm_out} (computed)")

    # =========================
    # CLV: load from sheet if available
    # =========================
    if clv_sheet is not None:
        clv_df = pd.read_excel(xl, sheet_name=clv_sheet)
        ensure_user_id(clv_df)

        if "predicted_clv_3m" not in clv_df.columns:
            for cand in ["Predicted_CLV_3M", "clv_3m", "CLV_3M", "predicted_clv", "CLV"]:
                if cand in clv_df.columns:
                    clv_df.rename(columns={cand: "predicted_clv_3m"}, inplace=True)
                    break

        if "predicted_clv_3m" in clv_df.columns:
            clv_df["predicted_clv_3m"] = pd.to_numeric(clv_df["predicted_clv_3m"], errors="coerce")

        clv_out = OUT_DIR / "clv.parquet"
        clv_df.to_parquet(clv_out, index=False)
        print(f"✅ Saved: {clv_out} (from Excel sheet)")
    else:
        print("ℹ️ Sheet CLV tidak ditemukan. Skip clv.parquet.")

    print("\n✅ Preprocess selesai. Semua output ada di data/processed/")


if __name__ == "__main__":
    main()
