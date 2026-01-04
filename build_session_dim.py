from pathlib import Path
import pandas as pd

IN_PATH = Path("data/processed/events.parquet")
OUT_PATH = Path("data/processed/session_dim.parquet")

def mode_or_unknown(s: pd.Series) -> str:
    s = s.dropna().astype(str)
    if s.empty:
        return "Unknown"
    m = s.mode()
    return m.iloc[0] if not m.empty else "Unknown"

def main():
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Missing {IN_PATH}")

    ev = pd.read_parquet(IN_PATH)

    # --- detect columns safely ---
    # session id
    session_col = None
    for cand in ["user_session", "session_id", "Session ID"]:
        if cand in ev.columns:
            session_col = cand
            break
    if session_col is None:
        raise ValueError("No session id column found (expected user_session/session_id).")

    # month
    if "month" not in ev.columns:
        # try derive from event_time
        if "event_time" in ev.columns:
            ev["event_time"] = pd.to_datetime(ev["event_time"], errors="coerce")
            ev["month"] = ev["event_time"].dt.to_period("M").dt.to_timestamp()
        else:
            raise ValueError("No 'month' or 'event_time' column found.")

    # brand/category
    if "brand" not in ev.columns:
        ev["brand"] = "Unknown"
    if "category_main" not in ev.columns:
        # fallback if category exists
        if "category" in ev.columns:
            ev["category_main"] = ev["category"].astype(str)
        else:
            ev["category_main"] = "Unknown"

    # --- build session_dim ---
    session_dim = (
        ev.groupby(["month", session_col], as_index=False)
          .agg(
              brand=("brand", mode_or_unknown),
              category_main=("category_main", mode_or_unknown),
          )
          .rename(columns={session_col: "user_session"})
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    session_dim.to_parquet(OUT_PATH, index=False)

    print("✅ Created:", OUT_PATH)
    print("Rows:", len(session_dim), "Cols:", session_dim.columns.tolist())

if __name__ == "__main__":
    main()
