from pathlib import Path
import pandas as pd

IN_PATH = Path("data/processed/events.parquet")
OUT_PATH = Path("data/processed/purchases.parquet")

def main():
    ev = pd.read_parquet(IN_PATH)

    if "month" not in ev.columns:
        ev["event_time"] = pd.to_datetime(ev["event_time"], errors="coerce")
        ev["month"] = ev["event_time"].dt.to_period("M").dt.to_timestamp()

    if "event_type" not in ev.columns:
        raise ValueError("No event_type column found.")

    ev["etype"] = ev["event_type"].astype(str).str.lower()
    pur = ev.loc[ev["etype"].eq("purchase")].copy()

    # user id
    if "user_id" not in pur.columns:
        for cand in ["User ID", "user"]:
            if cand in pur.columns:
                pur = pur.rename(columns={cand: "user_id"})
                break

    for c in ["brand", "category_main"]:
        if c not in pur.columns:
            pur[c] = "Unknown"

    if "price" in pur.columns:
        pur["price"] = pd.to_numeric(pur["price"], errors="coerce").fillna(0)
    else:
        pur["price"] = 0

    pur = pur[["user_id", "month", "brand", "category_main", "price"]].dropna(subset=["month"])
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pur.to_parquet(OUT_PATH, index=False)

    print("✅ Created:", OUT_PATH, "Rows:", len(pur))

if __name__ == "__main__":
    main()
