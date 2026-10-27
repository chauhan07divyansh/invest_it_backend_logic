"""
import_universe.py — expand stocks.db analysis universe from an NSE index CSV.

SOURCE (download from niftyindices.com, free):
  - ind_niftytotalmarket_list.csv   (~750 liquid names)   <- recommended
  - ind_nifty500list.csv            (~500 liquid names)   <- safer/more liquid
Both have columns: "Company Name","Industry","Symbol","Series","ISIN Code"

What it does:
  - reads the CSV, takes EQ-series rows
  - skips symbols already in stocks.db (no dupes, no overwrite)
  - adds new ones with the REAL sector from the CSV's Industry column
  - marks new long-tail as market_cap='Small', index_name='AnalysisOnly'
    so your portfolio queries (which filter by cap/index) never pick them
  - DRY RUN by default — set CONFIRM=True to actually write

Run in the Render shell:
  cd /app && python3 import_universe.py /path/to/ind_niftytotalmarket_list.csv
"""
import sys, csv, os
sys.path.insert(0, "/app")
from database.stock_db import get_stock_db

CONFIRM = False                      # <-- set True to actually write
TARGET_TOTAL = 1000                  # stop once the DB reaches this many active symbols
NEW_CAP      = "Small"               # tier for the added long-tail (keeps them out of portfolio)
NEW_INDEX    = "AnalysisOnly"        # index_name marker so portfolio filters exclude them

def main(csv_path):
    db = get_stock_db()
    existing = set(s.upper() for s in db.get_all_symbols(active_only=False))
    before = len(existing)
    print(f"stocks.db currently has {before} symbols.")

    # read CSV
    rows = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            sym = (r.get("Symbol") or r.get("SYMBOL") or "").strip().upper()
            name = (r.get("Company Name") or r.get("NAME OF COMPANY") or sym).strip()
            industry = (r.get("Industry") or "Unknown").strip() or "Unknown"
            series = (r.get("Series") or r.get("SERIES") or "EQ").strip().upper()
            if not sym or series not in ("EQ", "BE", ""):   # equities only
                continue
            rows.append((sym, name, industry))

    print(f"CSV has {len(rows)} equity rows.")
    to_add = [(s, n, ind) for (s, n, ind) in rows if s not in existing]
    print(f"{len(to_add)} are NOT already in stocks.db (candidates to add).")

    # respect the target cap
    room = max(0, TARGET_TOTAL - before)
    to_add = to_add[:room]
    print(f"Will add up to {room} to reach ~{TARGET_TOTAL}. Adding {len(to_add)} this run.\n")

    if not CONFIRM:
        print("DRY RUN — showing first 25 that WOULD be added:\n")
        for s, n, ind in to_add[:25]:
            print(f"  {s:14s} {ind:28s} {n}")
        print(f"\n... {len(to_add)} total. Set CONFIRM=True to write.")
        return

    added = skipped = 0
    for s, n, ind in to_add:
        ok = db.add_stock(symbol=s, name=n, sector_name=ind,
                          market_cap=NEW_CAP, index_name=NEW_INDEX)
        added += 1 if ok else 0
        skipped += 0 if ok else 1
    print(f"\n✅ Added {added}, skipped {skipped} (dupes).")
    print("New stats:", db.get_stats())

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python3 import_universe.py <path-to-nse-index-csv>")
        sys.exit(1)
    main(sys.argv[1])
