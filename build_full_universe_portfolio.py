#!/usr/bin/env python3
"""
ONE-OFF: build a portfolio from the FULL analyzable universe (~780 stocks).

Does NOT change the live app:
  - does not modify indian_stocks (portfolio universe)
  - does not write the 'batch_analysis:all_stocks' portfolio cache
  - only reads trading_api.swing_system and runs its existing pipeline
So there is nothing to "remove afterwards" — the running app is untouched.

It DOES consume real Fyers / EventRegistry / HF quota (one analysis per stock).
Run off-peak. Checkpoints raw results every chunk so a dropped shell is recoverable.

Usage (in the Render shell):
    cd /app && python3 build_full_universe_portfolio.py [BUDGET] [RISK]
    e.g.  python3 build_full_universe_portfolio.py 100000 MEDIUM
    BUDGET in rupees (10000 .. 10000000), RISK in LOW|MEDIUM|HIGH.
"""
import sys, json, time
sys.path.insert(0, "/app")

BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 100000.0
RISK   = (sys.argv[2] if len(sys.argv) > 2 else "MEDIUM").upper()
CHUNK  = 100
RAW    = "/app/full_universe_analysis.json"
OUT    = "/app/full_universe_portfolio.json"

if RISK not in ("LOW", "MEDIUM", "HIGH"):
    print(f"RISK must be LOW|MEDIUM|HIGH, got {RISK!r}"); sys.exit(1)
if not (10000 <= BUDGET <= 10000000):
    print(f"BUDGET must be 10,000 .. 10,000,000, got {BUDGET}"); sys.exit(1)

from main import trading_api

sw = getattr(trading_api, "swing_system", None)
if sw is None:
    print("trading_api.swing_system is not initialised in this process — "
          "the swing system didn't load on import. Tell Claude and we'll adapt.")
    sys.exit(1)

symbols = sw.get_all_analyzable_symbols()
print(f"Universe: {len(symbols)} symbols | budget Rs {BUDGET:,.0f} | risk {RISK}")
print("Analysing in chunks of", CHUNK, "— this is the slow, quota-spending part.\n")

all_results = []
t_start = time.time()
for i in range(0, len(symbols), CHUNK):
    chunk = symbols[i:i + CHUNK]
    t0 = time.time()
    res = sw.analyze_stocks_parallel(chunk, max_workers=10)
    all_results.extend(res)
    try:
        with open(RAW, "w") as f:
            json.dump(all_results, f, default=str)
    except Exception as e:
        print("  (checkpoint write failed:", e, ")")
    print(f"  chunk {i // CHUNK + 1}/{(len(symbols)+CHUNK-1)//CHUNK}: "
          f"sent {len(chunk)}, valid {len(res)}, cumulative {len(all_results)} "
          f"({time.time()-t0:.0f}s)")

# Union re-sort (each chunk was sorted on its own)
all_results.sort(key=lambda x: x.get("swing_score", 0), reverse=True)
filtered  = sw.filter_stocks_by_risk_appetite(all_results, RISK)
portfolio = sw.generate_portfolio_allocation(filtered, BUDGET, RISK)

summary = {
    "budget": BUDGET, "risk": RISK,
    "universe_size": len(symbols),
    "analyzed_valid": len(all_results),
    "passed_risk_filter": len(filtered),
    "elapsed_seconds": round(time.time() - t_start),
    "portfolio": portfolio,
    # full ranked universe (symbol + score + signal) so you can see all 780 ranked
    "ranked_universe": [
        {"symbol": r.get("symbol"), "score": r.get("swing_score"),
         "signal": r.get("trading_plan", {}).get("entry_signal"),
         "sector": r.get("sector"), "price": r.get("current_price")}
        for r in all_results
    ],
}
with open(OUT, "w") as f:
    json.dump(summary, f, indent=2, default=str)

print(f"\n=== PORTFOLIO  (risk {RISK}, budget Rs {BUDGET:,.0f}) ===")
print(f"from {len(all_results)} analysed, {len(filtered)} passed the risk/BUY filter, "
      f"top {len(portfolio)} allocated:\n")
for p in portfolio:
    print(f"  #{p['rank']:>2}  {p['symbol']:<12} {str(p['sector'])[:20]:<20} "
          f"score {p['score']:>5.1f}  {p['number_of_shares']:>4} sh  "
          f"Rs {p['investment_amount']:>12,.0f}  ({p['allocation_pct']:>4.1f}%)")
total = sum(p["investment_amount"] for p in portfolio)
print(f"\n  allocated Rs {total:,.0f} of Rs {BUDGET:,.0f} "
      f"({100*total/BUDGET:.1f}%) across {len(portfolio)} positions")
print(f"\nSaved: {OUT}")
print(f"Raw analysis (all {len(all_results)} valid): {RAW}")
print("\nLive app unchanged — nothing to revert.")
