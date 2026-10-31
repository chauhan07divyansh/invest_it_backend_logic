#!/usr/bin/env python3
"""
RECOVER a portfolio from an existing analysis checkpoint — NO new API calls,
NO token/quota spend. Reads /app/full_universe_analysis.json (written by the
main script as it ran) and re-runs only the in-memory filter + allocation.

You can also try different budget/risk on the SAME saved analysis for free,
because the checkpoint is the raw analysis (risk is applied afterwards).

Usage:
    cd /app && python3 recover_portfolio.py [BUDGET] [RISK]
    e.g.  python3 recover_portfolio.py 100000 HIGH
"""
import sys, json
sys.path.insert(0, "/app")

BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 100000.0
RISK   = (sys.argv[2] if len(sys.argv) > 2 else "HIGH").upper()
RAW    = "/app/full_universe_analysis.json"
OUT    = "/app/full_universe_portfolio.json"

if RISK not in ("LOW", "MEDIUM", "HIGH"):
    print(f"RISK must be LOW|MEDIUM|HIGH, got {RISK!r}"); sys.exit(1)
if not (10000 <= BUDGET <= 10000000):
    print(f"BUDGET must be 10,000 .. 10,000,000, got {BUDGET}"); sys.exit(1)

from main import trading_api
sw = getattr(trading_api, "swing_system", None)
if sw is None:
    print("trading_api.swing_system is None — can't access filter/allocation methods.")
    sys.exit(1)

try:
    with open(RAW) as f:
        all_results = json.load(f)
except FileNotFoundError:
    print(f"No checkpoint at {RAW} — nothing to recover. The run may not have")
    print("saved anything, or the container redeployed and wiped /app.")
    sys.exit(1)

print(f"Loaded {len(all_results)} analysed stocks from checkpoint (no new API calls).")

# JSON round-trip can stringify numeric fields; coerce the ones filter/alloc read.
def _f(x, d=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return d

for r in all_results:
    r["swing_score"]   = _f(r.get("swing_score"))
    r["current_price"] = _f(r.get("current_price"))
    rm = r.get("risk_metrics") or {}
    if "volatility" in rm:
        rm["volatility"] = _f(rm.get("volatility"), 1.0)
    r["risk_metrics"] = rm
    tp = r.get("trading_plan") or {}
    if "stop_loss" in tp:
        tp["stop_loss"] = _f(tp.get("stop_loss"))
    r["trading_plan"] = tp

all_results.sort(key=lambda x: x.get("swing_score", 0.0), reverse=True)
filtered  = sw.filter_stocks_by_risk_appetite(all_results, RISK)
portfolio = sw.generate_portfolio_allocation(filtered, BUDGET, RISK)

summary = {
    "budget": BUDGET, "risk": RISK,
    "analyzed_valid": len(all_results),
    "passed_risk_filter": len(filtered),
    "recovered_from_checkpoint": True,
    "portfolio": portfolio,
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
if portfolio:
    print(f"\n  allocated Rs {total:,.0f} of Rs {BUDGET:,.0f} "
          f"({100*total/BUDGET:.1f}%) across {len(portfolio)} positions")
else:
    print("\n  portfolio is EMPTY — nothing passed the BUY/volatility filter.")
    print("  See 'ranked_universe' in the output file for the full scored list.")
print(f"\nSaved: {OUT}")
