#!/usr/bin/env python3
"""
Build a portfolio from the full analyzable universe, PERSISTED TO NEON.

Why: Render wipes /app on every restart. This writes every analysed chunk to
your Neon database as it goes, so a restart/redeploy NEVER wastes the token
spend — on restart it resumes and only analyses what's left. The final
portfolio is saved to Neon too, queryable any time with read_portfolio.py.

Tables (auto-created, no migration):
  universe_analysis(run_id, symbol, data, created_at)   -- per-stock checkpoint
  universe_portfolio(run_id, created_at, budget, risk, analyzed_valid,
                     passed_filter, portfolio, ranked)  -- final result

Usage (Render shell; nohup so a dropped tab can't orphan it):
    cd /app && nohup python3 build_universe_portfolio_neon.py 100000 HIGH > neon_run.log 2>&1 &
    tail -f neon_run.log
RUN_ID defaults to today's date, so a same-day restart RESUMES the same run.
Re-running after it's fully analysed just rebuilds the portfolio (free) and
lets you try a different budget/risk with NO new API calls.

    python3 build_universe_portfolio_neon.py [BUDGET] [RISK] [RUN_ID]
"""
import sys, os, json, time, datetime
sys.path.insert(0, "/app")

BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 100000.0
RISK   = (sys.argv[2] if len(sys.argv) > 2 else "HIGH").upper()
RUN_ID = sys.argv[3] if len(sys.argv) > 3 else datetime.date.today().isoformat()
CHUNK  = 100

if RISK not in ("LOW", "MEDIUM", "HIGH"):
    print(f"RISK must be LOW|MEDIUM|HIGH, got {RISK!r}"); sys.exit(1)
if not (10000 <= BUDGET <= 10000000):
    print(f"BUDGET must be 10,000 .. 10,000,000, got {BUDGET}"); sys.exit(1)

from main import app, db, trading_api
from sqlalchemy import text

sw = getattr(trading_api, "swing_system", None)
if sw is None:
    print("trading_api.swing_system is None — swing system not initialised."); sys.exit(1)

DDL = [
    ("CREATE TABLE IF NOT EXISTS universe_analysis ("
     "run_id TEXT, symbol TEXT, data TEXT, created_at TEXT, "
     "PRIMARY KEY (run_id, symbol))"),
    ("CREATE TABLE IF NOT EXISTS universe_portfolio ("
     "run_id TEXT PRIMARY KEY, created_at TEXT, budget REAL, risk TEXT, "
     "analyzed_valid INTEGER, passed_filter INTEGER, portfolio TEXT, ranked TEXT)"),
]

def db_do(stmt, params=None, fetch=False):
    """Run one statement in a fresh session (fresh connection avoids Neon idle
    drops), with one retry."""
    last = None
    for attempt in (0, 1):
        try:
            with app.app_context():
                db.session.remove()
                res = db.session.execute(text(stmt), params if params is not None else {})
                out = res.fetchall() if fetch else None
                db.session.commit()
                return out
        except Exception as e:
            last = e
            print(f"   [db retry {attempt}] {e}")
            try:
                with app.app_context():
                    db.session.rollback(); db.session.remove()
            except Exception:
                pass
            time.sleep(3)
    raise last

for d in DDL:
    db_do(d)

done = {r[0] for r in db_do(
    "SELECT symbol FROM universe_analysis WHERE run_id=:r", {"r": RUN_ID}, fetch=True)}
symbols = sw.get_all_analyzable_symbols()
todo = [s for s in symbols if s not in done]
print(f"Run '{RUN_ID}' | universe {len(symbols)} | already analysed {len(done)} | "
      f"remaining {len(todo)} | budget Rs {BUDGET:,.0f} | risk {RISK}")
if len(symbols) < 600:
    print(f"  NOTE: only {len(symbols)} symbols — CSV merge may not be active; "
          "expected ~780.")

t0 = time.time()
for i in range(0, len(todo), CHUNK):
    chunk = todo[i:i + CHUNK]
    tc = time.time()
    res = sw.analyze_stocks_parallel(chunk, max_workers=10)
    now = datetime.datetime.utcnow().isoformat()
    rows = [{"r": RUN_ID, "s": x.get("symbol"),
             "d": json.dumps(x, default=str), "c": now}
            for x in res if x.get("symbol")]
    if rows:
        db_do("INSERT INTO universe_analysis (run_id, symbol, data, created_at) "
              "VALUES (:r,:s,:d,:c) "
              "ON CONFLICT (run_id, symbol) DO UPDATE SET data=EXCLUDED.data, "
              "created_at=EXCLUDED.created_at", rows)
    print(f"  chunk {i // CHUNK + 1}/{(len(todo)+CHUNK-1)//CHUNK}: "
          f"sent {len(chunk)}, valid {len(res)}, saved {len(rows)} to Neon "
          f"({time.time()-tc:.0f}s)")

# Build portfolio from ALL rows stored for this run (survives restarts).
allrows = db_do("SELECT data FROM universe_analysis WHERE run_id=:r",
                {"r": RUN_ID}, fetch=True)
all_results = [json.loads(r[0]) for r in allrows]

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
ranked = [{"symbol": r.get("symbol"), "score": r.get("swing_score"),
           "signal": r.get("trading_plan", {}).get("entry_signal"),
           "sector": r.get("sector"), "price": r.get("current_price")}
          for r in all_results]

db_do("INSERT INTO universe_portfolio "
      "(run_id, created_at, budget, risk, analyzed_valid, passed_filter, portfolio, ranked) "
      "VALUES (:r,:c,:b,:k,:av,:pf,:p,:rk) "
      "ON CONFLICT (run_id) DO UPDATE SET created_at=EXCLUDED.created_at, "
      "budget=EXCLUDED.budget, risk=EXCLUDED.risk, analyzed_valid=EXCLUDED.analyzed_valid, "
      "passed_filter=EXCLUDED.passed_filter, portfolio=EXCLUDED.portfolio, ranked=EXCLUDED.ranked",
      {"r": RUN_ID, "c": datetime.datetime.utcnow().isoformat(), "b": BUDGET, "k": RISK,
       "av": len(all_results), "pf": len(filtered),
       "p": json.dumps(portfolio, default=str), "rk": json.dumps(ranked, default=str)})

print(f"\n=== PORTFOLIO  (run {RUN_ID}, risk {RISK}, budget Rs {BUDGET:,.0f}) ===")
print(f"from {len(all_results)} analysed, {len(filtered)} passed the risk/BUY filter, "
      f"top {len(portfolio)} allocated:\n")
for p in portfolio:
    print(f"  #{p['rank']:>2}  {p['symbol']:<12} {str(p['sector'])[:20]:<20} "
          f"score {p['score']:>5.1f}  {p['number_of_shares']:>4} sh  "
          f"Rs {p['investment_amount']:>12,.0f}  ({p['allocation_pct']:>4.1f}%)")
tot = sum(p["investment_amount"] for p in portfolio)
if portfolio:
    print(f"\n  allocated Rs {tot:,.0f} of Rs {BUDGET:,.0f} "
          f"({100*tot/BUDGET:.1f}%) across {len(portfolio)} positions")
else:
    print("\n  portfolio EMPTY — nothing passed the BUY/volatility filter "
          "(see 'ranked' in Neon).")
print(f"\nSaved to Neon. Read any time (survives restarts):")
print(f"    python3 read_portfolio.py {RUN_ID}")
print(f"Elapsed {time.time()-t0:.0f}s")
