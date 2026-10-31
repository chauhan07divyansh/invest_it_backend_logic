#!/usr/bin/env python3
"""
Read a saved portfolio back from Neon. Survives every restart/redeploy.

Usage:
    cd /app && python3 read_portfolio.py [RUN_ID]
RUN_ID defaults to today's date. With no matching run it lists all saved runs.
"""
import sys, json, datetime
sys.path.insert(0, "/app")

RUN_ID = sys.argv[1] if len(sys.argv) > 1 else datetime.date.today().isoformat()

from main import app, db
from sqlalchemy import text

with app.app_context():
    db.session.remove()
    row = db.session.execute(
        text("SELECT created_at, budget, risk, analyzed_valid, passed_filter, portfolio "
             "FROM universe_portfolio WHERE run_id=:r"), {"r": RUN_ID}).fetchone()
    if row is None:
        print(f"No saved portfolio for run_id '{RUN_ID}'. Saved runs:")
        for r in db.session.execute(text(
                "SELECT run_id, created_at, risk, passed_filter FROM universe_portfolio "
                "ORDER BY created_at DESC")):
            print(f"   {r[0]:<14} {r[1]}  risk {r[2]}  passed {r[3]}")
        sys.exit(0)

created, budget, risk, av, pf, pjson = row
portfolio = json.loads(pjson)
print(f"Run {RUN_ID} | saved {created} | budget Rs {float(budget):,.0f} | risk {risk}")
print(f"analysed {av}, passed filter {pf}, positions {len(portfolio)}\n")
for p in portfolio:
    print(f"  #{p['rank']:>2}  {p['symbol']:<12} {str(p.get('sector',''))[:20]:<20} "
          f"score {p['score']:>5.1f}  {p['number_of_shares']:>4} sh  "
          f"Rs {p['investment_amount']:>12,.0f}  ({p['allocation_pct']:>4.1f}%)  "
          f"SL {p.get('stop_loss','-')}")
tot = sum(p["investment_amount"] for p in portfolio)
if portfolio:
    print(f"\n  allocated Rs {tot:,.0f} of Rs {float(budget):,.0f} "
          f"({100*tot/float(budget):.1f}%)")
