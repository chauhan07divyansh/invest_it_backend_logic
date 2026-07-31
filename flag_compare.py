# Off-vs-On comparison for ENABLE_NEW_LOGIC (regime exclude + sector cap).
# Analyzes the universe ONCE, then runs filter_stocks_by_risk_appetite BOTH ways
# by temporarily monkeypatching the module flag — NO prod flag change, NO deploy.
# Run in Render shell:  python3 flag_compare.py   (optionally set BUDGET / RISK below)
import sys, os
sys.path.insert(0, '/app')
from main import trading_api

BUDGET = 50000
RISK   = 'HIGH'   # HIGH = volatility cap 1.0 (widest), isolates regime/sector effect best

sw = trading_api.swing_system
import services.trading.swing_trading as swmod

# 1) analyze the FULL portfolio universe ONCE (same data feeds both variants)
symbols = sw.get_all_stock_symbols()
print(f'Analyzing {len(symbols)} portfolio symbols (this may take a bit)...')
results = sw.analyze_stocks_parallel(symbols, max_workers=8)
print(f'Got {len(results)} scored results\n')

# what regime are we actually in? (drives whether ON differs much)
regime = sw.detect_market_regime()
print(f'>>> CURRENT MARKET REGIME: {regime}')
print(f'    regime exclude sectors: {sw.REGIME_EXCLUDE.get(regime, [])}')
print(f'    max per sector (ON):    {sw.MAX_PER_SECTOR.get(regime, 2)}\n')

def run(flag_value):
    swmod.ENABLE_NEW_LOGIC = flag_value          # monkeypatch module-level flag
    filtered = sw.filter_stocks_by_risk_appetite(results, RISK)
    port = sw.generate_portfolio_allocation(filtered, BUDGET, RISK)
    return filtered, port

# 2) run BOTH ways
off_filtered, off_port = run(False)
on_filtered,  on_port  = run(True)
swmod.ENABLE_NEW_LOGIC = False                    # restore safe default in this process

def summarize(tag, filtered, port):
    syms = [p['symbol'].split('.')[0] for p in port]
    from collections import Counter
    sec = Counter(p['sector'] for p in port)
    print(f'--- {tag} ---')
    print(f'  passed filter: {len(filtered)} stocks')
    print(f'  portfolio: {len(port)} positions')
    print(f'  holdings: {syms}')
    print(f'  sectors:  {dict(sec)}')
    print(f'  avg score: {sum(p["score"] for p in port)/len(port):.1f}' if port else '  (empty)')
    print()
    return set(syms), sec

print('='*64)
off_syms, off_sec = summarize('FLAG OFF (current production)', off_filtered, off_port)
on_syms,  on_sec  = summarize('FLAG ON  (regime + sector cap)', on_filtered, on_port)

# 3) the diff
print('='*64)
print('DIFF (what turning the flag ON changes):')
dropped = off_syms - on_syms
added   = on_syms - off_syms
print(f'  dropped by new logic: {sorted(dropped) if dropped else "none"}')
print(f'  newly included:       {sorted(added) if added else "none"}')
print(f'  sector concentration OFF: {dict(off_sec)}')
print(f'  sector concentration ON:  {dict(on_sec)}')
if not dropped and not added:
    print('\n  >>> NO DIFFERENCE. In this regime the flag does not change the portfolio.')
    print(f'  >>> (regime={regime}: exclude={sw.REGIME_EXCLUDE.get(regime,[])}, cap={sw.MAX_PER_SECTOR.get(regime,2)}/sector)')
    print('  >>> Sector cap only bites when >cap stocks from one sector would have been picked.')
