"""
SentiQuant — Performance Metrics Suite
======================================
Computes the exact metrics that B2B diligence (RAs/brokers/PMS) will ask for,
from your real cycle data. Two levels:

  TRADE-LEVEL (from ~90 individual positions):
    win rate, profit factor, expectancy, avg winner / avg loser,
    payoff ratio, best/worst trade, avg holding return

  PORTFOLIO-LEVEL (from 9+ cycle outcomes + daily NAV series):
    total/annualized return, alpha, Sharpe, Sortino, max drawdown,
    volatility, up/down-capture, hit rate, cycle stats

HOW TO RUN (Render shell — has Fyers + your data provider):
    python3 compute_metrics.py

It pulls daily closes for each stock over each cycle window via your existing
data provider, builds an equal-weighted daily NAV per cycle, and computes
everything. NIFTY is fetched the same way you fetch it for regime detection.

⚠️ FILL IN THE `CYCLES` LIST BELOW with your real portfolios first.
   Each cycle = start date, end date, and the 10 symbols (bare tickers).
   Two are pre-filled (Jul 7, Jul 8-12:15) as templates — replace/extend.

No transaction costs are modeled by default (matches your paper track record).
Set COST_BPS > 0 to haircut for slippage+brokerage and see costed numbers too.
"""
import sys
sys.path.insert(0, '/app')
from datetime import date, datetime, timedelta
import numpy as np

# ───────────────────────── CONFIG ─────────────────────────
COST_BPS = 0          # round-trip cost in basis points (e.g. 30 = 0.30%). 0 = paper.
RISK_FREE_ANNUAL = 0.065   # ~6.5% Indian risk-free, for Sharpe
TRADING_DAYS = 252

# Each cycle: name, start (entry) date, end date, and the 10 bare symbols.
# Dates are the actual holding window. Symbols must be bare (no .NSE).
# >>> REPLACE THESE with your real 9 cycles. Two templates shown. <<<
# Each cycle: name, start, end, symbols. Optionally `shares` (rupee-weighted
# by actual holdings) — if omitted, the cycle is treated as equal-weighted.
# `entry` prices are optional; when present the script cross-checks them
# against the Fyers close on the start date and warns on mismatch.
CYCLES = [
    {
        "name": "Dec 11",
        "start": "2025-12-11", "end": "2025-12-24",
        "symbols": ["CLEAN","METROBRAND","GODREJCP","GRANULES","IDFCFIRSTB",
                    "HDFCLIFE","MANAPPURAM","PNBHOUSING","UTIAMC","FINPIPE"],
        "shares":  [11,8,8,17,124,12,35,11,8,60],
        "entry":   [898.05,1165.80,1147.90,564.75,80.61,775.20,285.15,900.15,1124.20,165.43],
    },
    {
        "name": "P5 (Feb 4-23)",
        "start": "2026-02-04", "end": "2026-02-23",
        "symbols": ["LUPIN","NESTLEIND","COALINDIA","SBIN","AXISBANK","TITAN",
                    "TCS","NTPC","HCLTECH","ADANIPORTS","HINDALCO","BPCL"],
        "shares":  [4,6,20,8,6,2,2,22,4,5,8,19],
        "entry":   [2185.90,1308.00,429.40,1064.20,1356.20,4068.60,
                    3225.30,358.55,1695.30,1530.80,955.30,373.45],
    },
    {
        "name": "P1 (Apr 7-27)",
        "start": "2026-04-07", "end": "2026-04-27",
        "symbols": ["JUBLPHARMA","INDIANB","HINDCOPPER","WIPRO","ABFRL",
                    "GODREJPROP","TATASTEEL","ADANIPORTS","JUSTDIAL","ROUTE"],
        "entry":   [854.25,899.85,503.35,197.29,58.94,
                    1584.70,196.10,1387.10,520.05,464.00],
    },
    {
        "name": "P2 (Apr 27-May 12)",
        "start": "2026-04-27", "end": "2026-05-12",
        "symbols": ["COALINDIA","BRITANNIA","GODREJCP","AMBUJACEM","PIIND",
                    "NYKAA","UPL","ACC","INDHOTEL","IRCTC"],
        "shares":  [21,1,9,22,3,38,15,7,15,18],
        "entry":   [456.00,5731.00,1090.00,451.00,3081.00,
                    263.00,631.00,1413.00,635.00,541.00],
    },
    {
        "name": "May 26-Jun 9 (was P3/P6)",
        "start": "2026-05-26", "end": "2026-06-09",
        "symbols": ["IOC","AMBUJACEM","ASHOKLEY","CANBK","PERSISTENT",
                    "BPCL","PNB","ASTRAL","FINPIPE","HDFCLIFE"],
        "shares":  [69,22,60,74,1,32,94,6,56,16],
        "entry":   [144,442,164,134,5038,308,106,1552,176,620],
    },
    {
        "name": "P8 (Jun 15)",
        "start": "2026-06-15", "end": "2026-06-29",
        "symbols": ["TVSMOTOR","HAVELLS","CANBK","IRCTC","M&M","UTIAMC",
                    "HEROMOTOCO","BATAINDIA","NAZARA","IGL"],
        "shares":  [2,8,74,19,3,10,1,14,34,60],
        "entry":   [3395,1184,134,526,3100,944,5037,687,287,166],
    },
    {
        "name": "Jul 7",
        "start": "2026-07-07", "end": "2026-07-21",
        "symbols": ["SIGNATURE","HAPPSTMNDS","SBIN","HINDUNILVR","AUBANK",
                    "PERSISTENT","BEL","DEVYANI","ESCORTS","MUTHOOTFIN"],
    },
    {
        "name": "Jul 8 (12:15)",
        "start": "2026-07-08", "end": "2026-07-22",
        "symbols": ["DEVYANI","PERSISTENT","TECHM","HAPPSTMNDS","SBIN",
                    "BALKRISIND","ESCORTS","MUTHOOTFIN","JUBLPHARMA","HEROMOTOCO"],
    },
    # Jan 14 cycle — add when tickers/entries recovered.
]

NIFTY_FYERS = "NSE:NIFTY50-INDEX"

# ───────────────────────── DATA ACCESS ─────────────────────────
def get_provider():
    """Reuse the app's initialized data provider + swing system."""
    from main import trading_api
    return trading_api.swing_system

def fetch_daily_closes(swing, symbol, start, end):
    """Return list of (date, close) for a symbol over [start,end] inclusive.
    Uses the same provider path the swing system uses. Falls back gracefully."""
    try:
        # period wide enough to cover the window; provider returns a DataFrame
        df, _, _ = swing.get_indian_stock_data(symbol, period="3mo")
        if df is None or df.empty:
            return None
        df = df[(df.index.date >= start) & (df.index.date <= end)]
        if df.empty:
            return None
        return [(d.date(), float(c)) for d, c in zip(df.index, df["Close"])]
    except Exception as e:
        print(f"    ! fetch failed {symbol}: {e}")
        return None

def fetch_nifty_closes(swing, start, end):
    try:
        df = swing._fetch_nifty_index()
        if df is None or df.empty:
            return None
        df = df[(df.index.date >= start) & (df.index.date <= end)]
        return [(d.date(), float(c)) for d, c in zip(df.index, df["Close"])]
    except Exception as e:
        print(f"    ! NIFTY fetch failed: {e}")
        return None

# ───────────────────────── CORE MATH ─────────────────────────
def parse(d): return datetime.strptime(d, "%Y-%m-%d").date()

def build_cycle_nav(swing, cyc):
    """Daily NAV for a cycle. Rupee-weighted by `shares` if provided, else
    equal-weighted. Returns (dates, nav_returns_pct, per_symbol_total_return)
    or (None,None,None) if data missing.

    If `entry` prices are provided, cross-checks each against the Fyers close
    on the start date and warns on >2% mismatch (catches transcription slips)."""
    start, end = parse(cyc["start"]), parse(cyc["end"])
    syms   = cyc["symbols"]
    shares = cyc.get("shares")          # None => equal weight
    entries= cyc.get("entry")           # None => no cross-check
    sym_shares = dict(zip(syms, shares)) if shares else None
    sym_entry  = dict(zip(syms, entries)) if entries else None

    series = {}          # sym -> {date: normalized_price (1.0 at entry)}
    per_symbol = {}      # sym -> total % return
    weights = {}         # sym -> rupee weight at entry (for weighted NAV)
    for sym in syms:
        rows = fetch_daily_closes(swing, sym, start, end)
        if not rows or len(rows) < 2:
            print(f"    ! insufficient data for {sym} — skipped in NAV")
            continue
        dates = [r[0] for r in rows]; closes = [r[1] for r in rows]
        entry = closes[0]
        # entry cross-check vs recorded price
        if sym_entry and sym in sym_entry:
            rec = sym_entry[sym]
            if rec and abs(closes[0]-rec)/rec > 0.02:
                print(f"    ~ {sym}: Fyers entry-day close ₹{closes[0]:.2f} vs "
                      f"recorded ₹{rec:.2f} ({(closes[0]-rec)/rec*100:+.1f}%) — check")
        series[sym] = {d: c/entry for d, c in zip(dates, closes)}
        per_symbol[sym] = (closes[-1]/entry - 1.0) * 100
        # rupee weight = entry_price * shares (if shares given) else 1 (equal)
        weights[sym] = (entry * sym_shares[sym]) if sym_shares else 1.0
    if not series:
        return None, None, None, None
    wsum = sum(weights.values())
    # weighted daily NAV: sum_i w_i * normalized_price_i / sum(w)
    all_dates = sorted(set().union(*[set(s.keys()) for s in series.values()]))
    nav = []
    for d in all_dates:
        num = sum(weights[s]*series[s][d] for s in series if d in series[s])
        den = sum(weights[s] for s in series if d in series[s])
        nav.append(num/den if den else 1.0)
    nav = np.array(nav)
    daily_ret = np.diff(nav)/nav[:-1] * 100  # daily % returns of the book
    book_ret_weighted = (nav[-1] - 1.0) * 100  # final weighted book return %
    return all_dates, daily_ret, per_symbol, book_ret_weighted

def trade_stats(all_trades):
    """all_trades = list of per-position % returns across all cycles."""
    a = np.array(all_trades)
    wins = a[a > 0]; losses = a[a < 0]
    n = len(a)
    win_rate = len(wins)/n*100 if n else 0
    avg_win = wins.mean() if len(wins) else 0
    avg_loss = losses.mean() if len(losses) else 0
    gross_win = wins.sum(); gross_loss = -losses.sum()
    profit_factor = gross_win/gross_loss if gross_loss > 0 else float('inf')
    expectancy = a.mean()
    payoff = abs(avg_win/avg_loss) if avg_loss != 0 else float('inf')
    return {
        "n_trades": n, "win_rate": win_rate,
        "avg_win": avg_win, "avg_loss": avg_loss,
        "payoff_ratio": payoff, "profit_factor": profit_factor,
        "expectancy": expectancy, "best": a.max(), "worst": a.min(),
        "gross_win": gross_win, "gross_loss": gross_loss,
    }

def portfolio_stats(cycle_rets, nifty_rets, all_daily_rets,
                    risk_free_annual=RISK_FREE_ANNUAL, trading_days=TRADING_DAYS):
    """cycle_rets / nifty_rets = per-cycle % (aligned). all_daily_rets = pooled
    daily % returns across cycles for Sharpe/vol/drawdown."""
    c = np.array(cycle_rets); nf = np.array(nifty_rets)
    alpha = c - nf
    hit = (alpha > 0).mean()*100
    # daily-based risk metrics
    dr = np.array(all_daily_rets)/100  # to decimal
    mu_d = dr.mean(); sd_d = dr.std(ddof=1) if len(dr) > 1 else 0
    ann_ret = mu_d*trading_days*100
    ann_vol = sd_d*np.sqrt(trading_days)*100
    rf_d = risk_free_annual/trading_days
    sharpe = ((mu_d-rf_d)/sd_d*np.sqrt(trading_days)) if sd_d > 0 else 0
    downside = dr[dr < 0]
    dd_sd = downside.std(ddof=1) if len(downside) > 1 else 0
    sortino = ((mu_d-rf_d)/dd_sd*np.sqrt(trading_days)) if dd_sd > 0 else 0
    # max drawdown on the pooled equity curve
    eq = np.cumprod(1+dr)
    peak = np.maximum.accumulate(eq)
    max_dd = ((eq-peak)/peak).min()*100 if len(eq) else 0
    # up/down capture (cycle-level)
    up = nf > 0; dn = nf < 0
    up_cap = (c[up].mean()/nf[up].mean()*100) if up.any() and nf[up].mean()!=0 else None
    dn_cap = (c[dn].mean()/nf[dn].mean()*100) if dn.any() and nf[dn].mean()!=0 else None
    return {
        "n_cycles": len(c), "avg_cycle": c.mean(), "median_cycle": np.median(c),
        "avg_alpha": alpha.mean(), "median_alpha": np.median(alpha),
        "hit_rate": hit, "best_cycle": c.max(), "worst_cycle": c.min(),
        "ann_ret": ann_ret, "ann_vol": ann_vol, "sharpe": sharpe,
        "sortino": sortino, "max_dd": max_dd,
        "up_capture": up_cap, "down_capture": dn_cap,
        "cycle_alpha_std": alpha.std(ddof=1) if len(alpha)>1 else 0,
    }

# ───────────────────────── RUN ─────────────────────────
def main():
    swing = get_provider()
    print("="*66)
    print("SENTIQUANT — PERFORMANCE METRICS")
    print(f"Cost model: {COST_BPS} bps round-trip {'(PAPER)' if COST_BPS==0 else ''}")
    print("="*66)

    all_trades = []
    cycle_rets, nifty_rets, pooled_daily = [], [], []

    for cyc in CYCLES:
        print(f"\n▶ {cyc['name']} ({cyc['start']} → {cyc['end']})")
        result = build_cycle_nav(swing, cyc)
        if result[2] is None:
            print("   skipped — no data"); continue
        dates, daily_ret, per_symbol, book_ret_w = result

        # trade-level: each position's total return, minus costs
        for sym, r in per_symbol.items():
            r_net = r - COST_BPS/100.0
            all_trades.append(r_net)
        # cycle-level: rupee-weighted (or equal-weighted) book return
        wtag = "wtd" if cyc.get("shares") else "eq"
        book_ret = book_ret_w - COST_BPS/100.0
        cycle_rets.append(book_ret)
        if daily_ret is not None:
            pooled_daily.extend(daily_ret.tolist())

        # nifty over same window
        nrows = fetch_nifty_closes(swing, parse(cyc["start"]), parse(cyc["end"]))
        if nrows and len(nrows) >= 2:
            nret = (nrows[-1][1]/nrows[0][1]-1)*100
        else:
            nret = 0.0
            print("   ! NIFTY data missing — using 0 (fix before trusting alpha)")
        nifty_rets.append(nret)
        print(f"   book {book_ret:+.2f}% [{wtag}]  nifty {nret:+.2f}%  alpha {book_ret-nret:+.2f}%")

    if not all_trades:
        print("\nNo data computed. Fill CYCLES with valid symbols/dates and rerun.")
        return

    ts = trade_stats(all_trades)
    ps = portfolio_stats(cycle_rets, nifty_rets, pooled_daily)

    print("\n" + "="*66)
    print("TRADE-LEVEL  (across all individual positions)")
    print("="*66)
    print(f"  Trades:            {ts['n_trades']}")
    print(f"  Win rate:          {ts['win_rate']:.1f}%")
    print(f"  Avg winner:        {ts['avg_win']:+.2f}%")
    print(f"  Avg loser:         {ts['avg_loss']:+.2f}%")
    print(f"  Payoff ratio:      {ts['payoff_ratio']:.2f}  (avg win / avg loss)")
    print(f"  Profit factor:     {ts['profit_factor']:.2f}  (gross win / gross loss)")
    print(f"  Expectancy:        {ts['expectancy']:+.2f}% per trade")
    print(f"  Best / worst:      {ts['best']:+.2f}% / {ts['worst']:+.2f}%")

    print("\n" + "="*66)
    print("PORTFOLIO-LEVEL")
    print("="*66)
    print(f"  Cycles:            {ps['n_cycles']}")
    print(f"  Avg cycle return:  {ps['avg_cycle']:+.2f}%   median {ps['median_cycle']:+.2f}%")
    print(f"  Avg alpha:         {ps['avg_alpha']:+.2f}%   median {ps['median_alpha']:+.2f}%")
    print(f"  Alpha hit rate:    {ps['hit_rate']:.0f}%   (cycles beating Nifty)")
    print(f"  Best / worst cyc:  {ps['best_cycle']:+.2f}% / {ps['worst_cycle']:+.2f}%")
    print(f"  Cycle-alpha std:   {ps['cycle_alpha_std']:.2f}%  (consistency)")
    print(f"  --- risk (daily-based) ---")
    print(f"  Annualized return: {ps['ann_ret']:+.1f}%")
    print(f"  Annualized vol:    {ps['ann_vol']:.1f}%")
    print(f"  Sharpe ratio:      {ps['sharpe']:.2f}")
    print(f"  Sortino ratio:     {ps['sortino']:.2f}")
    print(f"  Max drawdown:      {ps['max_dd']:.2f}%")
    uc = f"{ps['up_capture']:.0f}%" if ps['up_capture'] is not None else "n/a (no up cycles)"
    dc = f"{ps['down_capture']:.0f}%" if ps['down_capture'] is not None else "n/a (no down cycles)"
    print(f"  Up-capture:        {uc}")
    print(f"  Down-capture:      {dc}")

    print("\n" + "="*66)
    print("DILIGENCE NOTES (read before quoting these)")
    print("="*66)
    print(f"  • Sharpe/vol/drawdown are pooled across {ps['n_cycles']} short cycles —")
    print("    honest but small-sample. State n when quoting.")
    print(f"  • {'PAPER — zero costs.' if COST_BPS==0 else f'Costed at {COST_BPS}bps.'} "
          "Set COST_BPS=30 to see a realistic haircut.")
    print("  • Up-capture rests on however few up-cycles you have. If 1, say so.")
    print("  • These are computed from provider daily closes, not live fills.")

if __name__ == "__main__":
    main()
