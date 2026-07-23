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
# Must reach back to your OLDEST cycle. "1y" covers Dec-2025 onward.
# If a cycle prints "NO ROWS ... widen LOOKBACK", bump this to "2y".
LOOKBACK = "1y"

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
        "stop":    [3269,1137,127,506,2982,917,4851,658,274,159],
        "t1":      [3521,1231,140,545,3217,970,5223,716,301,173],
        "t2":      [3584,1254,143,554,3276,983,5316,730,308,176],
        "t3":      [3646,1277,146,564,3335,997,5409,745,315,179],
    },
    {
        "name": "Jul 7",
        "start": "2026-07-07", "end": "2026-07-21",
        "symbols": ["SIGNATURE","HAPPSTMNDS","SBIN","HINDUNILVR","AUBANK",
                    "PERSISTENT","BEL","DEVYANI","ESCORTS","MUTHOOTFIN"],
        "entry":   [783,349,1043,2205,1074,4877,422,114,2965,3128],
        "stop":    [740,335,1017,2149,1035,4602,407,109,2855,2989],
        "t1":      [826,363,1070,2262,1113,5151,436,119,3074,3267],
        "t2":      [847,370,1083,2290,1132,5289,444,121,3128,3336],
        "t3":      [868,377,1096,2318,1151,5426,451,123,3183,3406],
    },
    {
        "name": "Jul 8 (12:15)",
        "start": "2026-07-08", "end": "2026-07-22",
        "symbols": ["DEVYANI","PERSISTENT","TECHM","HAPPSTMNDS","SBIN",
                    "BALKRISIND","ESCORTS","MUTHOOTFIN","JUBLPHARMA","HEROMOTOCO"],
        "entry":   [115,4823,1442,356,1035,2266,2949,3147,981,4947],
        "stop":    [110,4542,1374,341,1008,2182,2840,3003,944,4786],
        "t1":      [120,5103,1510,370,1062,2349,3058,3290,1017,5109],
        "t2":      [123,5243,1544,377,1076,2391,3113,3362,1035,5189],
        "t3":      [125,5383,1578,384,1089,2433,3167,3434,1054,5270],
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

    NOTE: period must be wide enough to reach the OLDEST cycle. A "3mo" pull
    silently returns nothing for cycles older than ~3 months, which produces
    dropped/garbage cycles. "1y" covers Dec-2025 onward. If you add cycles
    older than 1 year, widen this to "2y"."""
    try:
        df, _, _ = swing.get_indian_stock_data(symbol, period=LOOKBACK)
        if df is None or df.empty:
            print(f"    ! no data returned at all for {symbol}")
            return None
        full_lo, full_hi = df.index.date.min(), df.index.date.max()
        df = df[(df.index.date >= start) & (df.index.date <= end)]
        if df.empty:
            print(f"    ! {symbol}: NO ROWS in {start}..{end} "
                  f"(provider only has {full_lo}..{full_hi}) — widen LOOKBACK")
            return None
        return [(d.date(), float(c)) for d, c in zip(df.index, df["Close"])]
    except Exception as e:
        print(f"    ! fetch failed {symbol}: {e}")
        return None

_NIFTY_CACHE = {}

def fetch_nifty_closes(swing, start, end):
    """Fetch NIFTY closes for an explicit window. The swing system's own
    _fetch_nifty_index() is hardcoded to ~120 days, which cannot reach older
    cycles — so pull directly from the paginated fetcher with real dates."""
    key = (start, end)
    if key in _NIFTY_CACHE:
        return _NIFTY_CACHE[key]
    try:
        df = swing.data_provider.fyers.fetch_paginated_history(
            symbol=NIFTY_FYERS, resolution='D',
            start_date=start - timedelta(days=5),   # pad for holidays
            end_date=end + timedelta(days=5),
        )
        if df is None or df.empty:
            print("    ! NIFTY: no data for window")
            return None
        df = df.copy()
        df.columns = [str(c).lower() for c in df.columns]
        if 'close' not in df.columns:
            print(f"    ! NIFTY: no close column ({list(df.columns)})")
            return None
        idx = df.index
        dates = [d.date() if hasattr(d, 'date') else d for d in idx]
        rows = [(d, float(c)) for d, c in zip(dates, df['close'])
                if start <= d <= end]
        if not rows:
            print(f"    ! NIFTY: no rows in {start}..{end}")
            return None
        _NIFTY_CACHE[key] = rows
        return rows
    except Exception as e:
        print(f"    ! NIFTY fetch failed: {e}")
        return None

# ───────────────────────── CORE MATH ─────────────────────────
def parse(d): return datetime.strptime(d, "%Y-%m-%d").date()

def simulate_exits(entry, stop, t1, t2, t3, bars, same_day_stop_first=True):
    """Simulate YOUR actual rules on daily OHLC:
         - low <= stop            -> exit ALL at stop
         - high >= T1             -> book 50% at T1, stop moves to entry (BE)
         - after T1, high >= T2/T3-> exit remainder at that target
         - both stop & target same day -> assume STOP first (conservative,
           because daily bars can't tell you which came first)
       Anything still held at window end is marked to the final close.
       Returns total % return on the position."""
    booked, rem, cur_stop, t1_done = 0.0, 1.0, stop, False
    for _d, h, l, c in bars:
        if rem <= 0: break
        stop_hit = l <= cur_stop
        t1_hit   = (not t1_done) and h >= t1
        tgt_hit  = t1_done and (h >= t2 or h >= t3)
        if stop_hit and (t1_hit or tgt_hit) and same_day_stop_first:
            booked += rem*(cur_stop-entry)/entry*100; rem = 0; break
        if stop_hit:
            booked += rem*(cur_stop-entry)/entry*100; rem = 0; break
        if t1_hit:
            booked += 0.5*(t1-entry)/entry*100
            rem, cur_stop, t1_done = 0.5, entry, True
            if h >= t3:   booked += rem*(t3-entry)/entry*100; rem = 0; break
            elif h >= t2: booked += rem*(t2-entry)/entry*100; rem = 0; break
            continue
        if tgt_hit:
            px = t3 if h >= t3 else t2
            booked += rem*(px-entry)/entry*100; rem = 0; break
    if rem > 0:
        booked += rem*(bars[-1][3]-entry)/entry*100
    return booked

def fetch_ohlc(swing, symbol, start, end):
    """Daily (date, high, low, close) over the window — needed for exit sim."""
    try:
        df, _, _ = swing.get_indian_stock_data(symbol, period=LOOKBACK)
        if df is None or df.empty: return None
        df = df[(df.index.date >= start) & (df.index.date <= end)]
        if df.empty: return None
        return [(d.date(), float(h), float(l), float(c))
                for d, h, l, c in zip(df.index, df["High"], df["Low"], df["Close"])]
    except Exception:
        return None

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
    sym_stop = dict(zip(syms, cyc["stop"]))  if cyc.get("stop")  else None
    sym_t1   = dict(zip(syms, cyc["t1"]))    if cyc.get("t1")    else None
    sym_t2   = dict(zip(syms, cyc["t2"]))    if cyc.get("t2")    else None
    sym_t3   = dict(zip(syms, cyc["t3"]))    if cyc.get("t3")    else None
    per_symbol_exit = {}

    series = {}          # sym -> {date: normalized_price (1.0 at ENTRY)}
    per_symbol = {}      # sym -> total % return (from recorded entry if given)
    per_symbol_cons = {} # sym -> total % return (conservative: entry-day close)
    weights = {}         # sym -> rupee weight at entry
    used_recorded = False
    for sym in syms:
        rows = fetch_daily_closes(swing, sym, start, end)
        if not rows or len(rows) < 2:
            print(f"    ! insufficient data for {sym} — skipped in NAV")
            continue
        dates = [r[0] for r in rows]; closes = [r[1] for r in rows]
        day0_close = closes[0]
        # BUY PRICE: your recorded entry if supplied, else the entry-day close.
        # Portfolios were generated intraday, so the recorded entry is the price
        # the signal actually fired at — using day0 close discards the day-1 move.
        rec = sym_entry.get(sym) if sym_entry else None
        if rec:
            entry = float(rec); used_recorded = True
            if abs(day0_close-rec)/rec > 0.02:
                print(f"    ~ {sym}: entry-day close ₹{day0_close:.2f} vs "
                      f"recorded ₹{rec:.2f} ({(day0_close-rec)/rec*100:+.1f}%)")
        else:
            entry = day0_close
        series[sym] = {d: c/entry for d, c in zip(dates, closes)}
        per_symbol[sym]      = (closes[-1]/entry - 1.0) * 100
        per_symbol_cons[sym] = (closes[-1]/day0_close - 1.0) * 100
        weights[sym] = (entry * sym_shares[sym]) if sym_shares else 1.0
        # exit-rule simulation (needs stop + T1/T2/T3 for this symbol)
        if sym_stop and sym_t1 and sym_stop.get(sym) and sym_t1.get(sym):
            bars = fetch_ohlc(swing, sym, start, end)
            if bars:
                per_symbol_exit[sym] = simulate_exits(
                    entry, sym_stop[sym], sym_t1[sym],
                    sym_t2.get(sym, sym_t1[sym]) if sym_t2 else sym_t1[sym],
                    sym_t3.get(sym, sym_t1[sym]) if sym_t3 else sym_t1[sym],
                    bars)
    if not series:
        return None, None, None, None, None, None
    # weighted daily NAV: sum_i w_i * normalized_price_i / sum(w)
    all_dates = sorted(set().union(*[set(s.keys()) for s in series.values()]))
    nav = []
    for d in all_dates:
        num = sum(weights[s]*series[s][d] for s in series if d in series[s])
        den = sum(weights[s] for s in series if d in series[s])
        nav.append(num/den if den else 1.0)
    # anchor at the entry itself (1.0) so the entry→first-close move counts
    nav = np.array([1.0] + nav)
    daily_ret = np.diff(nav)/nav[:-1] * 100  # daily % returns of the book
    book_ret_weighted = (nav[-1] - 1.0) * 100  # final weighted book return %
    # conservative book return (buy at entry-day close)
    wc = {s: (per_symbol_cons[s]) for s in per_symbol_cons}
    if sym_shares:
        tw = sum(sym_shares[s]*sym_entry[s] if sym_entry and sym_entry.get(s)
                 else sym_shares[s] for s in wc)
        book_cons = sum((sym_shares[s]*(sym_entry[s] if sym_entry and sym_entry.get(s) else 1))
                        * wc[s] for s in wc)/tw if tw else np.mean(list(wc.values()))
    else:
        book_cons = float(np.mean(list(wc.values())))
    return all_dates, daily_ret, per_symbol, book_ret_weighted, book_cons, per_symbol_exit

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
    cycle_rets, nifty_rets, pooled_daily, cons_rets = [], [], [], []
    exit_rets, exit_nifty = [], []
    ran, failed = [], []

    for cyc in CYCLES:
        print(f"\n▶ {cyc['name']} ({cyc['start']} → {cyc['end']})")
        result = build_cycle_nav(swing, cyc)
        if result[2] is None:
            print("   ✗ SKIPPED — no usable data")
            failed.append(cyc['name']); continue
        dates, daily_ret, per_symbol, book_ret_w, book_cons, per_sym_exit = result
        got, want = len(per_symbol), len(cyc['symbols'])
        if got < want:
            print(f"   ⚠ only {got}/{want} symbols had data — return is partial")

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
        ran.append(cyc['name'])
        cons_rets.append(book_cons - COST_BPS/100.0)
        print(f"   book {book_ret:+.2f}% [{wtag}]  nifty {nret:+.2f}%  "
              f"alpha {book_ret-nret:+.2f}%   (conservative entry: "
              f"{book_cons:+.2f}%, alpha {book_cons-nret:+.2f}%)")
        if per_sym_exit:
            if cyc.get("shares"):
                tw = sum(cyc["shares"][i]*cyc["entry"][i]
                         for i,sy in enumerate(cyc["symbols"]) if sy in per_sym_exit)
                be = sum(cyc["shares"][i]*cyc["entry"][i]*per_sym_exit[sy]
                         for i,sy in enumerate(cyc["symbols"]) if sy in per_sym_exit)/tw
            else:
                be = float(np.mean(list(per_sym_exit.values())))
            be -= COST_BPS/100.0
            exit_rets.append(be); exit_nifty.append(nret)
            print(f"   WITH EXIT RULES: {be:+.2f}%  alpha {be-nret:+.2f}%  "
                  f"({len(per_sym_exit)}/{len(cyc['symbols'])} positions modelled)")

    print("\n" + "="*66)
    print(f"COVERAGE: {len(ran)}/{len(CYCLES)} cycles computed")
    if failed:
        print(f"  ✗ FAILED: {', '.join(failed)}")
        print("    → Stats below EXCLUDE these. Fix before trusting any number.")
    else:
        print("  ✓ all cycles computed")

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

    if cons_rets:
        ca=np.array(cons_rets)-np.array(nifty_rets)
        print("\n  --- CONSERVATIVE BASIS (buy at entry-day close, not signal price) ---")
        print(f"  Avg cycle return:  {np.mean(cons_rets):+.2f}%")
        print(f"  Avg alpha:         {ca.mean():+.2f}%   median {np.median(ca):+.2f}%")
        print(f"  Alpha hit rate:    {(ca>0).mean()*100:.0f}%")
        print("  (Your recorded entries were intraday signal prices; this row")
        print("   shows what you'd have got buying at that day's CLOSE instead.)")

    if exit_rets:
        ea = np.array(exit_rets)-np.array(exit_nifty)
        print("\n  --- WITH YOUR EXIT RULES (stop / book 50% at T1 / T2-T3) ---")
        print(f"  Cycles modelled:   {len(exit_rets)}")
        print(f"  Avg cycle return:  {np.mean(exit_rets):+.2f}%")
        print(f"  Avg alpha:         {ea.mean():+.2f}%   median {np.median(ea):+.2f}%")
        print(f"  Alpha hit rate:    {(ea>0).mean()*100:.0f}%")
        print("  Assumptions: stop fills AT the stop price (no gap slippage);")
        print("  if stop and target hit the same day, stop assumed first.")

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
