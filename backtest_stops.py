"""
STOP-LOSS BACKTEST — old (fixed 1.5x ATR) vs new (regime-adaptive + hard cap).
Standalone. Imports the live system, walks each stock's OHLCV history, simulates
both stop regimes on identical entries, reports the 4 metrics + profit factor.

Run in Render shell:
    python3 backtest_stops.py

Honest limits (read before trusting numbers):
  - IN-SAMPLE on the same ~6mo window. Bear-tuned stops flatter themselves on a
    window containing the bear phase. Treat as RELATIVE (old vs new), not absolute.
  - No slippage/commission. Real fills are worse, and worse MORE for the
    higher-stop-out new system — so its real edge is below what prints here.
  - Intraday stop/target hits approximated from daily High/Low (did the bar's
    low pierce the stop? did the high reach the target?). If BOTH hit same day,
    assume STOP first (conservative).
  - Entry rule = same signal the live system uses (score >= 60 => BUY/STRONG BUY).
"""
import sys, os
sys.path.insert(0, '/app')
import numpy as np
import pandas as pd
from main import trading_api

sws = trading_api.swing_system

# ---- knobs (match generate_trading_plan) --------------------------------
HOLD_DAYS       = 20          # max bars to hold (swing 1-4wk); exit at close if neither hit
ENTRY_MIN_SCORE = 60          # BUY threshold
LOOKBACK_BARS   = 120         # ~6mo of daily bars to scan for entries
WARMUP          = 50          # need 50 bars for MAs before first entry

OLD = dict(stop_mult=1.5, max_sl_pct=None, t1=1.0, t2=1.5, t3=2.0)
NEW = {
    'BEAR':     dict(stop_mult=0.8, max_sl_pct=0.05, t1=1.0, t2=1.3, t3=1.7),
    'SIDEWAYS': dict(stop_mult=1.0, max_sl_pct=0.08, t1=1.0, t2=1.5, t3=2.0),
    'BULL':     dict(stop_mult=1.5, max_sl_pct=0.12, t1=1.0, t2=1.5, t3=2.0),
}

def atr_at(df, i, period=14):
    """ATR using bars up to index i (inclusive)."""
    h, l, c = df['High'], df['Low'], df['Close']
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.iloc[max(0, i - period + 1):i + 1].mean()

def regime_at(df, i):
    """Same logic as detect_market_regime but on the STOCK's own MAs at bar i
    (proxy — we don't have point-in-time NIFTY history cheaply here; the stock's
    own trend is a reasonable per-name regime proxy for stop sizing)."""
    c = df['Close']
    if i < WARMUP:
        return 'SIDEWAYS'
    cur = c.iloc[i]
    ma20 = c.iloc[i-19:i+1].mean()
    ma50 = c.iloc[i-49:i+1].mean()
    if cur > ma20 > ma50: return 'BULL'
    if cur < ma20 < ma50: return 'BEAR'
    return 'SIDEWAYS'

def simulate(df, entry_i, cfg, normal_atr):
    """Walk forward from entry. Return (outcome, pct_move).
    outcome in {'TP1','TP2','TP3','STOP','TIME'}."""
    entry = df['Close'].iloc[entry_i]
    sl_dist = normal_atr * cfg['stop_mult']
    if cfg['max_sl_pct'] is not None:
        sl_dist = min(sl_dist, entry * cfg['max_sl_pct'])
    stop = entry - sl_dist
    normal = normal_atr * 1.5
    t1 = entry + normal * cfg['t1']
    t2 = entry + normal * cfg['t2']
    t3 = entry + normal * cfg['t3']
    for j in range(entry_i + 1, min(entry_i + 1 + HOLD_DAYS, len(df))):
        lo, hi = df['Low'].iloc[j], df['High'].iloc[j]
        # conservative: stop checked before target on same bar
        if lo <= stop:
            return 'STOP', (stop - entry) / entry * 100
        if hi >= t3:
            return 'TP3', (t3 - entry) / entry * 100
        if hi >= t2:
            return 'TP2', (t2 - entry) / entry * 100
        if hi >= t1:
            return 'TP1', (t1 - entry) / entry * 100
    # time exit at last close
    exit_px = df['Close'].iloc[min(entry_i + HOLD_DAYS, len(df) - 1)]
    return 'TIME', (exit_px - entry) / entry * 100

def metrics(trades):
    """trades = list of pct moves. Return dict of the 4 metrics + PF."""
    if not trades:
        return None
    arr = np.array(trades)
    wins = arr[arr > 0]; losses = arr[arr < 0]
    win_rate = len(wins) / len(arr) * 100
    avg_win = wins.mean() if len(wins) else 0
    avg_loss = losses.mean() if len(losses) else 0
    gross_win = wins.sum() if len(wins) else 0
    gross_loss = -losses.sum() if len(losses) else 0
    pf = gross_win / gross_loss if gross_loss > 0 else float('inf')
    expectancy = arr.mean()
    return dict(n=len(arr), win_rate=win_rate, avg_win=avg_win,
                avg_loss=avg_loss, profit_factor=pf, expectancy=expectancy)

def run():
    symbols = sws.get_all_stock_symbols()
    print(f"Backtesting stops on {len(symbols)} stocks "
          f"({LOOKBACK_BARS} bars, {HOLD_DAYS}-day hold)...\n")
    old_trades, new_trades = [], []
    regime_counts = {'BULL':0,'BEAR':0,'SIDEWAYS':0}
    n_ok = 0
    for k, sym in enumerate(symbols):
        try:
            df, _, _ = sws.get_indian_stock_data(sym, period="1y")
            if df is None or len(df) < WARMUP + 25:
                continue
            df = df.tail(LOOKBACK_BARS + WARMUP).reset_index(drop=True)
            n_ok += 1
            # generate entries from the scoring engine's signal, walked historically
            for i in range(WARMUP, len(df) - HOLD_DAYS):
                window = df.iloc[:i+1]
                sent = ([], [], [], "None", "Sample")  # neutral — backtest stops, not sentiment
                score = sws.calculate_swing_trading_score(window, sent, "Unknown")
                if score < ENTRY_MIN_SCORE:
                    continue
                a = atr_at(df, i)
                if not a or a <= 0 or pd.isna(a):
                    continue
                rg = regime_at(df, i)
                regime_counts[rg] += 1
                _, old_pct = simulate(df, i, OLD, a)
                _, new_pct = simulate(df, i, NEW[rg], a)
                old_trades.append(old_pct)
                new_trades.append(new_pct)
            if (k+1) % 40 == 0:
                print(f"  ...{k+1}/{len(symbols)} scanned, {len(old_trades)} trades so far")
        except Exception as e:
            continue

    print(f"\nScanned {n_ok} stocks. Entry regime mix: {regime_counts}")
    print(f"Total simulated trades: {len(old_trades)}\n")
    mo, mn = metrics(old_trades), metrics(new_trades)
    if not mo or not mn:
        print("Not enough trades."); return
    print("="*64)
    print(f"{'METRIC':<18}{'OLD (1.5xATR)':>16}{'NEW (regime+cap)':>18}")
    print("-"*64)
    print(f"{'Trades':<18}{mo['n']:>16}{mn['n']:>18}")
    print(f"{'Win rate %':<18}{mo['win_rate']:>16.1f}{mn['win_rate']:>18.1f}")
    print(f"{'Avg win %':<18}{mo['avg_win']:>16.2f}{mn['avg_win']:>18.2f}")
    print(f"{'Avg loss %':<18}{mo['avg_loss']:>16.2f}{mn['avg_loss']:>18.2f}")
    print(f"{'Expectancy %':<18}{mo['expectancy']:>16.3f}{mn['expectancy']:>18.3f}")
    print(f"{'PROFIT FACTOR':<18}{mo['profit_factor']:>16.3f}{mn['profit_factor']:>18.3f}")
    print("="*64)
    pf_delta = mn['profit_factor'] - mo['profit_factor']
    print(f"\nProfit factor change: {pf_delta:+.3f}")
    if pf_delta > 0.05:
        print("✅ NEW stops improve profit factor — ship it.")
    elif pf_delta < -0.05:
        print("❌ NEW stops REDUCE profit factor — over-tightened. Loosen bear cap (5%→7%) and re-run.")
    else:
        print("➖ Roughly neutral — the change is safe but not a clear win. Judge on avg-loss reduction.")
    print("\nReminder: in-sample, no costs. Treat as relative signal, not a return forecast.")

if __name__ == "__main__":
    run()
