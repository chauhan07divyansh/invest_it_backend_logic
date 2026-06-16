import sys, os
sys.path.insert(0, '/app')
import numpy as np
import pandas as pd
from main import trading_api

sws = trading_api.swing_system

# Sweep these holding periods
HOLD_PERIODS = [5, 7, 10, 14, 20]
ENTRY_MIN_SCORE = 60
LOOKBACK_BARS = 120
WARMUP = 50

# Round-trip transaction cost (buy + sell), in percent. Indian retail:
# ~0.1-0.3% all-in (brokerage + STT + slippage). 0.15% is a fair default.
COST_PCT = 0.15

# Stop config to test the sweep under. Using the NEW adaptive stop so the
# horizon choice reflects the system you intend to ship.
NEW = {
    'BEAR': dict(stop_mult=0.8, max_sl_pct=0.05, t1=1.0, t2=1.3, t3=1.7),
    'SIDEWAYS': dict(stop_mult=1.0, max_sl_pct=0.08, t1=1.0, t2=1.5, t3=2.0),
    'BULL': dict(stop_mult=1.5, max_sl_pct=0.12, t1=1.0, t2=1.5, t3=2.0),
}

def atr_at(df, i, period=14):
    h, l, c = df['High'], df['Low'], df['Close']
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.iloc[max(0, i - period + 1):i + 1].mean()

def regime_at(df, i):
    c = df['Close']
    if i < WARMUP:
        return 'SIDEWAYS'
    cur = c.iloc[i]
    ma20 = c.iloc[i-19:i+1].mean()
    ma50 = c.iloc[i-49:i+1].mean()
    if cur > ma20 > ma50:
        return 'BULL'
    if cur < ma20 < ma50:
        return 'BEAR'
    return 'SIDEWAYS'

def simulate(df, entry_i, cfg, normal_atr, hold_days):
    entry = df['Close'].iloc[entry_i]
    sl_dist = normal_atr * cfg['stop_mult']
    if cfg['max_sl_pct'] is not None:
        sl_dist = min(sl_dist, entry * cfg['max_sl_pct'])
    stop = entry - sl_dist
    normal = normal_atr * 1.5
    t1 = entry + normal * cfg['t1']
    t2 = entry + normal * cfg['t2']
    t3 = entry + normal * cfg['t3']
    gross = None
    for j in range(entry_i + 1, min(entry_i + 1 + hold_days, len(df))):
        lo, hi = df['Low'].iloc[j], df['High'].iloc[j]
        if lo <= stop:
            gross = (stop - entry) / entry * 100; break
        if hi >= t3:
            gross = (t3 - entry) / entry * 100; break
        if hi >= t2:
            gross = (t2 - entry) / entry * 100; break
        if hi >= t1:
            gross = (t1 - entry) / entry * 100; break
    if gross is None:
        exit_px = df['Close'].iloc[min(entry_i + hold_days, len(df) - 1)]
        gross = (exit_px - entry) / entry * 100
    return gross - COST_PCT  # net of round-trip cost

def metrics(trades):
    if not trades:
        return None
    arr = np.array(trades)
    wins = arr[arr > 0]; losses = arr[arr < 0]
    gw = wins.sum() if len(wins) else 0
    gl = -losses.sum() if len(losses) else 0
    pf = gw / gl if gl > 0 else 999.0
    return dict(
        n=len(arr),
        win_rate=len(wins)/len(arr)*100,
        avg_win=wins.mean() if len(wins) else 0,
        avg_loss=losses.mean() if len(losses) else 0,
        pf=pf,
        expectancy=arr.mean(),
    )

def run():
    symbols = sws.get_all_stock_symbols()
    maxhold = max(HOLD_PERIODS)
    print("Holding-period sweep, NET of", COST_PCT, "pct round-trip cost")
    print("Scanning", len(symbols), "stocks...\n")

    # Collect entries once, then evaluate each entry across all hold periods
    # so every period is tested on IDENTICAL entries (clean comparison).
    by_hold = {h: [] for h in HOLD_PERIODS}
    n_ok = 0
    for k, sym in enumerate(symbols):
        try:
            df, _, _ = sws.get_indian_stock_data(sym, period="1y")
            if df is None or len(df) < WARMUP + 25:
                continue
            df = df.tail(LOOKBACK_BARS + WARMUP).reset_index(drop=True)
            n_ok += 1
            for i in range(WARMUP, len(df) - maxhold):
                window = df.iloc[:i+1]
                sent = ([], [], [], "None", "Sample")
                score = sws.calculate_swing_trading_score(window, sent, "Unknown")
                if score < ENTRY_MIN_SCORE:
                    continue
                a = atr_at(df, i)
                if not a or a <= 0 or pd.isna(a):
                    continue
                rg = regime_at(df, i)
                cfg = NEW[rg]
                for h in HOLD_PERIODS:
                    by_hold[h].append(simulate(df, i, cfg, a, h))
            if (k+1) % 40 == 0:
                print("  ", k+1, "/", len(symbols), "scanned")
        except Exception:
            continue

    print("\nScanned", n_ok, "stocks.")
    print("=" * 70)
    print("HOLD".rjust(5), "TRADES".rjust(9), "WIN%".rjust(8), "AVG_W".rjust(8),
          "AVG_L".rjust(8), "EXPECT".rjust(9), "PF".rjust(8))
    print("-" * 70)
    best_pf = -1; best_h = None
    best_exp = -1; best_h_exp = None
    for h in HOLD_PERIODS:
        m = metrics(by_hold[h])
        if not m:
            continue
        print(str(h).rjust(5), str(m['n']).rjust(9),
              ("%.1f" % m['win_rate']).rjust(8),
              ("%.2f" % m['avg_win']).rjust(8),
              ("%.2f" % m['avg_loss']).rjust(8),
              ("%.3f" % m['expectancy']).rjust(9),
              ("%.3f" % m['pf']).rjust(8))
        if m['pf'] > best_pf:
            best_pf = m['pf']; best_h = h
        if m['expectancy'] > best_exp:
            best_exp = m['expectancy']; best_h_exp = h
    print("=" * 70)
    print("Best PF        : %d-day hold (PF %.3f)" % (best_h, best_pf))
    print("Best expectancy: %d-day hold (%.3f%% per trade)" % (best_h_exp, best_exp))
    print("\nNote: in-sample, stock-trend regime proxy, %.2f%% cost applied." % COST_PCT)
    print("Expectancy is per-trade; shorter holds free capital faster so")
    print("compare expectancy-per-day too, not just per-trade PF.")
    print("\nExpectancy PER DAY held (capital-efficiency view):")
    for h in HOLD_PERIODS:
        m = metrics(by_hold[h])
        if m:
            print("  %2d-day: %.4f%%/day" % (h, m['expectancy'] / h))

if __name__ == "__main__":
    run()
