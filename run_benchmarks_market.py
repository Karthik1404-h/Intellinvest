#!/usr/bin/env python3
"""
Generate benchmark comparison for any market (US or INDIA).
Benchmarks included:
  - Equal_Weight_Portfolio   : equal weight across all market stocks
  - Cap_Weighted_Portfolio   : price-proxy cap weighted
  - Best_Stock / Worst_Stock : best and worst single stock over the period
  - Market_Index             : SPY for US, Nifty 50 (NSEI) for India
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.stdout.reconfigure(encoding='utf-8')

from config import Config
from src.evaluation.backtesting import PerformanceMetrics

# ── helpers ─────────────────────────────────────────────────────────────────

def results_dir(market: str) -> str:
    return os.path.join(Config.BASE_DIR, 'results', market.lower())

def raw_dir(market: str) -> str:
    return os.path.join(Config.BASE_DIR, 'data', 'raw', market.lower())

def processed_dir(market: str) -> str:
    return os.path.join(Config.BASE_DIR, 'data', 'processed', market.lower())


# ── load ML strategies ───────────────────────────────────────────────────────

def load_ml_strategies(market: str) -> dict:
    rdir = results_dir(market)
    strategies = {}
    names = ['risk_parity', 'mean_variance', 'max_sharpe', 'cluster_based', 'min_variance', 'momentum_filter', 'black_litterman', 'concentrated_momentum']

    for name in names:
        path = os.path.join(rdir, f'portfolio_values_{name}.csv')
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path)
            # Accept both "date" and index-based formats
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
            else:
                df.index = pd.to_datetime(df.index)

            series = df.iloc[:, 0]
            # Dollar values → convert to returns
            if series.abs().mean() > 10:
                series = series.pct_change().dropna()
            series = series.replace([np.inf, -np.inf], np.nan).dropna()
            strategies[f'ML_{name}'] = series
            print(f"  Loaded ML_{name}: {len(series)} returns")
        except Exception as e:
            print(f"  Warning: could not load {name}: {e}")

    return strategies


# ── build benchmark strategies from returns_data.csv ────────────────────────

def build_portfolio_benchmarks(market: str) -> dict:
    benchmarks = {}
    returns_path = os.path.join(processed_dir(market), 'returns_data.csv')
    if not os.path.exists(returns_path):
        print(f"  Warning: returns_data.csv not found at {returns_path}")
        return benchmarks

    try:
        df = pd.read_csv(returns_path, index_col=0, parse_dates=True)
        df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how='all')
        print(f"  Loaded returns for {df.shape[1]} stocks, {df.shape[0]} dates")

        # 1. Equal weight
        ew = df.mean(axis=1).dropna()
        benchmarks['Equal_Weight_Portfolio'] = ew
        print(f"  Equal Weight: {len(ew)} observations")

        # 2. Price-proxy cap weighted (use cumulative return as proxy for market cap)
        cum_ret = (1 + df).cumprod()
        # Average cumulative return as weight proxy
        avg_cum = cum_ret.mean()
        weights = avg_cum / avg_cum.sum()
        cw = (df * weights).sum(axis=1).dropna()
        benchmarks['Cap_Weighted_Portfolio'] = cw
        print(f"  Cap Weighted: {len(cw)} observations")

        # 3. Best and worst single stocks
        total_ret = (1 + df).prod() - 1
        best_sym = total_ret.idxmax()
        worst_sym = total_ret.idxmin()
        benchmarks[f'Best_Stock_{best_sym.replace(".NS", "").replace(".BSE", "")}'] = df[best_sym].dropna()
        benchmarks[f'Worst_Stock_{worst_sym.replace(".NS", "").replace(".BSE", "")}'] = df[worst_sym].dropna()
        print(f"  Best stock: {best_sym}  Worst stock: {worst_sym}")

    except Exception as e:
        print(f"  Error building portfolio benchmarks: {e}")
        import traceback; traceback.print_exc()

    return benchmarks


# ── load market index ────────────────────────────────────────────────────────

def load_index_benchmark(market: str) -> dict:
    benchmarks = {}
    rdir = raw_dir(market)

    if market.upper() == 'INDIA':
        index_file = os.path.join(rdir, 'NSEI_benchmark.csv')
        index_name = 'Nifty50_Index'
    else:
        index_file = os.path.join(rdir, 'SPY_benchmark.csv')
        index_name = 'SPY_SP500'

    if not os.path.exists(index_file):
        print(f"  Warning: index file not found: {index_file}")
        return benchmarks

    try:
        # Detect header rows: yfinance downloads often have 2-3 meta rows
        raw = pd.read_csv(index_file, header=None, nrows=5)
        # Find the first row where column 0 looks like a date
        skip = 0
        for i, row in raw.iterrows():
            val = str(row[0])
            try:
                pd.to_datetime(val, errors='raise')
                skip = i
                break
            except Exception:
                continue

        df = pd.read_csv(index_file, skiprows=skip, header=0)
        # First col = date, second col = close price
        df.columns = ['Date'] + list(df.columns[1:])
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date']).set_index('Date')
        close = pd.to_numeric(df.iloc[:, 0], errors='coerce').dropna()
        returns = close.pct_change().dropna()
        returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
        benchmarks[index_name] = returns
        print(f"  Loaded {index_name}: {len(returns)} observations  ({returns.index[0].date()} → {returns.index[-1].date()})")
    except Exception as e:
        print(f"  Error loading index {index_file}: {e}")
        import traceback; traceback.print_exc()

    return benchmarks


# ── calculate metrics ────────────────────────────────────────────────────────

def calculate_metrics(strategies: dict, risk_free: float = 0.06) -> pd.DataFrame:
    """Calculate standard performance metrics.
    risk_free default 6% for India, 2% can be passed for US."""
    rows = []
    for name, rets in strategies.items():
        if len(rets) < 50:
            print(f"  Skipping {name}: only {len(rets)} observations")
            continue
        try:
            rows.append({
                'Strategy': name,
                'Total_Return': PerformanceMetrics.calculate_total_return(rets),
                'Annual_Return': PerformanceMetrics.calculate_annualized_return(rets),
                'Volatility': PerformanceMetrics.calculate_volatility(rets),
                'Sharpe_Ratio': PerformanceMetrics.calculate_sharpe_ratio(rets, risk_free_rate=risk_free),
                'Sortino_Ratio': PerformanceMetrics.calculate_sortino_ratio(rets, risk_free_rate=risk_free),
                'Max_Drawdown': PerformanceMetrics.calculate_max_drawdown(rets),
                'Calmar_Ratio': PerformanceMetrics.calculate_calmar_ratio(rets),
                'Observations': len(rets),
            })
        except Exception as e:
            print(f"  Error for {name}: {e}")
    return pd.DataFrame(rows)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Generate benchmark comparison for a market')
    parser.add_argument('--market', default='US', choices=['US', 'INDIA', 'us', 'india'],
                        help='Market to generate benchmarks for')
    parser.add_argument('--risk-free', type=float, default=None,
                        help='Risk-free rate override (default: 0.06 for INDIA, 0.02 for US)')
    args = parser.parse_args()

    market = args.market.upper()
    rf = args.risk_free if args.risk_free is not None else (0.06 if market == 'INDIA' else 0.02)

    print("=" * 60)
    print(f"Generating Benchmark Comparison — {market} market")
    print(f"Risk-free rate: {rf*100:.1f}%")
    print("=" * 60)

    print("\n[1] Loading ML strategies...")
    ml = load_ml_strategies(market)
    print(f"  → {len(ml)} ML strategies loaded")

    print("\n[2] Building portfolio benchmarks...")
    port_bm = build_portfolio_benchmarks(market)
    print(f"  → {len(port_bm)} portfolio benchmarks")

    print("\n[3] Loading market index...")
    idx_bm = load_index_benchmark(market)
    print(f"  → {len(idx_bm)} index benchmarks")

    all_strats = {**ml, **port_bm, **idx_bm}
    print(f"\n[4] Total strategies: {len(all_strats)}")

    print("\n[5] Calculating metrics...")
    df = calculate_metrics(all_strats, risk_free=rf)
    print(df[['Strategy', 'Annual_Return', 'Sharpe_Ratio']].to_string(index=False))

    out_dir = results_dir(market)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'benchmark_comparison_detailed.csv')
    df.to_csv(out_path, index=False)
    print(f"\n✓ Saved to: {out_path}")
    print(f"  Strategies: {', '.join(df['Strategy'].tolist())}")


if __name__ == '__main__':
    main()
