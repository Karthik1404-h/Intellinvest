#!/usr/bin/env python3
"""
Regenerate complete benchmark comparison including actual benchmark strategies
"""
import os
import sys
import pandas as pd
import numpy as np
from config import Config
from src.evaluation.backtesting import PerformanceMetrics

# Suppress all Unicode output issues
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

def load_ml_strategies():
    """Load ML portfolio strategy results"""
    strategies = {}
    strategy_names = ['risk_parity', 'mean_variance', 'max_sharpe', 'cluster_based', 'min_variance']
    
    for strategy in strategy_names:
        try:
            values_path = os.path.join(Config.RESULTS_DIR, f'portfolio_values_{strategy}.csv')
            if os.path.exists(values_path):
                portfolio_data = pd.read_csv(values_path, index_col=0, parse_dates=True)
                data_series = portfolio_data.iloc[:, 0] if isinstance(portfolio_data, pd.DataFrame) else portfolio_data
                
                # Detect format
                if abs(data_series.mean()) > 10:  # Portfolio dollar values
                    returns = data_series.pct_change().dropna()
                else:  # Already returns
                    returns = data_series.dropna()
                
                returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
                strategies[f'ML_{strategy}'] = returns
                print(f"Loaded ML_{strategy}: {len(returns)} returns")
        except Exception as e:
            print(f"Error loading {strategy}: {e}")
    
    return strategies

def create_benchmark_strategies():
    """Create simple benchmark strategies"""
    benchmarks = {}
    
    try:
        # Load processed stock data
        processed_path = os.path.join(Config.PROCESSED_DATA_DIR, 'processed_stock_data.csv')
        df = pd.read_csv(processed_path)
        
        # Find where data starts
        first_valid_idx = None
        for i in range(len(df)):
            try:
                pd.to_datetime(df.iloc[i, 0], errors='raise')
                first_valid_idx = i
                break
            except:
                continue
        
        if first_valid_idx is None:
            print("Could not find valid dates")
            return benchmarks
        
        dates = pd.to_datetime(df.iloc[first_valid_idx:, 0])
        
        # Extract ticker symbols and close prices
        ticker_symbols = set()
        for col in df.columns[1:]:
            symbol = col.split('.')[0] if '.' in col else col
            ticker_symbols.add(symbol)
        
        symbol_prices = {}
        for symbol in sorted(ticker_symbols):
            # Find close price column (usually 4th column per symbol)
            for i, col in enumerate(df.columns[1:], 1):
                if col == symbol or col.startswith(f"{symbol}."):
                    close_prices = pd.to_numeric(df.iloc[first_valid_idx:, i+3], errors='coerce')
                    if close_prices.notna().sum() > 100:
                        symbol_prices[symbol] = close_prices.values
                    break
        
        prices_df = pd.DataFrame(symbol_prices, index=dates.values).dropna(axis=1, how='all')
        print(f"Loaded prices for {len(prices_df.columns)} stocks")
        
        # 1. Equal Weight Portfolio
        equal_returns = prices_df.pct_change().mean(axis=1).dropna()
        benchmarks['Equal_Weight_Portfolio'] = equal_returns
        
        # 2. Cap Weighted (proxy using avg prices)
        weights = prices_df.mean() / prices_df.mean().sum()
        cap_returns = (prices_df.pct_change() * weights).sum(axis=1).dropna()
        benchmarks['Cap_Weighted_Portfolio'] = cap_returns
        
        # 3. Best/Worst stocks
        stock_returns = prices_df.pct_change()
        total_returns = (1 + stock_returns).prod() - 1
        best = total_returns.idxmax()
        worst = total_returns.idxmin()
        benchmarks[f'Best_Stock_{best}'] = stock_returns[best].dropna()
        benchmarks[f'Worst_Stock_{worst}'] = stock_returns[worst].dropna()
        
        print(f"Created {len(benchmarks)} benchmark strategies")
        
    except Exception as e:
        print(f"Error creating benchmarks: {e}")
        import traceback
        traceback.print_exc()
    
    # Load SPY
    try:
        spy_path = os.path.join(Config.RAW_DATA_DIR, 'SPY_benchmark.csv')
        if os.path.exists(spy_path):
            spy_df = pd.read_csv(spy_path)
            spy_dates = pd.to_datetime(spy_df.iloc[:, 0])
            spy_close = pd.to_numeric(spy_df.iloc[:, 1], errors='coerce')
            spy_series = pd.Series(spy_close.values, index=spy_dates.values)
            spy_returns = spy_series.pct_change().dropna()
            benchmarks['SPY_SP500'] = spy_returns
            print(f"Loaded SPY benchmark")
    except Exception as e:
        print(f"Could not load SPY: {e}")
    
    return benchmarks

def calculate_metrics(strategies_dict):
    """Calculate performance metrics"""
    results = []
    
    for name, returns in strategies_dict.items():
        if len(returns) < 50:
            continue
        
        try:
            results.append({
                'Strategy': name,
                'Total_Return': PerformanceMetrics.calculate_total_return(returns),
                'Annual_Return': PerformanceMetrics.calculate_annualized_return(returns),
                'Volatility': PerformanceMetrics.calculate_volatility(returns),
                'Sharpe_Ratio': PerformanceMetrics.calculate_sharpe_ratio(returns, risk_free_rate=0.02),
                'Sortino_Ratio': PerformanceMetrics.calculate_sortino_ratio(returns, risk_free_rate=0.02),
                'Max_Drawdown': PerformanceMetrics.calculate_max_drawdown(returns),
                'Calmar_Ratio': PerformanceMetrics.calculate_calmar_ratio(returns),
                'Observations': len(returns)
            })
        except Exception as e:
            print(f"Error calculating metrics for {name}: {e}")
    
    return pd.DataFrame(results)

def main():
    print("="*60)
    print("Regenerating Complete Benchmark Comparison")
    print("="*60)
    
    # Load all strategies
    ml_strategies = load_ml_strategies()
    benchmark_strategies = create_benchmark_strategies()
    
    # Combine
    all_strategies = {**ml_strategies, **benchmark_strategies}
    print(f"\nTotal strategies: {len(all_strategies)}")
    print(f"  ML: {len(ml_strategies)}")
    print(f"  Benchmarks: {len(benchmark_strategies)}")
    
    # Calculate metrics
    comparison_df = calculate_metrics(all_strategies)
    
    # Save
    output_path = os.path.join(Config.RESULTS_DIR, 'benchmark_comparison_detailed.csv')
    comparison_df.to_csv(output_path, index=False)
    
    print(f"\nSaved to: {output_path}")
    print(f"\nStrategies included:")
    for strategy in comparison_df['Strategy']:
        print(f"  - {strategy}")
    
    print("\nDone!")

if __name__ == "__main__":
    main()
