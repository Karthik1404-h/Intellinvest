#!/usr/bin/env python3
"""
Simple script to update benchmark comparison without Unicode issues
"""
import os
import pandas as pd
import numpy as np
from config import Config
from src.evaluation.backtesting import PerformanceMetrics

def load_strategy_results():
    """Load all portfolio strategy results"""
    strategies = {}
    strategy_names = ['risk_parity', 'mean_variance', 'max_sharpe', 'cluster_based', 'min_variance']
    
    for strategy in strategy_names:
        try:
            values_path = os.path.join(Config.RESULTS_DIR, f'portfolio_values_{strategy}.csv')
            if os.path.exists(values_path):
                portfolio_data = pd.read_csv(values_path, index_col=0, parse_dates=True)
                
                if isinstance(portfolio_data, pd.DataFrame):
                    data_series = portfolio_data.iloc[:, 0]
                else:
                    data_series = portfolio_data
                
                # Detect if data is portfolio values (large numbers) or returns (small numbers)
                mean_val = abs(data_series.mean())
                
                if mean_val > 10:  # Likely portfolio dollar values
                    # Calculate returns from portfolio values
                    returns = data_series.pct_change().dropna()
                    print(f"Loaded {strategy}: {len(returns)} returns (converted from portfolio values)")
                else:  # Already returns
                    returns = data_series.dropna()
                    print(f"Loaded {strategy}: {len(returns)} returns (already in return format)")
                
                returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
                
                strategies[f'ML_{strategy}'] = returns
        except Exception as e:
            print(f"Could not load {strategy}: {e}")
    
    return strategies

def calculate_metrics(strategies_dict):
    """Calculate performance metrics for all strategies"""
    results = []
    
    for name, returns in strategies_dict.items():
        if len(returns) == 0:
            continue
        
        # Calculate each metric individually
        total_return = PerformanceMetrics.calculate_total_return(returns)
        annual_return = PerformanceMetrics.calculate_annualized_return(returns)
        volatility = PerformanceMetrics.calculate_volatility(returns)
        sharpe = PerformanceMetrics.calculate_sharpe_ratio(returns, risk_free_rate=0.02)
        sortino = PerformanceMetrics.calculate_sortino_ratio(returns, risk_free_rate=0.02)
        max_dd = PerformanceMetrics.calculate_max_drawdown(returns)
        calmar = PerformanceMetrics.calculate_calmar_ratio(returns)
        
        results.append({
            'Strategy': name,
            'Total_Return': total_return,
            'Annual_Return': annual_return,
            'Volatility': volatility,
            'Sharpe_Ratio': sharpe,
            'Sortino_Ratio': sortino,
            'Max_Drawdown': max_dd,
            'Calmar_Ratio': calmar,
            'Observations': len(returns)
        })
    
    return pd.DataFrame(results)

def main():
    print("Updating benchmark comparison data...")
    
    # Load ML strategies
    strategies = load_strategy_results()
    
    if not strategies:
        print("No strategies loaded! Exiting.")
        return
    
    # Calculate metrics
    comparison_df = calculate_metrics(strategies)
    
    # Save results
    output_path = os.path.join(Config.RESULTS_DIR, 'benchmark_comparison_detailed.csv')
    comparison_df.to_csv(output_path, index=False)
    print(f"\nSaved updated benchmark comparison to: {output_path}")
    
    # Print key values
    print("\nUpdated Annual Returns:")
    for _, row in comparison_df.iterrows():
        print(f"  {row['Strategy']}: {row['Annual_Return']*100:.2f}%")

if __name__ == "__main__":
    main()
