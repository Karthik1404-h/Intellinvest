"""
Regenerate portfolio optimization strategies with fixed implementations
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from datetime import datetime
from loguru import logger

from evaluation.backtesting import PortfolioBacktester
from config import Config

def regenerate_strategies():
    """Regenerate all optimization strategies"""
    logger.info("=" * 80)
    logger.info("REGENERATING PORTFOLIO STRATEGIES WITH FIXED OPTIMIZATIONS")
    logger.info("=" * 80)
    
    # Load processed data
    processed_data_path = os.path.join(Config.PROCESSED_DATA_DIR, 'processed_stock_data.csv')
    
    try:
        price_data = pd.read_csv(processed_data_path, index_col=0, header=[0, 1])
        price_data.index = pd.to_datetime(price_data.index)
        logger.info(f"Loaded data shape: {price_data.shape}")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return
    
    # Initialize backtester
    backtester = PortfolioBacktester()
    
    # Strategies to regenerate
    strategies = ['mean_variance', 'max_sharpe', 'min_variance', 'cluster_based', 'risk_parity']
    
    logger.info(f"\nRegenerating {len(strategies)} strategies: {strategies}")
    logger.info("=" * 80)
    
    results = {}
    
    for strategy in strategies:
        logger.info(f"\n{'='*80}")
        logger.info(f"STRATEGY: {strategy.upper()}")
        logger.info(f"{'='*80}")
        
        try:
            # Check if this is cluster_based and load cluster data
            cluster_data = None
            if strategy == 'cluster_based':
                cluster_path = os.path.join(Config.RESULTS_DIR, 'cluster_assignments_kmeans.csv')
                if os.path.exists(cluster_path):
                    cluster_data = pd.read_csv(cluster_path)
                    logger.info(f"Loaded cluster data: {len(cluster_data)} stocks")
            
            # Run backtest
            result = backtester.run_backtest(
                price_data=price_data,
                optimization_method=strategy,
                rebalancing_freq='monthly',
                start_date='2020-01-01',
                transaction_cost=0.001,
                cluster_data=cluster_data
            )
            
            results[strategy] = result
            
            # Display metrics
            metrics = result['performance_metrics']
            logger.info(f"\nPerformance Metrics:")
            logger.info(f"  Total Return:      {metrics.get('total_return', 0)*100:7.2f}%")
            logger.info(f"  Annual Return:     {metrics.get('annualized_return', 0)*100:7.2f}%")
            logger.info(f"  Sharpe Ratio:      {metrics.get('sharpe_ratio', 0):7.3f}")
            logger.info(f"  Volatility:        {metrics.get('volatility', 0)*100:7.2f}%")
            logger.info(f"  Max Drawdown:      {metrics.get('max_drawdown', 0)*100:7.2f}%")
            
            # Save results
            results_dir = Config.RESULTS_DIR
            
            # Save portfolio values (cumulative dollar values, not returns!)
            portfolio_values_path = os.path.join(results_dir, f'portfolio_values_{strategy}.csv')
            # Convert returns to cumulative portfolio values
            if 'portfolio_values' in result:
                result['portfolio_values'].to_csv(portfolio_values_path, header=['portfolio_value'])
            else:
                # Fallback: calculate from returns
                cumulative_value = (1 + result['portfolio_returns']).cumprod() * metrics.get('initial_capital', 1000000)
                cumulative_value.to_csv(portfolio_values_path, header=['portfolio_value'])
            logger.info(f"  Saved: {portfolio_values_path}")
            
            # Save portfolio weights
            if 'weights_history' in result and not result['weights_history'].empty:
                # Get the latest weights
                latest_weights_row = result['weights_history'].iloc[-1]
                weights_dict = latest_weights_row['weights'] if isinstance(latest_weights_row['weights'], dict) else {}
                
                weights_series = pd.Series(weights_dict)
                weights_path = os.path.join(results_dir, f'portfolio_weights_{strategy}.csv')
                weights_series.to_csv(weights_path)
                logger.info(f"  Saved: {weights_path}")
                
                # Display top 5 weights
                logger.info(f"\n  Top 5 Weights:")
                for symbol, weight in weights_series.nlargest(5).items():
                    logger.info(f"    {symbol:6s}: {weight*100:6.2f}%")
            
            # Save performance metrics
            import json
            metrics_path = os.path.join(results_dir, f'performance_metrics_{strategy}.json')
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"  Saved: {metrics_path}")
            
            logger.info(f"\n✓ {strategy} completed successfully!")
            
        except Exception as e:
            logger.error(f"✗ Error regenerating {strategy}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Create comparison summary
    logger.info(f"\n{'='*80}")
    logger.info("COMPARISON SUMMARY")
    logger.info(f"{'='*80}\n")
    
    comparison_data = []
    for strategy, result in results.items():
        metrics = result['performance_metrics']
        comparison_data.append({
            'Strategy': strategy,
            'Total Return': metrics.get('total_return', 0),
            'Annual Return': metrics.get('annualized_return', 0),
            'Sharpe Ratio': metrics.get('sharpe_ratio', 0),
            'Volatility': metrics.get('volatility', 0),
            'Max Drawdown': metrics.get('max_drawdown', 0)
        })
    
    comparison_df = pd.DataFrame(comparison_data).set_index('Strategy')
    print(comparison_df.round(4))
    
    # Check if strategies are different
    logger.info(f"\n{'='*80}")
    logger.info("VALIDATION: Checking if strategies are different")
    logger.info(f"{'='*80}\n")
    
    if len(results) >= 2:
        strategies_list = list(results.keys())
        for i in range(len(strategies_list)):
            for j in range(i+1, len(strategies_list)):
                strat1, strat2 = strategies_list[i], strategies_list[j]
                returns1 = results[strat1]['portfolio_returns']
                returns2 = results[strat2]['portfolio_returns']
                
                # Compare first few values
                are_same = np.allclose(returns1.head(10).values, returns2.head(10).values, rtol=1e-10)
                
                if are_same:
                    logger.warning(f"⚠️  {strat1} and {strat2} appear IDENTICAL!")
                else:
                    logger.info(f"✓  {strat1} and {strat2} are DIFFERENT")
    
    logger.info(f"\n{'='*80}")
    logger.info("REGENERATION COMPLETE!")
    logger.info(f"{'='*80}")

if __name__ == "__main__":
    regenerate_strategies()
