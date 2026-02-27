"""
Regenerate portfolio optimization strategies for a specific market (US or India)
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

def regenerate_strategies(market: str = 'US'):
    """Regenerate all optimization strategies for specified market"""
    market = market.upper()
    
    logger.info("=" * 80)
    logger.info(f"REGENERATING PORTFOLIO STRATEGIES FOR {market} MARKET")
    logger.info("=" * 80)
    
    # Get market-specific paths
    processed_data_dir = Config.get_market_data_dir(market, 'processed')
    results_dir = Config.get_market_results_dir(market)
    
    processed_data_path = os.path.join(processed_data_dir, 'processed_stock_data.csv')
    
    logger.info(f"\nMarket: {market}")
    logger.info(f"Data path: {processed_data_path}")
    logger.info(f"Results directory: {results_dir}")
    
    # Check if data exists
    if not os.path.exists(processed_data_path):
        logger.error(f"Data file not found: {processed_data_path}")
        logger.error(f"\nPlease run data collection first:")
        logger.error(f"  python collect_market_data.py --market {market}")
        return False
    
    try:
        price_data = pd.read_csv(processed_data_path, index_col=0, header=[0, 1])
        price_data.index = pd.to_datetime(price_data.index)
        logger.info(f"Loaded data shape: {price_data.shape}")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return False
    
    # Initialize backtester
    backtester = PortfolioBacktester()
    
    # Load cluster data for cluster_based strategy
    cluster_data = None
    cluster_file = os.path.join(results_dir, 'cluster_assignments_kmeans.csv')
    if os.path.exists(cluster_file):
        cluster_data = pd.read_csv(cluster_file)
        logger.info(f"Loaded cluster data: {len(cluster_data)} assignments from {cluster_file}")
    else:
        logger.warning(f"Cluster file not found: {cluster_file} — cluster_based will fall back to mean-variance")

    # Strategies to regenerate
    strategies = [
        'mean_variance', 'max_sharpe', 'min_variance', 'cluster_based',
        'risk_parity', 'momentum_filter', 'black_litterman', 'concentrated_momentum'
    ]
    
    logger.info(f"\nRegenerating {len(strategies)} strategies: {strategies}")
    logger.info("=" * 80)
    
    results = {}
    
    for strategy in strategies:
        logger.info(f"\n{'='*80}")
        logger.info(f"STRATEGY: {strategy.upper()}")
        logger.info(f"{'='*80}")
        
        try:
            # Determine backtest start date — allow 3 years warm-up from 2010
            start_date = Config.BACKTEST_START_DATE_INDIA if market == 'INDIA' \
                else Config.BACKTEST_START_DATE_US

            # Extra kwargs for strategies that need them
            extra_kwargs = {}
            if strategy == 'cluster_based' and cluster_data is not None:
                extra_kwargs['cluster_data'] = cluster_data

            # Run backtest  (training_window comes from Config.TRAINING_WINDOW_DAYS)
            result = backtester.run_backtest(
                price_data,
                optimization_method=strategy,
                start_date=start_date,
                rebalancing_freq='monthly',
                **extra_kwargs
            )
            
            results[strategy] = result
            
            # Save weights - extract last weights from weights_history
            weights_path = os.path.join(results_dir, f'portfolio_weights_{strategy}.csv')
            weights_history = result.get('weights_history', pd.DataFrame())
            if len(weights_history) > 0 and 'weights' in weights_history.columns:
                last_weights = pd.Series(weights_history.iloc[-1]['weights'])
                last_weights.to_csv(weights_path, header=['weight'])
                logger.info(f"✓ Saved weights to: {weights_path}")
            
            if 'portfolio_values' in result and len(result['portfolio_values']) > 0:
                portfolio_values_path = os.path.join(results_dir, f'portfolio_values_{strategy}.csv')
                result['portfolio_values'].to_csv(portfolio_values_path, header=['portfolio_value'])
                logger.info(f"✓ Saved portfolio values to: {portfolio_values_path}")
            elif 'portfolio_returns' in result and len(result['portfolio_returns']) > 0:
                # Fallback: calculate cumulative values from returns
                portfolio_values_path = os.path.join(results_dir, f'portfolio_values_{strategy}.csv')
                cumulative_value = (1 + result['portfolio_returns']).cumprod() * 1000000
                cumulative_value.to_csv(portfolio_values_path, header=['portfolio_value'])
                logger.info(f"✓ Saved portfolio values (from returns) to: {portfolio_values_path}")
            
            # Save metrics
            if 'performance_metrics' in result:
                import json
                metrics_path = os.path.join(results_dir, f'performance_metrics_{strategy}.json')
                metrics_to_save = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                                   for k, v in result['performance_metrics'].items() 
                                   if not isinstance(v, (pd.Series, pd.DataFrame))}
                with open(metrics_path, 'w') as f:
                    json.dump(metrics_to_save, f, indent=2)
                logger.info(f"✓ Saved metrics to: {metrics_path}")
            
            # Print performance summary
            metrics = result.get('performance_metrics', {})
            logger.info(f"\nPerformance Summary:")
            logger.info(f"  Total Return:        {metrics.get('total_return', 0)*100:8.2f}%")
            logger.info(f"  Annualized Return:   {metrics.get('annualized_return', 0)*100:8.2f}%")
            logger.info(f"  Volatility:          {metrics.get('volatility', 0)*100:8.2f}%")
            logger.info(f"  Sharpe Ratio:        {metrics.get('sharpe_ratio', 0):8.3f}")
            logger.info(f"  Max Drawdown:        {metrics.get('max_drawdown', 0)*100:8.2f}%")
            
        except Exception as e:
            logger.error(f"Error processing {strategy}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Create comparison summary
    logger.info(f"\n{'='*80}")
    logger.info("COMPARISON SUMMARY")
    logger.info(f"{'='*80}\n")
    
    comparison_data = []
    for strategy, result in results.items():
        metrics = result.get('performance_metrics', {})
        comparison_data.append({
            'Strategy': strategy,
            'Total Return': metrics.get('total_return', 0),
            'Annualized Return': metrics.get('annualized_return', 0),
            'Volatility': metrics.get('volatility', 0),
            'Sharpe Ratio': metrics.get('sharpe_ratio', 0),
            'Max Drawdown': metrics.get('max_drawdown', 0)
        })
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data).set_index('Strategy')
        print(comparison_df.round(4))
    else:
        logger.warning("No strategies completed successfully - comparison summary unavailable")
    
    # Save comparison
    comparison_path = os.path.join(results_dir, 'strategy_comparison_summary.csv')
    comparison_df.to_csv(comparison_path)
    logger.info(f"\n✓ Saved comparison to: {comparison_path}")
    
    # Check if strategies are different
    logger.info(f"\n{'='*80}")
    logger.info("VALIDATION: Checking if strategies are different")
    logger.info(f"{'='*80}\n")
    
    if len(results) >= 2:
        sharpe_ratios = [result.get('performance_metrics', {}).get('sharpe_ratio', 0) for result in results.values()]
        returns = [result.get('performance_metrics', {}).get('total_return', 0) for result in results.values()]
        
        if len(set([round(sr, 3) for sr in sharpe_ratios])) > 1:
            logger.info("✓ Strategies show DIFFERENT Sharpe ratios - Good!")
        else:
            logger.warning("⚠ Strategies have IDENTICAL Sharpe ratios - Check implementation")
        
        if len(set([round(r, 4) for r in returns])) > 1:
            logger.info("✓ Strategies show DIFFERENT returns - Good!")
        else:
            logger.warning("⚠ Strategies have IDENTICAL returns - Check implementation")
    
    logger.info(f"\n{'='*80}")
    logger.info(f"✓ REGENERATION COMPLETE FOR {market} MARKET!")
    logger.info(f"{'='*80}")
    
    return True

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Regenerate portfolio optimization strategies')
    parser.add_argument(
        '--market', 
        type=str, 
        default='US',
        choices=['US', 'INDIA', 'us', 'india'],
        help='Market to process (US or INDIA)'
    )
    
    args = parser.parse_args()
    
    success = regenerate_strategies(args.market)
    
    if success:
        logger.info(f"\n✓ Successfully regenerated {args.market} market strategies")
        logger.info(f"\nNext step:")
        logger.info(f"  View in dashboard: streamlit run dashboard.py")
    else:
        logger.error(f"\n✗ Failed to regenerate {args.market} market strategies")
        sys.exit(1)

if __name__ == "__main__":
    main()
