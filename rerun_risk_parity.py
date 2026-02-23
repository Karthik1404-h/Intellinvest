#!/usr/bin/env python3
"""
Re-run Risk Parity optimization after fixing the constraint bug
"""
import pandas as pd
import numpy as np
import os
import json
from loguru import logger

from config import Config
from src.optimization.portfolio_optimizer import PortfolioOptimizer
from src.evaluation.backtesting import PerformanceMetrics

def calculate_portfolio_returns(weights, returns_data):
    """Calculate portfolio returns given weights and historical returns"""
    # Align weights and returns columns
    common_stocks = weights.index.intersection(returns_data.columns)
    weights_aligned = weights[common_stocks]
    returns_aligned = returns_data[common_stocks]
    
    # Calculate daily portfolio returns
    portfolio_returns = (returns_aligned * weights_aligned).sum(axis=1)
    return portfolio_returns

def calculate_portfolio_value(portfolio_returns, initial_capital=1000000):
    """Calculate portfolio value over time"""
    cumulative_returns = (1 + portfolio_returns).cumprod()
    portfolio_value = initial_capital * cumulative_returns
    return portfolio_value

def main():
    logger.info("Re-running Risk Parity optimization with fixed constraints...")
    
    config = Config()
    
    # Load price data (not returns!)
    price_data_path = os.path.join(config.PROCESSED_DATA_DIR, 'processed_stock_data.csv')
    
    try:
        # Load with multi-index headers
        price_data = pd.read_csv(price_data_path, index_col=0, header=[0, 1])
        price_data.index = pd.to_datetime(price_data.index)
        logger.info(f"Loaded price data shape: {price_data.shape}")
    except Exception as e:
        logger.error(f"Error loading price data: {e}")
        return
    
    # Initialize optimizer
    optimizer = PortfolioOptimizer()
    
    # Run risk parity optimization
    logger.info("Running Risk Parity optimization...")
    try:
        result = optimizer.optimize_portfolio(
            price_data=price_data,
            method='risk_parity'
        )
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Save weights
    weights_path = os.path.join(config.RESULTS_DIR, 'portfolio_weights_risk_parity.csv')
    result['weights'].to_csv(weights_path, header=['weight'])
    logger.info(f"Saved weights to {weights_path}")
    
    # Print weight statistics
    weights = result['weights']
    logger.info(f"\n{'='*60}")
    logger.info(f"Weight Statistics:")
    logger.info(f"{'='*60}")
    logger.info(f"  Number of non-zero positions: {(weights > 0.001).sum()}")
    logger.info(f"  Max weight: {weights.max():.4f} ({weights.idxmax()})")
    logger.info(f"  Min weight (non-zero): {weights[weights > 0.001].min():.4f}")
    logger.info(f"  Mean weight: {weights[weights > 0.001].mean():.4f}")
    logger.info(f"  Std dev of weights: {weights.std():.4f}")
    logger.info(f"\n  Top 10 holdings:")
    for stock, weight in weights.nlargest(10).items():
        logger.info(f"    {stock}: {weight:.4f} ({weight*100:.2f}%)")
    
    # Save optimization summary
    summary = {
        'method': 'risk_parity',
        'expected_return': result.get('expected_return', 0),
        'volatility': result.get('volatility', 0),
        'sharpe_ratio': result.get('sharpe_ratio', 0),
        'optimization_status': result.get('optimization_status', 'success')
    }
    
    summary_path = os.path.join(config.RESULTS_DIR, 'optimization_summary_risk_parity.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    logger.info(f"\nSaved optimization summary to {summary_path}")
    
    # Calculate portfolio performance using price data
    logger.info("\nCalculating portfolio performance...")
    
    # Get clean closing prices for portfolio calculation
    closing_prices = {}
    for ticker in weights.index:
        if (ticker, 'Close') in price_data.columns:
            closing_prices[ticker] = price_data[(ticker, 'Close')]
    
    close_df = pd.DataFrame(closing_prices).dropna()
    returns_data = close_df.pct_change().dropna()
    
    # Calculate portfolio returns
    portfolio_returns = calculate_portfolio_returns(weights, returns_data)
    portfolio_value = calculate_portfolio_value(portfolio_returns)
    
    # Save portfolio values
    values_path = os.path.join(config.RESULTS_DIR, 'portfolio_values_risk_parity.csv')
    portfolio_value.to_csv(values_path, header=['portfolio_value'])
    logger.info(f"Saved portfolio values to {values_path}")
    
    # Calculate performance metrics
    metrics = {
        'total_return': PerformanceMetrics.calculate_total_return(portfolio_returns),
        'annualized_return': PerformanceMetrics.calculate_annualized_return(portfolio_returns),
        'volatility': PerformanceMetrics.calculate_volatility(portfolio_returns),
        'sharpe_ratio': PerformanceMetrics.calculate_sharpe_ratio(portfolio_returns),
        'sortino_ratio': PerformanceMetrics.calculate_sortino_ratio(portfolio_returns),
        'max_drawdown': PerformanceMetrics.calculate_max_drawdown(portfolio_returns),
        'calmar_ratio': PerformanceMetrics.calculate_calmar_ratio(portfolio_returns),
        'initial_capital': 1000000,
        'final_capital': portfolio_value.iloc[-1],
        'total_transaction_costs': 0.0,
        'average_turnover': 0.0,
        'number_of_rebalances': 0
    }
    
    # Save performance metrics
    metrics_path = os.path.join(config.RESULTS_DIR, 'performance_metrics_risk_parity.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Saved performance metrics to {metrics_path}")
    
    # Print performance summary
    logger.info(f"\n{'='*60}")
    logger.info(f"Performance Metrics:")
    logger.info(f"{'='*60}")
    logger.info(f"  Total Return: {metrics['total_return']*100:.2f}%")
    logger.info(f"  Annualized Return: {metrics['annualized_return']*100:.2f}%")
    logger.info(f"  Volatility: {metrics['volatility']*100:.2f}%")
    logger.info(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
    logger.info(f"  Sortino Ratio: {metrics['sortino_ratio']:.3f}")
    logger.info(f"  Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
    logger.info(f"  Calmar Ratio: {metrics['calmar_ratio']:.3f}")
    logger.info(f"{'='*60}")
    
    logger.info("\n✅ Risk Parity optimization completed successfully!")
    logger.info("Refresh your dashboard (press R) to see the updated results.")

if __name__ == "__main__":
    main()
