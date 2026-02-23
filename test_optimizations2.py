"""Test optimization methods with higher max_weight"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from src.optimization.portfolio_optimizer import PortfolioOptimizer

# Create synthetic data
np.random.seed(42)
n_stocks = 10
n_days = 500

# Generate random returns with DIFFERENT volatilities
returns = []
for i in range(n_stocks):
    vol = 0.01 * (i + 1)  # Vary volatility from 1% to 10%
    returns.append(np.random.randn(n_days) * vol)

returns = np.column_stack(returns)
symbols = [f'STOCK_{i}' for i in range(n_stocks)]

returns_df = pd.DataFrame(returns, columns=symbols)

# Create price data
prices = (1 + returns_df).cumprod() * 100

# Make multi-index
price_data = pd.DataFrame()
for symbol in symbols:
    price_data[(symbol, 'Close')] = prices[symbol]
    price_data[(symbol, 'returns_1d')] = returns_df[symbol]

price_data.columns = pd.MultiIndex.from_tuples(price_data.columns)

print("Testing portfolio optimizations with VARIED volatilities\n")
print(f"Number of stocks: {n_stocks}")
print(f"Number of days: {n_days}\n")

# Print volatilities
print("Stock Volatilities (annual):")
for i, symbol in enumerate(symbols):
    vol = returns_df[symbol].std() * np.sqrt(252)
    print(f"  {symbol}: {vol*100:.2f}%")

# Create optimizer
optimizer = PortfolioOptimizer()

# Define varied expected returns
expected_returns_varied = pd.Series({
    'STOCK_0': 0.002,  # High return, low vol
    'STOCK_1': 0.0015,
    'STOCK_2': 0.001,
    'STOCK_3': 0.0008,
    'STOCK_4': 0.0005,
    'STOCK_5': 0.0003,
    'STOCK_6': 0.0001,
    'STOCK_7': 0.00005,
    'STOCK_8': 0.00001,
    'STOCK_9': -0.0001  # Low return, high vol
})

# Test with HIGHER max_weight
constraints = {'max_weight': 0.5, 'min_weight': 0.0}

print("\n" + "="*80)
print("TESTING WITH max_weight=0.5 (50%)")
print("="*80)

print("\n" + "="*80)
print("MEAN VARIANCE")
print("="*80)
result_mv = optimizer.optimize_portfolio(price_data, expected_returns=expected_returns_varied, method='mean_variance', constraints=constraints)
mv_weights = result_mv['weights']
print(f"\nTop 5 weights:")
for stock, weight in mv_weights.nlargest(5).items():
    print(f"  {stock}: {weight*100:6.2f}%")
print(f"Bottom 5 weights:")
for stock, weight in mv_weights.nsmallest(5).items():
    print(f"  {stock}: {weight*100:6.2f}%")

print("\n" + "="*80)
print("MAX SHARPE")
print("="*80)
result_ms = optimizer.optimize_portfolio(price_data, expected_returns=expected_returns_varied, method='max_sharpe', constraints=constraints)
ms_weights = result_ms['weights']
print(f"\nTop 5 weights:")
for stock, weight in ms_weights.nlargest(5).items():
    print(f"  {stock}: {weight*100:6.2f}%")
print(f"Bottom 5 weights:")
for stock, weight in ms_weights.nsmallest(5).items():
    print(f"  {stock}: {weight*100:6.2f}%")

print("\n" + "="*80)
print("MIN VARIANCE")
print("="*80)
result_minv = optimizer.optimize_portfolio(price_data, expected_returns=expected_returns_varied, method='min_variance', constraints=constraints)
minv_weights = result_minv['weights']
print(f"\nTop 5 weights:")
for stock, weight in minv_weights.nlargest(5).items():
    print(f"  {stock}: {weight*100:6.2f}%")
print(f"Bottom 5 weights:")
for stock, weight in minv_weights.nsmallest(5).items():
    print(f"  {stock}: {weight*100:6.2f}%")

print("\n" + "="*80)
print("ARE THEY DIFFERENT?")
print("="*80)
print(f"Mean Variance vs Max Sharpe: {'SAME' if np.allclose(mv_weights, ms_weights, rtol=1e-4) else 'DIFFERENT'}")
print(f"Mean Variance vs Min Variance: {'SAME' if np.allclose(mv_weights, minv_weights, rtol=1e-4) else 'DIFFERENT'}")
print(f"Max Sharpe vs Min Variance: {'SAME' if np.allclose(ms_weights, minv_weights, rtol=1e-4) else 'DIFFERENT'}")

# Print weight differences
print("\n" + "="*80)
print("WEIGHT DIFFERENCES (top 3 stocks)")
print("="*80)
df = pd.DataFrame({
    'Mean Variance': mv_weights,
    'Max Sharpe': ms_weights,
    'Min Variance': minv_weights
})
print(df.head(3))
