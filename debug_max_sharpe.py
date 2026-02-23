"""Debug max_sharpe optimization specifically"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from src.optimization.portfolio_optimizer import PortfolioOptimizer
from config import Config

# Load real data
processed_data_path =os.path.join(Config.PROCESSED_DATA_DIR, 'processed_stock_data.csv')
price_data = pd.read_csv(processed_data_path, index_col=0, header=[0, 1])
price_data.index = pd.to_datetime(price_data.index)

# Use recent data
price_data = price_data[price_data.index <= '2020-06-01']

print("Testing MAX SHARPE optimization with REAL DATA")
print(f"Data shape: {price_data.shape}")
print(f"Date range: {price_data.index[0]} to {price_data.index[-1]}\n")

# Create optimizer with higher max_weight
optimizer = PortfolioOptimizer()

print(f"Config MAX_WEIGHT_PER_STOCK: {optimizer.config.MAX_WEIGHT_PER_STOCK}")
print(f"Config MIN_WEIGHT: {optimizer.config.MIN_WEIGHT}\n")

# Optimize
print("Running max_sharpe optimization...")
result = optimizer.optimize_portfolio(price_data, method='max_sharpe')

weights = result['weights']

print(f"\nOptimization Status: {result.get('optimization_status', 'unknown')}")
print(f"Expected Return: {result.get('expected_return', 0)*100:.2f}%")
print(f"Volatility: {result.get('volatility', 0)*100:.2f}%")
print(f"Sharpe Ratio: {result.get('sharpe_ratio', 0):.3f}")

print(f"\nTop 10 weights:")
for stock, weight in weights.nlargest(10).items():
    print(f"  {stock}: {weight*100:6.2f}%")

print(f"\nBottom 5 weights:")
for stock, weight in weights.nsmallest(5).items():
    print(f"  {stock}: {weight*100:6.2f}%")

print(f"\nWeight statistics:")
print(f"  Min: {weights.min()*100:.3f}%")
print(f"  Max: {weights.max()*100:.3f}%")
print(f"  Mean: {weights.mean()*100:.3f}%")
print(f"  Std: {weights.std()*100:.3f}%")
print(f"  Sum: {weights.sum():.6f}")

# Check if equal weights
n_stocks = len(weights)
expected_equal = 1.0 / n_stocks
is_equal = np.allclose(weights, expected_equal, rtol=1e-6)
print(f"\nAre weights equal? {is_equal}")
if is_equal:
    print(f"  Expected equal weight: {expected_equal*100:.3f}%")
    print(f"  This suggests the optimization failed or constraints are too restrictive")
