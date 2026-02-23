"""Debug script to check expected returns generation"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data.data_loader import PortfolioDataLoader
from ml.return_forecaster import ReturnForecaster
import pandas as pd

# Load stock data
loader = PortfolioDataLoader()
stocks = pd.read_csv('data/raw/dow_jones_stocks.csv')['Symbol'].tolist()
print(f"Loading data for {len(stocks)} stocks...")

price_data, _ = loader.load_stock_data(stocks, start_date='2018-01-01', end_date='2026-01-01')
print(f"Loaded price data shape: {price_data.shape}")

# Forecast returns
forecaster = ReturnForecaster()
expected_returns = forecaster.forecast_returns(price_data)

print("\n=== Expected Returns Summary ===")
print(expected_returns.describe())
print(f"\nMean expected return: {expected_returns.mean():.6f}")
print(f"Std expected return: {expected_returns.std():.6f}")
print(f"Number of zero returns: {(expected_returns == 0).sum()}")
print(f"Number of stocks: {len(expected_returns)}")

print("\n=== First 10 Stocks ===")
print(expected_returns.head(10))

print("\n=== Top 5 Expected Returns ===")
print(expected_returns.nlargest(5))

print("\n=== Bottom 5 Expected Returns ===")
print(expected_returns.nsmallest(5))
