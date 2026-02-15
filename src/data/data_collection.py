"""
Data collection and preprocessing module for portfolio optimization
"""
import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import os
from loguru import logger
from tqdm import tqdm

from config import Config

class DataCollector:
    """Collects financial data from various sources"""
    
    def __init__(self):
        self.config = Config()
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create necessary directories if they don't exist"""
        os.makedirs(self.config.RAW_DATA_DIR, exist_ok=True)
        os.makedirs(self.config.PROCESSED_DATA_DIR, exist_ok=True)
        os.makedirs(self.config.FEATURES_DIR, exist_ok=True)
    
    def collect_stock_data(self, 
                          symbols: Optional[List[str]] = None,
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None,
                          save_to_file: bool = True) -> pd.DataFrame:
        """
        Collect stock price data from Yahoo Finance
        
        Args:
            symbols: List of stock symbols to collect
            start_date: Start date for data collection
            end_date: End date for data collection
            save_to_file: Whether to save data to file
            
        Returns:
            DataFrame with multi-level columns (symbol, price_type)
        """
        if symbols is None:
            symbols = self.config.STOCK_SYMBOLS
        if start_date is None:
            start_date = self.config.DATA_START_DATE
        if end_date is None:
            end_date = self.config.DATA_END_DATE
        
        logger.info(f"Collecting data for {len(symbols)} symbols from {start_date} to {end_date}")
        
        # Download data for all symbols
        try:
            data = yf.download(symbols, start=start_date, end=end_date, group_by='ticker')
        except Exception as e:
            logger.error(f"Error downloading data: {e}")
            raise
        
        # Reshape data for easier handling
        if len(symbols) == 1:
            # Single symbol case
            data.columns = pd.MultiIndex.from_product([symbols, data.columns])
        
        if save_to_file:
            file_path = os.path.join(self.config.RAW_DATA_DIR, 'stock_prices.csv')
            data.to_csv(file_path)
            logger.info(f"Saved raw stock data to {file_path}")
        
        return data
    
    def collect_benchmark_data(self,
                              benchmark_symbol: Optional[str] = None,
                              start_date: Optional[str] = None,
                              end_date: Optional[str] = None) -> pd.DataFrame:
        """Collect benchmark data (e.g., S&P 500)"""
        if benchmark_symbol is None:
            benchmark_symbol = self.config.BENCHMARK_SYMBOL
        if start_date is None:
            start_date = self.config.DATA_START_DATE
        if end_date is None:
            end_date = self.config.DATA_END_DATE
        
        logger.info(f"Collecting benchmark data for {benchmark_symbol}")
        
        benchmark_data = yf.download(benchmark_symbol, start=start_date, end=end_date)
        
        # Save benchmark data
        file_path = os.path.join(self.config.RAW_DATA_DIR, f'{benchmark_symbol}_benchmark.csv')
        benchmark_data.to_csv(file_path)
        
        return benchmark_data

class DataPreprocessor:
    """Handles data cleaning and preprocessing"""
    
    def __init__(self):
        self.config = Config()
    
    def clean_stock_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean stock price data by handling missing values and outliers
        
        Args:
            data: Raw stock price data
            
        Returns:
            Cleaned stock price data
        """
        logger.info("Cleaning stock price data")
        
        cleaned_data = data.copy()
        
        # Get list of symbols
        symbols = cleaned_data.columns.get_level_values(0).unique()
        
        for symbol in tqdm(symbols, desc="Cleaning data"):
            try:
                symbol_data = cleaned_data[symbol]
                
                # Forward fill missing values first
                symbol_data = symbol_data.fillna(method='ffill')
                
                # Remove rows where all prices are missing
                symbol_data = symbol_data.dropna(subset=['Open', 'High', 'Low', 'Close'])
                
                # Check for price consistency (High >= Low, etc.)
                invalid_mask = (
                    (symbol_data['High'] < symbol_data['Low']) |
                    (symbol_data['Close'] < 0) |
                    (symbol_data['Volume'] < 0)
                )
                
                if invalid_mask.any():
                    logger.warning(f"Found {invalid_mask.sum()} invalid entries for {symbol}")
                    # Replace invalid entries with previous valid values
                    symbol_data[invalid_mask] = np.nan
                    symbol_data = symbol_data.fillna(method='ffill')
                
                cleaned_data[symbol] = symbol_data
                
            except Exception as e:
                logger.warning(f"Error cleaning data for {symbol}: {e}")
                continue
        
        # Remove any remaining rows with all NaN values
        cleaned_data = cleaned_data.dropna(how='all')
        
        logger.info(f"Data cleaning completed. Shape: {cleaned_data.shape}")
        return cleaned_data
    
    def calculate_returns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate various types of returns
        
        Args:
            data: Stock price data with Close prices
            
        Returns:
            DataFrame with calculated returns
        """
        logger.info("Calculating returns")
        
        symbols = data.columns.get_level_values(0).unique()
        returns_data = {}
        
        for symbol in symbols:
            try:
                close_prices = data[symbol]['Close']
                
                # Simple returns
                simple_returns = close_prices.pct_change()
                
                # Log returns
                log_returns = np.log(close_prices / close_prices.shift(1))
                
                # Multi-period returns
                returns_5d = close_prices.pct_change(periods=5)
                returns_20d = close_prices.pct_change(periods=20)
                
                returns_data[symbol] = pd.DataFrame({
                    'returns_1d': simple_returns,
                    'log_returns_1d': log_returns,
                    'returns_5d': returns_5d,
                    'returns_20d': returns_20d
                })
                
            except Exception as e:
                logger.warning(f"Error calculating returns for {symbol}: {e}")
                continue
        
        # Combine all returns into a single DataFrame
        returns_df = pd.concat(returns_data, axis=1)
        
        # Remove infinite and NaN values
        returns_df = returns_df.replace([np.inf, -np.inf], np.nan)
        returns_df = returns_df.fillna(0)
        
        return returns_df
    
    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to the dataset
        
        Args:
            data: Stock price data
            
        Returns:
            DataFrame with technical indicators added
        """
        try:
            import pandas_ta as ta
        except ImportError:
            logger.warning("pandas_ta not installed. Skipping technical indicators.")
            return data
        
        logger.info("Adding technical indicators")
        
        symbols = data.columns.get_level_values(0).unique()
        enhanced_data = data.copy()
        
        for symbol in tqdm(symbols, desc="Adding technical indicators"):
            try:
                symbol_data = data[symbol].copy()
                
                # Simple Moving Averages
                symbol_data['SMA_10'] = symbol_data['Close'].rolling(window=10).mean()
                symbol_data['SMA_20'] = symbol_data['Close'].rolling(window=20).mean()
                symbol_data['SMA_50'] = symbol_data['Close'].rolling(window=50).mean()
                
                # Exponential Moving Averages
                symbol_data['EMA_10'] = symbol_data['Close'].ewm(span=10).mean()
                symbol_data['EMA_20'] = symbol_data['Close'].ewm(span=20).mean()
                
                # RSI
                symbol_data['RSI'] = ta.rsi(symbol_data['Close'], length=14)
                
                # MACD
                macd_data = ta.macd(symbol_data['Close'])
                symbol_data = pd.concat([symbol_data, macd_data], axis=1)
                
                # Bollinger Bands
                bb_data = ta.bbands(symbol_data['Close'], length=20)
                symbol_data = pd.concat([symbol_data, bb_data], axis=1)
                
                # Volume indicators
                symbol_data['Volume_SMA'] = symbol_data['Volume'].rolling(window=20).mean()
                symbol_data['Volume_Ratio'] = symbol_data['Volume'] / symbol_data['Volume_SMA']
                
                enhanced_data[symbol] = symbol_data
                
            except Exception as e:
                logger.warning(f"Error adding technical indicators for {symbol}: {e}")
                continue
        
        return enhanced_data

def main():
    """Main function to collect and preprocess data"""
    logger.info("Starting data collection and preprocessing")
    
    # Initialize data collector and preprocessor
    collector = DataCollector()
    preprocessor = DataPreprocessor()
    
    # Collect stock data
    logger.info("Step 1: Collecting stock data")
    stock_data = collector.collect_stock_data()
    
    # Collect benchmark data
    logger.info("Step 2: Collecting benchmark data")
    benchmark_data = collector.collect_benchmark_data()
    
    # Clean data
    logger.info("Step 3: Cleaning data")
    cleaned_data = preprocessor.clean_stock_data(stock_data)
    
    # Calculate returns
    logger.info("Step 4: Calculating returns")
    returns_data = preprocessor.calculate_returns(cleaned_data)
    
    # Add technical indicators
    logger.info("Step 5: Adding technical indicators")
    enhanced_data = preprocessor.add_technical_indicators(cleaned_data)
    
    # Save processed data
    logger.info("Step 6: Saving processed data")
    processed_file = os.path.join(Config.PROCESSED_DATA_DIR, 'processed_stock_data.csv')
    enhanced_data.to_csv(processed_file)
    
    returns_file = os.path.join(Config.PROCESSED_DATA_DIR, 'returns_data.csv')
    returns_data.to_csv(returns_file)
    
    logger.info("Data collection and preprocessing completed successfully!")

if __name__ == "__main__":
    main()