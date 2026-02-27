#!/usr/bin/env python3
"""
Collect and process stock data for a specific market (US or India)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import yfinance as yf
from datetime import datetime
from loguru import logger

from config import Config

def collect_stock_data(market: str = 'US'):
    """
    Collect stock data for specified market
    
    Args:
        market: 'US' or 'INDIA'
    """
    market = market.upper()
    logger.info(f"="*80)
    logger.info(f"COLLECTING DATA FOR {market} MARKET")
    logger.info(f"="*80)
    
    # Get market-specific configuration
    symbols = Config.get_stock_symbols(market)
    benchmark = Config.get_benchmark_symbol(market)
    
    logger.info(f"\nMarket: {market}")
    logger.info(f"Number of stocks: {len(symbols)}")
    logger.info(f"Benchmark: {benchmark}")
    logger.info(f"Date range: {Config.DATA_START_DATE} to {Config.DATA_END_DATE}")
    
    # Get market-specific directories
    raw_data_dir = Config.get_market_data_dir(market, 'raw')
    processed_data_dir = Config.get_market_data_dir(market, 'processed')
    
    logger.info(f"Raw data directory: {raw_data_dir}")
    logger.info(f"Processed data directory: {processed_data_dir}")
    
    # Collect stock data
    logger.info(f"\n{'='*80}")
    logger.info("COLLECTING STOCK PRICE DATA")
    logger.info(f"{'='*80}")
    
    try:
        logger.info(f"Downloading data for {len(symbols)} stocks...")
        data = yf.download(
            symbols, 
            start=Config.DATA_START_DATE, 
            end=Config.DATA_END_DATE, 
            group_by='ticker',
            progress=True
        )
        
        if len(symbols) == 1:
            # Single symbol case - add ticker level
            data.columns = pd.MultiIndex.from_product([symbols, data.columns])
        
        # Save raw stock data
        stock_prices_path = os.path.join(raw_data_dir, 'stock_prices.csv')
        data.to_csv(stock_prices_path)
        logger.info(f"✓ Saved stock prices to: {stock_prices_path}")
        logger.info(f"  Data shape: {data.shape}")
        logger.info(f"  Date range: {data.index[0]} to {data.index[-1]}")
        
    except Exception as e:
        logger.error(f"Error downloading stock data: {e}")
        return False
    
    # Collect benchmark data
    logger.info(f"\n{'='*80}")
    logger.info("COLLECTING BENCHMARK DATA")
    logger.info(f"{'='*80}")
    
    try:
        logger.info(f"Downloading {benchmark}...")
        benchmark_data = yf.download(
            benchmark,
            start=Config.DATA_START_DATE,
            end=Config.DATA_END_DATE,
            progress=True
        )
        
        # Save benchmark data
        benchmark_path = os.path.join(raw_data_dir, f'{benchmark.replace("^", "")}_benchmark.csv')
        benchmark_data.to_csv(benchmark_path)
        logger.info(f"✓ Saved benchmark to: {benchmark_path}")
        logger.info(f"  Data shape: {benchmark_data.shape}")
        
    except Exception as e:
        logger.error(f"Error downloading benchmark data: {e}")
        logger.warning("Continuing without benchmark data...")
    
    # Process data
    logger.info(f"\n{'='*80}")
    logger.info("PROCESSING DATA")
    logger.info(f"{'='*80}")
    
    try:
        # Clean and process stock data
        logger.info("Cleaning data (removing NaN, forward-filling gaps)...")
        
        # Forward fill missing values (holidays, etc.)
        processed_data = data.ffill().bfill()
        
        # Remove stocks with too much missing data
        missing_threshold = 0.1  # 10%
        missing_pct = processed_data.isnull().sum() / len(processed_data)
        
        if isinstance(processed_data.columns, pd.MultiIndex):
            # Get ticker level
            tickers_to_keep = []
            for ticker in symbols:
                if ticker not in processed_data.columns.get_level_values(0):
                    logger.warning(f"  ✗ {ticker}: No data available")
                    continue
                    
                ticker_data = processed_data[ticker]
                ticker_missing = ticker_data.isnull().sum().sum() / (len(ticker_data) * len(ticker_data.columns))
                
                if ticker_missing > missing_threshold:
                    logger.warning(f"  ✗ {ticker}: Too much missing data ({ticker_missing*100:.1f}%)")
                else:
                    tickers_to_keep.append(ticker)
                    logger.info(f"  ✓ {ticker}: OK ({ticker_missing*100:.2f}% missing)")
            
            # Filter to valid tickers
            processed_data = processed_data[tickers_to_keep]
        
        # Save processed data
        processed_path = os.path.join(processed_data_dir, 'processed_stock_data.csv')
        processed_data.to_csv(processed_path)
        logger.info(f"\n✓ Saved processed data to: {processed_path}")
        logger.info(f"  Final shape: {processed_data.shape}")
        logger.info(f"  Stocks included: {len(tickers_to_keep)}/{len(symbols)}")
        
        # Calculate and save returns
        logger.info("\nCalculating returns...")
        returns_data = processed_data.xs('Close', level=1, axis=1).pct_change().dropna()
        returns_path = os.path.join(processed_data_dir, 'returns_data.csv')
        returns_data.to_csv(returns_path)
        logger.info(f"✓ Saved returns data to: {returns_path}")
        
    except Exception as e:
        logger.error(f"Error processing data: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    logger.info(f"\n{'='*80}")
    logger.info(f"✓ DATA COLLECTION COMPLETE FOR {market} MARKET")
    logger.info(f"{'='*80}")
    
    return True

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect market data for portfolio optimization')
    parser.add_argument(
        '--market', 
        type=str, 
        default='US',
        choices=['US', 'INDIA', 'us', 'india'],
        help='Market to collect data for (US or INDIA)'
    )
    
    args = parser.parse_args()
    
    # Collect data
    success = collect_stock_data(args.market)
    
    if success:
        logger.info(f"\n✓ Successfully collected and processed {args.market} market data")
        logger.info(f"\nNext steps:")
        logger.info(f"  1. Run clustering: python regenerate_strategies.py --market {args.market}")
        logger.info(f"  2. View in dashboard: streamlit run dashboard.py")
    else:
        logger.error(f"\n✗ Failed to collect {args.market} market data")
        sys.exit(1)

if __name__ == "__main__":
    main()
