#!/usr/bin/env python3
"""
Fix the data format issue in processed stock data
"""
import pandas as pd
import numpy as np
from config import Config
import os

def fix_processed_data():
    """Fix the processed data format"""
    print("Fixing processed stock data format...")
    
    # Read the problematic CSV
    processed_data_path = os.path.join(Config.PROCESSED_DATA_DIR, 'processed_stock_data.csv')
    df = pd.read_csv(processed_data_path)
    
    print(f"Original data shape: {df.shape}")
    
    # Extract the header information from row 0
    headers = df.iloc[0].values
    
    # Find the column structure
    tickers = []
    price_types = []
    
    current_ticker = None
    for i, header in enumerate(headers):
        if header != header:  # NaN check
            header = 'Date' if i == 0 else price_types[-1] if price_types else 'Close'
        
        if header == 'Date':
            continue
        elif header in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if current_ticker:
                tickers.append(current_ticker)
                price_types.append(header)
        else:
            # This is a ticker symbol
            current_ticker = header
            tickers.append(current_ticker)
            price_types.append('Open')  # Assume first column after ticker is Open
    
    # Get the data starting from row 2 (skip header and date row)
    data_rows = df.iloc[2:].copy()
    
    # Set the date as index
    dates = pd.to_datetime(data_rows.iloc[:, 0])
    data_values = data_rows.iloc[:, 1:]
    
    # Convert all data to numeric
    for col in data_values.columns:
        data_values[col] = pd.to_numeric(data_values[col], errors='coerce')
    
    # Create new DataFrame with proper structure
    # Group columns by ticker
    ticker_groups = {}
    col_idx = 0
    
    # Get unique tickers from config
    config = Config()
    expected_tickers = config.STOCK_SYMBOLS
    price_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    # Create MultiIndex manually
    multi_columns = []
    col_data = []
    
    for ticker in expected_tickers:
        if col_idx + 4 < len(data_values.columns):  # Make sure we have 5 columns per ticker
            for price_col in price_columns:
                multi_columns.append((ticker, price_col))
                col_data.append(data_values.iloc[:, col_idx])
                col_idx += 1
    
    # Create new DataFrame with MultiIndex
    multi_index = pd.MultiIndex.from_tuples(multi_columns, names=['Ticker', 'Price'])
    fixed_data = pd.DataFrame(np.column_stack(col_data), 
                             index=dates, 
                             columns=multi_index)
    
    # Save the fixed data
    fixed_path = os.path.join(Config.PROCESSED_DATA_DIR, 'processed_stock_data_fixed.csv')
    
    # Save with a flattened header that's easier to read
    # Flatten the MultiIndex for saving
    flat_columns = [f"{ticker}_{price}" for ticker, price in multi_columns]
    flat_data = pd.DataFrame(fixed_data.values, 
                           index=fixed_data.index, 
                           columns=flat_columns)
    
    flat_data.to_csv(fixed_path)
    print(f"Fixed data saved to: {fixed_path}")
    print(f"Fixed data shape: {flat_data.shape}")
    
    # Also fix returns data
    print("\nFixing returns data...")
    returns_data_path = os.path.join(Config.PROCESSED_DATA_DIR, 'returns_data.csv')
    
    if os.path.exists(returns_data_path):
        returns_df = pd.read_csv(returns_data_path)
        print(f"Returns data shape: {returns_df.shape}")
        
        # Skip header if it exists
        if returns_df.iloc[0, 0] in ['Price', 'Ticker']:
            returns_df = returns_df.iloc[1:]
        
        # Set date as index, but skip if first value is 'Date'
        if returns_df.iloc[0, 0] == 'Date':
            returns_df = returns_df.iloc[1:]
        
        returns_df = returns_df.set_index(returns_df.columns[0])
        
        try:
            returns_df.index = pd.to_datetime(returns_df.index)
        except:
            # If date parsing fails, use the dates from the price data
            print("Using dates from price data...")
            returns_df.index = dates[:len(returns_df)]
        
        # Convert to numeric
        for col in returns_df.columns:
            returns_df[col] = pd.to_numeric(returns_df[col], errors='coerce')
        
        # Create return column names
        return_types = ['returns_1d', 'log_returns_1d', 'returns_5d', 'returns_20d']
        return_columns = []
        
        for ticker in expected_tickers:
            for ret_type in return_types:
                return_columns.append(f"{ticker}_{ret_type}")
        
        # Adjust columns to match available data
        n_available_cols = min(len(return_columns), len(returns_df.columns))
        returns_df.columns = return_columns[:n_available_cols]
        
        # Save fixed returns data
        fixed_returns_path = os.path.join(Config.PROCESSED_DATA_DIR, 'returns_data_fixed.csv')
        returns_df.to_csv(fixed_returns_path)
        print(f"Fixed returns data saved to: {fixed_returns_path}")
    
    return fixed_path, fixed_returns_path if os.path.exists(returns_data_path) else None

if __name__ == "__main__":
    fix_processed_data()