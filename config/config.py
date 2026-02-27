"""
Configuration settings for the portfolio optimization project
"""
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any

class Config:
    """Main configuration class for portfolio optimization"""
    
    # Data settings
    DATA_START_DATE = "2010-01-01"   # Collect 15 years of history
    DATA_END_DATE = datetime.now().strftime("%Y-%m-%d")

    # Walk-forward rolling training window (trading days).
    # At each monthly rebalancing the optimizer uses only the most recent
    # TRAINING_WINDOW_DAYS of data so it adapts to the current market regime
    # rather than being anchored to distant historical regimes.
    # 756 ≈ 3 years of daily data.  Set to None to use all available history.
    TRAINING_WINDOW_DAYS = 756

    # First date at which we BEGIN recording portfolio performance.
    # Data collected before this date is used purely as warm-up for the
    # rolling training window.
    BACKTEST_START_DATE_US    = "2013-01-01"   # 3 year warm-up from 2010
    BACKTEST_START_DATE_INDIA = "2013-01-01"
    
    # Market selection
    AVAILABLE_MARKETS = ['US', 'INDIA']
    DEFAULT_MARKET = 'US'
    
    # US Stock universe
    US_STOCK_SYMBOLS = [
        # Technology
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'NFLX',
        # Finance
        'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C',
        # Healthcare
        'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'CVS',
        # Consumer
        'WMT', 'PG', 'KO', 'PEP', 'MCD', 'HD',
        # Energy
        'XOM', 'CVX', 'COP', 'SLB',
        # Industrial
        'BA', 'CAT', 'GE', 'MMM'
    ]
    
    # Indian Stock universe (Nifty 50)
    INDIA_STOCK_SYMBOLS = [
        # Add .NS suffix for NSE (National Stock Exchange of India)
        'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
        'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS',
        'LT.NS', 'AXISBANK.NS', 'ASIANPAINT.NS', 'MARUTI.NS', 'TITAN.NS',
        'SUNPHARMA.NS', 'ULTRACEMCO.NS', 'NESTLEIND.NS', 'BAJFINANCE.NS', 'WIPRO.NS',
        'HCLTECH.NS', 'ADANIENT.NS', 'ONGC.NS', 'NTPC.NS', 'POWERGRID.NS',
        'TATAMOTORS.NS', 'TATASTEEL.NS', 'BAJAJFINSV.NS', 'M&M.NS', 'TECHM.NS',
        'COALINDIA.NS', 'HINDALCO.NS', 'INDUSINDBK.NS', 'DRREDDY.NS', 'CIPLA.NS',
        'EICHERMOT.NS', 'DIVISLAB.NS', 'TATACONSUM.NS', 'GRASIM.NS', 'JSWSTEEL.NS',
        'APOLLOHOSP.NS', 'HEROMOTOCO.NS', 'BRITANNIA.NS', 'SBILIFE.NS', 'ADANIPORTS.NS',
        'BPCL.NS', 'HDFCLIFE.NS', 'SHRIRAMFIN.NS', 'LTIM.NS', 'BAJAJ-AUTO.NS'
    ]
    
    # Legacy compatibility - defaults to US
    STOCK_SYMBOLS = US_STOCK_SYMBOLS
    
    # Market-specific benchmarks
    MARKET_BENCHMARKS = {
        'US': 'SPY',      # S&P 500
        'INDIA': '^NSEI'  # Nifty 50 Index
    }
    
    # Market benchmark (legacy)
    BENCHMARK_SYMBOL = "SPY"
    
    # Clustering settings
    CLUSTERING_FEATURES = [
        'returns_mean', 'returns_std', 'volume_mean', 'market_cap',
        'pe_ratio', 'beta', 'sharpe_ratio', 'max_drawdown',
        'correlation_spy', 'volatility_ratio'
    ]
    
    CLUSTERING_ALGORITHMS = {
        'kmeans': {'n_clusters': 6},
        'hierarchical': {'n_clusters': 6, 'linkage': 'ward'},
        'gaussian_mixture': {'n_components': 6}
        # 'hdbscan': {'min_cluster_size': 3}  # Commented out - requires hdbscan package
    }
    
    # Feature engineering settings
    LOOKBACK_PERIODS = [5, 10, 20, 60]  # Trading days for rolling features
    TECHNICAL_INDICATORS = ['SMA', 'EMA', 'RSI', 'MACD', 'BB', 'ATR']
    
    # Model settings
    PREDICTION_HORIZONS = [1, 5, 20]  # Days ahead to predict
    
    ML_MODELS = {
        'linear_regression': {},
        'ridge': {'alpha': 1.0},
        'lasso': {'alpha': 0.1},
        'random_forest': {'n_estimators': 100, 'max_depth': 10},
        'xgboost': {'n_estimators': 100, 'max_depth': 6},
        'svm': {'C': 1.0, 'gamma': 'scale'}
    }
    
    DL_MODELS = {
        'lstm': {
            'sequence_length': 60,
            'hidden_size': 50,
            'num_layers': 2,
            'dropout': 0.2,
            'epochs': 100,
            'batch_size': 32
        },
        'gru': {
            'sequence_length': 60,
            'hidden_size': 50,
            'num_layers': 2,
            'dropout': 0.2,
            'epochs': 100,
            'batch_size': 32
        },
        'transformer': {
            'sequence_length': 60,
            'd_model': 64,
            'nhead': 8,
            'num_layers': 3,
            'dropout': 0.1,
            'epochs': 100,
            'batch_size': 32
        }
    }
    
    # Portfolio optimization settings
    OPTIMIZATION_METHODS = [
        'mean_variance',      # Modern Portfolio Theory
        'risk_parity',        # Risk Parity
        'hierarchical_risk_parity',  # HRP
        'black_litterman',    # Black-Litterman
        'robust_optimization' # Robust optimization
    ]
    
    # Risk constraints
    MAX_WEIGHT_PER_STOCK = 0.25  # Maximum 25% allocation per stock
    MAX_WEIGHT_PER_SECTOR = 0.3  # Maximum 30% allocation per sector
    MIN_WEIGHT = 0.0  # Minimum weight (allows short selling if negative)
    
    # Backtesting settings
    REBALANCING_FREQUENCY = 'monthly'  # 'daily', 'weekly', 'monthly', 'quarterly'
    TRANSACTION_COST = 0.001  # 0.1% transaction cost
    
    # File paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
    FEATURES_DIR = os.path.join(DATA_DIR, 'features')
    RESULTS_DIR = os.path.join(BASE_DIR, 'results')
    
    # Logging settings
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {module} | {message}"
    
    @staticmethod
    def get_stock_symbols(market: str = 'US') -> List[str]:
        """Get stock symbols for a specific market"""
        market = market.upper()
        if market == 'US':
            return Config.US_STOCK_SYMBOLS
        elif market == 'INDIA':
            return Config.INDIA_STOCK_SYMBOLS
        else:
            raise ValueError(f"Unknown market: {market}. Available markets: {Config.AVAILABLE_MARKETS}")
    
    @staticmethod
    def get_benchmark_symbol(market: str = 'US') -> str:
        """Get benchmark symbol for a specific market"""
        market = market.upper()
        return Config.MARKET_BENCHMARKS.get(market, Config.BENCHMARK_SYMBOL)
    
    @staticmethod
    def get_market_data_dir(market: str, data_type: str = 'raw') -> str:
        """Get market-specific data directory"""
        market = market.upper()
        base_dir = {
            'raw': Config.RAW_DATA_DIR,
            'processed': Config.PROCESSED_DATA_DIR,
            'features': Config.FEATURES_DIR
        }.get(data_type, Config.RAW_DATA_DIR)
        
        market_dir = os.path.join(base_dir, market.lower())
        os.makedirs(market_dir, exist_ok=True)
        return market_dir
    
    @staticmethod
    def get_market_results_dir(market: str) -> str:
        """Get market-specific results directory"""
        market = market.upper()
        results_dir = os.path.join(Config.RESULTS_DIR, market.lower())
        os.makedirs(results_dir, exist_ok=True)
        return results_dir

class ModelConfig:
    """Configuration for specific model parameters"""
    
    @staticmethod
    def get_model_config(model_type: str, model_name: str) -> Dict[str, Any]:
        """Get configuration for a specific model"""
        if model_type == 'ml':
            return Config.ML_MODELS.get(model_name, {})
        elif model_type == 'dl':
            return Config.DL_MODELS.get(model_name, {})
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def get_clustering_config(algorithm: str) -> Dict[str, Any]:
        """Get configuration for clustering algorithm"""
        return Config.CLUSTERING_ALGORITHMS.get(algorithm, {})