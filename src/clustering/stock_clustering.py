"""
Stock clustering module for portfolio optimization
Implements various clustering algorithms to group similar stocks
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from loguru import logger
import os
import warnings
warnings.filterwarnings('ignore')

from config import Config

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    print("Warning: hdbscan not available, will skip HDBSCAN clustering")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm

from config import Config

class FeatureEngineer:
    """Creates features for stock clustering"""
    
    def __init__(self):
        self.config = Config()
    
    def create_clustering_features(self, 
                                 price_data: pd.DataFrame, 
                                 returns_data: pd.DataFrame,
                                 market_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Create comprehensive features for stock clustering
        
        Args:
            price_data: Stock price data
            returns_data: Stock returns data  
            market_data: Market/benchmark data (optional)
            
        Returns:
            DataFrame with clustering features for each stock
        """
        logger.info("Creating clustering features")
        
        symbols = price_data.columns.get_level_values(0).unique()
        features_dict = {}
        
        for symbol in symbols:
            try:
                # Get symbol-specific data
                symbol_prices = price_data[symbol]
                symbol_returns = returns_data[symbol]['returns_1d'] if (symbol, 'returns_1d') in returns_data.columns else None
                
                if symbol_returns is None or len(symbol_returns.dropna()) < 50:
                    logger.warning(f"Insufficient data for {symbol}, skipping")
                    continue
                
                features = self._calculate_stock_features(symbol_prices, symbol_returns, market_data)
                features_dict[symbol] = features
                
            except Exception as e:
                logger.warning(f"Error calculating features for {symbol}: {e}")
                continue
        
        # Convert to DataFrame
        features_df = pd.DataFrame(features_dict).T
        
        # Remove any stocks with missing features
        features_df = features_df.dropna()
        
        logger.info(f"Created features for {len(features_df)} stocks with {len(features_df.columns)} features")
        return features_df
    
    def _calculate_stock_features(self, 
                                prices: pd.DataFrame, 
                                returns: pd.Series,
                                market_data: Optional[pd.DataFrame]) -> Dict[str, float]:
        """Calculate comprehensive features for a single stock"""
        features = {}
        
        # Return-based features
        valid_returns = returns.dropna()
        if len(valid_returns) > 0:
            features['returns_mean'] = valid_returns.mean()
            features['returns_std'] = valid_returns.std()
            features['returns_skewness'] = valid_returns.skew()
            features['returns_kurtosis'] = valid_returns.kurtosis()
            features['sharpe_ratio'] = self._calculate_sharpe_ratio(valid_returns)
            features['max_drawdown'] = self._calculate_max_drawdown(valid_returns)
            features['var_95'] = valid_returns.quantile(0.05)
            features['var_99'] = valid_returns.quantile(0.01)
        
        # Price-based features
        if 'Close' in prices.columns:
            close_prices = prices['Close'].dropna()
            if len(close_prices) > 0:
                # Volatility measures
                features['price_volatility'] = close_prices.pct_change().std()
                
                # Trend measures
                features['price_momentum_1m'] = (close_prices.iloc[-1] / close_prices.iloc[-22] - 1) if len(close_prices) >= 22 else 0
                features['price_momentum_3m'] = (close_prices.iloc[-1] / close_prices.iloc[-66] - 1) if len(close_prices) >= 66 else 0
                features['price_momentum_6m'] = (close_prices.iloc[-1] / close_prices.iloc[-132] - 1) if len(close_prices) >= 132 else 0
                
                # Moving average ratios
                sma_20 = close_prices.rolling(20).mean()
                sma_50 = close_prices.rolling(50).mean()
                if not sma_20.isna().all() and not sma_50.isna().all():
                    features['price_to_sma20'] = close_prices.iloc[-1] / sma_20.iloc[-1] if not np.isnan(sma_20.iloc[-1]) else 1
                    features['price_to_sma50'] = close_prices.iloc[-1] / sma_50.iloc[-1] if not np.isnan(sma_50.iloc[-1]) else 1
        
        # Volume-based features
        if 'Volume' in prices.columns:
            volume = prices['Volume'].dropna()
            if len(volume) > 0:
                features['volume_mean'] = volume.mean()
                features['volume_std'] = volume.std()
                features['volume_trend'] = self._calculate_trend(volume)
        
        # Market correlation (if market data provided)
        if market_data is not None and 'Close' in market_data.columns:
            market_returns = market_data['Close'].pct_change().dropna()
            
            # Align dates
            common_dates = returns.index.intersection(market_returns.index)
            if len(common_dates) > 30:
                stock_aligned = returns.loc[common_dates]
                market_aligned = market_returns.loc[common_dates]
                
                correlation = stock_aligned.corr(market_aligned)
                features['market_correlation'] = correlation if not np.isnan(correlation) else 0
                
                # Beta calculation
                covariance = stock_aligned.cov(market_aligned)
                market_variance = market_aligned.var()
                features['beta'] = covariance / market_variance if market_variance != 0 else 1
        
        # Replace any NaN or infinite values
        for key, value in features.items():
            if np.isnan(value) or np.isinf(value):
                features[key] = 0
        
        return features
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        excess_returns = returns.mean() * 252 - risk_free_rate  # Annualized
        volatility = returns.std() * np.sqrt(252)  # Annualized
        return excess_returns / volatility if volatility != 0 else 0
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        return drawdown.min()
    
    def _calculate_trend(self, series: pd.Series) -> float:
        """Calculate trend using linear regression slope"""
        try:
            x = np.arange(len(series))
            coeffs = np.polyfit(x, series.values, 1)
            return coeffs[0]  # Slope
        except:
            return 0

class StockClusterer:
    """Main class for stock clustering"""
    
    def __init__(self):
        self.config = Config()
        self.scaler = StandardScaler()
        self.cluster_models = {}
        self.cluster_labels = {}
        self.features = None
        self.scaled_features = None
    
    def fit_clustering_models(self, features: pd.DataFrame) -> Dict[str, Any]:
        """
        Fit multiple clustering algorithms to the features
        
        Args:
            features: DataFrame with clustering features
            
        Returns:
            Dictionary with model results and metrics
        """
        logger.info(f"Fitting clustering models on {len(features)} stocks")
        
        self.features = features
        self.scaled_features = self.scaler.fit_transform(features)
        
        results = {}
        
        for algorithm, params in self.config.CLUSTERING_ALGORITHMS.items():
            try:
                logger.info(f"Fitting {algorithm} clustering")
                
                if algorithm == 'kmeans':
                    model = KMeans(**params, random_state=42)
                    labels = model.fit_predict(self.scaled_features)
                    
                elif algorithm == 'hierarchical':
                    model = AgglomerativeClustering(**params)
                    labels = model.fit_predict(self.scaled_features)
                    
                elif algorithm == 'gaussian_mixture':
                    model = GaussianMixture(**params, random_state=42)
                    labels = model.fit_predict(self.scaled_features)
                    
                elif algorithm == 'hdbscan':
                    if not HDBSCAN_AVAILABLE:
                        logger.warning(f"HDBSCAN not available, skipping {algorithm}")
                        continue
                    model = hdbscan.HDBSCAN(**params)
                    labels = model.fit_predict(self.scaled_features)
                
                # Store results
                self.cluster_models[algorithm] = model
                self.cluster_labels[algorithm] = labels
                
                # Calculate metrics
                n_clusters = len(np.unique(labels[labels != -1]))  # Exclude noise for HDBSCAN
                
                if n_clusters > 1:
                    silhouette = silhouette_score(self.scaled_features, labels) if len(np.unique(labels)) > 1 else -1
                    calinski_harabasz = calinski_harabasz_score(self.scaled_features, labels) if len(np.unique(labels)) > 1 else 0
                else:
                    silhouette = -1
                    calinski_harabasz = 0
                
                results[algorithm] = {
                    'model': model,
                    'labels': labels,
                    'n_clusters': n_clusters,
                    'silhouette_score': silhouette,
                    'calinski_harabasz_score': calinski_harabasz,
                    'n_noise_points': np.sum(labels == -1) if algorithm == 'hdbscan' else 0
                }
                
                logger.info(f"{algorithm}: {n_clusters} clusters, silhouette={silhouette:.3f}")
                
            except Exception as e:
                logger.error(f"Error fitting {algorithm}: {e}")
                continue
        
        return results
    
    def get_cluster_assignments(self, algorithm: str = 'kmeans') -> pd.DataFrame:
        """
        Get cluster assignments for stocks
        
        Args:
            algorithm: Clustering algorithm to use
            
        Returns:
            DataFrame with stock symbols and their cluster assignments
        """
        if algorithm not in self.cluster_labels:
            raise ValueError(f"Algorithm {algorithm} not fitted yet")
        
        assignments = pd.DataFrame({
            'symbol': self.features.index,
            'cluster': self.cluster_labels[algorithm]
        })
        
        return assignments
    
    def analyze_clusters(self, algorithm: str = 'kmeans') -> pd.DataFrame:
        """
        Analyze cluster characteristics
        
        Args:
            algorithm: Clustering algorithm to analyze
            
        Returns:
            DataFrame with cluster statistics
        """
        if algorithm not in self.cluster_labels:
            raise ValueError(f"Algorithm {algorithm} not fitted yet")
        
        labels = self.cluster_labels[algorithm]
        cluster_stats = []
        
        unique_clusters = np.unique(labels)
        
        for cluster_id in unique_clusters:
            if cluster_id == -1:  # Skip noise cluster in HDBSCAN
                continue
            
            cluster_mask = labels == cluster_id
            cluster_features = self.features[cluster_mask]
            cluster_symbols = self.features.index[cluster_mask].tolist()
            
            stats = {
                'cluster_id': cluster_id,
                'n_stocks': len(cluster_symbols),
                'stocks': cluster_symbols,
                'mean_return': cluster_features['returns_mean'].mean() if 'returns_mean' in cluster_features.columns else 0,
                'mean_volatility': cluster_features['returns_std'].mean() if 'returns_std' in cluster_features.columns else 0,
                'mean_sharpe': cluster_features['sharpe_ratio'].mean() if 'sharpe_ratio' in cluster_features.columns else 0,
                'mean_market_corr': cluster_features['market_correlation'].mean() if 'market_correlation' in cluster_features.columns else 0
            }
            
            cluster_stats.append(stats)
        
        return pd.DataFrame(cluster_stats)
    
    def visualize_clusters(self, 
                          algorithm: str = 'kmeans',
                          save_plots: bool = True,
                          plot_type: str = 'pca') -> None:
        """
        Visualize clustering results
        
        Args:
            algorithm: Clustering algorithm to visualize
            save_plots: Whether to save plots to file
            plot_type: Type of visualization ('pca', 'features')
        """
        if algorithm not in self.cluster_labels:
            raise ValueError(f"Algorithm {algorithm} not fitted yet")
        
        labels = self.cluster_labels[algorithm]
        
        if plot_type == 'pca':
            self._plot_pca_clusters(labels, algorithm, save_plots)
        elif plot_type == 'features':
            self._plot_feature_clusters(labels, algorithm, save_plots)
    
    def _plot_pca_clusters(self, labels: np.ndarray, algorithm: str, save_plots: bool):
        """Plot clusters in PCA space"""
        # Apply PCA for visualization
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(self.scaled_features)
        
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                            c=labels, cmap='tab10', alpha=0.7, s=50)
        plt.colorbar(scatter, label='Cluster')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.title(f'Stock Clusters - {algorithm.title()} (PCA Visualization)')
        plt.grid(True, alpha=0.3)
        
        # Add stock labels for some points
        for i, symbol in enumerate(self.features.index):
            if i % 5 == 0:  # Label every 5th stock to avoid overcrowding
                plt.annotate(symbol, (features_2d[i, 0], features_2d[i, 1]), 
                           xytext=(5, 5), textcoords='offset points', 
                           fontsize=8, alpha=0.7)
        
        plt.tight_layout()
        
        if save_plots:
            os.makedirs(self.config.RESULTS_DIR, exist_ok=True)
            plt.savefig(os.path.join(self.config.RESULTS_DIR, f'clusters_pca_{algorithm}.png'), 
                       dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def _plot_feature_clusters(self, labels: np.ndarray, algorithm: str, save_plots: bool):
        """Plot cluster characteristics using key features"""
        n_clusters = len(np.unique(labels[labels != -1]))
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Feature pairs to plot
        feature_pairs = [
            ('returns_mean', 'returns_std'),
            ('sharpe_ratio', 'max_drawdown'),
            ('market_correlation', 'beta'),
            ('price_momentum_1m', 'price_volatility')
        ]
        
        for idx, (feat1, feat2) in enumerate(feature_pairs):
            row, col = idx // 2, idx % 2
            ax = axes[row, col]
            
            if feat1 in self.features.columns and feat2 in self.features.columns:
                scatter = ax.scatter(self.features[feat1], self.features[feat2], 
                                   c=labels, cmap='tab10', alpha=0.7, s=50)
                ax.set_xlabel(feat1.replace('_', ' ').title())
                ax.set_ylabel(feat2.replace('_', ' ').title())
                ax.grid(True, alpha=0.3)
                
                if idx == 0:  # Add colorbar only once
                    plt.colorbar(scatter, ax=ax, label='Cluster')
        
        plt.suptitle(f'Stock Clusters - {algorithm.title()} (Feature Space)', fontsize=16)
        plt.tight_layout()
        
        if save_plots:
            os.makedirs(self.config.RESULTS_DIR, exist_ok=True)
            plt.savefig(os.path.join(self.config.RESULTS_DIR, f'clusters_features_{algorithm}.png'), 
                       dpi=300, bbox_inches='tight')
        
        plt.show()

def main():
    """Main function to run stock clustering analysis"""
    logger.info("Starting stock clustering analysis")
    
    # Load processed data
    try:
        # Use original processed data file
        processed_data_path = os.path.join(Config.PROCESSED_DATA_DIR, 'processed_stock_data.csv')
        if not os.path.exists(processed_data_path):
            logger.error("Processed data not found. Please run data collection first.")
            return
        
        logger.info(f"Loading data from: {processed_data_path}")
        
        # Read the original CSV data
        df = pd.read_csv(processed_data_path)
        
        # Extract ticker symbols from column names (skip 'Ticker' column)
        ticker_symbols = set()
        for col in df.columns[1:]:  # Skip first column 'Ticker'
            # Extract base symbol (remove .1, .2, etc. suffixes)
            if '.' in col:
                base_symbol = col.split('.')[0]
            else:
                base_symbol = col
            ticker_symbols.add(base_symbol)
        
        ticker_symbols = sorted(list(ticker_symbols))
        logger.info(f"Found {len(ticker_symbols)} unique symbols: {ticker_symbols}")
        
        # Create date index (skip the header rows that contain price type info)
        # First find where actual data starts (should be rows with valid dates)
        date_col = df.iloc[:, 0]
        
        # Find the first row that contains a valid date
        first_valid_idx = None
        for i in range(len(date_col)):
            try:
                parsed_date = pd.to_datetime(date_col.iloc[i], errors='raise')
                if pd.notna(parsed_date):
                    first_valid_idx = i
                    break
            except:
                continue
        
        if first_valid_idx is None:
            logger.error("Could not find any valid dates in the data")
            return
            
        dates = pd.to_datetime(df.iloc[first_valid_idx:, 0])
        logger.info(f"Data starts at row {first_valid_idx}, found {len(dates)} dates")
        
        # Get price types from first row data (this tells us what each column represents)
        price_types = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        
        # Extract price data for each symbol
        multi_columns = []
        valid_columns = []
        
        for symbol in ticker_symbols:
            # Find all columns for this symbol
            symbol_cols = []
            symbol_col_indices = []
            
            for i, col in enumerate(df.columns[1:], 1):  # Start from 1 to skip 'Ticker'
                if col == symbol or col.startswith(f"{symbol}."):
                    symbol_cols.append(col)
                    symbol_col_indices.append(i)
            
            if len(symbol_cols) > 0:
                # Get the data for these columns (from first valid data row onwards)
                symbol_data = df.iloc[first_valid_idx:, symbol_col_indices]
                
                # Convert to numeric
                for col_idx, col_name in enumerate(symbol_cols):
                    col_data = pd.to_numeric(symbol_data.iloc[:, col_idx], errors='coerce')
                    
                    # Only include columns with sufficient valid data
                    if col_data.notna().sum() > 10:
                        # Assign appropriate price type
                        if col_idx < len(price_types):
                            price_type = price_types[col_idx]
                        else:
                            price_type = f"Price_{col_idx}"
                        
                        multi_columns.append((symbol, price_type))
                        valid_columns.append(col_data.values)
                        logger.debug(f"Added column for {symbol} - {price_type} with {col_data.notna().sum()} valid values")
        
        # Create the final MultiIndex DataFrame
        if valid_columns:
            multi_index = pd.MultiIndex.from_tuples(multi_columns, names=['Ticker', 'Price'])
            price_data = pd.DataFrame(
                np.column_stack(valid_columns),
                index=dates.values,
                columns=multi_index
            )
            
            # Convert to float
            price_data = price_data.astype(float)
            
            logger.info(f"Created price data with shape: {price_data.shape}")
            logger.info(f"Successfully loaded {len(price_data.columns.get_level_values(0).unique())} unique symbols")
        else:
            logger.error("No valid price data found")
            return
        
        # Calculate returns directly from price data
        symbols = price_data.columns.get_level_values(0).unique()
        returns_dict = {}
        
        logger.info(f"Calculating returns for {len(symbols)} symbols")
        logger.info(f"Available symbols: {symbols.tolist()}")
        logger.info(f"Sample columns: {price_data.columns[:10].tolist()}")
        
        for symbol in symbols:
            try:
                logger.debug(f"Processing returns for symbol: {symbol}")
                
                # Check what columns exist for this symbol
                symbol_columns = [col for col in price_data.columns if col[0] == symbol]
                logger.debug(f"Columns for {symbol}: {symbol_columns}")
                
                if (symbol, 'Close') in price_data.columns:
                    close_prices = price_data[symbol]['Close'].dropna()
                    
                    if len(close_prices) < 10:  # Need minimum data
                        logger.warning(f"Insufficient price data for {symbol}: {len(close_prices)} values")
                        continue
                    
                    # Calculate various return types
                    returns_1d = close_prices.pct_change()
                    log_returns_1d = np.log(close_prices / close_prices.shift(1))
                    returns_5d = close_prices.pct_change(periods=5)
                    returns_20d = close_prices.pct_change(periods=20)
                    
                    returns_dict[symbol] = pd.DataFrame({
                        'returns_1d': returns_1d,
                        'log_returns_1d': log_returns_1d,
                        'returns_5d': returns_5d,
                        'returns_20d': returns_20d
                    })
                    
                    logger.info(f"Calculated returns for {symbol}")
                    
                else:
                    logger.warning(f"No Close price column found for {symbol}")
                    
            except Exception as e:
                logger.warning(f"Error calculating returns for {symbol}: {e}")
                continue
        
        logger.info(f"Successfully calculated returns for {len(returns_dict)} symbols")
        
        # Combine returns into MultiIndex DataFrame
        if returns_dict:
            returns_data = pd.concat(returns_dict, axis=1)
        else:
            logger.warning("No returns data calculated, creating empty DataFrame")
            returns_data = pd.DataFrame()
        
        # Load benchmark data
        benchmark_path = os.path.join(Config.RAW_DATA_DIR, f'{Config.BENCHMARK_SYMBOL}_benchmark.csv')
        market_data = None
        if os.path.exists(benchmark_path):
            try:
                benchmark_df = pd.read_csv(benchmark_path)
                
                # Find where actual data starts in benchmark file (same structure as main file)
                benchmark_date_col = benchmark_df.iloc[:, 0]
                benchmark_first_valid_idx = None
                
                for i in range(len(benchmark_date_col)):
                    try:
                        parsed_date = pd.to_datetime(benchmark_date_col.iloc[i], errors='raise')
                        if pd.notna(parsed_date):
                            benchmark_first_valid_idx = i
                            break
                    except:
                        continue
                
                if benchmark_first_valid_idx is not None:
                    # Extract actual data rows
                    benchmark_dates = pd.to_datetime(benchmark_df.iloc[benchmark_first_valid_idx:, 0])
                    benchmark_prices = benchmark_df.iloc[benchmark_first_valid_idx:, 1:]
                    
                    # Convert to numeric
                    for col in benchmark_prices.columns:
                        benchmark_prices[col] = pd.to_numeric(benchmark_prices[col], errors='coerce')
                    
                    market_data = benchmark_prices.copy()
                    market_data.index = benchmark_dates.values
                    
                    # Rename columns to standard names if needed
                    if 'Close' not in market_data.columns and len(market_data.columns) > 0:
                        market_data['Close'] = market_data.iloc[:, 0]  # Use first column as Close
                    
                    logger.info(f"Loaded benchmark data with shape: {market_data.shape}")
                else:
                    logger.warning("Could not find valid dates in benchmark data")
                    
            except Exception as e:
                logger.warning(f"Error loading benchmark data: {e}")
                market_data = None
        
        logger.info("Data loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return
    
    # Create features
    feature_engineer = FeatureEngineer()
    features = feature_engineer.create_clustering_features(price_data, returns_data, market_data)
    
    # Perform clustering
    clusterer = StockClusterer()
    results = clusterer.fit_clustering_models(features)
    
    # Analyze and visualize results
    for algorithm in results.keys():
        logger.info(f"\nAnalysis for {algorithm}:")
        
        # Get cluster assignments
        assignments = clusterer.get_cluster_assignments(algorithm)
        
        # Analyze clusters
        cluster_analysis = clusterer.analyze_clusters(algorithm)
        print(cluster_analysis)
        
        # Visualize clusters
        clusterer.visualize_clusters(algorithm, save_plots=True, plot_type='pca')
        
        # Save results
        results_path = os.path.join(Config.RESULTS_DIR, f'cluster_assignments_{algorithm}.csv')
        assignments.to_csv(results_path, index=False)
        
        analysis_path = os.path.join(Config.RESULTS_DIR, f'cluster_analysis_{algorithm}.csv')
        cluster_analysis.to_csv(analysis_path, index=False)
    
    # Save features for future use
    features_path = os.path.join(Config.FEATURES_DIR, 'clustering_features.csv')
    features.to_csv(features_path)
    
    logger.info("Stock clustering analysis completed successfully!")

if __name__ == "__main__":
    main()