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
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from loguru import logger
import os
import warnings
warnings.filterwarnings('ignore')

from config import Config
from tslearn.clustering import TimeSeriesKMeans, silhouette_score as ts_silhouette_score
from tslearn.utils import to_time_series_dataset
import yfinance as yf
from src.data.loader import DataLoader

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
    """Creates features for clustering"""
    
    def __init__(self, market: str):
        self.config = Config()
        self.market = market
        self.results_dir = self.config.get_market_results_dir(market)
        os.makedirs(self.results_dir, exist_ok=True)

    def create_clustering_features(self, 
                                 price_data: pd.DataFrame, 
                                 returns_data: pd.DataFrame, 
                                 benchmark_data: pd.DataFrame) -> pd.DataFrame:
        """
        Create features for clustering analysis.
        This version is more robust against missing data from yfinance.
        """
        logger.info("Creating clustering features")
        
        self.price_data = price_data
        self.returns_data = returns_data.pct_change().dropna()
        self.benchmark_returns = benchmark_data['Close'].pct_change().dropna()
        
        symbols = self.price_data.columns.get_level_values(0).unique()
        features = []
        
        for symbol in tqdm(symbols, desc="Creating clustering features"):
            try:
                feature_dict = {}
                # Basic return and volume stats (usually reliable)
                symbol_returns = self.returns_data[(symbol, 'returns_1d')].dropna()
                feature_dict = {
                    'returns_mean': symbol_returns.mean(),
                    'returns_std': symbol_returns.std(),
                    'sharpe_ratio': (symbol_returns.mean() / symbol_returns.std()) * np.sqrt(252) if symbol_returns.std() != 0 else 0,
                    'volume_mean': self.price_data[(symbol, 'Volume')].mean(),
                }

                # Cumulative returns for max drawdown calculation
                cumulative_returns = (1 + symbol_returns).cumprod()
                peak = cumulative_returns.expanding(min_periods=1).max()
                drawdown = (cumulative_returns - peak) / peak
                feature_dict['max_drawdown'] = drawdown.min()

                # Correlation with benchmark
                if self.benchmark_returns is not None:
                    # Align indices before calculating correlation
                    aligned_returns, aligned_benchmark = symbol_returns.align(self.benchmark_returns, join='inner')
                    if not aligned_returns.empty:
                        feature_dict['correlation_benchmark'] = aligned_returns.corr(aligned_benchmark)
                    else:
                        feature_dict['correlation_benchmark'] = 0.0
                else:
                    feature_dict['correlation_benchmark'] = 0.0
                
                # Advanced metrics from yfinance (handle missing data gracefully)
                ticker = yf.Ticker(symbol)
                info = ticker.info

                feature_dict['market_cap'] = info.get('marketCap')
                feature_dict['pe_ratio'] = info.get('trailingPE') or info.get('forwardPE')
                feature_dict['beta'] = info.get('beta')

                # Add a fallback for volatility ratio if beta is not available
                if feature_dict['beta'] is not None and self.benchmark_returns is not None:
                     feature_dict['volatility_ratio'] = symbol_returns.std() / self.benchmark_returns.std()
                else:
                     feature_dict['volatility_ratio'] = 1.0


                features.append({'symbol': symbol, **feature_dict})
            except Exception as e:
                logger.warning(f"Could not create all features for {symbol}: {e}")
                # Append with whatever was successfully calculated
                if 'symbol' not in feature_dict:
                    feature_dict['symbol'] = symbol
                features.append(feature_dict)
                continue
        
        if not features:
            logger.error("Feature creation failed for all stocks.")
            return pd.DataFrame()

        features_df = pd.DataFrame(features).set_index('symbol')
        
        # Select only the features that were successfully created for most stocks
        # and are in the config, then fill NaNs
        available_cols = list(set(features_df.columns) & set(self.config.CLUSTERING_FEATURES))
        if not available_cols:
            logger.error("None of the desired clustering features could be created.")
            return pd.DataFrame()

        features_df = features_df[available_cols]
        features_df.fillna(features_df.median(), inplace=True) # Fill with median for robustness

        logger.info(f"Created features for {len(features_df)} stocks with {len(features_df.columns)} features")
        
        # Save features for inspection
        features_path = os.path.join(self.results_dir, 'clustering_features.csv')
        features_df.to_csv(features_path)
        
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

    def create_dtw_features(self, returns_data: pd.DataFrame) -> np.ndarray:
        """
        Prepare time series data for DTW clustering.
        
        Args:
            returns_data: DataFrame of daily returns for all stocks.
            
        Returns:
            A numpy array of shape (n_stocks, n_timesteps, 1)
        """
        logger.info("Preparing data for DTW clustering")
        
        # Pivot returns data to have symbols as rows and dates as columns
        returns_pivot = returns_data.unstack(level=0)['returns_1d']
        
        # Fill NaNs - forward fill is a reasonable approach for time series
        returns_pivot = returns_pivot.fillna(method='ffill').fillna(0)
        
        # Convert to tslearn format
        time_series_dataset = to_time_series_dataset(returns_pivot.values)
        
        logger.info(f"Created DTW dataset with shape: {time_series_dataset.shape}")
        return time_series_dataset, returns_pivot.index.tolist()


class StockClusterer:
    """Main class for stock clustering"""
    
    def __init__(self):
        self.config = Config()
        self.scaler = StandardScaler()
        self.cluster_models = {}
        self.cluster_labels = {}
        self.features = None
        self.scaled_features = None
        self.dtw_features = None
        self.dtw_symbols = None

    def set_dtw_features(self, dtw_features: np.ndarray, dtw_symbols: List[str]):
        """Set the pre-computed DTW features and corresponding symbols."""
        self.dtw_features = dtw_features
        self.dtw_symbols = dtw_symbols

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
        
        # --- Standard Feature-Based Clustering ---
        for algorithm, params in self.config.CLUSTERING_ALGORITHMS.items():
            if algorithm == 'dtw': # Skip DTW in this loop
                continue
            try:
                logger.info(f"Fitting {algorithm} clustering")
                
                if algorithm == 'kmeans':
                    model = KMeans(**params, random_state=42)
                    labels = model.fit_predict(self.scaled_features)
                    
                elif algorithm == 'dbscan':
                    model = DBSCAN(**params)
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
                n_clusters = len(np.unique(labels[labels != -1]))  # Exclude noise for DBSCAN/HDBSCAN
                
                if n_clusters > 1:
                    silhouette = silhouette_score(self.scaled_features, labels)
                    calinski_harabasz = calinski_harabasz_score(self.scaled_features, labels)
                else:
                    silhouette = -1
                    calinski_harabasz = 0
                
                results[algorithm] = {
                    'model': model,
                    'labels': labels,
                    'n_clusters': n_clusters,
                    'silhouette_score': silhouette,
                    'calinski_harabasz_score': calinski_harabasz,
                    'n_noise_points': np.sum(labels == -1)
                }
                
                logger.info(f"{algorithm}: {n_clusters} clusters, silhouette={silhouette:.3f}")
                
            except Exception as e:
                logger.error(f"Error fitting {algorithm}: {e}")
                continue

        # --- DTW Time Series Clustering ---
        if 'dtw' in self.config.CLUSTERING_ALGORITHMS and self.dtw_features is not None:
            algorithm = 'dtw'
            params = self.config.CLUSTERING_ALGORITHMS[algorithm]
            try:
                logger.info(f"Fitting {algorithm} clustering using DTW")
                
                # DTW needs a different clustering model from tslearn
                model = TimeSeriesKMeans(
                    n_clusters=params.get('n_clusters', 5),
                    metric="dtw",
                    verbose=False,
                    random_state=42,
                    n_jobs=-1
                )
                labels = model.fit_predict(self.dtw_features)
                
                self.cluster_models[algorithm] = model
                self.cluster_labels[algorithm] = labels
                
                n_clusters = len(np.unique(labels))
                
                # Silhouette score for time series is calculated differently
                silhouette = ts_silhouette_score(self.dtw_features, labels, metric="dtw") if n_clusters > 1 else -1
                
                results[algorithm] = {
                    'model': model,
                    'labels': labels,
                    'n_clusters': n_clusters,
                    'silhouette_score': silhouette,
                    'calinski_harabasz_score': -1, # Not applicable for DTW in this context
                    'n_noise_points': 0
                }
                logger.info(f"{algorithm}: {n_clusters} clusters, silhouette={silhouette:.3f}")

            except Exception as e:
                logger.error(f"Error fitting {algorithm}: {e}")

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
        
        # DTW uses a different set of symbols/ordering
        if algorithm == 'dtw':
            if self.dtw_symbols is None:
                raise ValueError("DTW symbols not set. Run feature engineering first.")
            symbols = self.dtw_symbols
        else:
            symbols = self.features.index

        assignments = pd.DataFrame({
            'symbol': symbols,
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
        
        # Determine which feature set and symbols to use
        if algorithm == 'dtw':
            if self.dtw_symbols is None:
                raise ValueError("DTW symbols not set.")
            # For DTW, analysis is based on the original features, not the time series itself
            feature_df = self.features.loc[self.dtw_symbols]
            symbols = self.dtw_symbols
        else:
            feature_df = self.features
            symbols = self.features.index

        cluster_stats = []
        
        unique_clusters = np.unique(labels)
        
        for cluster_id in unique_clusters:
            if cluster_id == -1:  # Skip noise cluster
                continue
            
            cluster_mask = labels == cluster_id
            cluster_features = feature_df[cluster_mask]
            cluster_symbols = np.array(symbols)[cluster_mask].tolist()
            
            stats = {
                'cluster_id': cluster_id,
                'n_stocks': len(cluster_symbols),
                'stocks': cluster_symbols,
                'mean_return': cluster_features['returns_mean'].mean(),
                'mean_volatility': cluster_features['returns_std'].mean(),
                'mean_sharpe': cluster_features['sharpe_ratio'].mean(),
                'mean_market_corr': cluster_features.get('market_correlation', pd.Series(0)).mean()
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
        
        # DTW visualization is different
        if algorithm == 'dtw':
            self._plot_dtw_clusters(labels, algorithm, save_plots)
        elif plot_type == 'pca':
            self._plot_pca_clusters(labels, algorithm, save_plots)
        elif plot_type == 'features':
            self._plot_feature_clusters(labels, algorithm, save_plots)

    def _plot_dtw_clusters(self, labels: np.ndarray, algorithm: str, save_plots: bool):
        """Plot the centroids of DTW clusters."""
        logger.info(f"Visualizing DTW cluster centroids for '{algorithm}'")
        
        model = self.cluster_models.get(algorithm)
        if not hasattr(model, 'cluster_centers_'):
            logger.warning("DTW model does not have cluster centers to plot.")
            return

        plt.figure(figsize=(12, 8))
        for i, center in enumerate(model.cluster_centers_):
            plt.plot(center.ravel(), label=f"Cluster {i}")
        
        plt.title(f"DTW Cluster Centroids - {algorithm.title()}")
        plt.xlabel("Time Step")
        plt.ylabel("Normalized Return")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_plots:
            os.makedirs(self.config.RESULTS_DIR, exist_ok=True)
            plt.savefig(os.path.join(self.config.RESULTS_DIR, f'clusters_dtw_{algorithm}.png'), 
                       dpi=300, bbox_inches='tight')
        plt.show()

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

def main(market: str = 'US'):
    """Main function for clustering analysis"""
    logger.info(f"Starting stock clustering analysis for {market} market")
    
    config = Config()
    
    # Load data
    data_loader = DataLoader(market=market)
    price_data, returns_data, benchmark_data = data_loader.load_all_data()
    
    if price_data.empty:
        logger.error("Price data is empty, cannot proceed with clustering.")
        return
        
    # Create features
    feature_engineer = FeatureEngineer(market=market)
    features = feature_engineer.create_clustering_features(price_data, returns_data, benchmark_data)
    
    if features.empty:
        logger.error("Feature creation failed. Aborting clustering.")
        return
        
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