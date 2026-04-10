"""
Machine Learning and Deep Learning models for stock return prediction
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import pickle
import os
from datetime import datetime
import warnings
import ta
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

# Deep Learning Libraries
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, callbacks
    HAS_TENSORFLOW = True
except ImportError:
    print("Warning: TensorFlow not available, deep learning models disabled")
    HAS_TENSORFLOW = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_PYTORCH = True
except ImportError:
    print("Warning: PyTorch not available, PyTorch models disabled")
    HAS_PYTORCH = False

from loguru import logger
from tqdm import tqdm

from config import Config

class FeatureGenerator:
    """Generate features for return prediction models"""
    
    def __init__(self):
        self.config = Config()
        self.scalers = {}
    
    def create_ml_features(self, 
                          price_data: pd.DataFrame,
                          returns_data: pd.DataFrame,
                          cluster_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Create features for traditional ML models
        
        Args:
            price_data: Stock price data
            returns_data: Stock returns data
            cluster_data: Stock cluster assignments (optional)
            
        Returns:
            DataFrame with ML features
        """
        logger.info("Creating ML features for return prediction")
        
        symbols = returns_data.columns.unique()
        all_features = []
        
        for symbol in tqdm(symbols, desc="Creating features"):
            try:
                # Ensure we are working with a single-level column index for the symbol
                if isinstance(price_data.columns, pd.MultiIndex):
                    symbol_prices = price_data[symbol]
                else:
                    # This case might occur if only one symbol is passed
                    symbol_prices = price_data

                if isinstance(returns_data.columns, pd.MultiIndex):
                    symbol_returns = returns_data[symbol]
                else:
                    symbol_returns = returns_data

                # Create features dataframe
                features_list = []
                
                # Ensure returns are numeric before calculations
                numeric_returns = pd.to_numeric(symbol_returns, errors='coerce').fillna(0)

                # Start from a point where all indicators and lookbacks are valid
                start_index = max(self.config.LOOKBACK_PERIODS) + 40 # Increased buffer for complex indicators

                for i in range(start_index, len(numeric_returns) - max(self.config.PREDICTION_HORIZONS)):
                    try:
                        feature_dict = {'symbol': symbol, 'date': numeric_returns.index[i]}
                        
                        # 1. Lagged Log Returns
                        log_returns = np.log(symbol_prices['Close'] / symbol_prices['Close'].shift(1))
                        for lag in range(1, 5):
                            feature_dict[f'log_return_lag_{lag}'] = log_returns.iloc[i-lag]

                        # 2. Paper's Technical Indicators
                        # SMA is already calculated by add_all_ta_features as 'trend_sma_fast' etc.
                        feature_dict['sma_20'] = temp_df['trend_sma_fast'].iloc[i] # Example, assuming 12/26 for MACD
                        
                        # MACD
                        feature_dict['macd'] = temp_df['trend_macd'].iloc[i]
                        feature_dict['macd_signal'] = temp_df['trend_macd_signal'].iloc[i]
                        
                        # PPO
                        feature_dict['ppo'] = temp_df['trend_ppo'].iloc[i]
                        
                        # ATR
                        feature_dict['atr'] = temp_df['volatility_atr'].iloc[i]
                        
                        # RSI
                        feature_dict['rsi'] = temp_df['momentum_rsi'].iloc[i]
                        
                        # Stochastic Oscillator
                        feature_dict['stoch_k'] = temp_df['momentum_stoch'].iloc[i]
                        feature_dict['stoch_d'] = temp_df['momentum_stoch_signal'].iloc[i]

                        # Keep some of the original valuable features
                        # Volatility
                        feature_dict['realized_vol_20d'] = log_returns.iloc[i-19:i+1].std() * np.sqrt(252)
                        
                        # Volume features
                        volumes = symbol_prices['Volume']
                        feature_dict['volume_ma_20d'] = volumes.iloc[i-19:i+1].mean()
                        feature_dict['volume_ratio'] = volumes.iloc[i] / volumes.iloc[i-19:i+1].mean() if volumes.iloc[i-19:i+1].mean() != 0 else 1
                        
                        # Target variables (future returns)
                        for horizon in self.config.PREDICTION_HORIZONS:
                            if i + horizon < len(numeric_returns):
                                if horizon == 1:
                                    feature_dict[f'target_{horizon}d'] = numeric_returns.iloc[i + horizon]
                                else:
                                    # Multi-period return
                                    feature_dict[f'target_{horizon}d'] = (
                                        numeric_returns.iloc[i+1:i+horizon+1].add(1).prod() - 1
                                    )
                        
                        features_list.append(feature_dict)
                        
                    except Exception as e:
                        # logger.debug(f"Skipping feature creation at index {i} for {symbol} due to: {e}")
                        continue
                
                if not features_list:
                    logger.warning(f"No features could be generated for {symbol} in the given date range.")
                    return pd.DataFrame()
                    
                return pd.DataFrame(features_list)
            
            except KeyError:
                logger.warning(f"Data not found for symbol {symbol}. Skipping feature creation.")
                continue
            except Exception as e:
                logger.warning(f"Error creating features for {symbol}: {e}")
                continue
        
        # Combine all features
        if not all_features:
            logger.warning("No features were created. Aborting.")
            return pd.DataFrame()
            
        features_df = pd.concat(all_features, ignore_index=True)
        
        # Merge with cluster data
        if cluster_data is not None:
            if features_df.empty:
                logger.error("Generated features dataframe is completely empty. Aborting merge.")
                return features_df
            # Ensure cluster_data has the right columns and no duplicates
            if 'symbol' in cluster_data.columns and 'cluster' in cluster_data.columns:
                cluster_info = cluster_data[['symbol', 'cluster']].drop_duplicates()
                features_df = pd.merge(features_df, cluster_info, on='symbol', how='left')
                logger.info(f"Merged features with cluster data. {features_df['cluster'].notna().sum()} of {len(features_df)} samples have cluster IDs.")
            else:
                logger.warning("`cluster_data` is missing 'symbol' or 'cluster' columns.")

        features_df = features_df.dropna(subset=[col for col in features_df.columns if col.startswith('target')])
        
        logger.info(f"Created {len(features_df)} feature samples with {features_df.shape[1]-len(self.config.PREDICTION_HORIZONS)-2} features") # -2 for symbol, date
        return features_df
    
    def _create_symbol_features(self,
                               symbol: str,
                               symbol_prices: pd.DataFrame,
                               symbol_returns: pd.DataFrame,
                               cluster_data: Optional[pd.DataFrame]) -> pd.DataFrame:
        """Create features for a single symbol"""
        
        # Create a temporary dataframe for easier feature calculation
        temp_df = pd.DataFrame({
            'high': symbol_prices['High'],
            'low': symbol_prices['Low'],
            'close': symbol_prices['Close'],
            'volume': symbol_prices['Volume']
        })

        # Add all technical indicators from the 'ta' library
        ta.add_all_ta_features(
            temp_df, 
            open="close", # Using close as open as we don't have open in some cases
            high="high", 
            low="low", 
            close="close", 
            volume="volume", 
            fillna=True
        )
        
        # Create features dataframe
        features_list = []
        
        # Ensure returns are numeric before calculations
        numeric_returns = pd.to_numeric(symbol_returns, errors='coerce').fillna(0)

        # Start from a point where all indicators and lookbacks are valid
        start_index = max(self.config.LOOKBACK_PERIODS) + 40 # Increased buffer for complex indicators

        for i in range(start_index, len(numeric_returns) - max(self.config.PREDICTION_HORIZONS)):
            try:
                feature_dict = {'symbol': symbol, 'date': numeric_returns.index[i]}
                
                # 1. Lagged Log Returns
                log_returns = np.log(symbol_prices['Close'] / symbol_prices['Close'].shift(1))
                for lag in range(1, 5):
                    feature_dict[f'log_return_lag_{lag}'] = log_returns.iloc[i-lag]

                # 2. Paper's Technical Indicators
                # SMA is already calculated by add_all_ta_features as 'trend_sma_fast' etc.
                feature_dict['sma_20'] = temp_df['trend_sma_fast'].iloc[i] # Example, assuming 12/26 for MACD
                
                # MACD
                feature_dict['macd'] = temp_df['trend_macd'].iloc[i]
                feature_dict['macd_signal'] = temp_df['trend_macd_signal'].iloc[i]
                
                # PPO
                feature_dict['ppo'] = temp_df['trend_ppo'].iloc[i]
                
                # ATR
                feature_dict['atr'] = temp_df['volatility_atr'].iloc[i]
                
                # RSI
                feature_dict['rsi'] = temp_df['momentum_rsi'].iloc[i]
                
                # Stochastic Oscillator
                feature_dict['stoch_k'] = temp_df['momentum_stoch'].iloc[i]
                feature_dict['stoch_d'] = temp_df['momentum_stoch_signal'].iloc[i]

                # Keep some of the original valuable features
                # Volatility
                feature_dict['realized_vol_20d'] = log_returns.iloc[i-19:i+1].std() * np.sqrt(252)
                
                # Volume features
                volumes = symbol_prices['Volume']
                feature_dict['volume_ma_20d'] = volumes.iloc[i-19:i+1].mean()
                feature_dict['volume_ratio'] = volumes.iloc[i] / volumes.iloc[i-19:i+1].mean() if volumes.iloc[i-19:i+1].mean() != 0 else 1
                
                # Target variables (future returns)
                for horizon in self.config.PREDICTION_HORIZONS:
                    if i + horizon < len(numeric_returns):
                        if horizon == 1:
                            feature_dict[f'target_{horizon}d'] = numeric_returns.iloc[i + horizon]
                        else:
                            # Multi-period return
                            feature_dict[f'target_{horizon}d'] = (
                                numeric_returns.iloc[i+1:i+horizon+1].add(1).prod() - 1
                            )
                
                features_list.append(feature_dict)
                
            except Exception as e:
                # logger.debug(f"Skipping feature creation at index {i} for {symbol} due to: {e}")
                continue
        
        if not features_list:
            logger.warning(f"No features could be generated for {symbol} in the given date range.")
            return pd.DataFrame()
            
        return pd.DataFrame(features_list)
    
    def create_lstm_sequences(self, 
                            price_data: pd.DataFrame,
                            sequence_length: int = 60) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Create sequences for LSTM training
        
        Args:
            price_data: Stock price data, now required for feature generation
            sequence_length: Length of input sequences
            
        Returns:
            Tuple of (X, y, symbols) where X is input sequences, y is targets
        """
        logger.info(f"Creating LSTM sequences with length {sequence_length}")
        
        symbols = price_data.columns.get_level_values(0).unique()
        X_list, y_list, symbol_list = [], [], []
        
        for symbol in tqdm(symbols, desc="Creating sequences"):
            try:
                symbol_prices = price_data[symbol]
                
                if len(symbol_prices) < sequence_length + 40: # Buffer for indicators
                    continue

                # Create features for the entire series
                temp_df = pd.DataFrame({
                    'high': symbol_prices['High'],
                    'low': symbol_prices['Low'],
                    'close': symbol_prices['Close'],
                    'volume': symbol_prices['Volume']
                })
                ta.add_all_ta_features(temp_df, open="close", high="high", low="low", close="close", volume="volume", fillna=True)
                log_returns = np.log(temp_df['close'] / temp_df['close'].shift(1)).fillna(0)

                # Combine features into a single dataframe
                features = pd.concat([
                    log_returns.rename('log_return'),
                    temp_df[[
                        'trend_macd', 'trend_ppo', 'volatility_atr', 
                        'momentum_rsi', 'momentum_stoch'
                    ]]
                ], axis=1).fillna(0)

                # Scale features
                scaler = StandardScaler()
                scaled_features = scaler.fit_transform(features)

                # Create sequences from scaled features
                for i in range(sequence_length, len(scaled_features) - 1):
                    X_seq = scaled_features[i-sequence_length:i, :]
                    # Target is the next day's log return
                    y_target = log_returns.iloc[i+1]
                    
                    X_list.append(X_seq)
                    y_list.append(y_target)
                    symbol_list.append(symbol)
                
            except Exception as e:
                logger.warning(f"Error creating sequences for {symbol}: {e}")
                continue
        
        if not X_list:
            logger.warning("No sequences were created. Check data and parameters.")
            return np.array([]), np.array([]), []

        X = np.array(X_list)
        y = np.array(y_list)
        
        logger.info(f"Created {len(X)} sequences from {len(set(symbol_list))} symbols with {X.shape[2]} features each")
        return X, y, symbol_list

class MLModels:
    """Traditional machine learning models for return prediction"""
    
    def __init__(self):
        self.config = Config()
        self.models = {}
        self.scalers = {}
        self.feature_columns = None
    
    def train_models(self, 
                    features_df: pd.DataFrame,
                    target_column: str = 'target_1d',
                    test_size: float = 0.2) -> Dict[str, Any]:
        """
        Train multiple ML models for return prediction
        
        Args:
            features_df: DataFrame with features and targets
            target_column: Name of target column
            test_size: Fraction of data for testing
            
        Returns:
            Dictionary with model results and metrics
        """
        logger.info(f"Training ML models to predict {target_column}")
        
        # Prepare data
        feature_cols = [col for col in features_df.columns 
                       if col not in ['symbol', 'date'] and not col.startswith('target_')]
        
        X = features_df[feature_cols].fillna(0)
        y = features_df[target_column].fillna(0)
        
        # Time-based split (important for financial data)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Scale features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers[target_column] = scaler
        self.feature_columns = feature_cols
        
        results = {}
        
        # Train each model
        for model_name, model_params in self.config.ML_MODELS.items():
            try:
                logger.info(f"Training {model_name}")
                
                # Initialize model
                if model_name == 'linear_regression':
                    model = LinearRegression(**model_params)
                elif model_name == 'ridge':
                    model = Ridge(**model_params)
                elif model_name == 'lasso':
                    model = Lasso(**model_params)
                elif model_name == 'random_forest':
                    model = RandomForestRegressor(**model_params, random_state=42)
                elif model_name == 'xgboost':
                    model = xgb.XGBRegressor(**model_params, random_state=42)
                elif model_name == 'svm':
                    model = SVR(**model_params)
                
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Make predictions
                y_pred_train = model.predict(X_train_scaled)
                y_pred_test = model.predict(X_test_scaled)
                
                # Calculate metrics
                train_metrics = self._calculate_metrics(y_train, y_pred_train)
                test_metrics = self._calculate_metrics(y_test, y_pred_test)
                
                # Store results
                self.models[f"{model_name}_{target_column}"] = model
                
                results[model_name] = {
                    'model': model,
                    'train_metrics': train_metrics,
                    'test_metrics': test_metrics,
                    'feature_importance': self._get_feature_importance(model, feature_cols)
                }
                
                logger.info(f"{model_name} - Test R²: {test_metrics['r2']:.4f}, "
                           f"Test RMSE: {test_metrics['rmse']:.4f}")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                continue
        
        return results
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate evaluation metrics"""
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
    
    def _get_feature_importance(self, model, feature_names: List[str]) -> Dict[str, float]:
        """Get feature importance if available"""
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_)
        else:
            return {}
        
        return dict(zip(feature_names, importance))

    def train_cluster_models(self, 
                             features_df: pd.DataFrame,
                             target_column: str = 'target_1d',
                             test_size: float = 0.2) -> Dict[str, Any]:
        """
        Train one model per cluster on representative data.
        
        Args:
            features_df: DataFrame with features, targets, and cluster_id
            target_column: Name of target column
            test_size: Fraction of data for testing
            
        Returns:
            Dictionary with model results and metrics per cluster
        """
        logger.info(f"Training models per cluster to predict {target_column}")
        
        if 'cluster_id' not in features_df.columns:
            logger.error("`cluster_id` not found in features. Cannot train cluster models.")
            return {}

        cluster_results = {}
        
        # Ensure date is a datetime object for proper sorting
        features_df['date'] = pd.to_datetime(features_df['date'])
        
        for cluster_id in sorted(features_df['cluster_id'].unique()):
            logger.info(f"--- Processing Cluster {cluster_id} ---")
            
            cluster_data = features_df[features_df['cluster_id'] == cluster_id].copy()
            
            # Create representative data by averaging features and target per date
            representative_data = cluster_data.groupby('date').mean(numeric_only=True)
            representative_data = representative_data.sort_index()
            
            logger.info(f"Created representative data for cluster {cluster_id} with {len(representative_data)} time steps.")

            # Prepare data for training
            feature_cols = [col for col in representative_data.columns 
                           if col not in ['cluster_id'] and not col.startswith('target_')]
            
            X = representative_data[feature_cols].fillna(0)
            y = representative_data[target_column].fillna(0)
            
            if len(X) < 50: # Not enough data to train
                logger.warning(f"Skipping cluster {cluster_id} due to insufficient data ({len(X)} samples).")
                continue

            # Time-based split
            split_idx = int(len(X) * (1 - test_size))
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # Scale features
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Store scaler for this cluster
            self.scalers[f"cluster_{cluster_id}_{target_column}"] = scaler
            
            cluster_model_results = {}
            
            # Train each model type for the cluster
            for model_name, model_params in self.config.ML_MODELS.items():
                if model_name in ['svm', 'linear_regression', 'ridge', 'lasso']: # Skip simpler models
                    continue
                try:
                    logger.info(f"Training {model_name} for cluster {cluster_id}")
                    
                    if model_name == 'random_forest':
                        model = RandomForestRegressor(**model_params, random_state=42, n_jobs=-1)
                    elif model_name == 'xgboost':
                        model = xgb.XGBRegressor(**model_params, random_state=42, n_jobs=-1)
                    else:
                        continue

                    model.fit(X_train_scaled, y_train)
                    
                    y_pred_test = model.predict(X_test_scaled)
                    test_metrics = self._calculate_metrics(y_test, y_pred_test)
                    
                    # Store the trained model, keyed by cluster and model name
                    model_key = f"cluster_{cluster_id}_{model_name}_{target_column}"
                    self.models[model_key] = model
                    
                    cluster_model_results[model_name] = {
                        'model': model,
                        'test_metrics': test_metrics,
                        'feature_importance': self._get_feature_importance(model, feature_cols)
                    }
                    logger.info(f"Cluster {cluster_id} | {model_name} - Test R²: {test_metrics['r2']:.4f}")

                except Exception as e:
                    logger.error(f"Error training {model_name} for cluster {cluster_id}: {e}")
            
            cluster_results[f"cluster_{cluster_id}"] = cluster_model_results

        self.feature_columns = feature_cols # Store feature columns for prediction
        
        # Save all trained models and scalers to a single file
        self._save_cluster_artifacts(target_column)
        
        return cluster_results

    def _save_cluster_artifacts(self, target_column: str):
        """Save all cluster models and scalers to pickle files."""
        market_models_dir = self.config.get_market_models_dir(self.config.MARKET)
        os.makedirs(market_models_dir, exist_ok=True)
        
        # Save models
        models_path = os.path.join(market_models_dir, 'cluster_ml_models.pkl')
        with open(models_path, 'wb') as f:
            pickle.dump(self.models, f)
        logger.info(f"Saved {len(self.models)} cluster models to {models_path}")
        
        # Save scalers
        scalers_path = os.path.join(market_models_dir, 'cluster_ml_scalers.pkl')
        with open(scalers_path, 'wb') as f:
            pickle.dump(self.scalers, f)
        logger.info(f"Saved {len(self.scalers)} cluster scalers to {scalers_path}")
        
        # Save feature columns
        features_path = os.path.join(market_models_dir, 'cluster_feature_columns.pkl')
        with open(features_path, 'wb') as f:
            pickle.dump(self.feature_columns, f)
        logger.info(f"Saved feature column list to {features_path}")


class DLModels:
    """Deep learning models for return prediction"""
    
    def __init__(self):
        self.config = Config()
        self.models = {}
        self.scalers = {}

    def build_cnn_model(self,
                        input_shape: Tuple[int, int],
                        model_config: Dict[str, Any]) -> Any:
        """Build 1D CNN model as per the paper's architecture."""
        if not HAS_TENSORFLOW:
            raise ImportError("TensorFlow not available for CNN model")
        
        model = models.Sequential([
            layers.Input(shape=input_shape),
            
            # First Conv1D layer
            layers.Conv1D(filters=128, kernel_size=3, activation='relu'),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(model_config['dropout']),
            
            # Second Conv1D layer
            layers.Conv1D(filters=32, kernel_size=3, activation='relu'),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(model_config['dropout']),
            
            # Flatten and Dense layers
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(model_config['dropout']),
            layers.Dense(1) # Output layer
        ])
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=model_config.get('learning_rate', 0.001))
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        return model

    def build_lstm_model(self, 
                        input_shape: Tuple[int, int],
                        model_config: Dict[str, Any]) -> Any:
        """Build LSTM model"""
        if not HAS_TENSORFLOW:
            raise ImportError("TensorFlow not available for LSTM model")
            
        model = models.Sequential([
            layers.LSTM(model_config['hidden_size'], 
                       return_sequences=True,
                       dropout=model_config['dropout'],
                       input_shape=input_shape),
            layers.LSTM(model_config['hidden_size'],
                       dropout=model_config['dropout']),
            layers.Dense(64, activation='relu'),
            layers.Dropout(model_config['dropout']),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def build_gru_model(self,
                       input_shape: Tuple[int, int],
                       model_config: Dict[str, Any]) -> Any:
        """Build GRU model"""
        if not HAS_TENSORFLOW:
            raise ImportError("TensorFlow not available for GRU model")
            
        model = models.Sequential([
            layers.GRU(model_config['hidden_size'],
                      return_sequences=True,
                      dropout=model_config['dropout'],
                      input_shape=input_shape),
            layers.GRU(model_config['hidden_size'],
                      dropout=model_config['dropout']),
            layers.Dense(64, activation='relu'),
            layers.Dropout(model_config['dropout']),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    def train_cluster_dl_models(self,
                                features_df: pd.DataFrame,
                                sequence_length: int = 60,
                                test_size: float = 0.2) -> Dict[str, Any]:
        """
        Train DL models (LSTM, CNN) on representative cluster data.
        """
        if not HAS_TENSORFLOW:
            logger.warning("TensorFlow not available, skipping DL models.")
            return {}
            
        logger.info(f"Training DL models per cluster on sequences of length {sequence_length}")

        if 'cluster_id' not in features_df.columns:
            logger.error("`cluster_id` not found. Cannot train cluster DL models.")
            return {}

        cluster_results = {}
        features_df['date'] = pd.to_datetime(features_df['date'])

        for cluster_id in sorted(features_df['cluster_id'].unique()):
            logger.info(f"--- Processing DL for Cluster {cluster_id} ---")
            
            cluster_data = features_df[features_df['cluster_id'] == cluster_id].copy()
            representative_data = cluster_data.groupby('date').mean(numeric_only=True).sort_index()
            
            feature_cols = [col for col in representative_data.columns if col not in ['cluster_id'] and not col.startswith('target_')]
            target_col = 'target_1d'
            
            if len(representative_data) < sequence_length + 10:
                logger.warning(f"Skipping DL for cluster {cluster_id}, not enough data.")
                continue

            # Prepare sequences from representative data
            X, y = self._create_sequences_from_df(representative_data, feature_cols, target_col, sequence_length)
            
            if len(X) == 0:
                logger.warning(f"Could not create sequences for cluster {cluster_id}.")
                continue

            # Time-based split
            split_idx = int(len(X) * (1 - test_size))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]

            cluster_model_results = {}
            for model_name, model_config in self.config.DL_MODELS.items():
                if model_name not in ['lstm', 'cnn']: # Only train specified models
                    continue
                
                try:
                    logger.info(f"Training {model_name} for cluster {cluster_id}")
                    
                    if model_name == 'lstm':
                        model = self.build_lstm_model(X_train.shape[1:], model_config)
                    elif model_name == 'cnn':
                        model = self.build_cnn_model(X_train.shape[1:], model_config)

                    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), **model_config['fit_params'])
                    
                    y_pred_test = model.predict(X_test)
                    test_metrics = self._calculate_metrics(y_test, y_pred_test)
                    
                    model_key = f"cluster_{cluster_id}_{model_name}_target_1d"
                    self.models[model_key] = model
                    
                    cluster_model_results[model_name] = {'model': model, 'test_metrics': test_metrics}
                    logger.info(f"Cluster {cluster_id} | {model_name} - Test RMSE: {test_metrics['rmse']:.6f}")

                except Exception as e:
                    logger.error(f"Error training {model_name} for cluster {cluster_id}: {e}")
            
            cluster_results[f"cluster_{cluster_id}"] = cluster_model_results
            
        return cluster_results

    def _create_sequences_from_df(self, df: pd.DataFrame, feature_cols: List[str], target_col: str, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Helper to create sequences from a single dataframe."""
        X_list, y_list = [], []
        
        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(df[feature_cols])
        # We don't scale the target here, models will predict the actual value
        targets = df[target_col].values

        for i in range(sequence_length, len(features_scaled)):
            X_list.append(features_scaled[i-sequence_length:i, :])
            y_list.append(targets[i])
            
        return np.array(X_list), np.array(y_list)

    def train_dl_models(self,
                       X: np.ndarray,
                       y: np.ndarray,
                       test_size: float = 0.2) -> Dict[str, Any]:
        """
        Train deep learning models
        
        Args:
            X: Input sequences
            y: Target values
            test_size: Fraction of data for testing
            
        Returns:
            Dictionary with model results
        """
        if not HAS_TENSORFLOW:
            logger.warning("TensorFlow not available, skipping deep learning models")
            return {}
            
        logger.info(f"Training DL models on {len(X)} sequences")
        
        # Scale targets
        y_scaler = StandardScaler()
        y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()
        
        # Time-based split
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y_scaled[:split_idx], y_scaled[split_idx:]
        
        results = {}
        
        for model_name, model_config in self.config.DL_MODELS.items():
            if model_name not in ['lstm', 'gru', 'cnn']:  # Added CNN
                continue
            
            try:
                logger.info(f"Training {model_name}")
                
                # Build model
                if model_name == 'lstm':
                    model = self.build_lstm_model(X_train.shape[1:], model_config)
                elif model_name == 'gru':
                    model = self.build_gru_model(X_train.shape[1:], model_config)
                elif model_name == 'cnn':
                    model = self.build_cnn_model(X_train.shape[1:], model_config)
                
                # Callbacks
                early_stopping = callbacks.EarlyStopping(
                    monitor='val_loss', patience=10, restore_best_weights=True
                )
                
                reduce_lr = callbacks.ReduceLROnPlateau(
                    monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6
                )
                
                # Train model
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=model_config['epochs'],
                    batch_size=model_config['batch_size'],
                    callbacks=[early_stopping, reduce_lr],
                    verbose=0
                )
                
                # Make predictions
                y_pred_test = model.predict(X_test)
                y_pred_test_unscaled = y_scaler.inverse_transform(y_pred_test.reshape(-1, 1)).flatten()
                y_test_unscaled = y_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
                
                # Calculate metrics
                test_metrics = self._calculate_metrics(y_test_unscaled, y_pred_test_unscaled)
                
                # Store results
                self.models[model_name] = model
                self.scalers[model_name] = y_scaler
                
                results[model_name] = {
                    'model': model,
                    'history': history.history,
                    'test_metrics': test_metrics,
                    'y_scaler': y_scaler
                }
                
                logger.info(f"{model_name} - Test RMSE: {test_metrics['rmse']:.6f}, "
                           f"Test R²: {test_metrics['r2']:.4f}")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                continue
        
        return results
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate evaluation metrics"""
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }

def main(market: str = 'US'):
    """Main function to train return prediction models"""
    logger.info(f"Starting return prediction model training for {market.upper()} market")
    
    try:
        # Load data
        config = Config()
        processed_data_dir = config.get_market_data_dir(market, 'processed')
        results_dir = config.get_market_results_dir(market)
        models_dir = os.path.join(config.BASE_DIR, 'models', market.lower())
        os.makedirs(models_dir, exist_ok=True)

        price_data = pd.read_csv(os.path.join(processed_data_dir, 'processed_stock_data.csv'), header=[0, 1], index_col=0, parse_dates=True)
        returns_data = pd.read_csv(os.path.join(processed_data_dir, 'returns_data.csv'), header=[0, 1], index_col=0, parse_dates=True)
        
        # For this example, we'll use kmeans assignments. In a real scenario, you'd loop through algorithms.
        cluster_assignments_path = os.path.join(results_dir, 'cluster_assignments_kmeans.csv')
        if not os.path.exists(cluster_assignments_path):
            logger.error(f"Cluster assignments not found at {cluster_assignments_path}. Please run clustering first.")
            return
        cluster_data = pd.read_csv(cluster_assignments_path)

        # Feature Generation
        feature_gen = FeatureGenerator()
        features_df = feature_gen.create_ml_features(price_data, returns_data, cluster_data)
        
        # --- Train Models on Representative Cluster Data ---
        ml_trainer = MLModels()
        cluster_ml_results = ml_trainer.train_cluster_models(features_df)
        
        # Save ML models and results
        with open(os.path.join(models_dir, 'cluster_ml_models.pkl'), 'wb') as f:
            pickle.dump(ml_trainer.models, f)
        with open(os.path.join(models_dir, 'cluster_ml_scalers.pkl'), 'wb') as f:
            pickle.dump(ml_trainer.scalers, f)
        logger.info(f"Saved trained cluster-based ML models and scalers to {models_dir}")

        # --- Train DL Models on Representative Cluster Data ---
        dl_trainer = DLModels()
        cluster_dl_results = dl_trainer.train_cluster_dl_models(features_df)

        # Save DL models
        for model_key, model in dl_trainer.models.items():
            model.save(os.path.join(models_dir, f"{model_key}.h5"))
        logger.info(f"Saved trained cluster-based DL models to {models_dir}")

    except Exception as e:
        logger.exception(f"An error occurred during model training: {e}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train return prediction models for a specific market')
    parser.add_argument(
        '--market', 
        type=str, 
        default='US',
        choices=['US', 'INDIA', 'us', 'india'],
        help='Market to train models for (US or INDIA)'
    )
    
    args = parser.parse_args()
    main(market=args.market.upper())