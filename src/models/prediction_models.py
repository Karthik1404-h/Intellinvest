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
        
        symbols = returns_data.columns.get_level_values(0).unique()
        all_features = []
        
        for symbol in tqdm(symbols, desc="Creating features"):
            try:
                symbol_features = self._create_symbol_features(
                    symbol, price_data, returns_data, cluster_data
                )
                all_features.append(symbol_features)
                
            except Exception as e:
                logger.warning(f"Error creating features for {symbol}: {e}")
                continue
        
        # Combine all features
        features_df = pd.concat(all_features, ignore_index=True)
        features_df = features_df.dropna()
        
        logger.info(f"Created {len(features_df)} feature samples with {features_df.shape[1]-3} features")
        return features_df
    
    def _create_symbol_features(self,
                               symbol: str,
                               price_data: pd.DataFrame,
                               returns_data: pd.DataFrame,
                               cluster_data: Optional[pd.DataFrame]) -> pd.DataFrame:
        """Create features for a single symbol"""
        
        # Get symbol data
        symbol_prices = price_data[symbol] if symbol in price_data.columns.get_level_values(0) else None
        symbol_returns = returns_data[symbol]['returns_1d'] if (symbol, 'returns_1d') in returns_data.columns else None
        
        if symbol_prices is None or symbol_returns is None:
            return pd.DataFrame()
        
        # Create features dataframe
        features_list = []
        
        for i in range(max(self.config.LOOKBACK_PERIODS) + 20, len(symbol_returns) - max(self.config.PREDICTION_HORIZONS)):
            try:
                feature_dict = {'symbol': symbol, 'date': symbol_returns.index[i]}
                
                # Historical returns features
                for lookback in self.config.LOOKBACK_PERIODS:
                    recent_returns = symbol_returns.iloc[i-lookback:i]
                    
                    feature_dict[f'return_mean_{lookback}d'] = recent_returns.mean()
                    feature_dict[f'return_std_{lookback}d'] = recent_returns.std()
                    feature_dict[f'return_skew_{lookback}d'] = recent_returns.skew()
                    feature_dict[f'return_kurt_{lookback}d'] = recent_returns.kurtosis()
                    feature_dict[f'return_min_{lookback}d'] = recent_returns.min()
                    feature_dict[f'return_max_{lookback}d'] = recent_returns.max()
                    
                    # Momentum features
                    feature_dict[f'momentum_{lookback}d'] = (symbol_prices['Close'].iloc[i] / 
                                                            symbol_prices['Close'].iloc[i-lookback] - 1)
                
                # Technical indicator features
                if 'Close' in symbol_prices.columns:
                    close_prices = symbol_prices['Close']
                    
                    # Moving averages
                    sma_5 = close_prices.iloc[i-4:i+1].mean()
                    sma_10 = close_prices.iloc[i-9:i+1].mean()
                    sma_20 = close_prices.iloc[i-19:i+1].mean()
                    
                    feature_dict['price_to_sma5'] = close_prices.iloc[i] / sma_5
                    feature_dict['price_to_sma10'] = close_prices.iloc[i] / sma_10
                    feature_dict['price_to_sma20'] = close_prices.iloc[i] / sma_20
                    feature_dict['sma5_to_sma20'] = sma_5 / sma_20
                    
                    # Volatility
                    feature_dict['realized_vol_5d'] = close_prices.iloc[i-4:i+1].pct_change().std() * np.sqrt(252)
                    feature_dict['realized_vol_20d'] = close_prices.iloc[i-19:i+1].pct_change().std() * np.sqrt(252)
                
                # Volume features
                if 'Volume' in symbol_prices.columns:
                    volumes = symbol_prices['Volume']
                    feature_dict['volume_ma_5d'] = volumes.iloc[i-4:i+1].mean()
                    feature_dict['volume_ma_20d'] = volumes.iloc[i-19:i+1].mean()
                    feature_dict['volume_ratio'] = volumes.iloc[i] / volumes.iloc[i-19:i+1].mean()
                
                # Cluster features (if available)
                if cluster_data is not None and symbol in cluster_data['symbol'].values:
                    cluster_id = cluster_data[cluster_data['symbol'] == symbol]['cluster'].iloc[0]
                    feature_dict['cluster_id'] = cluster_id
                
                # Target variables (future returns)
                for horizon in self.config.PREDICTION_HORIZONS:
                    if i + horizon < len(symbol_returns):
                        if horizon == 1:
                            feature_dict[f'target_{horizon}d'] = symbol_returns.iloc[i + horizon]
                        else:
                            # Multi-period return
                            feature_dict[f'target_{horizon}d'] = (
                                symbol_returns.iloc[i+1:i+horizon+1].add(1).prod() - 1
                            )
                
                features_list.append(feature_dict)
                
            except Exception as e:
                continue
        
        return pd.DataFrame(features_list)
    
    def create_lstm_sequences(self, 
                            returns_data: pd.DataFrame,
                            sequence_length: int = 60) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Create sequences for LSTM training
        
        Args:
            returns_data: Stock returns data
            sequence_length: Length of input sequences
            
        Returns:
            Tuple of (X, y, symbols) where X is input sequences, y is targets
        """
        logger.info(f"Creating LSTM sequences with length {sequence_length}")
        
        symbols = returns_data.columns.get_level_values(0).unique()
        X_list, y_list, symbol_list = [], [], []
        
        for symbol in tqdm(symbols, desc="Creating sequences"):
            try:
                if (symbol, 'returns_1d') not in returns_data.columns:
                    continue
                
                symbol_returns = returns_data[symbol]['returns_1d'].dropna()
                
                if len(symbol_returns) < sequence_length + 10:
                    continue
                
                # Create sequences
                for i in range(sequence_length, len(symbol_returns) - 1):
                    # Input sequence
                    X_seq = symbol_returns.iloc[i-sequence_length:i].values
                    # Target (next day return)
                    y_target = symbol_returns.iloc[i+1]
                    
                    X_list.append(X_seq)
                    y_list.append(y_target)
                    symbol_list.append(symbol)
                
            except Exception as e:
                logger.warning(f"Error creating sequences for {symbol}: {e}")
                continue
        
        X = np.array(X_list).reshape(-1, sequence_length, 1)
        y = np.array(y_list)
        
        logger.info(f"Created {len(X)} sequences from {len(set(symbol_list))} symbols")
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

class DLModels:
    """Deep learning models for return prediction"""
    
    def __init__(self):
        self.config = Config()
        self.models = {}
        self.scalers = {}
    
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
            if model_name not in ['lstm', 'gru']:  # Skip transformer for now
                continue
            
            try:
                logger.info(f"Training {model_name}")
                
                # Build model
                if model_name == 'lstm':
                    model = self.build_lstm_model(X_train.shape[1:], model_config)
                elif model_name == 'gru':
                    model = self.build_gru_model(X_train.shape[1:], model_config)
                
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

def main():
    """Main function to train return prediction models"""
    logger.info("Starting return prediction model training")
    
    try:
        # Load processed data
        processed_data_path = os.path.join(Config.PROCESSED_DATA_DIR, 'processed_stock_data.csv')
        returns_data_path = os.path.join(Config.PROCESSED_DATA_DIR, 'returns_data.csv')
        cluster_data_path = os.path.join(Config.RESULTS_DIR, 'cluster_assignments_kmeans.csv')
        
        price_data = pd.read_csv(processed_data_path, index_col=0, header=[0, 1])
        returns_data = pd.read_csv(returns_data_path, index_col=0, header=[0, 1])
        
        cluster_data = None
        if os.path.exists(cluster_data_path):
            cluster_data = pd.read_csv(cluster_data_path)
        
        logger.info("Data loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return
    
    # Generate features
    feature_generator = FeatureGenerator()
    
    # Train ML models
    logger.info("Training traditional ML models")
    ml_features = feature_generator.create_ml_features(price_data, returns_data, cluster_data)
    
    ml_trainer = MLModels()
    ml_results = ml_trainer.train_models(ml_features)
    
    # Train DL models
    logger.info("Training deep learning models") 
    X_sequences, y_sequences, symbols = feature_generator.create_lstm_sequences(returns_data)
    
    dl_trainer = DLModels()
    dl_results = dl_trainer.train_dl_models(X_sequences, y_sequences)
    
    # Save models and results
    results_dir = Config.RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)
    
    # Save ML models
    for model_name, result in ml_results.items():
        model_path = os.path.join(results_dir, f'ml_model_{model_name}.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(result['model'], f)
    
    # Save DL models
    for model_name, result in dl_results.items():
        model_path = os.path.join(results_dir, f'dl_model_{model_name}.h5')
        result['model'].save(model_path)
    
    # Save results summary
    results_summary = {
        'ml_models': {name: result['test_metrics'] for name, result in ml_results.items()},
        'dl_models': {name: result['test_metrics'] for name, result in dl_results.items()}
    }
    
    import json
    with open(os.path.join(results_dir, 'model_results_summary.json'), 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    logger.info("Model training completed successfully!")
    logger.info("Results summary:")
    for category, models in results_summary.items():
        print(f"\n{category.upper()}:")
        for model_name, metrics in models.items():
            print(f"  {model_name}: R² = {metrics.get('r2', 0):.4f}, "
                  f"RMSE = {metrics.get('rmse', 0):.6f}")

if __name__ == "__main__":
    main()