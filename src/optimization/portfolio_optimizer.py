"""
Portfolio optimization algorithms enhanced with machine learning predictions
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import cvxpy as cp
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

from loguru import logger

try:
    from pypfopt import EfficientFrontier, risk_models, expected_returns
    from pypfopt import HRPOpt, discrete_allocation
    PYPFOPT_AVAILABLE = True
except ImportError:
    PYPFOPT_AVAILABLE = False
    logger.warning("pypfopt not available. Some optimization methods will use fallback implementations.")
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

from config import Config

# Import prediction models
from src.models.prediction_models import FeatureGenerator

class ReturnForecaster:
    """Uses trained ML/DL models to forecast returns"""
    
    def __init__(self, market: str):
        self.config = Config()
        self.market = market.upper()
        self.models = {}
        self.scalers = {}
        self.feature_columns = None
        self.feature_generator = FeatureGenerator()
        self.cluster_assignments = None
        self._load_all_models()
        self._load_cluster_assignments()

    def _load_all_models(self):
        """Load all trained ML and DL models for the specified market."""
        import os
        import pickle
        import sys
        
        # Force absolute pathing to prevent directory mismatch errors
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        models_dir = os.path.join(base_dir, 'models', self.market.lower())
        
        logger.info(f"Looking for ML models in EXACT path: {models_dir}")
        
        if not os.path.exists(models_dir):
            logger.error(f"Models directory DOES NOT EXIST: {models_dir}")
            return
            
        try:
            ml_models_path = os.path.join(models_dir, 'cluster_ml_models.pkl')
            if os.path.exists(ml_models_path):
                with open(ml_models_path, 'rb') as f:
                    self.models.update(pickle.load(f))
                logger.info("Successfully loaded ML models from pickle.")
            else:
                logger.warning(f"Pickle file not found at: {ml_models_path}")
                
            scalers_path = os.path.join(models_dir, 'cluster_ml_scalers.pkl')
            if os.path.exists(scalers_path):
                with open(scalers_path, 'rb') as f:
                    self.scalers = pickle.load(f)
                    
        except Exception as e:
            logger.error(f"CRITICAL ERROR loading models: {str(e)}")

    def _load_cluster_assignments(self, algorithm: str = 'kmeans'):
        """Load the cluster assignments for the market."""
        results_dir = self.config.get_market_results_dir(self.market)
        assignments_path = os.path.join(results_dir, f'cluster_assignments_{algorithm}.csv')
        
        if os.path.exists(assignments_path):
            self.cluster_assignments = pd.read_csv(assignments_path).set_index('symbol')['cluster']
            logger.info(f"Loaded cluster assignments for {len(self.cluster_assignments)} symbols.")
        else:
            logger.warning(f"Cluster assignments not found at {assignments_path}. Predictions will not be cluster-aware.")

    def forecast_returns(self, 
                        price_data: pd.DataFrame,
                        returns_data: pd.DataFrame,
                        horizon: int = 1,
                        method: str = 'cluster_cnn') -> pd.Series:
        """
        Forecast stock returns using trained cluster-based models.
        
        Args:
            price_data: Historical price data for feature generation.
            returns_data: Historical returns data.
            horizon: Forecast horizon (currently supports 1-day).
            method: The model to use, e.g., 'cluster_cnn_target_1d'.
            
        Returns:
            Series with predicted returns for each stock.
        """
        logger.info(f"Forecasting returns for {horizon} day(s) using {method}")
        
        if self.cluster_assignments is None:
            logger.error("Cannot forecast: Cluster assignments not loaded.")
            return self._naive_forecast(returns_data, horizon)

        if not self.models:
            logger.error("Cannot forecast: No models loaded.")
            return self._naive_forecast(returns_data, horizon)
            
        logger.info(f"Available models in memory: {list(self.models.keys())}")

        # 1. Generate the latest features for all stocks
        # We need to create features for a single time step (the most recent one)
        latest_features = self._generate_latest_features(price_data, returns_data)
        
        if latest_features.empty:
            logger.error("Could not generate latest features. Falling back to naive forecast.")
            return self._naive_forecast(returns_data, horizon)

        # 2. Predict return for each stock using its cluster's model
        predictions = {}
        for symbol, features in latest_features.iterrows():
            try:
                cluster_id = self.cluster_assignments.get(symbol)
                if cluster_id is None:
                    continue

                model_key_base = method.replace('_target_1d', '') # e.g., 'cluster_random_forest'
                model_key = f"{model_key_base}_cluster_{cluster_id}"
                
                # Let's try a more robust key matching
                potential_keys = [
                    f"cluster_{cluster_id}_{model_key_base}_{horizon}d",
                    f"cluster_{cluster_id}_{model_key_base}_target_{horizon}d",
                    f"{model_key_base}_cluster_{cluster_id}_target_{horizon}d",
                ]
                
                model = None
                for key in potential_keys:
                    if key in self.models:
                        model = self.models[key]
                        model_key = key
                        break
                
                scaler_key = f"cluster_{cluster_id}_target_{horizon}d"
                scaler = self.scalers.get(scaler_key)

                if model is None:
                    # logger.warning(f"No model found for cluster {cluster_id} with base '{model_key_base}'")
                    continue
                if scaler is None:
                    # logger.warning(f"No scaler found for cluster {cluster_id} with key '{scaler_key}'")
                    continue

                # Prepare features for prediction (scale and reshape)
                feature_values = features.values.reshape(1, -1)
                scaled_features = scaler.transform(feature_values)
                
                # DL models need sequence input, ML models need flat input
                if 'cnn' in method or 'lstm' in method:
                    # For single prediction, we need to construct a sequence from historical data
                    # This part is complex and requires sequence generation for the last step
                    # For now, we'll use a simplified approach with ML models
                    pass # Fallback to ML for now
                
                # Using ML models (RandomForest, XGBoost)
                prediction = model.predict(scaled_features)[0]
                predictions[symbol] = prediction

            except Exception as e:
                logger.error(f"Error predicting for {symbol}: {e}")
                predictions[symbol] = 0 # Default to zero on error

        if not predictions:
            logger.error("No predictions were made. Falling back to naive forecast.")
            return self._naive_forecast(returns_data, horizon)

        return pd.Series(predictions)

    def _generate_latest_features(self, price_data: pd.DataFrame, returns_data: pd.DataFrame) -> pd.DataFrame:
        """Generates the most recent feature set for all symbols."""
        
        # This is a simplified version. A robust implementation would use the FeatureGenerator
        # and extract the last row for each symbol.
        logger.info("Generating latest features for all symbols...")
        full_features = self.feature_generator.create_ml_features(price_data, returns_data, self.cluster_assignments.reset_index())
        
        if full_features.empty:
            return pd.DataFrame()
            
        # Get the last valid feature row for each symbol
        latest_features = full_features.loc[full_features.groupby('symbol')['date'].idxmax()]
        
        # Ensure feature columns match what models were trained on
        if self.feature_columns:
             latest_features = latest_features[self.feature_columns]

        return latest_features.set_index('symbol')

    def _naive_forecast(self, data: pd.DataFrame, horizon: int) ->pd.Series:
        """Naive forecast using historical mean returns"""
        symbols = data.columns.get_level_values(0).unique()
        forecasts = {}
        
        for symbol in symbols:
            try:
                # Try returns_1d first, then calculate from Close
                if (symbol, 'returns_1d') in data.columns:
                    returns = data[symbol]['returns_1d'].dropna()
                elif (symbol, 'Close') in data.columns:
                    prices = data[symbol]['Close'].dropna()
                    returns = prices.pct_change().dropna()
                else:
                    continue
                
                if len(returns) >= 20:
                    # Use rolling average of recent returns
                    forecast = returns.tail(60).mean() * horizon  # Adjust for horizon
                    forecasts[symbol] = forecast
            except Exception as e:
                logger.warning(f"Error forecasting {symbol}: {e}")
                continue
        
        forecast_series = pd.Series(forecasts)
        logger.info(f"Naive forecast range: [{forecast_series.min():.6f}, {forecast_series.max():.6f}]")
        return forecast_series
    
    def _ensemble_forecast(self, data: pd.DataFrame, horizon: int) -> pd.Series:
        """Ensemble forecast combining multiple models"""
        # For now, implement a simple ensemble of naive + momentum
        naive_forecast = self._naive_forecast(data, horizon)
        momentum_forecast = self._momentum_forecast(data, horizon)
        
        # Simple average ensemble (can be enhanced with learned weights)
        ensemble_forecast = (naive_forecast + momentum_forecast) / 2
        return ensemble_forecast.fillna(0)
    
    def _momentum_forecast(self, data: pd.DataFrame, horizon: int) -> pd.Series:
        """Momentum-based forecast"""
        symbols = data.columns.get_level_values(0).unique()
        forecasts = {}
        
        for symbol in symbols:
            try:
                if 'Close' in data[symbol].columns:
                    prices = data[symbol]['Close'].dropna()
                    if len(prices) >= 20:
                        # Calculate momentum signals
                        short_ma = prices.tail(5).mean()
                        long_ma = prices.tail(20).mean()
                        momentum = (short_ma / long_ma - 1)
                        
                        # Convert to return forecast
                        forecasts[symbol] = momentum * 0.1  # Scale factor
                    
            except Exception as e:
                continue
        
        return pd.Series(forecasts)
    
    def _model_forecast(self, data: pd.DataFrame, horizon: int, model_name: str) -> pd.Series:
        """Forecast using a specific trained model"""
        # This is a placeholder for actual model prediction
        # In practice, you would use the trained ML/DL models here
        return self._naive_forecast(data, horizon)

class PortfolioOptimizer:
    """Main portfolio optimization class"""
    
    def __init__(self, market: str):
        self.config = Config()
        self.market = market
        self.forecaster = ReturnForecaster(market=self.market)
    
    def optimize_portfolio(self,
                          price_data: pd.DataFrame,
                          returns_data: pd.DataFrame, # Added returns_data
                          expected_returns: Optional[pd.Series] = None,
                          method: str = 'mean_variance',
                          constraints: Optional[Dict] = None,
                          cluster_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Optimize portfolio using specified method
        
        Args:
            price_data: Historical price data
            returns_data: Historical returns data
            expected_returns: Expected returns (if None, will be forecasted)
            method: Optimization method
            constraints: Additional constraints
            cluster_data: Stock cluster information for cluster-based constraints
            
        Returns:
            Dictionary with optimization results
        """
        logger.info(f"Optimizing portfolio using {method}")
        
        # Get symbols and clean data
        symbols = self._get_valid_symbols(price_data)
        clean_prices = self._clean_prices_for_optimization(price_data, symbols)
        
        # Calculate historical returns for covariance
        historical_returns = clean_prices.pct_change().dropna()
        
        # Get expected returns ONLY if the method requires them
        methods_requiring_forecasts = ['mean_variance', 'max_sharpe', 'cluster_based']
        
        if expected_returns is None and method in methods_requiring_forecasts:
            forecast_method = 'cluster_random_forest_target_1d'
            expected_returns = self.forecaster.forecast_returns(price_data, returns_data, method=forecast_method)
            # Align with available symbols and fill missing with historical mean
            expected_returns = expected_returns.reindex(symbols).fillna(historical_returns.mean())
        elif expected_returns is None:
            # For Risk Parity, Min Variance, etc., we don't need ML forecasts
            expected_returns = historical_returns.mean().reindex(symbols)
        
        # Calculate covariance matrix from historical returns
        cov_matrix = historical_returns.cov()
        
        # Align all inputs
        valid_symbols = list(set(expected_returns.index) & set(cov_matrix.index))
        expected_returns = expected_returns.loc[valid_symbols]
        cov_matrix = cov_matrix.loc[valid_symbols, valid_symbols]

        # Apply optimization method
        if method == 'mean_variance':
            result = self._mean_variance_optimization(expected_returns, cov_matrix, constraints)
        elif method == 'risk_parity':
            result = self._risk_parity_optimization(cov_matrix, constraints)
        elif method == 'hierarchical_risk_parity':
            result = self._hrp_optimization(returns_data, constraints)
        elif method == 'min_variance':
            result = self._min_variance_optimization(cov_matrix, constraints)
        elif method == 'max_sharpe':
            result = self._max_sharpe_optimization(expected_returns, cov_matrix, constraints)
        elif method == 'cluster_based':
            result = self._cluster_based_optimization(expected_returns, cov_matrix, cluster_data, constraints)
        else:
            raise ValueError(f"Unknown optimization method: {method}")
        
        # Add metadata
        result['method'] = method
        result['symbols'] = valid_symbols
        result['optimization_date'] = datetime.now().isoformat()
        result['expected_returns'] = expected_returns
        result['covariance_matrix'] = cov_matrix
        
        return result
    
    def _get_valid_symbols(self, price_data: pd.DataFrame) -> List[str]:
        """Get list of symbols with sufficient data"""
        symbols = price_data.columns.get_level_values(0).unique()
        valid_symbols = []
        
        for symbol in symbols:
            try:
                if (symbol, 'Close') in price_data.columns:
                    close_prices = price_data[symbol]['Close'].dropna()
                    if len(close_prices) >= 100:  # Minimum 100 data points
                        valid_symbols.append(symbol)
            except:
                continue
        
        logger.info(f"Found {len(valid_symbols)} valid symbols for optimization")
        return valid_symbols
    
    def _clean_prices_for_optimization(self, price_data: pd.DataFrame, symbols: List[str]) -> pd.DataFrame:
        """Clean and align price data for optimization"""
        clean_data = {}
        
        for symbol in symbols:
            if (symbol, 'Close') in price_data.columns:
                clean_data[symbol] = price_data[symbol]['Close']
        
        clean_df = pd.DataFrame(clean_data)
        clean_df = clean_df.dropna()
        
        return clean_df
    
    def _mean_variance_optimization(self, 
                                  expected_returns: pd.Series, 
                                  cov_matrix: pd.DataFrame,
                                  constraints: Optional[Dict] = None) -> Dict[str, Any]:
        """Modern Portfolio Theory optimization"""
        n_assets = len(expected_returns)
        
        if n_assets == 0:
            logger.warning("No assets to optimize.")
            return {'weights': pd.Series(), 'optimization_status': 'failed'}

        # Decision variables
        weights = cp.Variable(n_assets)
        
        # Objective: Maximize return - risk penalty
        risk_aversion = constraints.get('risk_aversion', 1.0) if constraints else 1.0
        portfolio_return = expected_returns.values @ weights
        portfolio_risk = cp.quad_form(weights, cov_matrix.values)
        
        objective = cp.Maximize(portfolio_return - risk_aversion * portfolio_risk)
        
        # Constraints
        constraint_list = [cp.sum(weights) == 1]  # Weights sum to 1
        
        # Weight limits
        max_weight = constraints.get('max_weight', self.config.MAX_WEIGHT_PER_STOCK) if constraints else self.config.MAX_WEIGHT_PER_STOCK
        min_weight = constraints.get('min_weight', self.config.MIN_WEIGHT) if constraints else self.config.MIN_WEIGHT
        
        constraint_list.append(weights >= min_weight)
        constraint_list.append(weights <= max_weight)
        
        # Solve optimization problem
        problem = cp.Problem(objective, constraint_list)
        problem.solve(solver=cp.SCS) # Specify solver for robustness
        
        if weights.value is None:
            logger.error("Mean-variance optimization failed. Trying fallback (equal weights).")
            equal_weights = pd.Series(index=expected_returns.index, data=1/n_assets)
            return {'weights': equal_weights, 'optimization_status': 'failed_fallback'}
        
        optimal_weights = pd.Series(index=expected_returns.index, data=weights.value)
        
        # Calculate portfolio metrics
        portfolio_return = (optimal_weights @ expected_returns) * 252
        portfolio_vol = np.sqrt(optimal_weights @ cov_matrix @ optimal_weights) * np.sqrt(252)
        sharpe_ratio = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0
        
        return {
            'weights': optimal_weights,
            'expected_return': portfolio_return,
            'volatility': portfolio_vol,
            'sharpe_ratio': sharpe_ratio,
            'optimization_status': problem.status
        }
    
    def _risk_parity_optimization(self, 
                                cov_matrix: pd.DataFrame,
                                constraints: Optional[Dict] = None) -> Dict[str, Any]:
        """Risk parity optimization"""
        n_assets = len(cov_matrix)

        if n_assets == 0:
            logger.warning("No assets for risk parity.")
            return {'weights': pd.Series(), 'optimization_status': 'failed'}
        
        def risk_parity_objective(weights):
            """Objective function for risk parity"""
            weights = np.array(weights)
            portfolio_vol = np.sqrt(weights @ cov_matrix.values @ weights)
            
            # Marginal risk contributions
            marginal_contribs = (cov_matrix.values @ weights) / portfolio_vol
            risk_contribs = weights * marginal_contribs
            
            # Target: equal risk contributions
            target_risk = portfolio_vol / n_assets
            deviations = risk_contribs - target_risk
            
            return np.sum(deviations ** 2)
        
        # Constraints
        constraints_list = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Weights sum to 1
        ]
        
        # Bounds
        max_weight = constraints.get('max_weight', self.config.MAX_WEIGHT_PER_STOCK) if constraints else self.config.MAX_WEIGHT_PER_STOCK
        min_weight = constraints.get('min_weight', self.config.MIN_WEIGHT) if constraints else self.config.MIN_WEIGHT
        bounds = [(min_weight, max_weight) for _ in range(n_assets)]
        
        # Initial guess: inverse volatility (better starting point for risk parity)
        vols = np.sqrt(np.diag(cov_matrix.values))
        inv_vol = 1 / vols
        x0 = inv_vol / np.sum(inv_vol)
        
        # Optimize
        result = minimize(
            risk_parity_objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list,
            options={'ftol': 1e-9, 'maxiter': 1000}
        )

        if not result.success:
            logger.warning(f"Risk parity optimization failed: {result.message}. Falling back to equal weights.")
            equal_weights = pd.Series(index=cov_matrix.index, data=1/n_assets)
            return {'weights': equal_weights, 'optimization_status': 'failed_fallback'}
        
        optimal_weights = pd.Series(index=cov_matrix.index, data=result.x)
        
        # Calculate portfolio metrics
        portfolio_vol = np.sqrt(optimal_weights @ cov_matrix @ optimal_weights) * np.sqrt(252)
        
        return {
            'weights': optimal_weights,
            'volatility': portfolio_vol,
            'optimization_status': 'success' if result.success else 'failed'
        }
    
    def _hrp_optimization(self, 
                         returns_data: pd.DataFrame,
                         constraints: Optional[Dict] = None) -> Dict[str, Any]:
        """Hierarchical Risk Parity optimization"""
        if not PYPFOPT_AVAILABLE:
            logger.warning("pypfopt not available, using equal weights")
            n_assets = len(returns_data.columns)
            equal_weights = pd.Series(index=returns_data.columns, data=1.0/n_assets)
            return {'weights': equal_weights}
        
        try:
            hrp = HRPOpt(returns_data)
            weights = hrp.optimize()
            
            optimal_weights = pd.Series(weights)
            
            # Calculate portfolio metrics
            cov_matrix = risk_models.sample_cov(returns_data)
            portfolio_vol = np.sqrt(optimal_weights @ cov_matrix @ optimal_weights) * np.sqrt(252)
            
            return {
                'weights': optimal_weights,
                'volatility': portfolio_vol,
                'optimization_status': 'success'
            }
            
        except Exception as e:
            logger.error(f"HRP optimization failed: {e}")
            n_assets = len(returns_data.columns)
            equal_weights = pd.Series(index=returns_data.columns, data=1.0/n_assets)
            return {'weights': equal_weights, 'optimization_status': 'failed_fallback'}
    
    def _min_variance_optimization(self, 
                                 cov_matrix: pd.DataFrame,
                                 constraints: Optional[Dict] = None) -> Dict[str, Any]:
        """Minimum variance optimization using scipy"""
        n_assets = len(cov_matrix)

        if n_assets == 0:
            logger.warning("No assets for min variance.")
            return {'weights': pd.Series(), 'optimization_status': 'failed'}
        
        # Objective: minimize portfolio variance
        def portfolio_variance(weights):
            return weights @ cov_matrix.values @ weights
        
        # Constraints
        max_weight = constraints.get('max_weight', self.config.MAX_WEIGHT_PER_STOCK) if constraints else self.config.MAX_WEIGHT_PER_STOCK
        min_weight = constraints.get('min_weight', self.config.MIN_WEIGHT) if constraints else self.config.MIN_WEIGHT
        
        cons = (
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Weights sum to 1
        )
        
        bounds = tuple((min_weight, max_weight) for _ in range(n_assets))
        
        # Initial guess: equal weights
        x0 = np.array([1.0 / n_assets] * n_assets)
        
        # Optimize
        result = minimize(
            portfolio_variance,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        if not result.success:
            logger.warning(f"Min variance optimization warning: {result.message}. Falling back to equal weights.")
            equal_weights = pd.Series(index=cov_matrix.index, data=1/n_assets)
            return {'weights': equal_weights, 'optimization_status': 'failed_fallback'}
        
        optimal_weights = pd.Series(index=cov_matrix.index, data=result.x)
        portfolio_vol = np.sqrt(result.fun) * np.sqrt(252)
        
        return {
            'weights': optimal_weights,
            'volatility': portfolio_vol,
            'optimization_status': 'success' if result.success else 'warning'
        }
    
    def _max_sharpe_optimization(self, 
                               expected_returns: pd.Series,
                               cov_matrix: pd.DataFrame,
                               constraints: Optional[Dict] = None) -> Dict[str, Any]:
        """Maximum Sharpe ratio optimization using scipy"""
        n_assets = len(expected_returns)

        if n_assets == 0:
            logger.warning("No assets for max sharpe.")
            return {'weights': pd.Series(), 'optimization_status': 'failed'}
        
        # Objective: maximize Sharpe ratio (minimize negative Sharpe)
        def negative_sharpe(weights):
            portfolio_return = np.sum(expected_returns.values * weights) * 252
            portfolio_vol = np.sqrt(weights @ cov_matrix.values @ weights) * np.sqrt(252)
            return -portfolio_return / portfolio_vol if portfolio_vol > 0 else 1e10
        
        # Constraints
        max_weight = constraints.get('max_weight', self.config.MAX_WEIGHT_PER_STOCK) if constraints else self.config.MAX_WEIGHT_PER_STOCK
        min_weight = constraints.get('min_weight', self.config.MIN_WEIGHT) if constraints else self.config.MIN_WEIGHT
        
        cons = (
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Weights sum to 1
        )
        
        bounds = tuple((min_weight, max_weight) for _ in range(n_assets))
        
        # Initial guess: weights proportional to expected returns
        x0 = np.abs(expected_returns.values) + 1e-8
        x0 = x0 / np.sum(x0)
        x0 = np.clip(x0, min_weight, max_weight)
        x0 = x0 / np.sum(x0)  # Renormalize
        
        # Optimize
        result = minimize(
            negative_sharpe,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        if not result.success:
            logger.warning(f"Max Sharpe optimization warning: {result.message}. Falling back to equal weights.")
            equal_weights = pd.Series(index=expected_returns.index, data=1/n_assets)
            return {'weights': equal_weights, 'optimization_status': 'failed_fallback'}
        
        optimal_weights = pd.Series(index=expected_returns.index, data=result.x)
        
        # Calculate metrics
        portfolio_return = (optimal_weights @ expected_returns) * 252
        portfolio_vol = np.sqrt(optimal_weights @ cov_matrix @ optimal_weights) * np.sqrt(252)
        sharpe_ratio = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0
        
        return {
            'weights': optimal_weights,
            'expected_return': portfolio_return,
            'volatility': portfolio_vol,
            'sharpe_ratio': sharpe_ratio,
            'optimization_status': 'success' if result.success else 'warning'
        }
    
    def _cluster_based_optimization(self,
                                  expected_returns: pd.Series,
                                  cov_matrix: pd.DataFrame,
                                  cluster_data: Optional[pd.DataFrame],
                                  constraints: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Robust Cluster-based Optimization combining ML forecasts with
        Tikhonov Regularized Mean-Variance (MVF) and Positivity Filtering.
        """
        import cvxpy as cp
        
        # 1. Filter out assets with negative predicted returns
        positive_returns_mask = expected_returns > 0
        valid_assets = expected_returns[positive_returns_mask].index.tolist()
        
        if len(valid_assets) == 0:
            logger.warning("No assets with positive expected returns. Returning equal cash weights or zeros.")
            return {
                'weights': pd.Series(0.0, index=expected_returns.index),
                'expected_return': 0.0,
                'volatility': 0.0,
                'sharpe_ratio': 0.0,
                'optimization_status': 'no_positive_returns'
            }
            
        # 2. Sub-select the covariance matrix to match exactly the positive assets
        subset_returns = expected_returns.loc[valid_assets].values
        subset_cov_matrix = cov_matrix.loc[valid_assets, valid_assets].values
        
        n_filtered_assets = len(valid_assets)
        
        # 3. Apply Tikhonov Regularization (L2 penalty) to the covariance matrix
        # This guarantees it is positive definite and prevents ECOS crashes
        tikhonov_penalty = 1e-8
        reg_cov_matrix = subset_cov_matrix + np.eye(n_filtered_assets) * tikhonov_penalty
        
        # 4. Define CVXPY problem for the valid subset
        w = cp.Variable(n_filtered_assets)
        
        # Maximize: w^T * mu - gamma * w^T * Cov * w
        # Assuming typical risk aversion gamma = 1 for this context
        gamma = 2.5
        portfolio_return = subset_returns @ w
        portfolio_variance = cp.quad_form(w, cp.psd_wrap(reg_cov_matrix))
        
        objective = cp.Maximize(portfolio_return - gamma * portfolio_variance)
        
        # Constraints: Long only, fully invested, max weight 25% (or based on config)
        max_weight = self.config.MAX_WEIGHT_PER_STOCK if hasattr(self.config, 'MAX_WEIGHT_PER_STOCK') else 0.25
        cp_constraints = [
            cp.sum(w) == 1,
            w >= 0,
            w <= max_weight
        ]
        
        prob = cp.Problem(objective, cp_constraints)
        
        # 5. Solve using the robust ECOS solver
        try:
            prob.solve(solver=cp.ECOS)
        except Exception as e:
            logger.error(f"ECOS solver failed in cluster optimization: {e}")
            # Fallback to general solver if ECOS specifically fails
            prob.solve()
            
        if prob.status not in ["optimal", "optimal_inaccurate"] or w.value is None:
            logger.warning(f"CVXPY Cluster Optimization failed with status: {prob.status}")
            # Fallback to equal weight on valid assets
            safe_weights = np.ones(n_filtered_assets) / n_filtered_assets
        else:
            # Clean tiny negative numerical errors from solver
            safe_weights = np.clip(w.value, 0, None)
            safe_weights /= safe_weights.sum() # Re-normalize
            
        # 6. Reconstruct the full weight series (0 for discarded assets)
        final_full_weights = pd.Series(0.0, index=expected_returns.index)
        final_full_weights.loc[valid_assets] = safe_weights
        
        # 7. Calculate final portfolio metrics on the FULL universe geometry
        # Note: (w @ R) and sqrt(w @ cov @ w)
        port_ret = (final_full_weights @ expected_returns) * 252
        port_vol = np.sqrt(final_full_weights @ cov_matrix.values @ final_full_weights) * np.sqrt(252)
        sharpe = port_ret / port_vol if port_vol > 0 else 0.0
        
        return {
            'weights': final_full_weights,
            'expected_return': port_ret,
            'volatility': port_vol,
            'sharpe_ratio': sharpe,
            'optimization_status': prob.status
        }
    
    def _calculate_cluster_allocation(self, weights: pd.Series, cluster_map: Dict[str, int]) -> Dict[int, float]:
        """Calculate allocation by cluster"""
        cluster_allocation = {}
        for symbol, weight in weights.items():
            cluster_id = cluster_map.get(symbol)
            if cluster_id is not None:
                cluster_allocation[cluster_id] = cluster_allocation.get(cluster_id, 0) + weight
        
        return cluster_allocation
    
    def generate_efficient_frontier(self,
                                  expected_returns: pd.Series,
                                  cov_matrix: pd.DataFrame,
                                  n_points: int = 50) -> Tuple[np.ndarray, np.ndarray, List[pd.Series]]:
        """Generate efficient frontier"""
        if not PYPFOPT_AVAILABLE:
            logger.warning("pypfopt not available, cannot generate efficient frontier")
            return np.array([]), np.array([]), []
        
        try:
            ef = EfficientFrontier(expected_returns, cov_matrix)
            
            # Get range of target returns
            min_ret = expected_returns.min() * 252
            max_ret = expected_returns.max() * 252
            target_returns = np.linspace(min_ret, max_ret, n_points)
            
            risks = []
            returns = []
            weights_list = []
            
            for target_return in target_returns:
                try:
                    ef_copy = EfficientFrontier(expected_returns, cov_matrix)
                    ef_copy.add_constraint(lambda w: w >= 0)
                    ef_copy.add_constraint(lambda w: w <= self.config.MAX_WEIGHT_PER_STOCK)
                    
                    weights = ef_copy.efficient_return(target_return / 252)
                    
                    weights_series = pd.Series(weights)
                    portfolio_return = (weights_series @ expected_returns) * 252
                    portfolio_vol = np.sqrt(weights_series @ cov_matrix @ weights_series) * np.sqrt(252)
                    
                    returns.append(portfolio_return)
                    risks.append(portfolio_vol)
                    weights_list.append(weights_series)
                    
                except:
                    continue
            
            return np.array(returns), np.array(risks), weights_list
            
        except Exception as e:
            logger.error(f"Error generating efficient frontier: {e}")
            return np.array([]), np.array([]), []

def test_strategy(self, method: str, returns_data: pd.DataFrame):
        """Test a single optimization strategy"""
        logger.info(f"Testing {method} optimization")
        try:
            # Pass the returns_data to the optimization method
            optimization_result = self.optimize_portfolio(
                price_data=self.price_data,
                returns_data=returns_data,
                method=method,
                cluster_data=self.cluster_assignments_df
            )
            # ... existing code ...
        except Exception as e:
            logger.error(f"Error in {method} optimization: {e}", exc_info=True)

    def run_all_tests(self):
        """Run tests for all optimization strategies"""
        # ... existing code ...
        
        # Load returns data once
        returns_path = os.path.join(self.config.get_market_processed_dir(self.market), 'returns_data.csv')
        if os.path.exists(returns_path):
            returns_data = pd.read_csv(returns_path, index_col=0, header=[0, 1])
            # Ensure the index is datetime
            returns_data.index = pd.to_datetime(returns_data.index)
        else:
            logger.error(f"Returns data not found at {returns_path}. Cannot run tests.")
            return

        # Pass returns_data to each test
        self.test_strategy('mean_variance', returns_data=returns_data)
        self.test_strategy('risk_parity', returns_data=returns_data)
        self.test_strategy('min_variance', returns_data=returns_data)
        self.test_strategy('max_sharpe', returns_data=returns_data)
        self.test_strategy('cluster_based', returns_data=returns_data)
    
    def main(market: str = 'US'):
        """Main function to run portfolio optimization"""
        logger.info(f"Starting portfolio optimization for {market} market")
        config = Config()
        
        try:
            # Load data
            processed_data_dir = config.get_market_data_dir(market, 'processed')
            results_dir = config.get_market_results_dir(market)

            processed_data_path = os.path.join(processed_data_dir, 'processed_stock_data.csv')
            cluster_data_path = os.path.join(results_dir, 'cluster_assignments_kmeans.csv')
            
            price_data = pd.read_csv(processed_data_path, index_col=0, header=[0, 1])
            price_data.index = pd.to_datetime(price_data.index)
            
            cluster_data = None
            if os.path.exists(cluster_data_path):
                cluster_data = pd.read_csv(cluster_data_path)
            
            logger.info("Data loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return
        
        # Initialize optimizer
        optimizer = PortfolioOptimizer(market=market)
        
        # Test different optimization methods
        methods = config.STRATEGIES
    
        results = {}
        
        # Load returns data once
        returns_path = os.path.join(processed_data_dir, 'returns_data.csv')
        if os.path.exists(returns_path):
            returns_data = pd.read_csv(returns_path, index_col=0, header=[0, 1])
            returns_data.index = pd.to_datetime(returns_data.index)
        else:
            logger.error(f"Returns data not found at {returns_path}. Cannot run optimization.")
            return

        for method in methods:
            logger.info(f"Testing {method} optimization")
            
            try:
                # Pass the returns_data to the optimization method
                optimization_result = optimizer.optimize_portfolio(
                    price_data=price_data,
                    returns_data=returns_data,
                    method=method,
                    cluster_data=cluster_data
                )
                
                results[method] = optimization_result
                
                # Print summary
                weights = optimization_result.get('weights')
                if weights is None or weights.empty:
                    logger.warning(f"No weights produced for {method}. Skipping summary.")
                    continue

                top_positions = weights.nlargest(5)
                
                print(f"\n{method.upper()} OPTIMIZATION:")
                print(f"Top 5 positions:")
                for symbol, weight in top_positions.items():
                    print(f"  {symbol}: {weight:.3f}")
                
                if 'sharpe_ratio' in optimization_result:
                    expected_return = optimization_result.get('expected_return', 'N/A')
                    volatility = optimization_result.get('volatility', 'N/A')
                    sharpe_ratio = optimization_result.get('sharpe_ratio', 'N/A')

                    # Check if the values are numeric before formatting
                    if isinstance(expected_return, (int, float)):
                        print(f"Expected Return: {expected_return:.3f}")
                    if isinstance(volatility, (int, float)):
                        print(f"Volatility: {volatility:.3f}")
                    if isinstance(sharpe_ratio, (int, float)):
                        print(f"Sharpe Ratio: {sharpe_ratio:.3f}")

        except Exception as e:
            logger.error(f"Error in {method} optimization: {e}", exc_info=True)
            continue
    
    # Save results
    for method, result in results.items():
        if 'weights' in result and not result['weights'].empty:
            weights_df = result['weights'].to_frame(name='weight')
            weights_df.to_csv(os.path.join(results_dir, f'portfolio_weights_{method}.csv'))
    
    logger.info("Portfolio optimization completed successfully!")