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

class ReturnForecaster:
    """Uses trained ML/DL models to forecast returns"""
    
    def __init__(self):
        self.config = Config()
        self.ml_models = {}
        self.dl_models = {}
        self.scalers = {}
    
    def load_trained_models(self, models_dir: str) -> bool:
        """Load previously trained models"""
        try:
            import pickle
            import tensorflow as tf
            
            # Load ML models
            for model_file in os.listdir(models_dir):
                if model_file.startswith('ml_model_') and model_file.endswith('.pkl'):
                    model_name = model_file.replace('ml_model_', '').replace('.pkl', '')
                    with open(os.path.join(models_dir, model_file), 'rb') as f:
                        self.ml_models[model_name] = pickle.load(f)
                
                # Load DL models
                elif model_file.startswith('dl_model_') and model_file.endswith('.h5'):
                    model_name = model_file.replace('dl_model_', '').replace('.h5', '')
                    self.dl_models[model_name] = tf.keras.models.load_model(
                        os.path.join(models_dir, model_file)
                    )
            
            logger.info(f"Loaded {len(self.ml_models)} ML models and {len(self.dl_models)} DL models")
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def forecast_returns(self, 
                        recent_data: pd.DataFrame,
                        horizon: int = 1,
                        method: str = 'ensemble') -> pd.Series:
        """
        Forecast stock returns using trained models
        
        Args:
            recent_data: Recent price/return data for prediction
            horizon: Forecast horizon in days
            method: 'ensemble', 'ml_only', 'dl_only', or specific model name
            
        Returns:
            Series with predicted returns for each stock
        """
        logger.info(f"Forecasting returns for {horizon} day(s) using {method}")
        
        if method == 'naive':
            # Simple historical mean as baseline
            return self._naive_forecast(recent_data, horizon)
        elif method == 'ensemble':
            return self._ensemble_forecast(recent_data, horizon)
        else:
            # Use specific model or method
            return self._model_forecast(recent_data, horizon, method)
    
    def _naive_forecast(self, data: pd.DataFrame, horizon: int) -> pd.Series:
        """Naive forecast using historical mean returns"""
        symbols = data.columns.get_level_values(0).unique()
        forecasts = {}
        
        for symbol in symbols:
            if (symbol, 'returns_1d') in data.columns:
                returns = data[symbol]['returns_1d'].dropna()
                # Use rolling average of recent returns
                forecast = returns.tail(20).mean() * horizon  # Adjust for horizon
                forecasts[symbol] = forecast
        
        return pd.Series(forecasts)
    
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
    
    def __init__(self):
        self.config = Config()
        self.forecaster = ReturnForecaster()
    
    def optimize_portfolio(self,
                          price_data: pd.DataFrame,
                          expected_returns: Optional[pd.Series] = None,
                          method: str = 'mean_variance',
                          constraints: Optional[Dict] = None,
                          cluster_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Optimize portfolio using specified method
        
        Args:
            price_data: Historical price data
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
        
        # Calculate returns
        returns_data = clean_prices.pct_change().dropna()
        
        # Get expected returns
        if expected_returns is None:
            expected_returns = self.forecaster.forecast_returns(price_data)
            # Align with available symbols
            expected_returns = expected_returns.reindex(symbols).fillna(0.08/252)  # Default daily return
        
        # Calculate covariance matrix
        cov_matrix = returns_data.cov()
        
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
        result['symbols'] = symbols
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
        problem.solve()
        
        if weights.value is None:
            logger.error("Optimization failed")
            return {'weights': pd.Series(index=expected_returns.index, data=0)}
        
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
        min_weight = constraints.get('min_weight', 0.01) if constraints else 0.01
        bounds = [(min_weight, max_weight) for _ in range(n_assets)]
        
        # Initial guess: equal weights
        x0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(
            risk_parity_objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list
        )
        
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
            cov_matrix = returns_data.cov()
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
            return {'weights': equal_weights}
    
    def _min_variance_optimization(self, 
                                 cov_matrix: pd.DataFrame,
                                 constraints: Optional[Dict] = None) -> Dict[str, Any]:
        """Minimum variance optimization"""
        if not PYPFOPT_AVAILABLE:
            return self._mean_variance_optimization(
                pd.Series(index=cov_matrix.index, data=0), 
                cov_matrix, 
                constraints
            )
        
        try:
            ef = EfficientFrontier(
                pd.Series(index=cov_matrix.index, data=0),  # Zero expected returns
                cov_matrix
            )
            
            # Apply constraints
            max_weight = constraints.get('max_weight', self.config.MAX_WEIGHT_PER_STOCK) if constraints else self.config.MAX_WEIGHT_PER_STOCK
            ef.add_constraint(lambda w: w <= max_weight)
            ef.add_constraint(lambda w: w >= 0)
            
            weights = ef.min_volatility()
            
            optimal_weights = pd.Series(weights)
            portfolio_vol = np.sqrt(optimal_weights @ cov_matrix @ optimal_weights) * np.sqrt(252)
            
            return {
                'weights': optimal_weights,
                'volatility': portfolio_vol,
                'optimization_status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Min variance optimization failed: {e}")
            return self._mean_variance_optimization(
                pd.Series(index=cov_matrix.index, data=0), 
                cov_matrix, 
                constraints
            )
    
    def _max_sharpe_optimization(self, 
                               expected_returns: pd.Series,
                               cov_matrix: pd.DataFrame,
                               constraints: Optional[Dict] = None) -> Dict[str, Any]:
        """Maximum Sharpe ratio optimization"""
        if not PYPFOPT_AVAILABLE:
            return self._mean_variance_optimization(expected_returns, cov_matrix, constraints)
        
        try:
            ef = EfficientFrontier(expected_returns, cov_matrix)
            
            # Apply constraints
            max_weight = constraints.get('max_weight', self.config.MAX_WEIGHT_PER_STOCK) if constraints else self.config.MAX_WEIGHT_PER_STOCK
            ef.add_constraint(lambda w: w <= max_weight)
            ef.add_constraint(lambda w: w >= 0)
            
            weights = ef.max_sharpe()
            
            optimal_weights = pd.Series(weights)
            
            # Calculate metrics
            portfolio_return = (optimal_weights @ expected_returns) * 252
            portfolio_vol = np.sqrt(optimal_weights @ cov_matrix @ optimal_weights) * np.sqrt(252)
            sharpe_ratio = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0
            
            return {
                'weights': optimal_weights,
                'expected_return': portfolio_return,
                'volatility': portfolio_vol,
                'sharpe_ratio': sharpe_ratio,
                'optimization_status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Max Sharpe optimization failed: {e}")
            return self._mean_variance_optimization(expected_returns, cov_matrix, constraints)
    
    def _cluster_based_optimization(self,
                                  expected_returns: pd.Series,
                                  cov_matrix: pd.DataFrame,
                                  cluster_data: Optional[pd.DataFrame],
                                  constraints: Optional[Dict] = None) -> Dict[str, Any]:
        """Cluster-based optimization with diversification constraints"""
        if cluster_data is None:
            logger.warning("No cluster data provided, using standard mean-variance")
            return self._mean_variance_optimization(expected_returns, cov_matrix, constraints)
        
        # Create cluster mapping
        cluster_map = {}
        for _, row in cluster_data.iterrows():
            if row['symbol'] in expected_returns.index:
                cluster_map[row['symbol']] = row['cluster']
        
        n_assets = len(expected_returns)
        weights = cp.Variable(n_assets)
        
        # Objective: Maximize return - risk penalty
        risk_aversion = constraints.get('risk_aversion', 1.0) if constraints else 1.0
        portfolio_return = expected_returns.values @ weights
        portfolio_risk = cp.quad_form(weights, cov_matrix.values)
        
        objective = cp.Maximize(portfolio_return - risk_aversion * portfolio_risk)
        
        # Standard constraints
        constraint_list = [cp.sum(weights) == 1]
        
        max_weight = constraints.get('max_weight', self.config.MAX_WEIGHT_PER_STOCK) if constraints else self.config.MAX_WEIGHT_PER_STOCK
        constraint_list.append(weights >= 0)
        constraint_list.append(weights <= max_weight)
        
        # Cluster diversification constraints
        max_cluster_weight = constraints.get('max_cluster_weight', 0.4) if constraints else 0.4
        
        clusters = set(cluster_map.values())
        for cluster_id in clusters:
            cluster_indices = [i for i, symbol in enumerate(expected_returns.index) 
                             if cluster_map.get(symbol) == cluster_id]
            if cluster_indices:
                constraint_list.append(cp.sum(weights[cluster_indices]) <= max_cluster_weight)
        
        # Solve
        problem = cp.Problem(objective, constraint_list)
        problem.solve()
        
        if weights.value is None:
            logger.error("Cluster-based optimization failed, using mean-variance")
            return self._mean_variance_optimization(expected_returns, cov_matrix, constraints)
        
        optimal_weights = pd.Series(index=expected_returns.index, data=weights.value)
        
        # Calculate metrics
        portfolio_return = (optimal_weights @ expected_returns) * 252
        portfolio_vol = np.sqrt(optimal_weights @ cov_matrix @ optimal_weights) * np.sqrt(252)
        sharpe_ratio = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0
        
        return {
            'weights': optimal_weights,
            'expected_return': portfolio_return,
            'volatility': portfolio_vol,
            'sharpe_ratio': sharpe_ratio,
            'cluster_allocation': self._calculate_cluster_allocation(optimal_weights, cluster_map),
            'optimization_status': problem.status
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

def main():
    """Main function to run portfolio optimization"""
    logger.info("Starting portfolio optimization")
    
    try:
        # Load data
        processed_data_path = os.path.join(Config.PROCESSED_DATA_DIR, 'processed_stock_data.csv')
        cluster_data_path = os.path.join(Config.RESULTS_DIR, 'cluster_assignments_kmeans.csv')
        
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
    optimizer = PortfolioOptimizer()
    
    # Test different optimization methods
    methods = ['mean_variance', 'risk_parity', 'min_variance', 'max_sharpe']
    if cluster_data is not None:
        methods.append('cluster_based')
    
    results = {}
    
    for method in methods:
        logger.info(f"Testing {method} optimization")
        
        try:
            result = optimizer.optimize_portfolio(
                price_data=price_data,
                method=method,
                cluster_data=cluster_data
            )
            
            results[method] = result
            
            # Print summary
            weights = result['weights']
            top_positions = weights.nlargest(5)
            
            print(f"\n{method.upper()} OPTIMIZATION:")
            print(f"Top 5 positions:")
            for symbol, weight in top_positions.items():
                print(f"  {symbol}: {weight:.3f}")
            
            if 'sharpe_ratio' in result:
                print(f"Expected Return: {result['expected_return']:.3f}")
                print(f"Volatility: {result['volatility']:.3f}")
                print(f"Sharpe Ratio: {result['sharpe_ratio']:.3f}")
            
        except Exception as e:
            logger.error(f"Error in {method} optimization: {e}")
            continue
    
    # Save results
    results_dir = Config.RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)
    
    for method, result in results.items():
        # Save weights
        weights_path = os.path.join(results_dir, f'portfolio_weights_{method}.csv')
        result['weights'].to_csv(weights_path, header=['weight'])
        
        # Save full results (excluding non-serializable objects)
        result_summary = {
            'method': method,
            'expected_return': result.get('expected_return', 0),
            'volatility': result.get('volatility', 0),
            'sharpe_ratio': result.get('sharpe_ratio', 0),
            'optimization_status': result.get('optimization_status', 'unknown')
        }
        
        import json
        summary_path = os.path.join(results_dir, f'optimization_summary_{method}.json')
        with open(summary_path, 'w') as f:
            json.dump(result_summary, f, indent=2)
    
    logger.info("Portfolio optimization completed successfully!")

if __name__ == "__main__":
    main()