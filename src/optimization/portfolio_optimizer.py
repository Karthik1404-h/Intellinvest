"""
Portfolio optimization algorithms enhanced with machine learning predictions
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import cvxpy as cp
from scipy.optimize import minimize, differential_evolution
from sklearn.covariance import LedoitWolf
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
                        method: str = 'advanced') -> pd.Series:
        """
        Forecast stock returns using trained models
        
        Args:
            recent_data: Recent price/return data for prediction
            horizon: Forecast horizon in days
            method: 'advanced' (default), 'ensemble', 'naive', or specific model name
            
        Returns:
            Series with predicted returns for each stock
        """
        logger.info(f"Forecasting returns for {horizon} day(s) using {method}")
        
        if method == 'naive':
            return self._naive_forecast(recent_data, horizon)
        elif method == 'advanced':
            return self._advanced_forecast(recent_data, horizon)
        elif method == 'ensemble':
            return self._advanced_forecast(recent_data, horizon)
        else:
            return self._advanced_forecast(recent_data, horizon)
    
    def _advanced_forecast(self, data: pd.DataFrame, horizon: int) -> pd.Series:
        """
        Multi-signal return forecast combining four academically-validated signals:
        
        1. 12-1 month momentum  : cumulative return from t-252 to t-21 days
           (Jegadeesh & Titman 1993 — most robust price momentum signal)
        2. EWMA mean return     : 63-day exponentially-weighted mean (annualised)
        3. Short-term reversal  : -1 × 1-month return (contrarian correction)
        4. 52-week high proximity: price / max(price, last 252d) — stocks near
           52-week highs reliably continue outperforming (George & Hwang 2004)
        
        All signals cross-sectionally z-scored before blending.
        """
        symbols = data.columns.get_level_values(0).unique()
        
        mom_dict   = {}   # 12-1 month momentum
        ewm_dict   = {}   # 63-day EWMA mean return (annualised)
        strev_dict = {}   # 1-month short-term reversal
        h52_dict   = {}   # 52-week high proximity

        for symbol in symbols:
            try:
                if (symbol, 'Close') not in data.columns:
                    continue
                prices = data[symbol]['Close'].dropna()
                if len(prices) < 260:
                    continue

                p_now = float(prices.iloc[-1])
                p_21  = float(prices.iloc[-21])
                p_252 = float(prices.iloc[-252])

                # Signal 1: 12-1 month momentum
                mom_dict[symbol] = (p_21 / p_252) - 1.0

                # Signal 2: 63-day EWMA mean daily return, annualised
                rets = prices.pct_change().dropna()
                ewm_mean_daily = rets.ewm(span=63, min_periods=21).mean().iloc[-1]
                ewm_dict[symbol] = ewm_mean_daily * 252

                # Signal 3: 1-month short-term reversal (contrarian)
                strev_dict[symbol] = (p_now / p_21) - 1.0

                # Signal 4: 52-week high proximity (higher = closer to 52w high = bullish)
                high_252 = float(prices.iloc[-252:].max())
                h52_dict[symbol] = p_now / high_252 if high_252 > 0 else 0.0

            except Exception:
                continue

        if len(mom_dict) < 3:
            return self._naive_forecast(data, horizon)

        mom_s   = pd.Series(mom_dict)
        ewm_s   = pd.Series(ewm_dict)
        strev_s = pd.Series(strev_dict)
        h52_s   = pd.Series(h52_dict)

        common = mom_s.index.intersection(ewm_s.index) \
                            .intersection(strev_s.index) \
                            .intersection(h52_s.index)
        mom_s   = mom_s.reindex(common)
        ewm_s   = ewm_s.reindex(common)
        strev_s = strev_s.reindex(common)
        h52_s   = h52_s.reindex(common)

        def _zscore(s: pd.Series) -> pd.Series:
            std = s.std()
            return (s - s.mean()) / std if std > 1e-8 else pd.Series(0.0, index=s.index)

        mom_z   = _zscore(mom_s)
        ewm_z   = _zscore(ewm_s)
        strev_z = _zscore(strev_s)
        h52_z   = _zscore(h52_s)

        # Blend: 35% momentum + 30% EWMA + 20% 52wk-high + -15% reversal
        # 52-week high gets weight because it captures breakout continuation
        combined_z = (0.35 * mom_z + 0.30 * ewm_z
                      + 0.20 * h52_z - 0.15 * strev_z)

        # Annualised return estimate: centre 8%, each z unit = 6% pa
        # (wider spread = more differentiation between stocks = optimizer has more signal)
        forecast = 0.08 + combined_z * 0.06

        forecast_daily = forecast / 252.0
        logger.info(
            f"Advanced forecast (4-factor): "
            f"range [{forecast.min():.4f}, {forecast.max():.4f}] ann."
        )
        return forecast_daily.reindex(common)

    def _naive_forecast(self, data: pd.DataFrame, horizon: int) -> pd.Series:
        """Naive forecast using historical mean returns — kept as fallback"""
        symbols = data.columns.get_level_values(0).unique()
        forecasts = {}
        
        for symbol in symbols:
            try:
                if (symbol, 'returns_1d') in data.columns:
                    returns = data[symbol]['returns_1d'].dropna()
                elif (symbol, 'Close') in data.columns:
                    prices = data[symbol]['Close'].dropna()
                    returns = prices.pct_change().dropna()
                else:
                    continue
                
                if len(returns) >= 20:
                    forecast = returns.tail(60).mean() * horizon
                    forecasts[symbol] = forecast
            except Exception as e:
                logger.warning(f"Error forecasting {symbol}: {e}")
                continue
        
        forecast_series = pd.Series(forecasts)
        return forecast_series


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
        
        # ── Ledoit-Wolf shrinkage covariance ──────────────────────────────────
        # LW shrinkage produces a better-conditioned, less noisy covariance
        # matrix than the raw sample estimator, especially when n_assets is
        # large relative to the number of observations.
        try:
            lw = LedoitWolf(assume_centered=False)
            lw.fit(returns_data.values)
            cov_matrix = pd.DataFrame(
                lw.covariance_,
                index=returns_data.columns,
                columns=returns_data.columns
            )
            logger.debug(f"Ledoit-Wolf covariance: shrinkage={lw.shrinkage_:.4f}")
        except Exception as e:
            logger.warning(f"Ledoit-Wolf failed ({e}), falling back to sample cov")
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
        elif method == 'momentum_filter':
            result = self._momentum_filter_optimization(expected_returns, cov_matrix, returns_data, constraints)
        elif method == 'black_litterman':
            result = self._black_litterman_optimization(expected_returns, cov_matrix, returns_data, constraints)
        elif method == 'concentrated_momentum':
            result = self._concentrated_momentum_optimization(expected_returns, cov_matrix, returns_data, constraints)
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
        """
        Modern Portfolio Theory optimization (Markowitz mean-variance).
        
        Operates in annualised units throughout so that the risk-aversion
        parameter λ has an intuitive interpretation: λ = 2 means the investor
        gives up 1 % of expected return to save 0.5 % of variance (a standard
        institutional risk appetite).
        """
        n_assets = len(expected_returns)
        
        # Annualise inputs so λ is scale-independent
        mu_ann = expected_returns.values * 252          # annualised returns
        cov_ann = cov_matrix.values * 252               # annualised covariance
        
        # Decision variables
        weights = cp.Variable(n_assets)
        
        # λ = 2 is a well-calibrated risk-aversion for long-only equity
        risk_aversion = constraints.get('risk_aversion', 2.0) if constraints else 2.0
        portfolio_return = mu_ann @ weights
        portfolio_risk = cp.quad_form(weights, cov_ann)
        
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
        problem.solve(solver=cp.CLARABEL)
        
        if weights.value is None:
            # Try fallback solver
            problem.solve(solver=cp.SCS)
        
        if weights.value is None:
            logger.error("Mean-variance optimization failed, using equal weights")
            ew = np.full(n_assets, 1.0 / n_assets)
            return {'weights': pd.Series(index=expected_returns.index, data=ew)}
        
        optimal_weights = pd.Series(index=expected_returns.index, data=weights.value)
        optimal_weights = optimal_weights.clip(lower=0)
        optimal_weights /= optimal_weights.sum()
        
        # Calculate portfolio metrics (annualised)
        port_return = float(optimal_weights @ expected_returns) * 252
        port_vol = float(np.sqrt(optimal_weights @ cov_matrix @ optimal_weights)) * np.sqrt(252)
        sharpe_ratio = (port_return - 0.02) / port_vol if port_vol > 0 else 0
        
        return {
            'weights': optimal_weights,
            'expected_return': port_return,
            'volatility': port_vol,
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
        """Minimum variance optimization using scipy"""
        n_assets = len(cov_matrix)
        
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
            logger.warning(f"Min variance optimization warning: {result.message}")
        
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
        """
        Maximum Sharpe ratio optimization using multi-start SLSQP.
        
        Running 20 random starting points greatly reduces the chance of
        getting stuck in a sub-optimal local minimum.
        The Sharpe formula includes the risk-free rate (2 % pa).
        """
        n_assets = len(expected_returns)
        rf_daily = 0.02 / 252  # risk-free rate per day
        
        mu_ann = expected_returns.values * 252
        cov_ann = cov_matrix.values * 252

        def negative_sharpe(weights):
            port_return = float(mu_ann @ weights)
            port_vol = float(np.sqrt(weights @ cov_ann @ weights))
            if port_vol < 1e-10:
                return 1e10
            return -(port_return - 0.02) / port_vol

        max_weight = constraints.get('max_weight', self.config.MAX_WEIGHT_PER_STOCK) if constraints else self.config.MAX_WEIGHT_PER_STOCK
        min_weight = constraints.get('min_weight', self.config.MIN_WEIGHT) if constraints else self.config.MIN_WEIGHT

        cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1},)
        bounds = tuple((min_weight, max_weight) for _ in range(n_assets))

        best_result = None
        best_sharpe = -np.inf

        rng = np.random.default_rng(42)

        # --- starting point 0: proportional to expected returns
        starting_points = []
        x0 = np.abs(expected_returns.values) + 1e-8
        x0 = np.clip(x0 / x0.sum(), min_weight, max_weight)
        x0 /= x0.sum()
        starting_points.append(x0)

        # --- starting points 1-19: random Dirichlet draws
        for _ in range(19):
            r = rng.dirichlet(np.ones(n_assets))
            r = np.clip(r, min_weight, max_weight)
            r /= r.sum()
            starting_points.append(r)

        for x0 in starting_points:
            res = minimize(
                negative_sharpe, x0,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 2000, 'ftol': 1e-10}
            )
            if res.fun < -best_sharpe:
                best_result = res
                best_sharpe = -res.fun

        if best_result is None:
            logger.error("Max Sharpe optimization failed entirely, using equal weights")
            ew = np.full(n_assets, 1.0 / n_assets)
            return {'weights': pd.Series(index=expected_returns.index, data=ew)}

        optimal_weights = pd.Series(index=expected_returns.index, data=best_result.x)
        optimal_weights = optimal_weights.clip(lower=0)
        optimal_weights /= optimal_weights.sum()

        port_return = float(optimal_weights @ expected_returns) * 252
        port_vol = float(np.sqrt(optimal_weights @ cov_matrix @ optimal_weights)) * np.sqrt(252)
        sharpe_ratio = (port_return - 0.02) / port_vol if port_vol > 0 else 0

        logger.info(f"Max Sharpe (best of 20 starts): Sharpe={sharpe_ratio:.4f}")
        return {
            'weights': optimal_weights,
            'expected_return': port_return,
            'volatility': port_vol,
            'sharpe_ratio': sharpe_ratio,
            'optimization_status': 'success' if best_result.success else 'warning'
        }
    
    def _cluster_based_optimization(self,
                                  expected_returns: pd.Series,
                                  cov_matrix: pd.DataFrame,
                                  cluster_data: Optional[pd.DataFrame],
                                  constraints: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Cluster-aware mean-variance with two improvements over plain MV:
        
        1. Hard cluster caps set to min(equal_share * 1.5, 0.30) so that no
           single sector/cluster can dominate even if it has the best forecast.
        2. Within each cluster the expected return is boosted by a small
           intra-cluster momentum tilt: the top-momentum stock in each cluster
           gets a +0.5 % pa premium, the bottom-momentum one a −0.5 % pa penalty.
           This ensures the optimizer genuinely differs from plain mean-variance.
        """
        if cluster_data is None:
            logger.warning("No cluster data provided, using standard mean-variance")
            return self._mean_variance_optimization(expected_returns, cov_matrix, constraints)
        
        # ── Build cluster map ─────────────────────────────────────────────
        cluster_map = {}
        for _, row in cluster_data.iterrows():
            if row['symbol'] in expected_returns.index:
                cluster_map[row['symbol']] = row['cluster']
        
        clusters = sorted(set(cluster_map.values()))
        n_clusters = len(clusters)
        n_assets = len(expected_returns)

        # ── Intra-cluster momentum tilt ───────────────────────────────────
        # Tilt expected returns slightly within each cluster so the cluster
        # strategy has a genuinely different signal from plain mean-variance.
        mu_tilted = expected_returns.copy()
        tilt = 0.005 / 252  # 0.5 % pa premium/penalty expressed as daily return
        for cid in clusters:
            members = [s for s in expected_returns.index if cluster_map.get(s) == cid]
            if len(members) < 2:
                continue
            cluster_mu = expected_returns[members].rank(ascending=True)
            # Best-ranked in cluster: +tilt; Worst-ranked: -tilt; rest linear
            mn, mx = cluster_mu.min(), cluster_mu.max()
            if mx > mn:
                normalized = (cluster_mu - mn) / (mx - mn)  # 0 (worst) → 1 (best)
                mu_tilted[members] += tilt * (normalized * 2 - 1)

        # ── Build CVXPY problem (annualised) ──────────────────────────────
        mu_ann  = mu_tilted.values * 252
        cov_ann = cov_matrix.values * 252
        
        weights = cp.Variable(n_assets)
        risk_aversion = constraints.get('risk_aversion', 2.0) if constraints else 2.0
        
        objective = cp.Maximize(mu_ann @ weights - risk_aversion * cp.quad_form(weights, cov_ann))
        
        max_weight = constraints.get('max_weight', self.config.MAX_WEIGHT_PER_STOCK) if constraints else self.config.MAX_WEIGHT_PER_STOCK
        
        constraint_list = [
            cp.sum(weights) == 1,
            weights >= 0,
            weights <= max_weight,
        ]
        
        # Tighter cluster cap: each cluster gets at most 1.5× its equal share
        # or 30 %, whichever is smaller — ensures genuine cross-cluster spread.
        equal_share = 1.0 / n_clusters if n_clusters > 0 else 0.4
        max_cluster_weight = min(equal_share * 1.5, 0.30)
        max_cluster_weight = constraints.get('max_cluster_weight', max_cluster_weight) if constraints else max_cluster_weight
        
        for cid in clusters:
            idx = [i for i, s in enumerate(expected_returns.index) if cluster_map.get(s) == cid]
            if idx:
                constraint_list.append(cp.sum(weights[idx]) <= max_cluster_weight)
        
        problem = cp.Problem(objective, constraint_list)
        problem.solve(solver=cp.CLARABEL)
        
        if weights.value is None:
            problem.solve(solver=cp.SCS)
        
        if weights.value is None:
            logger.error("Cluster-based optimization failed, falling back to mean-variance")
            return self._mean_variance_optimization(expected_returns, cov_matrix, constraints)
        
        optimal_weights = pd.Series(index=expected_returns.index, data=weights.value)
        optimal_weights = optimal_weights.clip(lower=0)
        optimal_weights /= optimal_weights.sum()
        
        port_return = float(optimal_weights @ expected_returns) * 252
        port_vol = float(np.sqrt(optimal_weights @ cov_matrix @ optimal_weights)) * np.sqrt(252)
        sharpe_ratio = (port_return - 0.02) / port_vol if port_vol > 0 else 0
        
        return {
            'weights': optimal_weights,
            'expected_return': port_return,
            'volatility': port_vol,
            'sharpe_ratio': sharpe_ratio,
            'cluster_allocation': self._calculate_cluster_allocation(optimal_weights, cluster_map),
            'optimization_status': problem.status
        }

    def _momentum_filter_optimization(self,
                                      expected_returns: pd.Series,
                                      cov_matrix: pd.DataFrame,
                                      returns_data: pd.DataFrame,
                                      constraints: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Momentum-filtered minimum-variance (academic momentum strategy).
        
        Step 1: Rank all stocks by 12-1 month momentum signal embedded in
                expected_returns (which are already momentum-tilted by
                _advanced_forecast).  Select top 50 %.
        Step 2: Apply minimum-variance optimisation on the selected subset.
        
        Literature: Blitz & Van Vliet (2007) — combining momentum with
        low-volatility gives better risk-adjusted returns than either alone.
        """
        n_total = len(expected_returns)
        
        # ── Step 1: momentum filter ───────────────────────────────────────
        # expected_returns are already momentum + EWMA blended (annualised daily).
        # Rank and keep top 50 % (or at least 5 stocks).
        n_keep = max(5, int(n_total * 0.50))
        top_symbols = expected_returns.nlargest(n_keep).index.tolist()
        
        mu_sub  = expected_returns[top_symbols]
        cov_sub = cov_matrix.loc[top_symbols, top_symbols]
        
        # ── Step 2: minimum variance on selected subset ───────────────────
        n_sub = len(top_symbols)
        
        max_weight = constraints.get('max_weight', self.config.MAX_WEIGHT_PER_STOCK) if constraints else self.config.MAX_WEIGHT_PER_STOCK
        min_weight = constraints.get('min_weight', self.config.MIN_WEIGHT) if constraints else self.config.MIN_WEIGHT
        
        def portfolio_variance(w):
            return float(w @ cov_sub.values @ w)
        
        cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1},)
        bounds = tuple((max(min_weight, 0), max_weight) for _ in range(n_sub))
        x0 = np.full(n_sub, 1.0 / n_sub)
        x0 = np.clip(x0, max(min_weight, 0), max_weight)
        x0 /= x0.sum()
        
        result = minimize(portfolio_variance, x0, method='SLSQP',
                          bounds=bounds, constraints=cons,
                          options={'maxiter': 2000, 'ftol': 1e-10})
        
        if not result.success:
            logger.warning(f"Momentum-filter min-var: {result.message}")
        
        # Build full-universe weight series (zeros for excluded stocks)
        sub_weights = pd.Series(index=top_symbols, data=result.x)
        sub_weights = sub_weights.clip(lower=0)
        sub_weights /= sub_weights.sum()
        
        optimal_weights = pd.Series(0.0, index=expected_returns.index)
        optimal_weights[top_symbols] = sub_weights
        
        port_return = float(optimal_weights @ expected_returns) * 252
        port_vol = float(np.sqrt(optimal_weights @ cov_matrix @ optimal_weights)) * np.sqrt(252)
        sharpe_ratio = (port_return - 0.02) / port_vol if port_vol > 0 else 0
        
        logger.info(
            f"Momentum-filter: {n_keep}/{n_total} stocks selected, "
            f"Sharpe={sharpe_ratio:.4f}"
        )
        return {
            'weights': optimal_weights,
            'expected_return': port_return,
            'volatility': port_vol,
            'sharpe_ratio': sharpe_ratio,
            'n_selected': n_keep,
            'selected_symbols': top_symbols,
            'optimization_status': 'success' if result.success else 'warning'
        }

    
    def _black_litterman_optimization(self,
                                       expected_returns: pd.Series,
                                       cov_matrix: pd.DataFrame,
                                       returns_data: pd.DataFrame,
                                       constraints: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Black-Litterman (BL) portfolio optimization.

        BL starts from the market-implied equilibrium returns (reverse-engineered
        from market cap weights) and then blends in our factor signal as
        'investor views', producing a posterior return estimate that is both
        well-diversified AND tilted toward high-conviction factor stocks.

        Why this beats passive cap-weighted:
        • The BL prior IS the cap-weighted portfolio, so it never strays far
          from the market.
        • Our factor views (momentum + EWMA + 52w high) tilt away from the prior
          where there is strong signal, compounding the factor premium on top of
          the general market return.

        Algorithm (He & Litterman 1999 / Idzorek 2005):
        1. Compute market-cap proxy weights w_mkt = 1/σ_i (inverse-vol proxy).
        2. Implied equilibrium excess returns: π = λ Σ w_mkt  (λ = 2.5).
        3. Views: one absolute view per stock, Q_i = composite factor score (ann.)
           P = I  (identity — we have a view on every stock).
        4. Uncertainty: Ω = diag((1-|z_i|/z_max)^2 * τ Σ_ii)
           High-conviction stocks (large |z|) get smaller uncertainty omega.
        5. Posterior: μ* = [(τΣ)⁻¹ + PᵀΩ⁻¹P]⁻¹ [(τΣ)⁻¹π + PᵀΩ⁻¹Q]
        6. Run multi-start max-Sharpe optimisation on μ*.
        """
        n = len(expected_returns)
        symbols = expected_returns.index

        # ── 1. Market-cap proxy weights (price-proxy, same logic as passive benchmark)
        # Weights proportional to recent price level = (1 + cumulative return)
        # This matches what the passive cap-weighted benchmark does, so BL starts
        # from the same point and our views simply tilt away from it.
        try:
            cum_ret = (1.0 + returns_data[symbols.tolist()]).cumprod()
            # Use the final row (most recent) price level
            price_levels = cum_ret.iloc[-1].values.astype(float)
            price_levels = np.where(price_levels <= 0, 1e-8, price_levels)
            w_mkt = price_levels / price_levels.sum()
        except Exception:
            # Fall back to inverse-vol if price computation fails
            vols = np.sqrt(np.diag(cov_matrix.values))
            vols = np.where(vols < 1e-8, 1e-8, vols)
            w_mkt = (1.0 / vols)
            w_mkt /= w_mkt.sum()

        # ── 2. Implied equilibrium excess returns (annualised) ────────────────
        lam = 2.5  # risk-aversion coefficient (standard BL literature value)
        cov_ann = cov_matrix.values * 252
        pi = lam * (cov_ann @ w_mkt)          # shape (n,)

        # ── 3. Views — our 4-factor score as absolute return views ────────────
        # expected_returns is already our composite 4-factor signal (daily scale)
        Q = expected_returns.values * 252      # annualised views vector (n,)
        P = np.eye(n)                          # one view per asset

        # ── 4. View uncertainty Ω ─────────────────────────────────────────────
        # τ = 1/T is the standard scaling (T = number of obs).
        T = max(len(returns_data), 1)
        tau = 1.0 / T

        # Relative confidence: normalise combined-z to [0,1]
        # large |z| → small omega (high confidence in view)
        z_raw = Q - pi                         # view excess over equilibrium
        z_norm = z_raw / (np.abs(z_raw).max() + 1e-8)   # ∈ [-1, 1]
        confidence = z_norm ** 2              # ∈ [0, 1]; larger → higher confidence

        # Ω_ii = (1 - confidence_i) * τ * Σ_ii  — lower for high-confidence views
        omega_diag = (1.0 - confidence) * tau * np.diag(cov_ann)
        omega_diag = np.where(omega_diag < 1e-10, 1e-10, omega_diag)
        Omega = np.diag(omega_diag)

        # ── 5. BL posterior expected returns ─────────────────────────────────
        try:
            tau_cov = tau * cov_ann
            tau_cov_inv = np.linalg.inv(tau_cov + np.eye(n) * 1e-8)
            omega_inv = np.diag(1.0 / omega_diag)

            A = tau_cov_inv + P.T @ omega_inv @ P
            b = tau_cov_inv @ pi + P.T @ omega_inv @ Q

            mu_bl = np.linalg.solve(A + np.eye(n) * 1e-8, b)
        except np.linalg.LinAlgError:
            logger.warning("BL matrix inversion failed, using pure factor forecast")
            mu_bl = Q

        mu_bl_series = pd.Series(mu_bl / 252.0, index=symbols)  # back to daily scale

        logger.info(
            f"BL posterior: range [{mu_bl.min():.4f}, {mu_bl.max():.4f}] ann., "
            f"vs equilibrium [{pi.min():.4f}, {pi.max():.4f}]"
        )

        # ── 6. Multi-start max-Sharpe on posterior returns ───────────────────
        max_weight = (constraints.get('max_weight', 0.35)
                      if constraints else 0.35)  # allow 35% — BL naturally anchors diversification
        min_weight = constraints.get('min_weight', self.config.MIN_WEIGHT) if constraints else self.config.MIN_WEIGHT

        return self._max_sharpe_optimization(
            mu_bl_series, cov_matrix,
            {'max_weight': max_weight, 'min_weight': min_weight}
        )

    def _concentrated_momentum_optimization(self,
                                             expected_returns: pd.Series,
                                             cov_matrix: pd.DataFrame,
                                             returns_data: pd.DataFrame,
                                             constraints: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Concentrated momentum — a high-conviction factor strategy.

        Philosophy: passive cap-weighted wins partly because it inadvertently
        concentrates in momentum winners.  This strategy explicitly:
        1. Selects the TOP 30 % of stocks by composite factor score.
        2. Runs a multi-start MAX-SHARPE (not min-var) on the subset.
        3. Uses a 40 % weight cap (higher than other strategies) to allow
           meaningful concentration while remaining investable.

        The combination of a tighter universe + higher weight cap + max-Sharpe
        objective is designed to beat cap-weighted on a Sharpe basis.
        """
        n_total = len(expected_returns)
        n_keep = max(5, int(n_total * 0.25))   # top 25 % — tighter than 30 % for more conviction
        top_symbols = expected_returns.nlargest(n_keep).index.tolist()

        mu_sub  = expected_returns[top_symbols]
        cov_sub = cov_matrix.loc[top_symbols, top_symbols]

        # Higher weight cap: 50 % to allow real concentration in single winners
        # (cap-weighted can hit 50-60%+ in the top stock; we match that ability)
        max_weight = constraints.get('max_weight', 0.50) if constraints else 0.50
        min_weight = constraints.get('min_weight', self.config.MIN_WEIGHT) if constraints else self.config.MIN_WEIGHT

        result_sub = self._max_sharpe_optimization(
            mu_sub, cov_sub,
            {'max_weight': max_weight, 'min_weight': min_weight}
        )

        sub_weights = result_sub['weights'].clip(lower=0)
        sub_weights /= sub_weights.sum()

        optimal_weights = pd.Series(0.0, index=expected_returns.index)
        optimal_weights[top_symbols] = sub_weights

        port_return = float(optimal_weights @ expected_returns) * 252
        port_vol = float(np.sqrt(optimal_weights @ cov_matrix @ optimal_weights)) * np.sqrt(252)
        sharpe_ratio = (port_return - 0.02) / port_vol if port_vol > 0 else 0

        logger.info(
            f"Concentrated momentum: {n_keep}/{n_total} stocks, "
            f"max_w={max_weight:.0%}, Sharpe={sharpe_ratio:.4f}"
        )
        return {
            'weights': optimal_weights,
            'expected_return': port_return,
            'volatility': port_vol,
            'sharpe_ratio': sharpe_ratio,
            'n_selected': n_keep,
            'selected_symbols': top_symbols,
            'optimization_status': result_sub.get('optimization_status', 'success')
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