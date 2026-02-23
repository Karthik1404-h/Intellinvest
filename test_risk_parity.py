#!/usr/bin/env python3
"""
Test Risk Parity optimization to debug the issue
"""
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import os

from config import Config

def test_risk_parity():
    config = Config()
    
    # Load returns data
    returns_path = os.path.join(config.PROCESSED_DATA_DIR, 'returns_data.csv')
    returns_df = pd.read_csv(returns_path, index_col=0, parse_dates=True)
    
    print(f"Returns data shape: {returns_df.shape}")
    print(f"Number of stocks: {len(returns_df.columns)}")
    
    # Calculate covariance matrix
    cov_matrix = returns_df.cov() * 252  # Annualized
    
    print(f"\nCovariance matrix shape: {cov_matrix.shape}")
    print(f"Volatilities (annualized):")
    vols = np.sqrt(np.diag(cov_matrix))
    vol_series = pd.Series(vols, index=cov_matrix.index)
    print(vol_series.sort_values(ascending=False).head(10))
    
    n_assets = len(cov_matrix)
    
    def risk_parity_objective(weights):
        """Objective function for risk parity"""
        weights = np.array(weights)
        portfolio_vol = np.sqrt(weights @ cov_matrix.values @ weights)
        
        if portfolio_vol < 1e-10:
            return 1e10
        
        # Marginal risk contributions
        marginal_contribs = (cov_matrix.values @ weights) / portfolio_vol
        risk_contribs = weights * marginal_contribs
        
        # Target: equal risk contributions
        target_risk = portfolio_vol / n_assets
        deviations = risk_contribs - target_risk
        
        return np.sum(deviations ** 2)
    
    # Test equal weights
    equal_weights = np.ones(n_assets) / n_assets
    obj_equal = risk_parity_objective(equal_weights)
    print(f"\nObjective value with equal weights: {obj_equal}")
    
    # Test inverse volatility weights
    inv_vol_weights = 1 / vols
    inv_vol_weights = inv_vol_weights / inv_vol_weights.sum()
    obj_inv_vol = risk_parity_objective(inv_vol_weights)
    print(f"Objective value with inverse volatility weights: {obj_inv_vol}")
    
   # Constraints
    constraints_list = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    ]
    
    # Bounds
    bounds = [(0.0, 0.1) for _ in range(n_assets)]
    
    # Try different initial guesses
    print("\n" + "="*60)
    print("Testing different optimization approaches:")
    print("="*60)
    
    # 1. Equal weights initial guess
    x0_equal = equal_weights
    result_equal = minimize(
        risk_parity_objective,
        x0_equal,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints_list,
        options={'ftol': 1e-9, 'maxiter': 1000}
    )
    print(f"\n1. Starting from equal weights:")
    print(f"   Success: {result_equal.success}")
    print(f"   Objective: {result_equal.fun}")
    print(f"weights range: [{result_equal.x.min():.4f}, {result_equal.x.max():.4f}]")
    print(f"   Weights std: {result_equal.x.std():.4f}")
    
    # 2. Inverse volatility initial guess
    x0_inv_vol = inv_vol_weights
    result_inv_vol = minimize(
        risk_parity_objective,
        x0_inv_vol,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints_list,
        options={'ftol': 1e-9, 'maxiter': 1000}
    )
    print(f"\n2. Starting from inverse volatility weights:")
    print(f"   Success: {result_inv_vol.success}")
    print(f"   Objective: {result_inv_vol.fun}")
    print(f"   Weights range: [{result_inv_vol.x.min():.4f}, {result_inv_vol.x.max():.4f}]")
    print(f"   Weights std: {result_inv_vol.x.std():.4f}")
    
    # 3. Try with tighter bounds
    bounds_tight = [(0.01, 0.05) for _ in range(n_assets)]
    x0_tight = np.ones(n_assets) * 0.03
    result_tight = minimize(
        risk_parity_objective,
        x0_tight,
        method='SLSQP',
        bounds=bounds_tight,
        constraints=constraints_list,
        options={'ftol': 1e-9, 'maxiter': 1000}
    )
    print(f"\n3. With tighter bounds [0.01, 0.05]:")
    print(f"   Success: {result_tight.success}")
    print(f"   Objective: {result_tight.fun}")
    print(f"   Weights range: [{result_tight.x.min():.4f}, {result_tight.x.max():.4f}]")
    print(f"   Weights std: {result_tight.x.std():.4f}")
    
    # Show which is best
    print(f"\n" + "="*60)
    results = [
        ("Equal weights start", result_equal),
        ("Inverse vol start", result_inv_vol),
        ("Tighter bounds", result_tight)
    ]
    
    best_result = min(results, key=lambda x: x[1].fun)
    print(f"Best result: {best_result[0]}")
    print(f"Objective value: {best_result[1].fun}")
    print(f"\nTop 10 holdings:")
    weights_series = pd.Series(best_result[1].x, index=cov_matrix.index)
    print(weights_series.sort_values(ascending=False).head(10))

if __name__ == "__main__":
    test_risk_parity()
