"""
Run stock clustering analysis for a specific market (US or INDIA)
Generates cluster_assignments_*.csv and cluster_analysis_*.csv files
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import argparse
from loguru import logger

from config import Config


def build_features(returns_df: pd.DataFrame) -> pd.DataFrame:
    """Build simple clustering features from returns data."""
    features = {}
    for symbol in returns_df.columns:
        r = returns_df[symbol].dropna()
        if len(r) < 50:
            continue
        cumret = (1 + r).cumprod()
        roll_max = cumret.expanding().max()
        drawdown = (cumret - roll_max) / roll_max
        ann_ret = r.mean() * 252
        ann_vol = r.std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        features[symbol] = {
            'ann_return': ann_ret,
            'ann_volatility': ann_vol,
            'sharpe_ratio': sharpe,
            'skewness': float(r.skew()),
            'kurtosis': float(r.kurtosis()),
            'max_drawdown': float(drawdown.min()),
            'var_95': float(r.quantile(0.05)),
            'momentum_6m': float((cumret.iloc[-1] / cumret.iloc[-min(132, len(cumret))] - 1)),
        }
    df = pd.DataFrame(features).T.dropna()
    return df


def run_clustering(market: str = 'US'):
    market = market.upper()
    logger.info("=" * 60)
    logger.info(f"RUNNING CLUSTERING FOR {market} MARKET")
    logger.info("=" * 60)

    processed_dir = Config.get_market_data_dir(market, 'processed')
    results_dir = Config.get_market_results_dir(market)

    returns_path = os.path.join(processed_dir, 'returns_data.csv')
    if not os.path.exists(returns_path):
        logger.error(f"Returns data not found at {returns_path}")
        return False

    returns_df = pd.read_csv(returns_path, index_col=0, parse_dates=True)
    logger.info(f"Loaded returns data: {returns_df.shape}")

    features_df = build_features(returns_df)
    logger.info(f"Built features for {len(features_df)} stocks")

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features_df)

    algorithms = {
        'kmeans': KMeans(n_clusters=5, random_state=42, n_init=10),
        'hierarchical': AgglomerativeClustering(n_clusters=5),
        'gaussian_mixture': GaussianMixture(n_components=5, random_state=42),
    }

    for algo_name, model in algorithms.items():
        logger.info(f"Running {algo_name}...")
        try:
            labels = model.fit_predict(scaled)

            # cluster_assignments
            assignments = pd.DataFrame({
                'symbol': features_df.index,
                'cluster': labels
            })
            assign_path = os.path.join(results_dir, f'cluster_assignments_{algo_name}.csv')
            assignments.to_csv(assign_path, index=False)
            logger.info(f"  Saved assignments -> {assign_path}")

            # cluster_analysis
            stats_rows = []
            for cid in sorted(np.unique(labels)):
                mask = labels == cid
                cfeats = features_df.iloc[mask]
                n = mask.sum()
                stats_rows.append({
                    'cluster_id': int(cid),
                    'n_stocks': int(n),
                    'mean_return': float(cfeats['ann_return'].mean()),
                    'mean_volatility': float(cfeats['ann_volatility'].mean()),
                    'mean_sharpe': float(cfeats['sharpe_ratio'].mean()),
                    'mean_drawdown': float(cfeats['max_drawdown'].mean()),
                    'stocks': ', '.join(cfeats.index.tolist()),
                })
            analysis_df = pd.DataFrame(stats_rows)
            analysis_path = os.path.join(results_dir, f'cluster_analysis_{algo_name}.csv')
            analysis_df.to_csv(analysis_path, index=False)
            logger.info(f"  Saved analysis  -> {analysis_path}")

            if len(np.unique(labels)) > 1:
                sil = silhouette_score(scaled, labels)
                logger.info(f"  Silhouette score: {sil:.3f}")

        except Exception as e:
            logger.error(f"  Failed {algo_name}: {e}")
            import traceback; traceback.print_exc()

    logger.info(f"\n✓ CLUSTERING COMPLETE FOR {market} MARKET")
    return True


def main():
    parser = argparse.ArgumentParser(description='Run stock clustering for a market')
    parser.add_argument('--market', type=str, default='US', choices=['US', 'INDIA'],
                        help='Market to cluster (US or INDIA)')
    args = parser.parse_args()
    run_clustering(args.market)


if __name__ == "__main__":
    main()
