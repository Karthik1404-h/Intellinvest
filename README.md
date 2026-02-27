# ML-Enhanced Portfolio Optimizer

An end-to-end portfolio optimization system that combines **machine learning return forecasting**, **stock clustering**, and **modern portfolio theory** to build and backtest optimized investment portfolios — for both **US equities** and **Indian (Nifty 50) stocks**.

Results are visualized through an interactive **Streamlit dashboard** with six analysis pages and a benchmark comparison section that quantifies how much value the ML strategies add over passive investing.

---

## Features

- **5 Portfolio Strategies**: Risk Parity, Mean Variance, Maximum Sharpe, Minimum Variance, Cluster-Based
- **ML Return Forecasting**: Ridge, Lasso, Random Forest, SVM, XGBoost, Linear Regression — ensemble-averaged for monthly rebalancing
- **Stock Clustering**: K-Means, Agglomerative, Gaussian Mixture for sector-agnostic grouping
- **Multi-Market Support**: US equities (S&P 500 universe) and Indian equities (Nifty 50)
- **Benchmark Comparison**: ML strategies vs market index, equal-weight, cap-weighted, and best/worst single stocks
- **Interactive Dashboard**: 6 pages — Overview, Performance Analysis, Portfolio Composition, Clustering Analysis, Risk Analysis, Benchmark Comparison

---

## Project Structure

```
portfolio_optimizer/
│
├── dashboard.py                    # Streamlit dashboard (main UI)
├── main.py                         # Pipeline entry point (CLI)
├── collect_market_data.py          # Download US / India market data
├── regenerate_strategies_market.py # Run all 5 strategies for a market
├── run_benchmarks_market.py        # Generate benchmark comparison CSV
├── run_clustering_market.py        # Generate cluster files for a market
├── setup.py                        # Dependency installer
├── requirements.txt
│
├── config/
│   └── config.py                   # Central configuration (symbols, paths, params)
│
├── src/
│   ├── data/
│   │   └── data_collection.py      # Data download & preprocessing
│   ├── clustering/
│   │   └── stock_clustering.py     # K-Means, Hierarchical, GMM clustering
│   ├── models/
│   │   └── prediction_models.py    # ML return forecasting models
│   ├── optimization/
│   │   └── portfolio_optimizer.py  # Portfolio optimization algorithms
│   └── evaluation/
│       └── backtesting.py          # Backtesting engine & performance metrics
│
├── data/                           # gitignored — download with collect_market_data.py
│   ├── raw/
│   │   ├── us/                     # SPY_benchmark.csv, stock_prices.csv
│   │   └── india/                  # NSEI_benchmark.csv, stock_prices.csv
│   └── processed/
│       ├── us/                     # processed_stock_data.csv, returns_data.csv
│       └── india/
│
├── results/
│   ├── us/                         # All US strategy outputs
│   │   ├── portfolio_values_*.csv
│   │   ├── portfolio_weights_*.csv
│   │   ├── performance_metrics_*.json
│   │   ├── cluster_assignments_*.csv
│   │   ├── cluster_analysis_*.csv
│   │   ├── benchmark_comparison_detailed.csv
│   │   └── strategy_comparison_summary.csv
│   └── india/                      # Same structure for India
│
└── notebooks/
    ├── 01_data_exploration.ipynb
    └── 02_advanced_analysis.ipynb
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
# or use the helper script:
python setup.py
```

### 2. Collect Market Data

```bash
# US market (S&P 500 subset)
python collect_market_data.py --market US

# Indian market (Nifty 50)
python collect_market_data.py --market INDIA
```

### 3. Generate Strategy Backtests

```bash
python regenerate_strategies_market.py --market US
python regenerate_strategies_market.py --market INDIA
```

### 4. Generate Benchmarks & Clustering

```bash
python run_benchmarks_market.py --market US
python run_benchmarks_market.py --market INDIA

python run_clustering_market.py --market US
python run_clustering_market.py --market INDIA
```

### 5. Launch Dashboard

```bash
streamlit run dashboard.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Dashboard Pages

| Page | Description |
|------|-------------|
| 🏠 Overview | Cumulative returns chart, key metrics, strategy ranking |
| 📈 Performance Analysis | Detailed metrics, drawdown curves, rolling Sharpe/volatility |
| 🥧 Portfolio Composition | Sector breakdown, weight heatmaps, rebalancing history |
| 🔬 Clustering Analysis | PCA cluster plots, inter/intra-cluster statistics |
| ⚠️ Risk Analysis | VaR, CVaR, correlation matrix, tail risk |
| 🏆 Benchmark Comparison | **Alpha verdict**, value-add scorecard, cumulative wealth chart, ML vs benchmark full table |

---

## Sample Results (India — Nifty 50, 2020–2026)

| Strategy | Annual Return | Sharpe Ratio | Max Drawdown |
|----------|--------------|-------------|-------------|
| Risk Parity | 24.7% | 1.27 | -18.1% |
| Mean Variance | 22.8% | 0.81 | -31.2% |
| Max Sharpe | 20.7% | 0.81 | -28.7% |
| Min Variance | 16.9% | 0.86 | -23.3% |
| *Nifty 50 Index* | *11.8%* | *0.39* | — |
| *Equal Weight* | *20.0%* | *0.81* | — |

---

## Sample Results (US, 2018–2023)

| Strategy | Annual Return | Sharpe Ratio |
|----------|--------------|-------------|
| Max Sharpe | 25.8% | 1.14 |
| Cluster-Based | 28.0% | 1.03 |
| Risk Parity | 18.6% | 1.09 |
| *S&P 500 (SPY)* | — | — |
| *Equal Weight* | *18.5%* | *0.85* |

---

## Configuration

All parameters are in `config/config.py`:

```python
# Key parameters
MAX_WEIGHT_PER_STOCK = 0.25      # Max allocation per stock
MIN_WEIGHT_PER_STOCK = 0.01      # Min allocation per stock
REBALANCING_FREQ = 'monthly'     # Rebalancing frequency
LOOKBACK_PERIOD = 252            # Days of history for optimization
RISK_FREE_RATE_US = 0.02         # For Sharpe ratio (US)
RISK_FREE_RATE_INDIA = 0.06      # For Sharpe ratio (India)
```

---

## Research Basis

Based on:
> *"Deep learning and machine learning models for portfolio optimization: Enhancing return prediction with stock clustering"*

The project implements the core idea of using **cluster membership as a feature** for return prediction and as a **diversification constraint** in portfolio construction.

---

## Requirements

- Python 3.9+
- Key packages: `pandas`, `numpy`, `scikit-learn`, `scipy`, `streamlit`, `plotly`, `yfinance`, `loguru`
- Optional: `pypfopt` (for additional optimization methods), `xgboost`, `lightgbm`

See `requirements.txt` for full list.
