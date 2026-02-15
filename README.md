# Portfolio Optimization with Machine Learning and Deep Learning

## Project Overview
This project implements portfolio optimization using machine learning and deep learning models enhanced with stock clustering techniques for improved return prediction.

## Key Features
- **Stock Clustering**: Group similar stocks using various clustering algorithms
- **Return Prediction**: ML/DL models for forecasting stock returns
- **Portfolio Optimization**: Modern portfolio theory with ML enhancements
- **Backtesting**: Comprehensive evaluation framework

## Project Structure
```
portfolio_optimizer/
├── data/                    # Data storage and management
│   ├── raw/                # Raw stock data
│   ├── processed/          # Cleaned and processed data
│   └── features/           # Engineered features
├── src/                    # Source code
│   ├── data/               # Data collection and preprocessing
│   ├── clustering/         # Stock clustering algorithms
│   ├── models/             # ML/DL prediction models
│   ├── optimization/       # Portfolio optimization
│   └── evaluation/         # Backtesting and metrics
├── notebooks/              # Jupyter notebooks for analysis
├── config/                 # Configuration files
├── tests/                  # Unit tests
├── results/                # Model outputs and results
└── requirements.txt        # Dependencies
```

## Getting Started
1. Install dependencies: `pip install -r requirements.txt`
2. Run data collection: `python src/data/collect_data.py`
3. Execute clustering analysis: `python src/clustering/cluster_stocks.py`
4. Train prediction models: `python src/models/train_models.py`
5. Optimize portfolio: `python src/optimization/optimize_portfolio.py`

## Research Reference
Based on "Deep learning and machine learning models for portfolio optimization: Enhancing return prediction with stock clustering"