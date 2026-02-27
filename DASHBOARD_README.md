# 📊 Portfolio Optimizer Dashboard

## Quick Start

### Install Dashboard Dependencies
```bash
pip install streamlit plotly
```

### Launch Dashboard
```bash
streamlit run dashboard.py
```

The dashboard will automatically open in your default web browser at `http://localhost:8501`

## Features

### 🏠 Overview
- Cumulative portfolio value chart with optional market-index overlay
- Key performance metrics at a glance (Sharpe, return, drawdown)
- Strategy ranking table
- Quick comparison across all 5 strategies

### 📈 Performance Analysis
- Detailed strategy metrics
- Cumulative returns visualization
- Drawdown curves
- Returns distribution histogram
- Rolling Sharpe ratio and rolling volatility

### 💼 Portfolio Composition
- Top 10 holdings bar chart
- Portfolio allocation pie chart
- Complete holdings table with weights
- Rebalancing history heatmap

### 🔬 Clustering Analysis
- PCA scatter of stock clusters (K-Means / Hierarchical / Gaussian Mixture)
- Cluster characteristics: mean return, volatility, Sharpe, drawdown
- Stocks-per-cluster listing

### ⚠️ Risk Analysis
- VaR (95%) and CVaR estimates
- Correlation matrix heatmap
- Volatility and drawdown comparison across strategies
- Risk-adjusted metrics table (Sharpe, Sortino, Calmar)

### 🏆 Benchmark Comparison
- **Verdict banner**: "✅ YES — ML Delivers Real Alpha" / "⚠️ PARTIALLY" / "❌ NO"  
- Value-add scorecard (Sharpe & return delta vs market index, equal-weight, cap-weighted)
- Cumulative wealth bar chart ($1 invested → $X for each strategy + benchmark)
- Side-by-side Annual Return and Sharpe Ratio bar charts (blue = ML, orange = Benchmark)
- Risk-return scatter (ML vs benchmarks colour-coded)
- Expandable full alpha table (ML vs every individual benchmark)
- Complete performance table with gradient styling

## Dashboard Controls

- **Market Selector** (sidebar): Switch between 🇺🇸 US and 🇮🇳 India (Nifty 50) markets
- **Sidebar Navigation**: Switch between analysis pages
- **Interactive Charts**: Hover for details, zoom, pan, download PNG
- **Data Tables**: Sort and filter all performance tables

## Tips

1. Start with **Overview** for a quick summary across all strategies
2. Use **Benchmark Comparison** to read the alpha verdict and value-add scorecard
3. Check **Performance Analysis** for drawdown curves and rolling metrics
4. Explore **Clustering Analysis** to understand how stocks were grouped
5. Use **Risk Analysis** for VaR/CVaR and correlation breakdown

## Troubleshooting

### Dashboard won't start
```bash
pip install --upgrade streamlit plotly
streamlit run dashboard.py
```

### No data showing for a market
Run the full pipeline for that market:
```bash
python collect_market_data.py --market INDIA
python regenerate_strategies_market.py --market INDIA
python run_benchmarks_market.py --market INDIA
python run_clustering_market.py --market INDIA
```

### Port already in use
```bash
streamlit run dashboard.py --server.port 8502
```

