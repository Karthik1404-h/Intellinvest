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
- Key performance metrics at a glance
- Cumulative performance charts
- Quick strategy comparison
- Performance rankings

### 📈 Performance Analysis
- Detailed strategy metrics
- Cumulative returns visualization
- Drawdown analysis
- Returns distribution
- Rolling performance metrics

### 💼 Portfolio Composition
- Top 10 holdings visualization
- Portfolio allocation pie charts
- Complete holdings table
- Concentration metrics

### 🎯 Clustering Analysis
- Stock clustering results
- Cluster characteristics
- Risk-return profile by cluster
- Stocks grouped by similarity

### ⚖️ Risk Analysis
- Volatility comparison
- Maximum drawdown analysis
- Risk-adjusted returns (Sharpe, Sortino, Calmar ratios)
- Detailed risk metrics table

### 🏆 Benchmark Comparison
- ML strategies vs simple benchmarks
- Risk-return scatter plot
- Performance gap analysis
- Complete comparison metrics

## Dashboard Controls

- **Sidebar Navigation**: Switch between different analysis pages
- **Strategy Selector**: Choose specific strategies to analyze in detail
- **Interactive Charts**: Hover for details, zoom, pan, download
- **Data Tables**: Sort and filter performance metrics

## Tips

1. Start with the **Overview** page to get a quick summary
2. Use **Performance Analysis** to deep-dive into specific strategies
3. Check **Benchmark Comparison** to validate ML advantage
4. Explore **Clustering Analysis** to understand stock groupings
5. Review **Risk Analysis** for comprehensive risk metrics

## Keyboard Shortcuts

- `R` - Rerun the dashboard (refresh data)
- `C` - Clear cache
- `S` - View settings

## Troubleshooting

### Dashboard won't start
```bash
pip install --upgrade streamlit plotly
```

### Data not loading
Make sure you've run the optimization first:
```bash
python main.py --full
```

### Port already in use
```bash
streamlit run dashboard.py --server.port 8502
```

## Customization

You can modify `dashboard.py` to:
- Add new metrics
- Change color schemes
- Add custom visualizations
- Export reports

## Screenshots

The dashboard includes:
- 📊 Interactive charts powered by Plotly
- 🎨 Modern, clean interface
- 📱 Responsive design
- 💾 Data caching for performance
- 🔄 Real-time calculations

Enjoy analyzing your ML-optimized portfolios! 🚀
