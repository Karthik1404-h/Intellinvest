# Multi-Market Portfolio Optimization Guide

## Overview

The portfolio optimizer now supports **multiple markets**:
- 🇺🇸 **United States** - 35+ US stocks
- 🇮🇳 **India** - Nifty 50 stocks

All portfolio optimization strategies and analytics are available for both markets with market-specific benchmarks.

## Quick Start

### 1. Collect Market Data

#### For US Stocks:
```bash
python collect_market_data.py --market US
```

#### For Indian Stocks (Nifty 50):
```bash
python collect_market_data.py --market INDIA
```

**What this does:**
- Downloads historical price data from Yahoo Finance
- Collects market benchmark (S&P 500 for US, Nifty 50 for India)
- Processes and cleans the data
- Saves to market-specific directories

**Expected output:**
```
data/
├── raw/
│   ├── us/
│   │   ├── stock_prices.csv
│   │   └── SPY_benchmark.csv
│   └── india/
│       ├── stock_prices.csv
│       └── NSEI_benchmark.csv
├── processed/
│   ├── us/
│   │   ├── processed_stock_data.csv
│   │   └── returns_data.csv
│   └── india/
│       ├── processed_stock_data.csv
│       └── returns_data.csv
```

### 2. Generate Portfolio Strategies

#### For US Stocks:
```bash
python regenerate_strategies_market.py --market US
```

#### For Indian Stocks:
```bash
python regenerate_strategies_market.py --market INDIA
```

**What this does:**
- Runs 5 optimization strategies: Mean Variance, Max Sharpe, Min Variance, Cluster-Based, Risk Parity
- Backtests each strategy with monthly rebalancing
- Saves results to market-specific directories (`results/us/` or `results/india/`)

### 3. Generate Benchmark Comparison

```bash
python run_benchmarks_market.py --market US
python run_benchmarks_market.py --market INDIA
```

**What this does:**
- Computes Equal-Weight, Cap-Weighted, Best/Worst single-stock, and market-index benchmarks
- Produces `benchmark_comparison_detailed.csv` used by the dashboard's Benchmark Comparison page
- Uses market-appropriate risk-free rate (2% US, 6% India)

### 4. Generate Clustering Data

```bash
python run_clustering_market.py --market US
python run_clustering_market.py --market INDIA
```

**What this does:**
- Runs K-Means, Agglomerative, and Gaussian Mixture clustering on stock return features
- Saves `cluster_assignments_*.csv` and `cluster_analysis_*.csv` files

### 5. View in Dashboard

```bash
streamlit run dashboard.py
```

**Features:**
- 🌍 **Market Selector** in sidebar — switch between US and Indian markets instantly
- All 6 pages update automatically for the selected market
- Market-specific benchmarks in charts
- Benchmark Comparison page shows a **verdict banner** (▶ Do ML strategies add value?)

## Supported Markets

### 🇺🇸 United States (US)
- **Stocks**: 35 major US stocks across sectors (Tech, Finance, Healthcare, Consumer, Energy, Industrial)
- **Benchmark**: S&P 500 (SPY)
- **Data Source**: Yahoo Finance
- **Symbols**: AAPL, MSFT, GOOGL, AMZN, etc.

### 🇮🇳 India (INDIA)
- **Stocks**: 50 Nifty stocks
- **Benchmark**: Nifty 50 Index (^NSEI)
- **Data Source**: Yahoo Finance (NSE)
- **Symbols**: RELIANCE.NS, TCS.NS, HDFCBANK.NS, INFY.NS, etc.

**Note**: Indian stock symbols have `.NS` suffix for National Stock Exchange (NSE)

## Directory Structure

```
portfolio_optimizer/
├── config/
│   └── config.py                        # Multi-market configuration
├── data/
│   ├── raw/
│   │   ├── us/                          # US raw data
│   │   └── india/                       # Indian raw data
│   ├── processed/
│   │   ├── us/                          # US processed data
│   │   └── india/                       # Indian processed data
│   └── features/
│       ├── us/                          # US features
│       └── india/                       # Indian features
├── results/
│   ├── us/                              # US optimization results
│   └── india/                           # Indian optimization results
├── collect_market_data.py               # Data collection script
├── regenerate_strategies_market.py      # Optimization script
└── dashboard.py                         # Multi-market dashboard
```

## Configuration

Market configurations are defined in `config/config.py`:

```python
class Config:
    # Available markets
    AVAILABLE_MARKETS = ['US', 'INDIA']
    
    # Stock symbols by market
    US_STOCK_SYMBOLS = [...]
    INDIA_STOCK_SYMBOLS = [...]
    
    # Market-specific benchmarks
    MARKET_BENCHMARKS = {
        'US': 'SPY',      # S&P 500
        'INDIA': '^NSEI'  # Nifty 50
    }
    
    # Helper methods
    @staticmethod
    def get_stock_symbols(market: str)
    
    @staticmethod
    def get_benchmark_symbol(market: str)
    
    @staticmethod
    def get_market_data_dir(market: str, data_type: str)
    
    @staticmethod
    def get_market_results_dir(market: str)
```

## Workflow Example

### Complete Setup for Both Markets

```bash
# Step 1: Collect data for both markets
python collect_market_data.py --market US
python collect_market_data.py --market INDIA

# Step 2: Generate strategies for both markets
python regenerate_strategies_market.py --market US
python regenerate_strategies_market.py --market INDIA

# Step 3: Launch dashboard
streamlit run dashboard.py

# Now use the market selector in dashboard to switch between markets!
```

### Updating Data

To refresh data for a specific market:

```bash
# Re-collect data
python collect_market_data.py --market INDIA

# Re-run optimization
python regenerate_strategies_market.py --market INDIA
```

Dashboard will automatically pick up the new data on refresh.

## Dashboard Features by Market

### Market Selector
- Located at top of sidebar
- Dropdown with flag emojis (🇺🇸 United States / 🇮🇳 India)
- Automatically loads correct data
- Shows market-specific benchmark

### All Pages Support Both Markets:

#### 🏠 Overview
- Cumulative performance vs market benchmark
- Key metrics summary
- Strategy comparison

#### 📈 Performance Analysis
- Detailed metrics by strategy
- Daily returns distribution
- Rolling Sharpe ratio
- Rolling volatility

#### 💼 Portfolio Composition
- Top 10 holdings
- Allocation pie charts
- Portfolio statistics
- Complete allocation table

#### 🎯 Clustering Analysis
- Stock clusters for selected market
- Cluster characteristics
- Risk-return profiles

#### ⚖️ Risk Analysis
- Volatility comparison
- Maximum drawdown
- Risk-adjusted returns
- Detailed risk metrics

#### 🏆 Benchmark Comparison
- ML strategies vs benchmarks
- Risk-return scatter plot
- Performance comparison

## Technical Details

### Data Collection
- Uses `yfinance` library
- Handles market-specific symbols (e.g., `.NS` for NSE)
- Automatic data cleaning and processing
- Forward-fills missing values (holidays, etc.)
- Removes stocks with >10% missing data

### Optimization Strategies
Same strategies available for both markets:
1. **Mean Variance**: Traditional MPT
2. **Maximum Sharpe**: Maximize risk-adjusted returns
3. **Minimum Variance**: Minimize portfolio volatility
4. **Cluster-Based**: Use clustering for diversification
5. **Risk Parity**: Equal risk contribution

### Backtesting
- Start date: 2020-01-01 (configurable)
- Rebalancing: Monthly
- Initial capital: $1,000,000
- Transaction costs: 0.1%

## Troubleshooting

### Issue: "No data found for INDIA market"
**Solution**: Run data collection first
```bash
python collect_market_data.py --market INDIA
python regenerate_strategies_market.py --market INDIA
```

### Issue: Some Indian stocks missing
**Cause**: Yahoo Finance may not have data for all symbols
**Solution**: Script automatically filters out stocks with insufficient data and continues with available stocks

### Issue: Benchmark data not loading
**Check**: 
- File exists in `data/raw/india/NSEI_benchmark.csv` (or `SPY_benchmark.csv` for US)
- File format is correct (Date, Close columns)

### Issue: Dashboard shows old data
**Solution**: 
1. Stop Streamlit (Ctrl+C)
2. Re-run strategies: `python regenerate_strategies_market.py --market INDIA`
3. Restart dashboard: `streamlit run dashboard.py`

## Adding More Markets

To add a new market (e.g., Europe, Asia):

1. **Update config.py**:
```python
AVAILABLE_MARKETS = ['US', 'INDIA', 'EUROPE']
EUROPE_STOCK_SYMBOLS = [...]
MARKET_BENCHMARKS = {
    'EUROPE': '^STOXX50E'
}
```

2. **Data Collection**: Works automatically with new market

3. **Dashboard**: Add to selectbox options
```python
options=['US', 'INDIA', 'EUROPE']
```

## Performance Notes

- Data collection: ~2-5 minutes per market
- Strategy generation: ~3-5 minutes per market
- Dashboard: Real-time switching between markets

## Next Steps

1. ✅ Set up both US and Indian markets
2. 🔄 Compare strategies across markets
3. 📊 Analyze different market behaviors
4. 💡 Find market-specific opportunities

## Support

For issues or questions:
1. Check data collection logs
2. Verify file paths and permissions
3. Ensure Yahoo Finance access
4. Check symbol format (.NS for NSE, etc.)
