#!/usr/bin/env python3
"""
Interactive Portfolio Optimization Dashboard
Modern web-based interface for analyzing ML portfolio strategies
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from datetime import datetime

from config import Config

# Page configuration
st.set_page_config(
    page_title="Portfolio Optimizer Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stMetric label {
        color: #31333F !important;
        font-weight: 600 !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #0e1117 !important;
        font-size: 1.5rem !important;
        font-weight: 600 !important;
    }
    .stMetric [data-testid="stMetricDelta"] {
        font-weight: 500 !important;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
    h1 {
        color: #1f77b4;
        font-weight: 700;
    }
    h2 {
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 10px;
    }
    h3 {
        color: #2c3e50;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 16px 24px;
        background-color: #f0f2f6;
        border-radius: 8px 8px 0 0;
    }
    </style>
""", unsafe_allow_html=True)

def load_data(market: str = 'US'):
    """Load all project data for specified market"""
    data = {}
    market = market.upper()
    
    try:
        # Get market-specific directories
        results_dir = Config.get_market_results_dir(market)
        raw_data_dir = Config.get_market_data_dir(market, 'raw')
        benchmark_symbol = Config.get_benchmark_symbol(market)
        
        # Load strategy performance
        strategies = ['risk_parity', 'mean_variance', 'max_sharpe', 'cluster_based', 'min_variance', 'momentum_filter', 'black_litterman', 'concentrated_momentum']
        data['portfolio_values'] = {}
        data['portfolio_weights'] = {}
        data['performance_metrics'] = {}
        
        for strategy in strategies:
            # Portfolio values (returns)
            values_path = os.path.join(results_dir, f'portfolio_values_{strategy}.csv')
            if os.path.exists(values_path):
                data['portfolio_values'][strategy] = pd.read_csv(values_path, index_col=0, parse_dates=True)
            
            # Portfolio weights
            weights_path = os.path.join(results_dir, f'portfolio_weights_{strategy}.csv')
            if os.path.exists(weights_path):
                data['portfolio_weights'][strategy] = pd.read_csv(weights_path, index_col=0)
            
            # Performance metrics
            metrics_path = os.path.join(results_dir, f'performance_metrics_{strategy}.json')
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    data['performance_metrics'][strategy] = json.load(f)
        
        # Load clustering results
        cluster_path = os.path.join(results_dir, 'cluster_assignments_kmeans.csv')
        if os.path.exists(cluster_path):
            data['clusters'] = pd.read_csv(cluster_path)
        
        cluster_analysis_path = os.path.join(results_dir, 'cluster_analysis_kmeans.csv')
        if os.path.exists(cluster_analysis_path):
            data['cluster_analysis'] = pd.read_csv(cluster_analysis_path)
        
        # Load benchmark comparison
        benchmark_path = os.path.join(results_dir, 'benchmark_comparison_detailed.csv')
        if os.path.exists(benchmark_path):
            data['benchmark_comparison'] = pd.read_csv(benchmark_path)
        
        # Fallback: build benchmark_comparison from performance_metrics JSONs if file not found
        if 'benchmark_comparison' not in data and data.get('performance_metrics'):
            rows = []
            for strategy, metrics in data['performance_metrics'].items():
                rows.append({
                    'Strategy': f'ML_{strategy}',
                    'Total_Return': metrics.get('total_return', 0),
                    'Annual_Return': metrics.get('annualized_return', 0),
                    'Volatility': metrics.get('volatility', 0),
                    'Sharpe_Ratio': metrics.get('sharpe_ratio', 0),
                    'Sortino_Ratio': metrics.get('sortino_ratio', 0),
                    'Max_Drawdown': metrics.get('max_drawdown', 0),
                    'Calmar_Ratio': metrics.get('calmar_ratio', 0),
                })
            if rows:
                data['benchmark_comparison'] = pd.DataFrame(rows)
        
        # Load strategy comparison
        strategy_comparison_path = os.path.join(results_dir, 'strategy_comparison_summary.csv')
        if os.path.exists(strategy_comparison_path):
            data['strategy_comparison'] = pd.read_csv(strategy_comparison_path, index_col=0)
        
        # Load market benchmark
        benchmark_filename = benchmark_symbol.replace('^', '') + '_benchmark.csv'
        benchmark_path = os.path.join(raw_data_dir, benchmark_filename)
        
        if os.path.exists(benchmark_path):
            benchmark_df = pd.read_csv(benchmark_path)
            # Get date column (skip first 2 rows if they are headers)
            if len(benchmark_df) > 2 and benchmark_df.iloc[0, 0] == 'Ticker':
                benchmark_df = benchmark_df.iloc[2:]  # Skip header rows
            
            benchmark_dates = pd.to_datetime(benchmark_df.iloc[:, 0], errors='coerce')
            benchmark_close = pd.to_numeric(benchmark_df.iloc[:, 1], errors='coerce')
            # Remove any NaN values
            valid_idx = ~(benchmark_dates.isna() | benchmark_close.isna())
            benchmark_series = pd.Series(benchmark_close[valid_idx].values, index=benchmark_dates[valid_idx].values)
            benchmark_series = benchmark_series.sort_index()
            data['benchmark'] = benchmark_series
            data['benchmark_name'] = 'S&P 500' if market == 'US' else 'Nifty 50'
        
        data['market'] = market
        data['has_data'] = len(data['portfolio_values']) > 0
        
    except Exception as e:
        st.error(f"Error loading data for {market} market: {e}")
        data['has_data'] = False
        data['market'] = market
    
    return data

def create_performance_chart(data):
    """Create interactive cumulative performance chart"""
    fig = go.Figure()
    
    colors = {
        'risk_parity': '#1f77b4',
        'mean_variance': '#ff7f0e',
        'max_sharpe': '#2ca02c',
        'cluster_based': '#d62728',
        'min_variance': '#9467bd',
        'momentum_filter': '#8c564b',
        'black_litterman': '#e377c2',
        'concentrated_momentum': '#17becf',
        'benchmark': '#808080'
    }
    
    benchmark_name = data.get('benchmark_name', 'Benchmark')
    
    # Add benchmark first as reference
    if 'benchmark' in data:
        benchmark_series = data['benchmark']
        # Align with portfolio dates
        if data['portfolio_values']:
            first_strategy = list(data['portfolio_values'].keys())[0]
            portfolio_dates = data['portfolio_values'][first_strategy].index
            # Filter benchmark to matching dates
            benchmark_aligned = benchmark_series[benchmark_series.index.isin(portfolio_dates)]
            if len(benchmark_aligned) > 0:
                benchmark_cumulative = benchmark_aligned / benchmark_aligned.iloc[0]
                fig.add_trace(go.Scatter(
                    x=benchmark_cumulative.index,
                    y=benchmark_cumulative.values,
                    mode='lines',
                    name=f'{benchmark_name} (Benchmark)',
                    line=dict(width=2, color=colors['benchmark'], dash='dash'),
                    hovertemplate='%{y:.2f}<br>%{x|%Y-%m-%d}<extra></extra>'
                ))
    
    for strategy, values in data['portfolio_values'].items():
        if isinstance(values, pd.DataFrame):
            portfolio_values = values.iloc[:, 0]
        else:
            portfolio_values = values
        
        # Normalize to start at 1 (these are dollar values, not returns)
        cumulative = portfolio_values / portfolio_values.iloc[0]
        
        fig.add_trace(go.Scatter(
            x=cumulative.index,
            y=cumulative.values,
            mode='lines',
            name=strategy.replace('_', ' ').title(),
            line=dict(width=2.5, color=colors.get(strategy, '#000000')),
            hovertemplate='%{y:.2f}<br>%{x|%Y-%m-%d}<extra></extra>'
        ))
    
    fig.update_layout(
        title=f'Cumulative Portfolio Performance vs {benchmark_name}',
        xaxis_title='Date',
        yaxis_title='Cumulative Return (Normalized to 1.0)',
        hovermode='x unified',
        template='plotly_white',
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_risk_return_scatter(comparison_df):
    """Create risk-return scatter plot"""
    plot_df = comparison_df.copy()
    # Separate ML strategies from benchmarks
    plot_df['Type'] = plot_df['Strategy'].apply(
        lambda x: 'ML Strategy' if x.startswith('ML_') else 'Benchmark'
    )
    # Bubble size must be positive
    plot_df['BubbleSize'] = plot_df['Sharpe_Ratio'].clip(lower=0.05)

    fig = px.scatter(
        plot_df,
        x='Volatility',
        y='Annual_Return',
        size='BubbleSize',
        color='Type',
        hover_name='Strategy',
        hover_data={
            'Volatility': ':.2%',
            'Annual_Return': ':.2%',
            'Sharpe_Ratio': ':.3f',
            'Type': False
        },
        color_discrete_map={'ML Strategy': '#1f77b4', 'Benchmark': '#ff7f0e'},
        labels={
            'Volatility': 'Volatility (Risk)',
            'Annual_Return': 'Annual Return'
        },
        title='Risk-Return Profile: ML Strategies vs Benchmarks'
    )
    
    fig.update_traces(marker=dict(line=dict(width=2, color='DarkSlateGray')))
    fig.update_layout(
        template='plotly_white',
        height=500,
        showlegend=True
    )
    
    return fig

def create_weights_chart(weights_series, strategy_name):
    """Create portfolio weights visualization"""
    try:
        # Get top 10 holdings - weights_series should already be a Series
        weights_top = weights_series.sort_values(ascending=False).head(10)
        
        fig = go.Figure(data=[
            go.Bar(
                x=weights_top.values * 100,
                y=weights_top.index,
                orientation='h',
                marker=dict(
                    color=weights_top.values,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Weight %")
                ),
                text=[f'{v*100:.1f}%' for v in weights_top.values],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title=f'Top 10 Holdings - {strategy_name.replace("_", " ").title()}',
            xaxis_title='Weight (%)',
            yaxis_title='Stock Symbol',
            template='plotly_white',
            height=400
        )
        
        return fig
    except Exception as e:
        # Return empty figure with error message
        fig = go.Figure()
        fig.add_annotation(text=f"Error creating chart: {e}", 
                          xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig

def create_cluster_visualization(clusters_df, cluster_analysis_df):
    """Create cluster visualization"""
    fig = px.scatter(
        clusters_df,
        x='cluster',
        y='symbol',
        color='cluster',
        title='Stock Clustering Results',
        labels={'cluster': 'Cluster ID', 'symbol': 'Stock Symbol'},
        color_continuous_scale='viridis',
        height=500
    )
    
    fig.update_layout(
        template='plotly_white',
        showlegend=False
    )
    
    return fig

def create_drawdown_chart(returns):
    """Create drawdown chart"""
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown.values * 100,
        fill='tozeroy',
        name='Drawdown',
        line=dict(color='red', width=1),
        fillcolor='rgba(255, 0, 0, 0.2)'
    ))
    
    fig.update_layout(
        title='Portfolio Drawdown Over Time',
        xaxis_title='Date',
        yaxis_title='Drawdown (%)',
        template='plotly_white',
        height=400,
        hovermode='x unified'
    )
    
    return fig

def create_metrics_comparison(comparison_df, metric):
    """Create horizontal bar chart for metric comparison"""
    sorted_df = comparison_df.sort_values(metric, ascending=True)
    
    colors = ['#1f77b4' if s.startswith('ML_') else '#ff7f0e' for s in sorted_df['Strategy']]
    
    fig = go.Figure(data=[
        go.Bar(
            x=sorted_df[metric],
            y=sorted_df['Strategy'].str.replace('ML_', '').str.replace('_', ' '),
            orientation='h',
            marker=dict(color=colors),
            text=[f'{v:.3f}' for v in sorted_df[metric]],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title=f'{metric.replace("_", " ").title()} Comparison',
        xaxis_title=metric.replace('_', ' ').title(),
        yaxis_title='Strategy',
        template='plotly_white',
        height=500
    )
    
    return fig

def main():
    """Main dashboard function"""
    
    # Sidebar
    st.sidebar.image("https://img.icons8.com/fluency/96/000000/bar-chart.png", width=80)
    st.sidebar.title("📊 Portfolio Optimizer")
    st.sidebar.markdown("### ML-Enhanced Portfolio Analysis")
    
    # Market Selector
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🌍 Select Market")
    market = st.sidebar.selectbox(
        "Market",
        options=['US', 'INDIA'],
        format_func=lambda x: '🇺🇸 United States' if x == 'US' else '🇮🇳 India (Nifty 50)',
        help="Choose between US stocks or Indian stocks (Nifty 50)"
    )
    
    # Load data for selected market
    with st.spinner(f"Loading {market} market data..."):
        data = load_data(market)
    
    if not data.get('has_data', False):
        st.error(f"⚠️ No data found for {market} market. Please run data collection and optimization first.")
        st.markdown(f"### Setup {market} Market:")
        st.markdown(f"1. **Collect Data:**")
        st.code(f"python collect_market_data.py --market {market}")
        st.markdown(f"2. **Run Optimization:**")
        st.code(f"python regenerate_strategies_market.py --market {market}")
        st.stop()
    
    st.sidebar.success(f"✓ {market} market data loaded")
    st.sidebar.markdown("---")
    
    # Navigation
    page = st.sidebar.radio(
        "Navigation",
        ["🏠 Overview", "📈 Performance Analysis", "💼 Portfolio Composition", 
         "🎯 Clustering Analysis", "⚖️ Risk Analysis", "🏆 Benchmark Comparison"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        f"**Market**: {market}  \n"
        f"**Benchmark**: {data.get('benchmark_name', 'N/A')}  \n"
        f"**About**: This dashboard displays results from ML-enhanced portfolio optimization "
        "using stock clustering and advanced optimization algorithms."
    )
    
    # ============================================
    # 🏠 OVERVIEW PAGE
    # ============================================
    if page == "🏠 Overview":
        st.title(f"🏠 Portfolio Optimization Dashboard - {market} Market")
        st.markdown("### ML-Enhanced Investment Strategy Analysis")
        
        # Key Metrics Row
        if 'benchmark_comparison' in data:
            st.markdown("## 📊 Key Performance Metrics")
            
            # Get best ML strategy
            ml_strategies = data['benchmark_comparison'][data['benchmark_comparison']['Strategy'].str.startswith('ML_')]
            if not ml_strategies.empty:
                best_strategy = ml_strategies.loc[ml_strategies['Sharpe_Ratio'].idxmax()]
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Best Strategy",
                        best_strategy['Strategy'].replace('ML_', '').replace('_', ' ').title(),
                        "Sharpe: {:.3f}".format(best_strategy['Sharpe_Ratio'])
                    )
                
                with col2:
                    st.metric(
                        "Annual Return",
                        f"{best_strategy['Annual_Return']*100:.2f}%",
                        f"{best_strategy['Annual_Return']*100:.2f}%"
                    )
                
                with col3:
                    st.metric(
                        "Volatility",
                        f"{best_strategy['Volatility']*100:.2f}%",
                        f"-{(20-best_strategy['Volatility']*100):.1f}% vs market"
                    )
                
                with col4:
                    st.metric(
                        "Max Drawdown",
                        f"{best_strategy['Max_Drawdown']*100:.2f}%",
                        "Risk managed"
                    )
        
        st.markdown("---")
        
        # Cumulative Performance Chart
        if data['portfolio_values']:
            st.markdown("## 📈 Cumulative Performance")
            fig = create_performance_chart(data)
            st.plotly_chart(fig, width='stretch')
        
        # Quick Stats
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎯 Portfolio Strategies")
            if 'strategy_comparison' in data:
                st.dataframe(
                    data['strategy_comparison'][['Total Return', 'Sharpe Ratio', 'Max Drawdown']].style.format({
                        'Total Return': '{:.2%}',
                        'Sharpe Ratio': '{:.3f}',
                        'Max Drawdown': '{:.2%}'
                    }).background_gradient(cmap='RdYlGn', subset=['Sharpe Ratio'])
                )
        
        with col2:
            st.markdown("### 🏆 Performance Rankings")
            if 'benchmark_comparison' in data:
                top_strategies = data['benchmark_comparison'].nlargest(5, 'Sharpe_Ratio')[['Strategy', 'Sharpe_Ratio', 'Annual_Return']]
                top_strategies['Strategy'] = top_strategies['Strategy'].str.replace('ML_', '').str.replace('_', ' ')
                st.dataframe(
                    top_strategies.style.format({
                        'Sharpe_Ratio': '{:.3f}',
                        'Annual_Return': '{:.2%}'
                    }).background_gradient(cmap='Blues')
                )
    
    # ============================================
    # 📈 PERFORMANCE ANALYSIS PAGE
    # ============================================
    elif page == "📈 Performance Analysis":
        st.title("📈 Performance Analysis")
        
        # Strategy selector
        strategy = st.selectbox(
            "Select Strategy",
            list(data['portfolio_values'].keys()),
            format_func=lambda x: x.replace('_', ' ').title()
        )
        
        if strategy in data['portfolio_values']:
            values = data['portfolio_values'][strategy]
            if isinstance(values, pd.DataFrame):
                portfolio_values = values.iloc[:, 0]
            else:
                portfolio_values = values
            
            # Calculate returns from portfolio values
            returns = portfolio_values.pct_change().fillna(0)
            
            # Metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            
            metrics = data['performance_metrics'].get(strategy, {})
            
            with col1:
                st.metric("Total Return", f"{metrics.get('total_return', 0)*100:.2f}%")
            with col2:
                st.metric("Annual Return", f"{metrics.get('annualized_return', 0)*100:.2f}%")
            with col3:
                st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.3f}")
            with col4:
                st.metric("Sortino Ratio", f"{metrics.get('sortino_ratio', 0):.3f}")
            with col5:
                st.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0)*100:.2f}%")
            
            st.markdown("---")
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Cumulative returns
                cumulative = (1 + returns).cumprod()
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=cumulative.index,
                    y=cumulative.values,
                    mode='lines',
                    fill='tozeroy',
                    name='Cumulative Return',
                    line=dict(color='#1f77b4', width=2)
                ))
                fig.update_layout(
                    title='Cumulative Returns',
                    xaxis_title='Date',
                    yaxis_title='Cumulative Return',
                    template='plotly_white',
                    height=400
                )
                st.plotly_chart(fig, width='stretch')
            
            with col2:
                # Drawdown
                st.plotly_chart(create_drawdown_chart(returns), width='stretch')
            
            # Returns distribution
            st.markdown("### Returns Distribution")
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=returns * 100,
                nbinsx=50,
                name='Returns',
                marker=dict(color='#1f77b4', line=dict(color='white', width=1))
            ))
            fig.update_layout(
                title='Distribution of Daily Returns',
                xaxis_title='Daily Return (%)',
                yaxis_title='Frequency',
                template='plotly_white',
                height=400
            )
            st.plotly_chart(fig, width='stretch')
            
            # Rolling metrics
            st.markdown("### Rolling Performance Metrics")
            window = st.slider("Rolling Window (days)", 20, 252, 60)
            
            rolling_sharpe = returns.rolling(window).mean() / returns.rolling(window).std() * np.sqrt(252)
            rolling_vol = returns.rolling(window).std() * np.sqrt(252)
            
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Rolling Sharpe Ratio', 'Rolling Volatility'),
                vertical_spacing=0.15
            )
            
            fig.add_trace(
                go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe.values, name='Sharpe', line=dict(color='green')),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=rolling_vol.index, y=rolling_vol.values * 100, name='Volatility', line=dict(color='red')),
                row=2, col=1
            )
            
            fig.update_xaxes(title_text="Date", row=2, col=1)
            fig.update_yaxes(title_text="Sharpe Ratio", row=1, col=1)
            fig.update_yaxes(title_text="Volatility (%)", row=2, col=1)
            
            fig.update_layout(height=600, template='plotly_white', showlegend=False)
            st.plotly_chart(fig, width='stretch')
    
    # ============================================
    # 💼 PORTFOLIO COMPOSITION PAGE
    # ============================================
    elif page == "💼 Portfolio Composition":
        st.title("💼 Portfolio Composition")
        
        strategy = st.selectbox(
            "Select Strategy",
            list(data['portfolio_weights'].keys()),
            format_func=lambda x: x.replace('_', ' ').title()
        )
        
        if strategy in data['portfolio_weights']:
            weights_df = data['portfolio_weights'][strategy]
            
            # Convert to Series if DataFrame
            if isinstance(weights_df, pd.DataFrame):
                if 'weight' in weights_df.columns:
                    weights_series = weights_df['weight']
                else:
                    weights_series = weights_df.iloc[:, 0]
            else:
                weights_series = weights_df
            
            # Top holdings chart
            st.markdown("### Top 10 Holdings")
            st.plotly_chart(create_weights_chart(weights_series, strategy), width='stretch')
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Pie chart of top holdings
                try:
                    weights_series_top = weights_series.sort_values(ascending=False).head(10)
                    
                    fig = px.pie(
                        values=weights_series_top.values,
                        names=weights_series_top.index,
                        title='Portfolio Allocation (Top 10)',
                        hole=0.4
                    )
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    fig.update_layout(template='plotly_white', height=400)
                    st.plotly_chart(fig, width='stretch')
                except Exception as e:
                    st.error(f"Error creating pie chart: {e}")
            
            with col2:
                # Portfolio statistics
                st.markdown("### Portfolio Statistics")
                
                try:
                    stats = {
                        "Number of Holdings": (weights_series > 0.001).sum(),
                        "Largest Position": f"{weights_series.max()*100:.2f}%",
                        "Smallest Position": f"{weights_series[weights_series > 0.001].min()*100:.2f}%",
                        "Average Weight": f"{weights_series[weights_series > 0.001].mean()*100:.2f}%",
                        "Concentration (Top 5)": f"{weights_series.nlargest(5).sum()*100:.2f}%",
                        "HHI (Concentration Index)": f"{(weights_series**2).sum():.4f}"
                    }
                    
                    for key, value in stats.items():
                        st.metric(key, value)
                except Exception as e:
                    st.error(f"Error calculating statistics: {e}")
            
            # Full allocation table
            st.markdown("### Complete Portfolio Allocation")
            
            try:
                # Create display dataframe
                weights_display = pd.DataFrame({
                    'Symbol': weights_series.index,
                    'Weight': weights_series.values,
                    'Weight %': weights_series.values * 100
                })
                
                # Sort by weight descending and filter small positions
                weights_display = weights_display[weights_display['Weight'] > 0.001].sort_values('Weight', ascending=False)
                weights_display = weights_display.reset_index(drop=True)
                
                st.dataframe(
                    weights_display.style.format({
                        'Weight': '{:.4f}',
                        'Weight %': '{:.2f}%'
                    }).background_gradient(cmap='YlGnBu', subset=['Weight %']),
                    height=400
                )
            except Exception as e:
                st.error(f"Error displaying portfolio allocation: {e}")
                st.write("Raw data shape:", weights_df.shape if hasattr(weights_df, 'shape') else 'Unknown')
    
    # ============================================
    # 🎯 CLUSTERING ANALYSIS PAGE
    # ============================================
    elif page == "🎯 Clustering Analysis":
        st.title("🎯 Stock Clustering Analysis")
        
        if 'clusters' in data and 'cluster_analysis' in data:
            st.markdown("### Cluster Summary")
            
            # Cluster statistics
            cluster_stats = data['cluster_analysis']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Number of Clusters", len(cluster_stats))
            with col2:
                st.metric("Stocks Analyzed", len(data['clusters']))
            with col3:
                st.metric("Average Cluster Size", f"{data['clusters']['cluster'].value_counts().mean():.1f}")
            
            st.markdown("---")
            
            # Cluster characteristics
            st.markdown("### Cluster Characteristics")
            
            display_cols = ['cluster_id', 'n_stocks', 'mean_return', 'mean_volatility', 'mean_sharpe']
            if all(col in cluster_stats.columns for col in display_cols):
                cluster_display = cluster_stats[display_cols].copy()
                cluster_display.columns = ['Cluster', '# Stocks', 'Avg Return', 'Avg Volatility', 'Avg Sharpe']
                
                st.dataframe(
                    cluster_display.style.format({
                        'Avg Return': '{:.4f}',
                        'Avg Volatility': '{:.4f}',
                        'Avg Sharpe': '{:.3f}'
                    }).background_gradient(cmap='RdYlGn', subset=['Avg Sharpe']),
                    height=300
                )
            
            # Cluster visualization
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Cluster Distribution")
                cluster_counts = data['clusters']['cluster'].value_counts().sort_index()
                fig = px.bar(
                    x=cluster_counts.index,
                    y=cluster_counts.values,
                    labels={'x': 'Cluster ID', 'y': 'Number of Stocks'},
                    title='Stocks per Cluster',
                    color=cluster_counts.values,
                    color_continuous_scale='viridis'
                )
                fig.update_layout(template='plotly_white', height=400)
                st.plotly_chart(fig, width='stretch')
            
            with col2:
                st.markdown("### Cluster Risk-Return Profile")
                if all(col in cluster_stats.columns for col in ['mean_return', 'mean_volatility', 'cluster_id']):
                    fig = px.scatter(
                        cluster_stats,
                        x='mean_volatility',
                        y='mean_return',
                        size='n_stocks',
                        color='cluster_id',
                        hover_data=['n_stocks'],
                        labels={
                            'mean_volatility': 'Average Volatility',
                            'mean_return': 'Average Return',
                            'cluster_id': 'Cluster'
                        },
                        title='Cluster Risk-Return Profile'
                    )
                    fig.update_layout(template='plotly_white', height=400)
                    st.plotly_chart(fig, width='stretch')
            
            # Stocks by cluster
            st.markdown("### Stocks by Cluster")
            selected_cluster = st.selectbox("Select Cluster", sorted(data['clusters']['cluster'].unique()))
            
            cluster_stocks = data['clusters'][data['clusters']['cluster'] == selected_cluster]['symbol'].tolist()
            
            cols = st.columns(5)
            for idx, stock in enumerate(cluster_stocks):
                with cols[idx % 5]:
                    st.info(f"**{stock}**")
        else:
            st.warning("⚠️ Clustering results not found. Please run clustering analysis first.")
    
    # ============================================
    # ⚖️ RISK ANALYSIS PAGE
    # ============================================
    elif page == "⚖️ Risk Analysis":
        st.title("⚖️ Risk Analysis")
        
        if 'benchmark_comparison' not in data:
            st.warning("⚠️ Risk analysis data not available. Please run the optimization pipeline first.")
        elif 'benchmark_comparison' in data:
            comparison_df = data['benchmark_comparison']
            
            # Risk metrics comparison
            st.markdown("### Risk Metrics Comparison")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(
                    create_metrics_comparison(comparison_df, 'Volatility'),
                    width='stretch'
                )
            
            with col2:
                st.plotly_chart(
                    create_metrics_comparison(comparison_df, 'Max_Drawdown'),
                    width='stretch'
                )
            
            # Risk-adjusted returns
            st.markdown("### Risk-Adjusted Returns")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(
                    create_metrics_comparison(comparison_df, 'Sharpe_Ratio'),
                    width='stretch'
                )
            
            with col2:
                st.plotly_chart(
                    create_metrics_comparison(comparison_df, 'Sortino_Ratio'),
                    width='stretch'
                )
            
            # Detailed risk table
            st.markdown("### Detailed Risk Metrics")
            risk_cols = ['Strategy', 'Volatility', 'Max_Drawdown', 'Sharpe_Ratio', 'Sortino_Ratio', 'Calmar_Ratio']
            risk_table = comparison_df[risk_cols].copy()
            risk_table['Strategy'] = risk_table['Strategy'].str.replace('ML_', '').str.replace('_', ' ')
            
            st.dataframe(
                risk_table.style.format({
                    'Volatility': '{:.2%}',
                    'Max_Drawdown': '{:.2%}',
                    'Sharpe_Ratio': '{:.3f}',
                    'Sortino_Ratio': '{:.3f}',
                    'Calmar_Ratio': '{:.3f}'
                }).background_gradient(cmap='RdYlGn', subset=['Sharpe_Ratio', 'Sortino_Ratio', 'Calmar_Ratio'])
                .background_gradient(cmap='RdYlGn_r', subset=['Volatility', 'Max_Drawdown']),
                height=600
            )
    
    # ============================================
    # 🏆 BENCHMARK COMPARISON PAGE
    # ============================================
    elif page == "🏆 Benchmark Comparison":
        st.title("🏆 Benchmark Comparison")

        if 'benchmark_comparison' not in data:
            st.warning("⚠️ Benchmark comparison data not found.")
        else:
            comparison_df = data['benchmark_comparison']
            ml_strategies = comparison_df[comparison_df['Strategy'].str.startswith('ML_')]
            benchmarks = comparison_df[~comparison_df['Strategy'].str.startswith('ML_')]
            has_benchmarks = len(benchmarks) > 0

            best_ml = ml_strategies.loc[ml_strategies['Sharpe_Ratio'].idxmax()]
            best_ml_name = best_ml['Strategy'].replace('ML_', '').replace('_', ' ').title()

            # identify market index row (SPY/Nifty) and passive benchmarks
            INDEX_KEYWORDS = ['SPY', 'Nifty50', 'SP500', 'Index']
            index_rows = benchmarks[benchmarks['Strategy'].apply(
                lambda s: any(k.lower() in s.lower() for k in INDEX_KEYWORDS))]
            passive_rows = benchmarks[~benchmarks['Strategy'].apply(
                lambda s: any(k.lower() in s.lower() for k in INDEX_KEYWORDS + ['Best_Stock', 'Worst_Stock']))]

            best_passive = (passive_rows if len(passive_rows) > 0 else benchmarks).loc[
                (passive_rows if len(passive_rows) > 0 else benchmarks)['Sharpe_Ratio'].idxmax()
            ] if has_benchmarks else None

            market_index = index_rows.iloc[0] if len(index_rows) > 0 else None
            ew_rows = comparison_df[comparison_df['Strategy'].str.contains('Equal_Weight', case=False)]
            equal_weight = ew_rows.iloc[0] if len(ew_rows) > 0 else None
            cw_rows = comparison_df[comparison_df['Strategy'].str.contains('Cap_Weighted', case=False)]
            cap_weight = cw_rows.iloc[0] if len(cw_rows) > 0 else None

            # ═══════════════════════════════════════════════════════════════
            # SECTION 1 — VERDICT BANNER
            # ═══════════════════════════════════════════════════════════════
            st.markdown("## 🎯 Are ML Strategies Worth Following?")

            if has_benchmarks and best_passive is not None:
                sharpe_vs_passive = best_ml['Sharpe_Ratio'] - best_passive['Sharpe_Ratio']
                ret_vs_passive    = best_ml['Annual_Return'] - best_passive['Annual_Return']
                ml_beats_passive  = sharpe_vs_passive > 0

                if market_index is not None:
                    sharpe_vs_index = best_ml['Sharpe_Ratio'] - market_index['Sharpe_Ratio']
                    ret_vs_index    = best_ml['Annual_Return'] - market_index['Annual_Return']
                    ml_beats_index  = sharpe_vs_index > 0
                else:
                    sharpe_vs_index = sharpe_vs_passive
                    ret_vs_index    = ret_vs_passive
                    ml_beats_index  = ml_beats_passive

                beats_both = ml_beats_passive and ml_beats_index
                beats_one  = ml_beats_passive or ml_beats_index

                if beats_both:
                    verdict_color = "#1a7a1a"
                    verdict_icon  = "✅"
                    verdict_text  = "YES — ML Strategies Deliver Real Alpha"
                    verdict_sub   = (
                        f"The best ML strategy (<b>{best_ml_name}</b>) outperforms both the market index "
                        f"and simple passive portfolios on a risk-adjusted basis. "
                        f"It generates <b>{ret_vs_index*100:+.1f}% extra annual return</b> vs the market index "
                        f"with a Sharpe ratio improvement of <b>{sharpe_vs_index:+.3f}</b>."
                    )
                elif beats_one:
                    verdict_color = "#b38000"
                    verdict_icon  = "⚠️"
                    verdict_text  = "PARTIALLY — ML Adds Value Over Some Benchmarks"
                    verdict_sub   = (
                        f"<b>{best_ml_name}</b> beats some benchmarks but not all. "
                        f"Sharpe vs best passive benchmark: <b>{sharpe_vs_passive:+.3f}</b>. "
                        f"Consider the strategy selectively."
                    )
                else:
                    verdict_color = "#8b0000"
                    verdict_icon  = "❌"
                    verdict_text  = "NO — Passive Strategies Currently Outperform"
                    verdict_sub   = (
                        f"Simple passive strategies beat the best ML strategy on risk-adjusted returns. "
                        f"Sharpe deficit vs best passive: <b>{sharpe_vs_passive:.3f}</b>. "
                        f"Review the model or market regime."
                    )

                st.markdown(
                    f"""
                    <div style="background:{verdict_color}22; border-left:6px solid {verdict_color};
                                padding:20px 24px; border-radius:8px; margin-bottom:16px;">
                        <h2 style="color:{verdict_color}; margin:0 0 6px 0;">{verdict_icon} {verdict_text}</h2>
                        <p style="font-size:15px; margin:0; color:#ddd;">{verdict_sub}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # ── Value-add scorecard ───────────────────────────────────
                st.markdown("### 📐 Value Added vs Benchmarks")
                cols = st.columns(4)

                def delta_metric(col, label, ml_val, bm_val, fmt_pct=False):
                    diff = ml_val - bm_val
                    bm_str = f"{bm_val*100:.2f}%" if fmt_pct else f"{bm_val:.3f}"
                    diff_str = f"{diff*100:+.2f}%" if fmt_pct else f"{diff:+.3f}"
                    col.metric(label, f"{ml_val*100:.2f}%" if fmt_pct else f"{ml_val:.3f}",
                               delta=f"{diff_str} vs {bm_str}",
                               delta_color="normal" if diff >= 0 else "inverse")

                with cols[0]:
                    st.caption("vs Market Index" if market_index is not None else "Best ML")
                    if market_index is not None:
                        delta_metric(cols[0], "Sharpe Ratio", best_ml['Sharpe_Ratio'], market_index['Sharpe_Ratio'])
                    else:
                        st.metric("Sharpe Ratio", f"{best_ml['Sharpe_Ratio']:.3f}")
                with cols[1]:
                    st.caption("vs Market Index" if market_index is not None else "Best ML")
                    if market_index is not None:
                        delta_metric(cols[1], "Annual Return", best_ml['Annual_Return'], market_index['Annual_Return'], fmt_pct=True)
                    else:
                        st.metric("Annual Return", f"{best_ml['Annual_Return']*100:.2f}%")
                with cols[2]:
                    st.caption("vs Equal Weight")
                    if equal_weight is not None:
                        delta_metric(cols[2], "Sharpe Ratio", best_ml['Sharpe_Ratio'], equal_weight['Sharpe_Ratio'])
                    else:
                        st.metric("Max Drawdown", f"{best_ml['Max_Drawdown']*100:.2f}%")
                with cols[3]:
                    st.caption("vs Cap Weighted")
                    if cap_weight is not None:
                        delta_metric(cols[3], "Sharpe Ratio", best_ml['Sharpe_Ratio'], cap_weight['Sharpe_Ratio'])
                    else:
                        st.metric("Calmar Ratio", f"{best_ml['Calmar_Ratio']:.3f}")

                st.markdown("---")

                # ── Cumulative wealth bar ─────────────────────────────────
                st.markdown("### 💰 Cumulative Wealth: $1 Invested")
                wealth_rows = []
                for _, row in comparison_df.iterrows():
                    is_ml = row['Strategy'].startswith('ML_')
                    label = row['Strategy'].replace('ML_', '').replace('_', ' ').title()
                    wealth_rows.append({
                        'Strategy': label,
                        'Final Value': 1 + row['Total_Return'],
                        'Type': 'ML Strategy' if is_ml else 'Benchmark',
                    })
                wealth_df = pd.DataFrame(wealth_rows).sort_values('Final Value', ascending=True)
                fig_wealth = go.Figure(go.Bar(
                    x=wealth_df['Final Value'],
                    y=wealth_df['Strategy'],
                    orientation='h',
                    marker=dict(
                        color=wealth_df['Type'].map({'ML Strategy': '#2196F3', 'Benchmark': '#FF9800'}),
                    ),
                    text=[f"${v:.2f}" for v in wealth_df['Final Value']],
                    textposition='auto',
                ))
                fig_wealth.add_vline(x=1.0, line_dash='dash', line_color='red',
                                     annotation_text='Break-even ($1)', annotation_position='top right')
                fig_wealth.update_layout(
                    title='Final portfolio value for every $1 invested (blue = ML, orange = Benchmark)',
                    xaxis_title='Portfolio Value ($)',
                    template='plotly_white', height=420,
                    showlegend=False,
                )
                st.plotly_chart(fig_wealth, width='stretch')

                # ── Alpha table ───────────────────────────────────────────
                with st.expander("📋 Full Alpha Table — ML vs Every Benchmark", expanded=False):
                    alpha_rows = []
                    for _, bm in benchmarks.iterrows():
                        bm_label = bm['Strategy'].replace('_', ' ')
                        alpha_rows.append({
                            'Benchmark': bm_label,
                            'Benchmark Sharpe': bm['Sharpe_Ratio'],
                            'Best ML Sharpe': best_ml['Sharpe_Ratio'],
                            'Sharpe Alpha': best_ml['Sharpe_Ratio'] - bm['Sharpe_Ratio'],
                            'Benchmark Ann. Return': bm['Annual_Return'],
                            'Best ML Ann. Return': best_ml['Annual_Return'],
                            'Return Alpha': best_ml['Annual_Return'] - bm['Annual_Return'],
                            'ML Max DD': best_ml['Max_Drawdown'],
                            'BM Max DD': bm['Max_Drawdown'],
                            'DD Improvement': abs(bm['Max_Drawdown']) - abs(best_ml['Max_Drawdown']),
                        })
                    alpha_df = pd.DataFrame(alpha_rows)
                    st.dataframe(
                        alpha_df.style.format({
                            'Benchmark Sharpe': '{:.3f}', 'Best ML Sharpe': '{:.3f}', 'Sharpe Alpha': '{:+.3f}',
                            'Benchmark Ann. Return': '{:.2%}', 'Best ML Ann. Return': '{:.2%}', 'Return Alpha': '{:+.2%}',
                            'ML Max DD': '{:.2%}', 'BM Max DD': '{:.2%}', 'DD Improvement': '{:+.2%}',
                        }).background_gradient(cmap='RdYlGn', subset=['Sharpe Alpha', 'Return Alpha', 'DD Improvement']),
                        use_container_width=True,
                    )

                st.markdown("---")

            else:
                # No benchmark data — just show best ML summary
                st.info("ℹ️ No benchmark data available. Showing ML strategies only.")

            # ═══════════════════════════════════════════════════════════════
            # SECTION 2 — BEST STRATEGY DETAILS
            # ═══════════════════════════════════════════════════════════════
            st.markdown(f"### 🥇 Best ML Strategy: {best_ml_name}")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Sharpe Ratio", f"{best_ml['Sharpe_Ratio']:.3f}")
            with col2:
                st.metric("Annual Return", f"{best_ml['Annual_Return']*100:.2f}%")
            with col3:
                st.metric("Volatility", f"{best_ml['Volatility']*100:.2f}%")
            with col4:
                st.metric("Max Drawdown", f"{best_ml['Max_Drawdown']*100:.2f}%")

            st.markdown("---")

            # ═══════════════════════════════════════════════════════════════
            # SECTION 3 — SIDE-BY-SIDE BAR CHARTS
            # ═══════════════════════════════════════════════════════════════
            st.markdown("### 📊 Strategy Comparison")
            plot_df = comparison_df.copy()
            plot_df['Label'] = plot_df['Strategy'].str.replace('ML_', '').str.replace('_', ' ').str.title()
            plot_df['Color'] = plot_df['Strategy'].apply(lambda s: '#2196F3' if s.startswith('ML_') else '#FF9800')

            col1, col2 = st.columns(2)
            with col1:
                sorted_ret = plot_df.sort_values('Annual_Return', ascending=True)
                fig_ret = go.Figure(go.Bar(
                    x=sorted_ret['Annual_Return'] * 100,
                    y=sorted_ret['Label'],
                    orientation='h',
                    marker_color=sorted_ret['Color'].tolist(),
                    text=[f"{v*100:.1f}%" for v in sorted_ret['Annual_Return']],
                    textposition='auto',
                ))
                fig_ret.update_layout(title='Annual Return (blue=ML, orange=Benchmark)',
                                      xaxis_title='Annual Return (%)',
                                      template='plotly_white', height=420)
                st.plotly_chart(fig_ret, width='stretch')

            with col2:
                sorted_sr = plot_df.sort_values('Sharpe_Ratio', ascending=True)
                fig_sr = go.Figure(go.Bar(
                    x=sorted_sr['Sharpe_Ratio'],
                    y=sorted_sr['Label'],
                    orientation='h',
                    marker_color=sorted_sr['Color'].tolist(),
                    text=[f"{v:.3f}" for v in sorted_sr['Sharpe_Ratio']],
                    textposition='auto',
                ))
                fig_sr.update_layout(title='Sharpe Ratio (blue=ML, orange=Benchmark)',
                                     xaxis_title='Sharpe Ratio',
                                     template='plotly_white', height=420)
                st.plotly_chart(fig_sr, width='stretch')

            # ═══════════════════════════════════════════════════════════════
            # SECTION 4 — RISK-RETURN SCATTER
            # ═══════════════════════════════════════════════════════════════
            st.markdown("### 📈 Risk-Return Profile")
            st.plotly_chart(create_risk_return_scatter(comparison_df), width='stretch')

            # ═══════════════════════════════════════════════════════════════
            # SECTION 5 — FULL TABLE
            # ═══════════════════════════════════════════════════════════════
            st.markdown("### Complete Performance Table")
            display_df = comparison_df.copy()
            display_df['Strategy'] = display_df['Strategy'].str.replace('ML_', 'ML: ').str.replace('_', ' ')
            st.dataframe(
                display_df.style.format({
                    'Total_Return': '{:.2%}',
                    'Annual_Return': '{:.2%}',
                    'Volatility': '{:.2%}',
                    'Sharpe_Ratio': '{:.3f}',
                    'Sortino_Ratio': '{:.3f}',
                    'Max_Drawdown': '{:.2%}',
                    'Calmar_Ratio': '{:.3f}',
                }).background_gradient(cmap='RdYlGn', subset=['Sharpe_Ratio', 'Annual_Return', 'Calmar_Ratio'])
                .background_gradient(cmap='RdYlGn_r', subset=['Volatility', 'Max_Drawdown']),
                height=420,
                use_container_width=True,
            )
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <div style='text-align: center'>
            <p style='font-size: 12px; color: #666;'>
                Portfolio Optimizer v1.0<br>
                ML-Enhanced Investment Analysis<br>
                © 2026
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
