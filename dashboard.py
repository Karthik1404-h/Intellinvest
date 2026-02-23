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

@st.cache_data
def load_data():
    """Load all project data"""
    data = {}
    
    try:
        # Load strategy performance
        strategies = ['risk_parity', 'mean_variance', 'max_sharpe', 'cluster_based', 'min_variance']
        data['portfolio_values'] = {}
        data['portfolio_weights'] = {}
        data['performance_metrics'] = {}
        
        for strategy in strategies:
            # Portfolio values (returns)
            values_path = os.path.join(Config.RESULTS_DIR, f'portfolio_values_{strategy}.csv')
            if os.path.exists(values_path):
                data['portfolio_values'][strategy] = pd.read_csv(values_path, index_col=0, parse_dates=True)
            
            # Portfolio weights
            weights_path = os.path.join(Config.RESULTS_DIR, f'portfolio_weights_{strategy}.csv')
            if os.path.exists(weights_path):
                data['portfolio_weights'][strategy] = pd.read_csv(weights_path, index_col=0)
            
            # Performance metrics
            metrics_path = os.path.join(Config.RESULTS_DIR, f'performance_metrics_{strategy}.json')
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    data['performance_metrics'][strategy] = json.load(f)
        
        # Load clustering results
        cluster_path = os.path.join(Config.RESULTS_DIR, 'cluster_assignments_kmeans.csv')
        if os.path.exists(cluster_path):
            data['clusters'] = pd.read_csv(cluster_path)
        
        cluster_analysis_path = os.path.join(Config.RESULTS_DIR, 'cluster_analysis_kmeans.csv')
        if os.path.exists(cluster_analysis_path):
            data['cluster_analysis'] = pd.read_csv(cluster_analysis_path)
        
        # Load benchmark comparison
        benchmark_path = os.path.join(Config.RESULTS_DIR, 'benchmark_comparison_detailed.csv')
        if os.path.exists(benchmark_path):
            data['benchmark_comparison'] = pd.read_csv(benchmark_path)
        
        # Load strategy comparison
        strategy_comparison_path = os.path.join(Config.RESULTS_DIR, 'strategy_comparison_summary.csv')
        if os.path.exists(strategy_comparison_path):
            data['strategy_comparison'] = pd.read_csv(strategy_comparison_path, index_col=0)
        
        # Load S&P 500 benchmark
        spy_path = os.path.join(Config.RAW_DATA_DIR, 'SPY_benchmark.csv')
        if os.path.exists(spy_path):
            spy_df = pd.read_csv(spy_path)
            # Get date column (skip first 2 rows if they are headers)
            if spy_df.iloc[0, 0] == 'Ticker':
                spy_df = spy_df.iloc[2:]  # Skip header rows
            spy_dates = pd.to_datetime(spy_df.iloc[:, 0], errors='coerce')
            spy_close = pd.to_numeric(spy_df.iloc[:, 1], errors='coerce')
            # Remove any NaN values
            valid_idx = ~(spy_dates.isna() | spy_close.isna())
            spy_series = pd.Series(spy_close[valid_idx].values, index=spy_dates[valid_idx].values)
            spy_series = spy_series.sort_index()
            data['sp500'] = spy_series
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
    
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
        'sp500': '#808080'
    }
    
    # Add S&P 500 first as reference
    if 'sp500' in data:
        sp500_series = data['sp500']
        # Align with portfolio dates
        if data['portfolio_values']:
            first_strategy = list(data['portfolio_values'].keys())[0]
            portfolio_dates = data['portfolio_values'][first_strategy].index
            # Filter S&P 500 to matching dates
            sp500_aligned = sp500_series[sp500_series.index.isin(portfolio_dates)]
            if len(sp500_aligned) > 0:
                sp500_cumulative = sp500_aligned / sp500_aligned.iloc[0]
                fig.add_trace(go.Scatter(
                    x=sp500_cumulative.index,
                    y=sp500_cumulative.values,
                    mode='lines',
                    name='S&P 500 (Benchmark)',
                    line=dict(width=2, color=colors['sp500'], dash='dash'),
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
        title='Cumulative Portfolio Performance vs S&P 500',
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
    # Separate ML strategies from benchmarks
    comparison_df['Type'] = comparison_df['Strategy'].apply(
        lambda x: 'ML Strategy' if x.startswith('ML_') else 'Benchmark'
    )
    
    fig = px.scatter(
        comparison_df,
        x='Volatility',
        y='Annual_Return',
        size='Sharpe_Ratio',
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
    
    # Load data
    with st.spinner("Loading portfolio data..."):
        data = load_data()
    
    if not data:
        st.error("⚠️ No data found. Please run the optimization pipeline first.")
        st.code("python main.py --full")
        st.stop()
    
    # Navigation
    page = st.sidebar.radio(
        "Navigation",
        ["🏠 Overview", "📈 Performance Analysis", "💼 Portfolio Composition", 
         "🎯 Clustering Analysis", "⚖️ Risk Analysis", "🏆 Benchmark Comparison"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        "**About**: This dashboard displays results from ML-enhanced portfolio optimization "
        "using stock clustering and advanced optimization algorithms."
    )
    
    # ============================================
    # 🏠 OVERVIEW PAGE
    # ============================================
    if page == "🏠 Overview":
        st.title("🏠 Portfolio Optimization Dashboard")
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
        
        if 'benchmark_comparison' in data:
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
        
        if 'benchmark_comparison' in data:
            comparison_df = data['benchmark_comparison']
            
            st.markdown("### ML Strategies vs Simple Benchmarks")
            
            # Key insight
            ml_strategies = comparison_df[comparison_df['Strategy'].str.startswith('ML_')]
            benchmarks = comparison_df[~comparison_df['Strategy'].str.startswith('ML_')]
            
            # Check if we have both ML strategies and benchmarks
            if len(ml_strategies) == 0 or len(benchmarks) == 0:
                st.warning("⚠️ Benchmark comparison data incomplete. Showing available strategies only.")
                st.markdown("### Available Strategies")
                st.dataframe(
                    comparison_df.style.format({
                        'Total_Return': '{:.2%}',
                        'Annual_Return': '{:.2%}',
                        'Volatility': '{:.2%}',
                        'Sharpe_Ratio': '{:.3f}',
                        'Sortino_Ratio': '{:.3f}',
                        'Max_Drawdown': '{:.2%}',
                        'Calmar_Ratio': '{:.3f}'
                    }).background_gradient(cmap='RdYlGn', subset=['Sharpe_Ratio'])
                )
                return
            
            best_ml = ml_strategies.loc[ml_strategies['Sharpe_Ratio'].idxmax()]
            best_benchmark = benchmarks.loc[benchmarks['Sharpe_Ratio'].idxmax()]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.success(f"**Best ML Strategy**\n\n{best_ml['Strategy'].replace('ML_', '').replace('_', ' ').title()}")
                st.metric("Sharpe Ratio", f"{best_ml['Sharpe_Ratio']:.3f}")
                st.metric("Annual Return", f"{best_ml['Annual_Return']*100:.2f}%")
            
            with col2:
                st.info(f"**Best Benchmark**\n\n{best_benchmark['Strategy'].replace('_', ' ').title()}")
                st.metric("Sharpe Ratio", f"{best_benchmark['Sharpe_Ratio']:.3f}")
                st.metric("Annual Return", f"{best_benchmark['Annual_Return']*100:.2f}%")
            
            with col3:
                improvement = ((best_ml['Sharpe_Ratio'] - best_benchmark['Sharpe_Ratio']) / abs(best_benchmark['Sharpe_Ratio'])) * 100
                st.warning("**Performance Gap**")
                st.metric("Sharpe Improvement", f"{improvement:+.2f}%")
                
                if improvement > 0:
                    st.success("✅ ML adds value!")
                else:
                    st.error("❌ Underperforming")
            
            st.markdown("---")
            
            # Risk-return scatter
            st.plotly_chart(create_risk_return_scatter(comparison_df), width='stretch')
            
            # Complete comparison table
            st.markdown("### Complete Performance Comparison")
            
            display_df = comparison_df.copy()
            display_df['Strategy'] = display_df['Strategy'].str.replace('ML_', '').str.replace('_', ' ')
            
            st.dataframe(
                display_df.style.format({
                    'Total_Return': '{:.2%}',
                    'Annual_Return': '{:.2%}',
                    'Volatility': '{:.2%}',
                    'Sharpe_Ratio': '{:.3f}',
                    'Sortino_Ratio': '{:.3f}',
                    'Max_Drawdown': '{:.2%}',
                    'Calmar_Ratio': '{:.3f}'
                }).background_gradient(cmap='RdYlGn', subset=['Sharpe_Ratio', 'Annual_Return', 'Calmar_Ratio'])
                .background_gradient(cmap='RdYlGn_r', subset=['Volatility', 'Max_Drawdown']),
                height=600
            )
        else:
            st.warning("⚠️ Benchmark comparison data not found.")
    
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
