#!/usr/bin/env python3
"""
Benchmark Comparison: ML Portfolio Strategies vs Simple Benchmarks
Compare the ML-optimized portfolios against buy-and-hold strategies
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from datetime import datetime
from loguru import logger

from config import Config
from src.evaluation.backtesting import PerformanceMetrics

def load_strategy_results():
    """Load all portfolio strategy results"""
    logger.info("Loading ML portfolio strategy results...")
    
    strategies = {}
    strategy_names = ['risk_parity', 'mean_variance', 'max_sharpe', 'cluster_based', 'min_variance']
    
    for strategy in strategy_names:
        try:
            # Load portfolio returns (these are already returns, not cumulative values!)
            values_path = os.path.join(Config.RESULTS_DIR, f'portfolio_values_{strategy}.csv')
            if os.path.exists(values_path):
                portfolio_data = pd.read_csv(values_path, index_col=0, parse_dates=True)
                
                # The CSV actually contains returns, not cumulative values
                if isinstance(portfolio_data, pd.DataFrame):
                    returns = portfolio_data.iloc[:, 0]
                else:
                    returns = portfolio_data
                
                # Remove any NaN values
                returns = returns.dropna()
                
                strategies[f'ML_{strategy}'] = returns
                logger.info(f"Loaded {strategy}: {len(returns)} return observations")
        except Exception as e:
            logger.warning(f"Could not load {strategy}: {e}")
    
    return strategies

def create_benchmark_strategies():
    """Create simple benchmark strategies from the data"""
    logger.info("Creating benchmark strategies...")
    
    benchmarks = {}
    
    # Load the processed stock data
    processed_data_path = os.path.join(Config.PROCESSED_DATA_DIR, 'processed_stock_data.csv')
    
    try:
        # Read the CSV
        df = pd.read_csv(processed_data_path)
        
        # Find where actual data starts
        date_col = df.iloc[:, 0]
        first_valid_idx = None
        for i in range(len(date_col)):
            try:
                parsed_date = pd.to_datetime(date_col.iloc[i], errors='raise')
                if pd.notna(parsed_date):
                    first_valid_idx = i
                    break
            except:
                continue
        
        if first_valid_idx is None:
            logger.error("Could not find valid dates in processed data")
            return benchmarks
        
        # Extract dates
        dates = pd.to_datetime(df.iloc[first_valid_idx:, 0])
        
        # Extract ticker symbols from column names
        ticker_symbols = set()
        for col in df.columns[1:]:
            if '.' in col:
                base_symbol = col.split('.')[0]
            else:
                base_symbol = col
            ticker_symbols.add(base_symbol)
        
        ticker_symbols = sorted(list(ticker_symbols))
        logger.info(f"Found {len(ticker_symbols)} unique symbols")
        
        # Extract close prices for each symbol
        symbol_prices = {}
        price_types = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        
        for symbol in ticker_symbols:
            # Find columns for this symbol
            symbol_cols = []
            symbol_col_indices = []
            
            for i, col in enumerate(df.columns[1:], 1):
                if col == symbol or col.startswith(f"{symbol}."):
                    symbol_cols.append(col)
                    symbol_col_indices.append(i)
            
            if len(symbol_cols) >= 4:  # Need at least Close price
                # Get close price (4th column = index 3)
                close_idx = symbol_col_indices[3] if len(symbol_col_indices) > 3 else symbol_col_indices[0]
                close_prices = pd.to_numeric(df.iloc[first_valid_idx:, close_idx], errors='coerce')
                
                if close_prices.notna().sum() > 100:  # Need sufficient data
                    symbol_prices[symbol] = close_prices.values
        
        logger.info(f"Extracted prices for {len(symbol_prices)} symbols")
        
        # Create DataFrame with all close prices
        prices_df = pd.DataFrame(symbol_prices, index=dates.values)
        prices_df = prices_df.dropna(axis=1, how='all')
        
        # 1. Equal Weight Portfolio - buy all stocks equally
        equal_weight_returns = prices_df.pct_change().mean(axis=1).dropna()
        benchmarks['Equal_Weight_Portfolio'] = equal_weight_returns
        logger.info(f"Equal Weight Portfolio: {len(equal_weight_returns)} returns")
        
        # 2. Market Cap Weight Proxy - weight by average price (simplified)
        avg_prices = prices_df.mean()
        market_weights = avg_prices / avg_prices.sum()
        cap_weighted_returns = (prices_df.pct_change() * market_weights).sum(axis=1).dropna()
        benchmarks['Cap_Weighted_Portfolio'] = cap_weighted_returns
        logger.info(f"Cap Weighted Portfolio: {len(cap_weighted_returns)} returns")
        
        # 3. Single Best Stock (in hindsight - for comparison)
        stock_returns = prices_df.pct_change()
        total_returns = (1 + stock_returns).prod() - 1
        best_stock = total_returns.idxmax()
        benchmarks[f'Best_Stock_{best_stock}'] = stock_returns[best_stock].dropna()
        logger.info(f"Best Stock ({best_stock}): {total_returns[best_stock]:.2%} total return")
        
        # 4. Worst Stock (for reference)
        worst_stock = total_returns.idxmin()
        benchmarks[f'Worst_Stock_{worst_stock}'] = stock_returns[worst_stock].dropna()
        logger.info(f"Worst Stock ({worst_stock}): {total_returns[worst_stock]:.2%} total return")
        
        # 5. Random Selection (10 stocks)
        np.random.seed(42)
        random_stocks = np.random.choice(list(symbol_prices.keys()), size=min(10, len(symbol_prices)), replace=False)
        random_portfolio_returns = prices_df[random_stocks].pct_change().mean(axis=1).dropna()
        benchmarks['Random_10_Stocks'] = random_portfolio_returns
        logger.info(f"Random 10 Stocks Portfolio: {len(random_portfolio_returns)} returns")
        
    except Exception as e:
        logger.error(f"Error creating benchmarks: {e}")
        import traceback
        traceback.print_exc()
    
    # Load SPY benchmark if available
    try:
        spy_path = os.path.join(Config.RAW_DATA_DIR, 'SPY_benchmark.csv')
        if os.path.exists(spy_path):
            spy_df = pd.read_csv(spy_path)
            
            # Find valid dates in SPY data
            spy_date_col = spy_df.iloc[:, 0]
            spy_first_valid_idx = None
            for i in range(len(spy_date_col)):
                try:
                    parsed = pd.to_datetime(spy_date_col.iloc[i], errors='raise')
                    if pd.notna(parsed):
                        spy_first_valid_idx = i
                        break
                except:
                    continue
            
            if spy_first_valid_idx is not None:
                spy_dates = pd.to_datetime(spy_df.iloc[spy_first_valid_idx:, 0])
                spy_close = pd.to_numeric(spy_df.iloc[spy_first_valid_idx:, 1], errors='coerce')
                spy_series = pd.Series(spy_close.values, index=spy_dates.values)
                spy_returns = spy_series.pct_change().dropna()
                benchmarks['SPY_SP500'] = spy_returns
                logger.info(f"SPY (S&P 500): {len(spy_returns)} returns")
    except Exception as e:
        logger.warning(f"Could not load SPY benchmark: {e}")
    
    return benchmarks

def calculate_all_metrics(strategies_dict):
    """Calculate performance metrics for all strategies"""
    logger.info("Calculating performance metrics...")
    
    results = []
    
    for name, returns in strategies_dict.items():
        try:
            if len(returns) < 50:
                logger.warning(f"Insufficient data for {name}")
                continue
            
            metrics = {
                'Strategy': name,
                'Total_Return': PerformanceMetrics.calculate_total_return(returns),
                'Annual_Return': PerformanceMetrics.calculate_annualized_return(returns),
                'Volatility': PerformanceMetrics.calculate_volatility(returns),
                'Sharpe_Ratio': PerformanceMetrics.calculate_sharpe_ratio(returns),
                'Sortino_Ratio': PerformanceMetrics.calculate_sortino_ratio(returns),
                'Max_Drawdown': PerformanceMetrics.calculate_max_drawdown(returns),
                'Calmar_Ratio': PerformanceMetrics.calculate_calmar_ratio(returns),
                'Observations': len(returns)
            }
            results.append(metrics)
            
        except Exception as e:
            logger.warning(f"Error calculating metrics for {name}: {e}")
    
    return pd.DataFrame(results)

def analyze_and_visualize(comparison_df):
    """Analyze results and create visualizations"""
    
    # Separate ML strategies from benchmarks
    ml_strategies = comparison_df[comparison_df['Strategy'].str.startswith('ML_')]
    benchmark_strategies = comparison_df[~comparison_df['Strategy'].str.startswith('ML_')]
    
    print("\n" + "="*100)
    print("📊 COMPREHENSIVE PERFORMANCE COMPARISON: ML STRATEGIES vs BENCHMARKS")
    print("="*100)
    
    print("\n🤖 ML-OPTIMIZED STRATEGIES:")
    print(ml_strategies.to_string(index=False))
    
    print("\n📈 SIMPLE BENCHMARK STRATEGIES:")
    print(benchmark_strategies.to_string(index=False))
    
    print("\n" + "="*100)
    print("🏆 OVERALL RANKINGS")
    print("="*100)
    
    # Rankings
    metrics_to_rank = ['Sharpe_Ratio', 'Annual_Return', 'Calmar_Ratio', 'Sortino_Ratio']
    
    for metric in metrics_to_rank:
        sorted_df = comparison_df.sort_values(metric, ascending=False)
        top_strategy = sorted_df.iloc[0]
        print(f"\n{metric.replace('_', ' ')}:")
        print(f"  🥇 Winner: {top_strategy['Strategy']} = {top_strategy[metric]:.4f}")
        print(f"  Top 3: {', '.join([f'{s} ({v:.3f})' for s, v in zip(sorted_df['Strategy'].head(3), sorted_df[metric].head(3))])}")
    
    # Value-add analysis
    print("\n" + "="*100)
    print("💡 VALUE-ADD ANALYSIS: DO ML STRATEGIES BEAT SIMPLE BENCHMARKS?")
    print("="*100)
    
    if len(ml_strategies) > 0 and len(benchmark_strategies) > 0:
        # Best ML vs Best Benchmark
        best_ml_sharpe = ml_strategies.loc[ml_strategies['Sharpe_Ratio'].idxmax()]
        best_benchmark_sharpe = benchmark_strategies.loc[benchmark_strategies['Sharpe_Ratio'].idxmax()]
        
        print(f"\n📊 SHARPE RATIO COMPARISON:")
        print(f"  Best ML Strategy: {best_ml_sharpe['Strategy']}")
        print(f"    Sharpe Ratio: {best_ml_sharpe['Sharpe_Ratio']:.4f}")
        print(f"    Annual Return: {best_ml_sharpe['Annual_Return']:.2%}")
        print(f"    Volatility: {best_ml_sharpe['Volatility']:.2%}")
        
        print(f"\n  Best Benchmark: {best_benchmark_sharpe['Strategy']}")
        print(f"    Sharpe Ratio: {best_benchmark_sharpe['Sharpe_Ratio']:.4f}")
        print(f"    Annual Return: {best_benchmark_sharpe['Annual_Return']:.2%}")
        print(f"    Volatility: {best_benchmark_sharpe['Volatility']:.2%}")
        
        sharpe_improvement = ((best_ml_sharpe['Sharpe_Ratio'] - best_benchmark_sharpe['Sharpe_Ratio']) / 
                             abs(best_benchmark_sharpe['Sharpe_Ratio'])) * 100
        
        print(f"\n  📈 Sharpe Ratio Improvement: {sharpe_improvement:+.2f}%")
        
        # Return comparison
        best_ml_return = ml_strategies.loc[ml_strategies['Annual_Return'].idxmax()]
        best_benchmark_return = benchmark_strategies.loc[benchmark_strategies['Annual_Return'].idxmax()]
        
        return_improvement = ((best_ml_return['Annual_Return'] - best_benchmark_return['Annual_Return']) / 
                             abs(best_benchmark_return['Annual_Return'])) * 100
        
        print(f"\n  💰 Annual Return Improvement: {return_improvement:+.2f}%")
        
        # Verdict
        print("\n" + "="*100)
        print("⚖️  FINAL VERDICT")
        print("="*100)
        
        points = 0
        total_points = 4
        
        if best_ml_sharpe['Sharpe_Ratio'] > best_benchmark_sharpe['Sharpe_Ratio']:
            print("✅ ML strategies have BETTER risk-adjusted returns (Sharpe)")
            points += 1
        else:
            print("❌ ML strategies have WORSE risk-adjusted returns (Sharpe)")
        
        if best_ml_return['Annual_Return'] > best_benchmark_return['Annual_Return']:
            print("✅ ML strategies generate HIGHER absolute returns")
            points += 1
        else:
            print("❌ ML strategies generate LOWER absolute returns")
        
        if ml_strategies['Volatility'].mean() < benchmark_strategies['Volatility'].mean():
            print("✅ ML strategies have LOWER average volatility")
            points += 1
        else:
            print("❌ ML strategies have HIGHER average volatility")
        
        if ml_strategies['Max_Drawdown'].mean() > benchmark_strategies['Max_Drawdown'].mean():
            print("✅ ML strategies have BETTER drawdown control")
            points += 1
        else:
            print("❌ ML strategies have WORSE drawdown control")
        
        score_pct = (points / total_points) * 100
        
        print(f"\n📊 ML Strategy Score: {points}/{total_points} ({score_pct:.0f}%)")
        
        if score_pct >= 75:
            print("\n🎉 STRONG ADVANTAGE: ML approach significantly outperforms simple strategies!")
            print("   Your sophisticated optimization is WORTH IT!")
        elif score_pct >= 50:
            print("\n✅ MODERATE ADVANTAGE: ML approach shows benefits over simple strategies")
            print("   Your optimization adds value, especially in risk-adjusted terms")
        else:
            print("\n⚠️  LIMITED ADVANTAGE: ML approach shows mixed results vs simple strategies")
            print("   Consider simplification or parameter tuning")
    
    # Create visualizations
    create_comparison_plots(comparison_df, ml_strategies, benchmark_strategies)
    
    return comparison_df

def create_comparison_plots(comparison_df, ml_strategies, benchmark_strategies):
    """Create comprehensive comparison visualizations"""
    
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Risk-Return Scatter
    ax1 = plt.subplot(2, 3, 1)
    
    # Plot ML strategies
    ax1.scatter(ml_strategies['Volatility'], ml_strategies['Annual_Return'], 
               s=200, alpha=0.7, c='blue', marker='o', label='ML Strategies', edgecolors='black', linewidths=2)
    
    # Plot benchmarks
    ax1.scatter(benchmark_strategies['Volatility'], benchmark_strategies['Annual_Return'],
               s=200, alpha=0.7, c='red', marker='s', label='Benchmarks', edgecolors='black', linewidths=2)
    
    # Add labels
    for _, row in comparison_df.iterrows():
        ax1.annotate(row['Strategy'].replace('ML_', '').replace('_', ' ')[:15], 
                    (row['Volatility'], row['Annual_Return']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax1.set_xlabel('Volatility (Risk)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Annual Return', fontsize=12, fontweight='bold')
    ax1.set_title('Risk-Return Profile: ML vs Benchmarks', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. Sharpe Ratio Comparison
    ax2 = plt.subplot(2, 3, 2)
    sharpe_sorted = comparison_df.sort_values('Sharpe_Ratio', ascending=True)
    colors = ['blue' if s.startswith('ML_') else 'red' for s in sharpe_sorted['Strategy']]
    y_pos = np.arange(len(sharpe_sorted))
    ax2.barh(y_pos, sharpe_sorted['Sharpe_Ratio'], color=colors, alpha=0.7, edgecolor='black')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([s.replace('ML_', '').replace('_', ' ')[:20] for s in sharpe_sorted['Strategy']], fontsize=9)
    ax2.set_xlabel('Sharpe Ratio', fontsize=12, fontweight='bold')
    ax2.set_title('Sharpe Ratio Comparison', fontsize=14, fontweight='bold')
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax2.grid(True, alpha=0.3, axis='x')
    
    # 3. Annual Return Comparison
    ax3 = plt.subplot(2, 3, 3)
    return_sorted = comparison_df.sort_values('Annual_Return', ascending=True)
    colors = ['blue' if s.startswith('ML_') else 'red' for s in return_sorted['Strategy']]
    y_pos = np.arange(len(return_sorted))
    ax3.barh(y_pos, return_sorted['Annual_Return'] * 100, color=colors, alpha=0.7, edgecolor='black')
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels([s.replace('ML_', '').replace('_', ' ')[:20] for s in return_sorted['Strategy']], fontsize=9)
    ax3.set_xlabel('Annual Return (%)', fontsize=12, fontweight='bold')
    ax3.set_title('Annual Return Comparison', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')
    
    # 4. Max Drawdown Comparison
    ax4 = plt.subplot(2, 3, 4)
    dd_sorted = comparison_df.sort_values('Max_Drawdown', ascending=False)
    colors = ['blue' if s.startswith('ML_') else 'red' for s in dd_sorted['Strategy']]
    y_pos = np.arange(len(dd_sorted))
    ax4.barh(y_pos, dd_sorted['Max_Drawdown'] * 100, color=colors, alpha=0.7, edgecolor='black')
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels([s.replace('ML_', '').replace('_', ' ')[:20] for s in dd_sorted['Strategy']], fontsize=9)
    ax4.set_xlabel('Max Drawdown (%)', fontsize=12, fontweight='bold')
    ax4.set_title('Maximum Drawdown Comparison', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x')
    
    # 5. Sortino Ratio Comparison
    ax5 = plt.subplot(2, 3, 5)
    sortino_sorted = comparison_df.sort_values('Sortino_Ratio', ascending=True)
    colors = ['blue' if s.startswith('ML_') else 'red' for s in sortino_sorted['Strategy']]
    y_pos = np.arange(len(sortino_sorted))
    ax5.barh(y_pos, sortino_sorted['Sortino_Ratio'], color=colors, alpha=0.7, edgecolor='black')
    ax5.set_yticks(y_pos)
    ax5.set_yticklabels([s.replace('ML_', '').replace('_', ' ')[:20] for s in sortino_sorted['Strategy']], fontsize=9)
    ax5.set_xlabel('Sortino Ratio', fontsize=12, fontweight='bold')
    ax5.set_title('Sortino Ratio Comparison', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='x')
    
    # 6. Metrics Heatmap
    ax6 = plt.subplot(2, 3, 6)
    metrics_for_heatmap = comparison_df.set_index('Strategy')[['Sharpe_Ratio', 'Sortino_Ratio', 'Calmar_Ratio', 'Annual_Return']]
    metrics_normalized = (metrics_for_heatmap - metrics_for_heatmap.min()) / (metrics_for_heatmap.max() - metrics_for_heatmap.min())
    sns.heatmap(metrics_normalized.T, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax6, 
                cbar_kws={'label': 'Normalized Score'}, linewidths=0.5)
    ax6.set_title('Normalized Performance Metrics', fontsize=14, fontweight='bold')
    ax6.set_xlabel('')
    ax6.set_xticklabels([s.replace('ML_', '').replace('_', ' ')[:15] for s in metrics_normalized.index], 
                        rotation=45, ha='right', fontsize=8)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(Config.RESULTS_DIR, 'benchmark_comparison_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved comparison plot to: {output_path}")
    
    plt.show()

def main():
    """Main benchmark comparison function"""
    logger.info("Starting Benchmark Comparison Analysis")
    logger.info("="*80)
    
    # Load ML strategy results
    ml_strategies = load_strategy_results()
    
    if not ml_strategies:
        logger.error("No ML strategy results found. Please run the optimization pipeline first.")
        return
    
    # Create benchmark strategies
    benchmarks = create_benchmark_strategies()
    
    if not benchmarks:
        logger.error("Could not create benchmark strategies")
        return
    
    # Combine all strategies
    all_strategies = {**ml_strategies, **benchmarks}
    
    logger.info(f"Total strategies to compare: {len(all_strategies)}")
    logger.info(f"  - ML Strategies: {len(ml_strategies)}")
    logger.info(f"  - Benchmarks: {len(benchmarks)}")
    
    # Calculate metrics for all
    comparison_df = calculate_all_metrics(all_strategies)
    
    # Analyze and visualize
    results = analyze_and_visualize(comparison_df)
    
    # Save results
    output_path = os.path.join(Config.RESULTS_DIR, 'benchmark_comparison_detailed.csv')
    results.to_csv(output_path, index=False)
    logger.info(f"\nDetailed results saved to: {output_path}")
    
    logger.info("\n✅ Benchmark comparison completed successfully!")
    
    return results

if __name__ == "__main__":
    main()
