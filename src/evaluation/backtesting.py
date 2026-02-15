"""
Backtesting and evaluation framework for portfolio optimization strategies
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    import pyfolio as pf
    PYFOLIO_AVAILABLE = True
except ImportError:
    PYFOLIO_AVAILABLE = False
    print("Warning: pyfolio not available. Some evaluation metrics may not be computed.")

from loguru import logger
import os

from config import Config

class PerformanceMetrics:
    """Calculate portfolio performance metrics"""
    
    @staticmethod
    def calculate_returns(prices: pd.Series) -> pd.Series:
        """Calculate returns from price series"""
        return prices.pct_change().dropna()
    
    @staticmethod
    def calculate_cumulative_returns(returns: pd.Series) -> pd.Series:
        """Calculate cumulative returns"""
        return (1 + returns).cumprod() - 1
    
    @staticmethod
    def calculate_total_return(returns: pd.Series) -> float:
        """Calculate total return over the period"""
        return (1 + returns).prod() - 1
    
    @staticmethod
    def calculate_annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
        """Calculate annualized return"""
        total_return = PerformanceMetrics.calculate_total_return(returns)
        n_periods = len(returns)
        years = n_periods / periods_per_year
        return (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    
    @staticmethod
    def calculate_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
        """Calculate annualized volatility"""
        return returns.std() * np.sqrt(periods_per_year)
    
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02, 
                              periods_per_year: int = 252) -> float:
        """Calculate Sharpe ratio"""
        excess_returns = returns - risk_free_rate / periods_per_year
        return np.sqrt(periods_per_year) * excess_returns.mean() / returns.std() if returns.std() != 0 else 0
    
    @staticmethod
    def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02,
                               periods_per_year: int = 252) -> float:
        """Calculate Sortino ratio"""
        excess_returns = returns - risk_free_rate / periods_per_year
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else returns.std()
        return np.sqrt(periods_per_year) * excess_returns.mean() / downside_std if downside_std != 0 else 0
    
    @staticmethod
    def calculate_max_drawdown(returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        return drawdown.min()
    
    @staticmethod
    def calculate_calmar_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
        """Calculate Calmar ratio (annualized return / max drawdown)"""
        annual_return = PerformanceMetrics.calculate_annualized_return(returns, periods_per_year)
        max_dd = abs(PerformanceMetrics.calculate_max_drawdown(returns))
        return annual_return / max_dd if max_dd != 0 else 0
    
    @staticmethod
    def calculate_information_ratio(portfolio_returns: pd.Series, 
                                  benchmark_returns: pd.Series) -> float:
        """Calculate information ratio vs benchmark"""
        excess_returns = portfolio_returns - benchmark_returns
        tracking_error = excess_returns.std()
        return excess_returns.mean() / tracking_error if tracking_error != 0 else 0
    
    @staticmethod
    def calculate_beta(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate portfolio beta vs benchmark"""
        aligned_data = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
        if len(aligned_data) < 2:
            return 1.0
        
        covariance = aligned_data.cov().iloc[0, 1]
        benchmark_variance = aligned_data.iloc[:, 1].var()
        
        return covariance / benchmark_variance if benchmark_variance != 0 else 1.0
    
    @staticmethod
    def calculate_alpha(portfolio_returns: pd.Series, benchmark_returns: pd.Series,
                       risk_free_rate: float = 0.02, periods_per_year: int = 252) -> float:
        """Calculate Jensen's alpha"""
        portfolio_annual = PerformanceMetrics.calculate_annualized_return(portfolio_returns, periods_per_year)
        benchmark_annual = PerformanceMetrics.calculate_annualized_return(benchmark_returns, periods_per_year)
        beta = PerformanceMetrics.calculate_beta(portfolio_returns, benchmark_returns)
        
        expected_return = risk_free_rate + beta * (benchmark_annual - risk_free_rate)
        return portfolio_annual - expected_return

class PortfolioBacktester:
    """Main backtesting engine"""
    
    def __init__(self):
        self.config = Config()
        self.results = {}
    
    def run_backtest(self,
                    price_data: pd.DataFrame,
                    optimization_method: str,
                    rebalancing_freq: str = 'monthly',
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None,
                    initial_capital: float = 1000000,
                    transaction_cost: float = 0.001,
                    **optimization_kwargs) -> Dict[str, Any]:
        """
        Run backtest for a portfolio optimization strategy
        
        Args:
            price_data: Historical price data
            optimization_method: Portfolio optimization method to use
            rebalancing_freq: Rebalancing frequency ('daily', 'weekly', 'monthly', 'quarterly')
            start_date: Backtest start date
            end_date: Backtest end date
            initial_capital: Starting capital
            transaction_cost: Transaction cost as fraction of trade value
            **optimization_kwargs: Additional arguments for optimization
            
        Returns:
            Dictionary with backtest results
        """
        logger.info(f"Running backtest for {optimization_method} strategy")
        
        # Prepare data
        if start_date:
            price_data = price_data[price_data.index >= start_date]
        if end_date:
            price_data = price_data[price_data.index <= end_date]
        
        # Get rebalancing dates
        rebalancing_dates = self._get_rebalancing_dates(price_data.index, rebalancing_freq)
        
        # Initialize tracking variables
        portfolio_values = []
        portfolio_weights_history = []
        turnover_history = []
        transaction_costs_history = []
        
        current_capital = initial_capital
        current_weights = None
        
        from src.optimization import PortfolioOptimizer
        optimizer = PortfolioOptimizer()
        
        # Run backtest
        for i, rebal_date in enumerate(rebalancing_dates[:-1]):
            try:
                # Get data up to rebalancing date
                historical_data = price_data[price_data.index <= rebal_date]
                
                if len(historical_data) < 100:  # Need sufficient history
                    continue
                
                # Optimize portfolio
                optimization_result = optimizer.optimize_portfolio(
                    historical_data,
                    method=optimization_method,
                    **optimization_kwargs
                )
                
                new_weights = optimization_result['weights']
                
                # Calculate turnover and transaction costs
                if current_weights is not None:
                    turnover = self._calculate_turnover(current_weights, new_weights)
                    transaction_costs = turnover * transaction_cost * current_capital
                    current_capital -= transaction_costs
                else:
                    turnover = 0
                    transaction_costs = 0
                
                # Simulate portfolio performance until next rebalancing
                next_rebal_date = rebalancing_dates[i + 1]
                period_data = price_data[(price_data.index > rebal_date) & 
                                       (price_data.index <= next_rebal_date)]
                
                if len(period_data) > 0:
                    period_returns = self._calculate_portfolio_returns(
                        period_data, new_weights
                    )
                    
                    # Update capital
                    period_value = current_capital * (1 + period_returns).cumprod()
                    current_capital = period_value.iloc[-1]
                    
                    # Store results
                    for date, value in period_value.items():
                        portfolio_values.append({'date': date, 'value': value})
                
                # Update weights
                current_weights = new_weights.copy()
                portfolio_weights_history.append({
                    'date': rebal_date, 
                    'weights': new_weights.to_dict()
                })
                turnover_history.append({'date': rebal_date, 'turnover': turnover})
                transaction_costs_history.append({
                    'date': rebal_date, 
                    'cost': transaction_costs
                })
                
            except Exception as e:
                logger.warning(f"Error at rebalancing date {rebal_date}: {e}")
                continue
        
        # Convert results to DataFrames
        portfolio_df = pd.DataFrame(portfolio_values).set_index('date')['value']
        portfolio_returns = portfolio_df.pct_change().dropna()
        
        # Calculate performance metrics
        metrics = self._calculate_performance_metrics(portfolio_returns)
        
        # Add additional backtest information
        metrics.update({
            'initial_capital': initial_capital,
            'final_capital': portfolio_df.iloc[-1] if len(portfolio_df) > 0 else initial_capital,
            'total_transaction_costs': sum([tc['cost'] for tc in transaction_costs_history]),
            'average_turnover': np.mean([th['turnover'] for th in turnover_history]),
            'number_of_rebalances': len(rebalancing_dates) - 1
        })
        
        results = {
            'portfolio_values': portfolio_df,
            'portfolio_returns': portfolio_returns,
            'weights_history': pd.DataFrame(portfolio_weights_history),
            'turnover_history': pd.DataFrame(turnover_history),
            'transaction_costs_history': pd.DataFrame(transaction_costs_history),
            'performance_metrics': metrics,
            'optimization_method': optimization_method,
            'rebalancing_frequency': rebalancing_freq
        }
        
        logger.info(f"Backtest completed. Final capital: ${metrics['final_capital']:,.0f}")
        return results
    
    def _get_rebalancing_dates(self, 
                              date_index: pd.DatetimeIndex, 
                              frequency: str) -> List[datetime]:
        """Get rebalancing dates based on frequency"""
        if frequency == 'daily':
            return list(date_index)
        elif frequency == 'weekly':
            # Use Monday as rebalancing day
            return [d for d in date_index if d.weekday() == 0]
        elif frequency == 'monthly':
            # Use first trading day of each month
            monthly_dates = []
            current_month = None
            for date in date_index:
                if current_month != date.month:
                    monthly_dates.append(date)
                    current_month = date.month
            return monthly_dates
        elif frequency == 'quarterly':
            # Use first trading day of each quarter
            quarterly_dates = []
            current_quarter = None
            for date in date_index:
                quarter = (date.month - 1) // 3 + 1
                if current_quarter != quarter:
                    quarterly_dates.append(date)
                    current_quarter = quarter
            return quarterly_dates
        else:
            raise ValueError(f"Unknown rebalancing frequency: {frequency}")
    
    def _calculate_turnover(self, old_weights: pd.Series, new_weights: pd.Series) -> float:
        """Calculate portfolio turnover"""
        aligned_old = old_weights.reindex(new_weights.index, fill_value=0)
        return (aligned_old - new_weights).abs().sum() / 2
    
    def _calculate_portfolio_returns(self, 
                                   price_data: pd.DataFrame, 
                                   weights: pd.Series) -> pd.Series:
        """Calculate portfolio returns for a given period"""
        # Get valid symbols that exist in both weights and price data
        valid_symbols = []
        for symbol in weights.index:
            if (symbol, 'Close') in price_data.columns:
                valid_symbols.append(symbol)
        
        if not valid_symbols:
            return pd.Series(index=price_data.index, data=0)
        
        # Get price data for valid symbols
        symbol_prices = {}
        for symbol in valid_symbols:
            symbol_prices[symbol] = price_data[symbol]['Close']
        
        prices_df = pd.DataFrame(symbol_prices)
        
        # Calculate returns
        returns_df = prices_df.pct_change().dropna()
        
        # Align weights with available symbols
        aligned_weights = weights.reindex(valid_symbols, fill_value=0)
        aligned_weights = aligned_weights / aligned_weights.sum()  # Renormalize
        
        # Calculate portfolio returns
        portfolio_returns = (returns_df * aligned_weights).sum(axis=1)
        
        return portfolio_returns
    
    def _calculate_performance_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        if len(returns) == 0:
            return {}
        
        metrics = {
            'total_return': PerformanceMetrics.calculate_total_return(returns),
            'annualized_return': PerformanceMetrics.calculate_annualized_return(returns),
            'volatility': PerformanceMetrics.calculate_volatility(returns),
            'sharpe_ratio': PerformanceMetrics.calculate_sharpe_ratio(returns),
            'sortino_ratio': PerformanceMetrics.calculate_sortino_ratio(returns),
            'max_drawdown': PerformanceMetrics.calculate_max_drawdown(returns),
            'calmar_ratio': PerformanceMetrics.calculate_calmar_ratio(returns)
        }
        
        return metrics
    
    def compare_strategies(self,
                          price_data: pd.DataFrame,
                          strategies: List[str],
                          benchmark_symbol: str = 'SPY',
                          **backtest_kwargs) -> Dict[str, Any]:
        """
        Compare multiple portfolio strategies
        
        Args:
            price_data: Historical price data
            strategies: List of optimization methods to compare
            benchmark_symbol: Benchmark symbol for comparison
            **backtest_kwargs: Arguments for backtesting
            
        Returns:
            Comparison results
        """
        logger.info(f"Comparing {len(strategies)} strategies")
        
        results = {}
        
        # Run backtest for each strategy
        for strategy in strategies:
            try:
                logger.info(f"Backtesting {strategy}")
                result = self.run_backtest(
                    price_data=price_data,
                    optimization_method=strategy,
                    **backtest_kwargs
                )
                results[strategy] = result
            except Exception as e:
                logger.error(f"Error backtesting {strategy}: {e}")
                continue
        
        # Add benchmark
        if (benchmark_symbol, 'Close') in price_data.columns:
            benchmark_prices = price_data[benchmark_symbol]['Close']
            benchmark_returns = benchmark_prices.pct_change().dropna()
            
            # Align with strategy returns
            common_dates = None
            for strategy_result in results.values():
                if common_dates is None:
                    common_dates = strategy_result['portfolio_returns'].index
                else:
                    common_dates = common_dates.intersection(strategy_result['portfolio_returns'].index)
            
            if common_dates is not None and len(common_dates) > 0:
                aligned_benchmark = benchmark_returns.reindex(common_dates, method='ffill').dropna()
                
                results['benchmark'] = {
                    'portfolio_returns': aligned_benchmark,
                    'performance_metrics': self._calculate_performance_metrics(aligned_benchmark),
                    'optimization_method': benchmark_symbol
                }
        
        # Create comparison summary
        comparison_metrics = self._create_comparison_summary(results)
        
        return {
            'individual_results': results,
            'comparison_metrics': comparison_metrics
        }
    
    def _create_comparison_summary(self, results: Dict[str, Any]) -> pd.DataFrame:
        """Create summary comparison of strategies"""
        summary_data = []
        
        for strategy_name, result in results.items():
            metrics = result['performance_metrics']
            
            summary_data.append({
                'Strategy': strategy_name,
                'Total Return': metrics.get('total_return', 0),
                'Annual Return': metrics.get('annualized_return', 0),
                'Volatility': metrics.get('volatility', 0),
                'Sharpe Ratio': metrics.get('sharpe_ratio', 0),
                'Sortino Ratio': metrics.get('sortino_ratio', 0),
                'Max Drawdown': metrics.get('max_drawdown', 0),
                'Calmar Ratio': metrics.get('calmar_ratio', 0)
            })
        
        return pd.DataFrame(summary_data).set_index('Strategy')

class VisualizationTools:
    """Tools for visualizing backtest results"""
    
    @staticmethod
    def plot_performance_comparison(comparison_results: Dict[str, Any], 
                                  save_path: Optional[str] = None):
        """Plot performance comparison of strategies"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Cumulative returns
        ax1 = axes[0, 0]
        for strategy_name, result in comparison_results['individual_results'].items():
            returns = result['portfolio_returns']
            cumulative_returns = (1 + returns).cumprod()
            ax1.plot(cumulative_returns.index, cumulative_returns.values, 
                    label=strategy_name, linewidth=2)
        
        ax1.set_title('Cumulative Returns')
        ax1.set_ylabel('Cumulative Return')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Rolling Sharpe ratio
        ax2 = axes[0, 1]
        for strategy_name, result in comparison_results['individual_results'].items():
            returns = result['portfolio_returns']
            rolling_sharpe = returns.rolling(252).apply(
                lambda x: PerformanceMetrics.calculate_sharpe_ratio(x)
            )
            ax2.plot(rolling_sharpe.index, rolling_sharpe.values, 
                    label=strategy_name, linewidth=2)
        
        ax2.set_title('Rolling 1-Year Sharpe Ratio')
        ax2.set_ylabel('Sharpe Ratio')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Drawdown
        ax3 = axes[1, 0]
        for strategy_name, result in comparison_results['individual_results'].items():
            returns = result['portfolio_returns']
            cumulative = (1 + returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative - rolling_max) / rolling_max
            ax3.fill_between(drawdown.index, drawdown.values, 0, 
                           alpha=0.3, label=strategy_name)
        
        ax3.set_title('Drawdown')
        ax3.set_ylabel('Drawdown')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Performance metrics comparison
        ax4 = axes[1, 1]
        metrics_df = comparison_results['comparison_metrics']
        
        # Plot Sharpe vs Volatility
        scatter = ax4.scatter(metrics_df['Volatility'], metrics_df['Sharpe Ratio'], 
                            s=100, alpha=0.7)
        
        for i, strategy in enumerate(metrics_df.index):
            ax4.annotate(strategy, 
                        (metrics_df['Volatility'].iloc[i], metrics_df['Sharpe Ratio'].iloc[i]),
                        xytext=(5, 5), textcoords='offset points')
        
        ax4.set_xlabel('Volatility')
        ax4.set_ylabel('Sharpe Ratio')
        ax4.set_title('Risk-Return Profile')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def plot_correlation_matrix(returns_df: pd.DataFrame, 
                              title: str = "Strategy Correlation Matrix",
                              save_path: Optional[str] = None):
        """Plot correlation matrix of strategy returns"""
        plt.figure(figsize=(10, 8))
        
        correlation_matrix = returns_df.corr()
        
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f', cbar_kws={"shrink": .8})
        
        plt.title(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

def main():
    """Main function to run comprehensive backtesting"""
    logger.info("Starting comprehensive portfolio backtesting")
    
    try:
        # Load data
        processed_data_path = os.path.join(Config.PROCESSED_DATA_DIR, 'processed_stock_data.csv')
        
        price_data = pd.read_csv(processed_data_path, index_col=0, header=[0, 1])
        price_data.index = pd.to_datetime(price_data.index)
        
        logger.info("Data loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return
    
    # Initialize backtester
    backtester = PortfolioBacktester()
    
    # Define strategies to test
    strategies = [
        'mean_variance',
        'risk_parity', 
        'min_variance',
        'max_sharpe'
    ]
    
    # Check if cluster data exists
    cluster_data_path = os.path.join(Config.RESULTS_DIR, 'cluster_assignments_kmeans.csv')
    if os.path.exists(cluster_data_path):
        strategies.append('cluster_based')
    
    # Run comparison
    logger.info(f"Comparing strategies: {strategies}")
    
    comparison_results = backtester.compare_strategies(
        price_data=price_data,
        strategies=strategies,
        rebalancing_freq='monthly',
        start_date='2020-01-01',  # Use recent data for faster testing
        transaction_cost=0.001
    )
    
    # Display results
    print("\nSTRATEGY COMPARISON RESULTS:")
    print("=" * 80)
    print(comparison_results['comparison_metrics'].round(4))
    
    # Create visualizations
    results_dir = Config.RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)
    
    # Plot performance comparison
    VisualizationTools.plot_performance_comparison(
        comparison_results,
        save_path=os.path.join(results_dir, 'strategy_comparison.png')
    )
    
    # Save detailed results
    for strategy_name, result in comparison_results['individual_results'].items():
        # Save portfolio values
        portfolio_values_path = os.path.join(results_dir, f'portfolio_values_{strategy_name}.csv')
        result['portfolio_returns'].to_csv(portfolio_values_path)
        
        # Save performance metrics
        metrics_path = os.path.join(results_dir, f'performance_metrics_{strategy_name}.json')
        import json
        with open(metrics_path, 'w') as f:
            json.dump(result['performance_metrics'], f, indent=2)
    
    # Save comparison summary
    summary_path = os.path.join(results_dir, 'strategy_comparison_summary.csv')
    comparison_results['comparison_metrics'].to_csv(summary_path)
    
    # Final summary
    best_sharpe = comparison_results['comparison_metrics']['Sharpe Ratio'].idxmax()
    best_return = comparison_results['comparison_metrics']['Annual Return'].idxmax()
    
    print(f"\nBEST PERFORMING STRATEGIES:")
    print(f"Highest Sharpe Ratio: {best_sharpe} ({comparison_results['comparison_metrics'].loc[best_sharpe, 'Sharpe Ratio']:.3f})")
    print(f"Highest Annual Return: {best_return} ({comparison_results['comparison_metrics'].loc[best_return, 'Annual Return']:.3f})")
    
    logger.info("Backtesting completed successfully!")

if __name__ == "__main__":
    main()