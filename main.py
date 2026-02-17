#!/usr/bin/env python3
"""
Main entry point for the Portfolio Optimizer project
Run the complete machine learning enhanced portfolio optimization pipeline
"""

import argparse
import sys
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import with error handling
try:
    from loguru import logger
    LOGURU_AVAILABLE = True
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    LOGURU_AVAILABLE = False
    logger.warning("loguru not available, using standard logging")

import logging  # Always import for fallback usage

try:
    from config import Config
except ImportError as e:
    logger.error(f"Failed to import config: {e}")
    logger.error("Please install required dependencies: pip install pyyaml")
    sys.exit(1)
import importlib


def _get_step_callable(module_path: str):
    """Lazily import a module and return its `main` callable if present.

    This prevents heavy optional dependencies from being imported at module
    import time so users can run lightweight commands like `--info`.
    """
    try:
        mod = importlib.import_module(module_path)
        if hasattr(mod, 'main'):
            return getattr(mod, 'main')
        else:
            def _noop():
                logger.warning(f"Module {module_path} has no main(), skipping.")
            return _noop
    except Exception as e:
        error_msg = str(e)
        def _err():
            logger.error(f"Failed to import {module_path}: {error_msg}")
        return _err


# Define step callables lazily
def _run_data_collection():
    return _get_step_callable('src.data.data_collection')()

def _run_clustering():
    return _get_step_callable('src.clustering.stock_clustering')()

def _run_model_training():
    return _get_step_callable('src.models.prediction_models')()

def _run_optimization():
    return _get_step_callable('src.optimization.portfolio_optimizer')()

def _run_backtesting():
    return _get_step_callable('src.evaluation.backtesting')()

def setup_logging():
    """Setup logging configuration"""
    if LOGURU_AVAILABLE:
        logger.remove()  # Remove default handler
        logger.add(
            sys.stdout,
            format=Config.LOG_FORMAT,
            level=Config.LOG_LEVEL,
            colorize=True
        )
        
        # Also log to file
        log_file = os.path.join(Config.RESULTS_DIR, 'portfolio_optimizer.log')
        os.makedirs(Config.RESULTS_DIR, exist_ok=True)
        
        logger.add(
            log_file,
            format=Config.LOG_FORMAT,
            level=Config.LOG_LEVEL,
            rotation="10 MB"
        )
    else:
        # Use standard logging as fallback
        logging.basicConfig(
            level=getattr(logging, Config.LOG_LEVEL),
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

def run_full_pipeline():
    """Run the complete portfolio optimization pipeline"""
    logger.info("=" * 80)
    logger.info("PORTFOLIO OPTIMIZATION WITH MACHINE LEARNING")
    logger.info("=" * 80)
    
    try:
        # Step 1: Data Collection and Preprocessing
        logger.info("STEP 1: Data Collection and Preprocessing")
        logger.info("-" * 50)
        _run_data_collection()
        
        # Step 2: Stock Clustering  
        logger.info("\nSTEP 2: Stock Clustering Analysis")
        logger.info("-" * 50)
        _run_clustering()
        
        # Step 3: ML/DL Model Training
        logger.info("\nSTEP 3: ML/DL Model Training for Return Prediction")
        logger.info("-" * 50)
        _run_model_training()
        
        # Step 4: Portfolio Optimization
        logger.info("\nSTEP 4: Portfolio Optimization")
        logger.info("-" * 50)
        _run_optimization()
        
        # Step 5: Backtesting and Evaluation
        logger.info("\nSTEP 5: Backtesting and Performance Evaluation")
        logger.info("-" * 50)
        _run_backtesting()
        
        logger.info("\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("\nResults saved in:")
        logger.info(f"  📊 Data: {Config.PROCESSED_DATA_DIR}")
        logger.info(f"  📈 Results: {Config.RESULTS_DIR}")
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        raise

def run_individual_step(step: str):
    """Run an individual step of the pipeline"""
    steps = {
        'data': ('Data Collection and Preprocessing', run_data_collection),
        'clustering': ('Stock Clustering Analysis', run_clustering),
        'models': ('ML/DL Model Training', run_model_training), 
        'optimization': ('Portfolio Optimization', run_optimization),
        'backtesting': ('Backtesting and Evaluation', run_backtesting)
    }
    
    if step not in steps:
        logger.error(f"Unknown step: {step}. Available steps: {list(steps.keys())}")
        return
    
    step_name, step_function = steps[step]
    logger.info(f"Running: {step_name}")
    logger.info("-" * 50)
    
    try:
        step_function()
        logger.info(f"✅ {step_name} completed successfully!")
    except Exception as e:
        logger.error(f"❌ {step_name} failed: {e}")
        raise

def print_project_info():
    """Print project information and status"""
    logger.info("Portfolio Optimization with Machine Learning")
    logger.info("=" * 50)
    logger.info("This project implements portfolio optimization enhanced with:")
    logger.info("• Stock clustering for similarity grouping")
    logger.info("• ML/DL models for return prediction")
    logger.info("• Multiple optimization algorithms")  
    logger.info("• Comprehensive backtesting framework")
    logger.info("")
    
    logger.info("Project Structure:")
    logger.info(f"  📁 Data Directory: {Config.DATA_DIR}")
    logger.info(f"  📁 Results Directory: {Config.RESULTS_DIR}")
    logger.info(f"  📊 Stock Universe: {len(Config.STOCK_SYMBOLS)} stocks")
    logger.info(f"  📈 Benchmark: {Config.BENCHMARK_SYMBOL}")
    logger.info("")
    
    # Check data availability
    processed_data = os.path.join(Config.PROCESSED_DATA_DIR, 'processed_stock_data.csv')
    cluster_data = os.path.join(Config.RESULTS_DIR, 'cluster_assignments_kmeans.csv')
    
    logger.info("Data Status:")
    logger.info(f"  📊 Processed Data: {'✅' if os.path.exists(processed_data) else '❌'}")
    logger.info(f"  🎯 Clustering Results: {'✅' if os.path.exists(cluster_data) else '❌'}")

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(
        description='Portfolio Optimization with Machine Learning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --full                    # Run complete pipeline
  python main.py --step data              # Run only data collection
  python main.py --step clustering        # Run only clustering
  python main.py --step models           # Run only model training  
  python main.py --step optimization     # Run only optimization
  python main.py --step backtesting      # Run only backtesting
  python main.py --info                  # Show project information
        """
    )
    
    parser.add_argument('--full', action='store_true', 
                       help='Run the complete pipeline')
    parser.add_argument('--step', choices=['data', 'clustering', 'models', 'optimization', 'backtesting'],
                       help='Run individual step of the pipeline')
    parser.add_argument('--info', action='store_true',
                       help='Show project information and status')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    if args.verbose:
        Config.LOG_LEVEL = 'DEBUG'
    
    setup_logging()
    
    # Handle arguments
    if args.info:
        print_project_info()
    elif args.full:
        run_full_pipeline()
    elif args.step:
        run_individual_step(args.step)
    else:
        print_project_info()
        logger.info("\nTo get started, run:")
        logger.info("  python main.py --full")
        logger.info("\nOr run individual steps:")
        logger.info("  python main.py --step data")
        logger.info("\nFor help: python main.py --help")

if __name__ == "__main__":
    main()