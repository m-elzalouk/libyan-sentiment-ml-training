"""
Script to tune the number of components for TruncatedSVD.
"""
import os
import logging
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from scripts.train import train_with_ngram, parse_args
from scripts.config import Config
from scripts.data_loader import load_and_prepare_data
from scripts.utils import setup_logging, log_duration

def tune_svd_components(X_train, X_test, y_train, y_test, config, ngram_name, ngram_range, min_components=100, max_components=500, step=100):
    """
    Tune the number of components for TruncatedSVD.
    
    Args:
        X_train: Training features DataFrame
        X_test: Test features DataFrame
        y_train: Training labels
        y_test: Test labels
        config: Configuration object
        ngram_name: Name of the n-gram configuration
        ngram_range: N-gram range to use
        min_components: Minimum number of components to try
        max_components: Maximum number of components to try
        step: Step size for component search
        
    Returns:
        DataFrame with results for each component size
    """
    logger = logging.getLogger(__name__)
    results = []
    
    # Test different component sizes
    for n_components in tqdm(range(min_components, max_components + 1, step), 
                           desc=f"Tuning SVD components for {ngram_name}"):
        try:
            logger.info(f"\n{'='*50}")
            logger.info(f"Testing with {n_components} components")
            logger.info(f"{'='*50}")
            
            # Train with current component size
            metrics = train_with_ngram(
                X_train, X_test, y_train, y_test, config,
                ngram_name, ngram_range,
                use_svd=True,
                n_components=n_components
            )
            
            # Add component size and n-gram info to metrics
            metrics['svd_components'] = n_components
            metrics['ngram_name'] = ngram_name
            metrics['ngram_range'] = str(ngram_range)  # Convert tuple to string for CSV
            
            # Get vocab size from metrics if available, otherwise set to 0
            if 'vocab_size' not in metrics:
                metrics['vocab_size'] = 0
            
            results.append(metrics)
            
            # Save intermediate results
            results_df = pd.DataFrame(results)
            results_file = os.path.join(config.RESULTS_DIR, f"svd_tuning_{ngram_name}.csv")
            results_df.to_csv(results_file, index=False)
            logger.info(f"Saved results to {results_file}")
            
        except Exception as e:
            logger.error(f"Error with {n_components} components: {str(e)}", exc_info=True)
            
            # Create a basic metrics dictionary with error information
            error_metrics = {
                'svd_components': n_components,
                'error': str(e),
                'ngram_name': ngram_name,
                'ngram_range': str(ngram_range),
                'accuracy': 0.0,
                'f1': 0.0
            }
            results.append(error_metrics)
            
            # Save intermediate results
            results_df = pd.DataFrame(results)
            results_file = os.path.join(config.RESULTS_DIR, f"svd_tuning_{ngram_name}.csv")
            results_df.to_csv(results_file, index=False)
            logger.info(f"Saved intermediate results to {results_file}")
            continue
    
    return pd.DataFrame(results)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Tune TruncatedSVD components.')
    parser.add_argument('--min_components', type=int, default=100,
                        help='Minimum number of components to try')
    parser.add_argument('--max_components', type=int, default=500,
                        help='Maximum number of components to try')
    parser.add_argument('--step', type=int, default=50,
                        help='Step size for component search')
    parser.add_argument('--ngram', type=str, default=None,
                        help='Specific n-gram configuration to tune (e.g., "Unigrams")')
    args = parser.parse_args()
    
    # Set up logging
    config = Config()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(config.LOGS_DIR, f'svd_tuning_{timestamp}.log')
    setup_logging(log_file=log_file)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting SVD component tuning")
    logger.info(f"Min components: {args.min_components}")
    logger.info(f"Max components: {args.max_components}")
    logger.info(f"Step size: {args.step}")
    
    # Load and prepare data
    logger.info("Loading and preparing data...")
    X_train, X_test, y_train, y_test = load_and_prepare_data(config)
    
    # Get n-gram configurations to test
    ngrams_to_test = config.NGRAMS
    if args.ngram and args.ngram in ngrams_to_test:
        ngrams_to_test = {args.ngram: ngrams_to_test[args.ngram]}
    
    # Tune components for each n-gram configuration
    all_results = []
    for ngram_name, ngram_range in ngrams_to_test.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"Tuning SVD components for {ngram_name}: {ngram_range}")
        logger.info(f"{'='*50}")
        
        results = tune_svd_components(
            X_train, X_test, y_train, y_test, config,
            ngram_name, ngram_range,
            min_components=args.min_components,
            max_components=args.max_components,
            step=args.step
        )
        
        if not results.empty:
            all_results.append(results)
    
    # Save combined results
    if all_results:
        results_df = pd.concat(all_results, ignore_index=True)
        results_file = os.path.join(config.RESULTS_DIR, 'svd_tuning_results.csv')
        results_df.to_csv(results_file, index=False)
        logger.info(f"\nSaved all results to {results_file}")
        
        # Log best configuration
        best_idx = results_df['f1'].idxmax()
        best = results_df.loc[best_idx]
        logger.info("\nBest Configuration:")
        logger.info(f"N-gram: {best['ngram']} ({best['ngram_range']})")
        logger.info(f"SVD Components: {int(best['svd_components'])}")
        logger.info(f"Accuracy: {best['accuracy']:.4f}")
        logger.info(f"F1 Score: {best['f1']:.4f}")
    else:
        logger.warning("No results were generated.")

if __name__ == "__main__":
    main()
