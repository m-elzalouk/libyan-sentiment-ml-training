"""
Test script to verify n-gram configuration integration.
"""
import os
import sys
import logging
from pathlib import Path
from scripts.config import Config
from scripts.train import train_with_ngram
from scripts.data_loader import load_and_prepare_data
from scripts.utils import setup_logging, log_duration

def main():
    # Set up logging
    logger = setup_logging()
    logger.info("Testing n-gram configuration integration")
    
    # Load configuration
    config = Config()
    
    # Create test results directory
    test_dir = os.path.join(config.RESULTS_DIR, "test_ngrams")
    os.makedirs(test_dir, exist_ok=True)
    
    # Load a small subset of data for testing
    with log_duration("Data loading"):
        X_train, X_test, y_train, y_test = load_and_prepare_data(config)
        # Use a small subset for testing
        X_train, y_train = X_train[:100], y_train[:100]
        X_test, y_test = X_test[:20], y_test[:20]
    
    # Test with a subset of n-gram configurations
    test_ngrams = {
        'test_unigrams': (1, 1),
        'test_bigrams': (2, 2),
        'test_unibi': (1, 2)
    }
    
    results = {}
    
    for ngram_name, ngram_range in test_ngrams.items():
        try:
            logger.info(f"\n{'='*50}")
            logger.info(f"Testing n-gram configuration: {ngram_name} {ngram_range}")
            logger.info(f"{'='*50}")
            
            # Create output directory for this n-gram config
            ngram_dir = os.path.join(test_dir, ngram_name)
            os.makedirs(ngram_dir, exist_ok=True)
            
            # Train and evaluate with this n-gram config
            metrics = train_with_ngram(
                X_train, X_test, y_train, y_test, 
                config, ngram_name, ngram_range
            )
            
            # Save results
            results[ngram_name] = {
                'ngram_range': ngram_range,
                'f1': metrics['f1'],
                'accuracy': metrics['accuracy']
            }
            
            logger.info(f"Completed {ngram_name} with F1: {metrics['f1']:.4f}, Accuracy: {metrics['accuracy']:.4f}")
            
        except Exception as e:
            logger.error(f"Error testing {ngram_name}: {str(e)}", exc_info=True)
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("N-gram Configuration Test Summary")
    logger.info("="*50)
    for name, result in results.items():
        logger.info(f"{name} ({result['ngram_range']}): F1={result['f1']:.4f}, Accuracy={result['accuracy']:.4f}")
    
    logger.info(f"\nTest complete. Results saved to: {test_dir}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.critical("Test failed", exc_info=True)
        raise
