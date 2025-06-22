"""
CLI entry point for Libyan Dialect Sentiment Analysis Training Pipeline.
"""
import os
import logging
import json
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
from scripts.config import Config
from scripts.data_loader import load_and_prepare_data
from scripts.features import extract_features
from scripts.models import train_and_select_model
from scripts.evaluate import evaluate_and_save_results
from scripts.optimization import optimize_ngram_for_classifier_with_hyperparams
from scripts.utils import log_duration, log_step, setup_logging

def train_with_ngram(X_train, X_test, y_train, y_test, config, ngram_name, ngram_range):
    """Train and evaluate a model with a specific n-gram configuration."""
    logger = logging.getLogger(__name__)
    logger.info(f"\n{'='*50}")
    logger.info(f"Training with {ngram_name}: {ngram_range}")
    logger.info(f"{'='*50}")
    
    # Create a results directory for this n-gram configuration
    ngram_dir = os.path.join(config.RESULTS_DIR, f"ngram_{ngram_name.lower().replace(' ', '_')}")
    os.makedirs(ngram_dir, exist_ok=True)
    
    # Extract features with the current n-gram range
    with log_duration("Feature extraction"):
        X_train_vec, X_test_vec, vectorizer = extract_features(
            X_train, X_test, config, ngram_range=ngram_range
        )
        logger.info(f"Feature extraction complete. Vocabulary size: {len(vectorizer.vocabulary_)}")
    
    # Train model
    with log_duration("Model training"):
        model, params = train_and_select_model(X_train_vec, y_train, config)
        logger.info(f"Model training complete. Best parameters: {params}")
    
    # Save the vectorizer with n-gram info
    vectorizer_path = os.path.join(ngram_dir, "vectorizer.pkl")
    import joblib
    joblib.dump(vectorizer, vectorizer_path)
    logger.info(f"Saved vectorizer to {vectorizer_path}")
    
    # Evaluate model
    with log_duration("Model evaluation"):
        metrics = evaluate_and_save_results(
            model, X_test_vec, y_test, vectorizer, config, params, results_dir=ngram_dir
        )
    
    # Add n-gram info to metrics
    metrics['ngram_config'] = {
        'name': ngram_name,
        'range': ngram_range,
        'vocab_size': len(vectorizer.vocabulary_)
    }
    
    return metrics

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train sentiment analysis model with hyperparameter optimization.')
    parser.add_argument('--optimize', action='store_true',
                      help='Run hyperparameter optimization')
    parser.add_argument('--fine-tune', action='store_true',
                      help='Use extended parameter grid for fine-tuning')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Initialize logging
    logger = setup_logging()
    logger.info("Initializing training pipeline with n-gram analysis")
    
    # Load configuration
    config = Config()
    logger.info(f"Configuration loaded. Data path: {config.DATA_PATH}")
    
    # Create necessary directories
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    logger.info(f"Output directories verified/created")
    
    # Load and prepare data (only once)
    with log_duration("Data loading and preparation"):
        if args.optimize:
            # For optimization, we need the full DataFrame
            logger.info("Loading data with return_dataframe=True")
            data = load_and_prepare_data(config, return_dataframe=True)
            logger.info(f"Data type: {type(data)}")
            if isinstance(data, tuple):
                logger.error(f"Expected DataFrame but got tuple with {len(data)} elements")
                logger.error(f"First element type: {type(data[0]) if len(data) > 0 else 'N/A'}")
            else:
                logger.info(f"Data loaded. Samples: {len(data)}")
                logger.info(f"Columns: {data.columns.tolist() if hasattr(data, 'columns') else 'N/A'}")
            
            # Split data into train and test sets
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                data['Processed_Text_base_ai'], 
                data['Sentiment'],
                test_size=config.TEST_SIZE,
                random_state=config.RANDOM_STATE,
                stratify=data['Sentiment']
            )
            logger.info(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
        else:
            # For regular training, use the standard split
            X_train, X_test, y_train, y_test = load_and_prepare_data(config, return_dataframe=False)
            logger.info(f"Data loaded. Train samples: {len(X_train)}, Test samples: {len(X_test)}")
            
            # Ensure we have the text data in the right format
            if isinstance(X_train, pd.Series):
                X_train = X_train.values
            if isinstance(X_test, pd.Series):
                X_test = X_test.values
            if isinstance(y_train, pd.Series):
                y_train = y_train.values
            if isinstance(y_test, pd.Series):
                y_test = y_test.values
    
    if args.optimize:
        # Run hyperparameter optimization
        with log_duration("Hyperparameter optimization"):
            logger.info("Starting hyperparameter optimization...")
            results_df = optimize_ngram_for_classifier_with_hyperparams(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                results_dir=config.RESULTS_DIR,
                fine_tune=args.fine_tune
            )
            
            # Save the best model
            best_idx = results_df['f1'].idxmax()
            best_result = results_df.loc[best_idx]
            logger.info(f"\n{'='*50}")
            logger.info("Best Model Configuration:")
            logger.info(f"Classifier: {best_result['classifier']}")
            logger.info(f"N-gram: {best_result['ngram']} ({best_result['ngram_range']})")
            logger.info(f"Accuracy: {best_result['accuracy']:.4f}")
            logger.info(f"F1 Score: {best_result['f1']:.4f}")
            logger.info(f"Parameters: {best_result['best_params']}")
            logger.info(f"{'='*50}")
            
            # Exit after optimization if not in fine-tuning mode
            if not args.fine_tune:
                return
    
    # Track results for all n-gram configurations
    all_results = {}
    best_score = -1
    best_ngram = None
    
    # Train and evaluate models for each n-gram configuration
    for ngram_name, ngram_range in config.NGRAMS.items():
        try:
            metrics = train_with_ngram(
                X_train, X_test, y_train, y_test, config, ngram_name, ngram_range
            )
            all_results[ngram_name] = metrics
            
            # Track best performing n-gram configuration
            if metrics['f1'] > best_score:  # Using F1 score as the main metric
                best_score = metrics['f1']
                best_ngram = ngram_name
                
        except Exception as e:
            logger.error(f"Error processing {ngram_name} ({ngram_range}): {str(e)}", exc_info=True)
            continue
    
    # Save summary of all results
    summary_path = os.path.join(config.RESULTS_DIR, "ngram_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        # Convert numpy types to native Python types for JSON serialization
        import numpy as np
        def convert_numpy_types(obj):
            if isinstance(obj, (np.integer, np.floating, np.bool_)):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_numpy_types(x) for x in obj]
            return obj
        
        json.dump(convert_numpy_types(all_results), f, indent=2)
    
    logger.info(f"\n{'='*50}")
    logger.info("N-gram Analysis Complete")
    logger.info(f"{'='*50}")
    logger.info(f"Best performing n-gram: {best_ngram} (F1: {best_score:.4f})")
    logger.info(f"Detailed results saved to: {summary_path}")
    logger.info("Training pipeline completed successfully")

if __name__ == "__main__":
    try:
        with log_duration("Total pipeline execution"):
            main()
    except Exception as e:
        logging.critical("Pipeline execution failed", exc_info=True)
        raise
