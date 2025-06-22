"""
CLI entry point for Libyan Dialect Sentiment Analysis Training Pipeline.
"""
import os
import logging
import json
from pathlib import Path
from datetime import datetime
from scripts.config import Config
from scripts.data_loader import load_and_prepare_data
from scripts.features import extract_features
from scripts.models import train_and_select_model
from scripts.evaluate import evaluate_and_save_results
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

def main():
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
        X_train, X_test, y_train, y_test = load_and_prepare_data(config)
        logger.info(f"Data loaded. Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
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
