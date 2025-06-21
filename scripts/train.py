"""
CLI entry point for Libyan Dialect Sentiment Analysis Training Pipeline.
"""
import os
import logging
from scripts.config import Config
from scripts.data_loader import load_and_prepare_data
from scripts.features import extract_features
from scripts.models import train_and_select_model
from scripts.evaluate import evaluate_and_save_results
from scripts.utils import log_duration, log_step, setup_logging

def main():
    # Initialize logging
    logger = setup_logging()
    logger.info("Initializing training pipeline")
    
    # Load configuration
    config = Config()
    logger.info(f"Configuration loaded. Data path: {config.DATA_PATH}")
    
    # Create necessary directories
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    logger.info(f"Output directories verified/created")
    
    # Load and prepare data
    with log_duration("Data loading and preparation"):
        X_train, X_test, y_train, y_test = load_and_prepare_data(config)
        logger.info(f"Data loaded. Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Extract features
    with log_duration("Feature extraction"):
        X_train_vec, X_test_vec, vectorizer = extract_features(X_train, X_test, config)
        logger.info(f"Feature extraction complete. Vectorizer vocabulary size: {len(vectorizer.vocabulary_)}")
    
    # Train model
    with log_duration("Model training"):
        best_model, best_params = train_and_select_model(X_train_vec, y_train, config)
        logger.info(f"Model training complete. Best parameters: {best_params}")
    
    # Evaluate model
    with log_duration("Model evaluation"):
        evaluate_and_save_results(best_model, X_test_vec, y_test, vectorizer, config, best_params)
    
    logger.info("Training pipeline completed successfully")

if __name__ == "__main__":
    try:
        with log_duration("Total pipeline execution"):
            main()
    except Exception as e:
        logging.critical("Pipeline execution failed", exc_info=True)
        raise
