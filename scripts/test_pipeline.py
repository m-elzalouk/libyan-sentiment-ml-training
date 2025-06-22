"""
Test script for quick verification of the sentiment analysis pipeline.
This script uses a small subset of data and a simple model for fast execution.
"""
import os
import sys
import logging
import numpy as np
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.data_loader import load_and_prepare_data
from scripts.features import extract_features
from scripts.models import train_and_select_model
from scripts.evaluate import evaluate_and_save_results
from scripts.config import Config
from scripts.utils import setup_logging, log_duration

# Configure test settings
TEST_SAMPLE_SIZE = 500  # Number of samples to use for testing
TEST_RANDOM_STATE = 42
TEST_OUTPUT_DIR = "test_results"

# Test configuration
TEST_CONFIG = Config()
# Override some settings for faster testing
TEST_CONFIG.TEST_SIZE = 0.2  # Use 20% for testing
TEST_CONFIG.RANDOM_STATE = TEST_RANDOM_STATE
# Limit the number of features for faster testing
TEST_CONFIG.VECTORIZER_PARAMS = {
    'max_features': 2000,
    'ngram_range': (1, 1),
    'stop_words': None
}
# Include all required model parameters for testing
TEST_CONFIG.GRID_SEARCH_PARAMS = {
    'svm_linear': {
        'C': [1.0],
        'kernel': ['linear'],
        'class_weight': ['balanced']
    },
    'svm_rbf': {
        'C': [1.0],
        'kernel': ['rbf'],
        'class_weight': ['balanced']
    },
    'logreg': {
        'C': [1.0],
        'max_iter': [100],
        'solver': ['liblinear'],
        'random_state': [TEST_RANDOM_STATE]
    },
    'nb': {
        'alpha': [0.1, 1.0]
    }
}
# Set the scoring metric
TEST_CONFIG.SCORING = 'f1_weighted'

def run_test_pipeline():
    """Run a quick test of the pipeline with a small dataset."""
    # Set up logging
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
    log_file = os.path.join(TEST_OUTPUT_DIR, f"test_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    setup_logging(logger_name=log_file)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("STARTING TEST PIPELINE")
    logger.info("=" * 80)
    
    try:
        # Load and prepare data
        with log_duration("Data loading and preparation"):
            logger.info("Loading and preparing data...")
            
            # Load the full dataset using our test config
            X_train_full, X_test_full, y_train_full, y_test_full = load_and_prepare_data(TEST_CONFIG)
            
            # For testing, use a smaller subset of the training data
            if len(X_train_full) > TEST_SAMPLE_SIZE:
                from sklearn.model_selection import train_test_split
                _, X_train, _, y_train = train_test_split(
                    X_train_full, y_train_full,
                    train_size=TEST_SAMPLE_SIZE,
                    random_state=TEST_RANDOM_STATE,
                    stratify=y_train_full
                )
                logger.info(f"Using {len(X_train)} random samples for testing")
            else:
                X_train, y_train = X_train_full, y_train_full
            
            # Use a portion of the test set for faster evaluation
            if len(X_test_full) > (TEST_SAMPLE_SIZE // 4):
                _, X_test, _, y_test = train_test_split(
                    X_test_full, y_test_full,
                    test_size=min(TEST_SAMPLE_SIZE // 4, len(X_test_full)),
                    random_state=TEST_RANDOM_STATE,
                    stratify=y_test_full
                )
                logger.info(f"Using {len(X_test)} test samples")
            else:
                X_test, y_test = X_test_full, y_test_full
            
            logger.info(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
            logger.info(f"Class distribution in training: {dict(zip(*np.unique(y_train, return_counts=True)))}")
            logger.info(f"Class distribution in testing: {dict(zip(*np.unique(y_test, return_counts=True)))}")
        
        # Initialize and fit vectorizer
        with log_duration("Text vectorization"):
            logger.info("Extracting features...")
            # extract_features returns (X_train_vec, X_test_vec, vectorizer)
            X_train_vec, X_test_vec, vectorizer = extract_features(X_train, X_test, Config)
            logger.info(f"Vectorizer vocabulary size: {len(vectorizer.vocabulary_)}")
            logger.info(f"Training features shape: {X_train_vec.shape}")
            logger.info(f"Test features shape: {X_test_vec.shape}")
        
        # Train model
        with log_duration("Model training"):
            logger.info("Training model...")
            # Use our test config which has the simplified model parameters
            best_model, best_params = train_and_select_model(
                X_train_vec, y_train, TEST_CONFIG
            )
            logger.info(f"Best model: {best_model.__class__.__name__}")
            logger.info(f"Best parameters: {best_params}")
        
        # Evaluate
        with log_duration("Model evaluation"):
            logger.info("Evaluating model...")
            # Create test output directory
            test_results_dir = os.path.join(TEST_OUTPUT_DIR, "results")
            os.makedirs(test_results_dir, exist_ok=True)
            
            # Temporarily override config paths for test
            original_results_path = Config.RESULTS_PATH
            original_models_dir = Config.MODELS_DIR
            
            try:
                Config.RESULTS_PATH = os.path.join(test_results_dir, "scores.json")
                Config.MODELS_DIR = os.path.join(test_results_dir, "models")
                
                metrics = evaluate_and_save_results(
                    best_model, X_test_vec, y_test, vectorizer, Config, best_params
                )
                
                logger.info("\n" + "="*50)
                logger.info("TEST PIPELINE COMPLETED SUCCESSFULLY!")
                logger.info("="*50)
                logger.info(f"Test accuracy: {metrics.get('accuracy', 0):.4f}")
                logger.info(f"Test F1 score: {metrics.get('f1', 0):.4f}")
                logger.info(f"Results saved to: {test_results_dir}")
                
            finally:
                # Restore original config
                Config.RESULTS_PATH = original_results_path
                Config.MODELS_DIR = original_models_dir
        
        return True, "Test pipeline completed successfully!"
        
    except Exception as e:
        logger.error("TEST PIPELINE FAILED!", exc_info=True)
        return False, f"Test pipeline failed: {str(e)}"

if __name__ == "__main__":
    success, message = run_test_pipeline()
    print(f"\n{message}")
    sys.exit(0 if success else 1)
