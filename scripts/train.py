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

def train_with_ngram(X_train, X_test, y_train, y_test, config, ngram_name, ngram_range, use_svd=True, n_components=300):
    """
    Train and evaluate a model with a specific n-gram configuration and engineered features.
    
    Args:
        X_train: Training features DataFrame
        X_test: Test features DataFrame
        y_train: Training labels
        y_test: Test labels
        config: Configuration object
        ngram_name: Name of the n-gram configuration (for logging)
        ngram_range: N-gram range to use
        use_svd: Whether to apply TruncatedSVD for dimensionality reduction
        n_components: Number of components for TruncatedSVD
    """
    logger = logging.getLogger(__name__)
    logger.info(f"\n{'='*50}")
    logger.info(f"Training with {ngram_name}: {ngram_range}")
    if use_svd:
        logger.info(f"Using TruncatedSVD with {n_components} components")
    logger.info(f"{'='*50}")
    
    # Create a results directory for this n-gram configuration
    ngram_dir = os.path.join(config.RESULTS_DIR, f"ngram_{ngram_name.lower().replace(' ', '_')}")
    if use_svd:
        ngram_dir += f"_svd{n_components}"
    os.makedirs(ngram_dir, exist_ok=True)
    
    # Extract text and numeric features
    with log_duration("Feature extraction"):
        # Separate text and numeric features
        X_train_text = X_train['text']
        X_test_text = X_test['text']
        
        # Extract text features (TF-IDF + optional SVD)
        X_train_vec, X_test_vec, feature_pipeline = extract_features(
            X_train_text, 
            X_test_text, 
            config, 
            ngram_range=ngram_range,
            use_svd=use_svd,
            n_components=n_components
        )
        
        # Get numeric features (excluding the text column)
        numeric_cols = [col for col in X_train.columns if col != 'text']
        X_train_numeric = X_train[numeric_cols].values
        X_test_numeric = X_test[numeric_cols].values
        
        # Combine TF-IDF (possibly reduced with SVD) and numeric features
        from scipy.sparse import hstack, csr_matrix
        import numpy as np
        
        # Convert numeric features to sparse if needed
        if not hasattr(X_train_numeric, 'toarray'):
            X_train_numeric = csr_matrix(X_train_numeric.astype(float))
            X_test_numeric = csr_matrix(X_test_numeric.astype(float))
        
        # Combine features
        X_train_combined = hstack([X_train_vec, X_train_numeric])
        X_test_combined = hstack([X_test_vec, X_test_numeric])
        
        logger.info(f"Combined feature matrix shape (train): {X_train_combined.shape}")
        logger.info(f"Combined feature matrix shape (test): {X_test_combined.shape}")
        
        logger.info(f"Feature extraction complete. "
                   f"TF-IDF dim: {X_train_vec.shape[1]}, "
                   f"Numeric features: {X_train_numeric.shape[1]}, "
                   f"Total features: {X_train_combined.shape[1]}")
    
    # Train model with combined features
    with log_duration("Model training"):
        model, metrics = train_and_select_model(
            X_train_combined, 
            y_train,
            config
        )
        logger.info(f"Model training complete. Best parameters: {metrics}")
        
        # Evaluate on test set
        test_accuracy = model.score(X_test_combined, y_test)
        logger.info(f"Test set accuracy: {test_accuracy:.4f}")
        
        # Log feature importances if available
        if hasattr(model, 'feature_importances_'):
            # Get feature names
            feature_names = (list(feature_pipeline.named_steps['vectorizer'].get_feature_names_out()) + 
                           [f'numeric_{i}' for i in range(X_train_numeric.shape[1])])
            
            # Get top 20 important features
            importances = model.feature_importances_
            top_indices = importances.argsort()[-20:][::-1]
            top_features = [(feature_names[i], importances[i]) for i in top_indices]
            
            logger.info("Top 20 important features:")
            for feat, imp in top_features:
                logger.info(f"  {feat}: {imp:.4f}")
    
    # Save the feature pipeline (TF-IDF + SVD)
    import joblib
    os.makedirs(ngram_dir, exist_ok=True)  # Ensure directory exists
    pipeline_path = os.path.join(ngram_dir, "feature_pipeline.joblib")
    joblib.dump(feature_pipeline, pipeline_path)
    logger.info(f"Saved feature pipeline to {pipeline_path}")
    
    # Prepare n-gram config
    ngram_config = {
        'name': ngram_name,
        'range': ngram_range,
        'vocab_size': len(feature_pipeline.named_steps['tfidf'].vocabulary_) if 'tfidf' in feature_pipeline.named_steps and hasattr(feature_pipeline.named_steps['tfidf'], 'vocabulary_') else 0
    }
    
    # Get model parameters if available
    model_params = getattr(model, 'best_params_', None) or getattr(model, 'get_params', lambda: {})()
    
    # Evaluate model with combined features
    with log_duration("Model evaluation"):
        metrics = evaluate_and_save_results(
            model=model,
            X_test_combined=X_test_combined,
            y_test=y_test,
            results_dir=ngram_dir,
            config=config,
            best_params=model_params,
            ngram_config=ngram_config
        )
    
    return metrics

def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Libyan Dialect Sentiment Analysis Training Pipeline. "
                    "Train and optimize sentiment analysis models with different configurations."
    )
    
    # Main execution modes (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--optimize", 
        action="store_true", 
        help="Run hyperparameter optimization for all classifiers and n-gram configurations"
    )
    mode_group.add_argument(
        "--tune-svd", 
        action="store_true",
        help="Tune the number of SVD components for dimensionality reduction"
    )
    
    # Fine-tuning options
    parser.add_argument(
        "--fine-tune", 
        action="store_true", 
        help="Perform fine-grained hyperparameter tuning (only with --optimize)"
    )
    
    # SVD options
    svd_group = parser.add_argument_group('SVD Options')
    svd_group.add_argument(
        "--no-svd", 
        action="store_false", 
        dest="use_svd",
        help="Disable TruncatedSVD dimensionality reduction"
    )
    svd_group.add_argument(
        "--svd-components", 
        type=int, 
        default=300,
        help="Number of components for TruncatedSVD (default: 300)"
    )
    
    # SVD tuning options (only used with --tune-svd)
    tune_group = parser.add_argument_group('SVD Tuning Options')
    tune_group.add_argument(
        "--min-components",
        type=int,
        default=100,
        help="Minimum number of SVD components to try (default: 100)"
    )
    tune_group.add_argument(
        "--max-components",
        type=int,
        default=500,
        help="Maximum number of SVD components to try (default: 500)"
    )
    tune_group.add_argument(
        "--step",
        type=int,
        default=50,
        help="Step size for SVD component search (default: 50)"
    )
    tune_group.add_argument(
        "--ngram",
        type=str,
        choices=['unigrams', 'bigrams', 'trigrams', 'unigrams_bigrams'],
        default=None,
        help="Specific n-gram configuration to tune (default: all)"
    )
    
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
    
    # Get SVD parameters from command line
    use_svd = args.use_svd
    n_components = args.svd_components
    
    if use_svd:
        logger.info(f"TruncatedSVD enabled with {n_components} components")
    else:
        logger.info("TruncatedSVD disabled")
    
    # Check execution mode
    if args.tune_svd:
        logger.info("Starting SVD component tuning...")
        from scripts.tune_svd_components import tune_svd_components
        
        # Get n-gram configurations to test
        ngrams_to_test = config.NGRAMS
        if args.ngram and args.ngram.upper() in ngrams_to_test:
            ngrams_to_test = {args.ngram.upper(): ngrams_to_test[args.ngram.upper()]}
        
        # Run SVD component tuning
        all_results = []
        for ngram_name, ngram_range in ngrams_to_test.items():
            logger.info(f"\n{'='*80}")
            logger.info(f"Tuning SVD components for {ngram_name}: {ngram_range}")
            logger.info(f"{'='*80}")
            
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
            logger.info(f"N-gram: {best.get('ngram_name', 'N/A')} ({best.get('ngram_range', 'N/A')})")
            logger.info(f"SVD Components: {int(best.get('svd_components', 0))}")
            logger.info(f"Accuracy: {best.get('accuracy', 0):.4f}")
            logger.info(f"F1 Score: {best.get('f1', 0):.4f}")
        
        return
    
    elif args.optimize:
        logger.info("Starting hyperparameter optimization...")
        results_df = optimize_ngram_for_classifier_with_hyperparams(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            results_dir=config.RESULTS_DIR,
            fine_tune=args.fine_tune,
            use_svd=use_svd,
            n_components=n_components
        )
        
        # Save the best model
        best_idx = results_df['f1'].idxmax()
        best_result = results_df.loc[best_idx]
        logger.info(f"\n{'='*80}")
        logger.info("Best Model Configuration:")
        logger.info(f"Classifier: {best_result['classifier']}")
        logger.info(f"N-gram: {best_result['ngram']} ({best_result['ngram_range']})")
        if use_svd:
            logger.info(f"SVD Components: {n_components}")
        logger.info(f"Accuracy: {best_result['accuracy']:.4f}")
        logger.info(f"F1 Score: {best_result['f1']:.4f}")
        logger.info(f"Parameters: {best_result['best_params']}")
        logger.info(f"{'='*80}")
        
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
                X_train, 
                X_test, 
                y_train, 
                y_test, 
                config, 
                ngram_name, 
                ngram_range,
                use_svd=use_svd,
                n_components=n_components
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
