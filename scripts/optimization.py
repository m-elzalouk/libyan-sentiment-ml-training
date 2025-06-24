"""
Hyperparameter optimization for sentiment analysis models.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, roc_auc_score, ConfusionMatrixDisplay
)
import joblib
import os
import logging
import json
from typing import Dict, Tuple, Any, List, Union
from .config import Config
from .features import extract_features

logger = logging.getLogger(__name__)

# Initialize config to access parameter grids
config = Config()

def optimize_ngram_for_classifier_with_hyperparams(
    X_train: List[str],
    y_train: List[str],
    X_test: List[str],
    y_test: List[str],
    results_dir: str = 'results',
    fine_tune: bool = False,
    use_svd: bool = True,
    n_components: int = 300
) -> pd.DataFrame:
    """
    Optimize n-gram ranges and hyperparameters for multiple classifiers.
    
    Args:
        X_train: Training text data
        y_train: Training labels
        X_test: Test text data
        y_test: Test labels
        results_dir: Directory to save results and plots
        fine_tune: Whether to perform fine-tuning with extended parameter grid
        use_svd: Whether to use TruncatedSVD for dimensionality reduction
        n_components: Number of components for TruncatedSVD if use_svd is True
    
    Returns:
        DataFrame containing results for all experiments
    """
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    # Log optimization configuration
    logger.info("\n" + "="*80)
    logger.info("Starting Hyperparameter Optimization")
    logger.info("="*80)
    logger.info(f"Results directory: {os.path.abspath(results_dir)}")
    logger.info(f"Fine-tuning mode: {fine_tune}")
    logger.info(f"Using TruncatedSVD: {use_svd}")
    if use_svd:
        logger.info(f"SVD components: {n_components}")
    logger.info("\n" + "-"*40 + "\n")
    
    # Initialize results storage
    results = []
    
    # Get classifiers and parameter grids from config
    classifiers = config.CLASSIFIERS
    ngrams = config.NGRAMS
    
    for clf_name, (clf, param_grid) in classifiers.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"Training {clf_name}")
        logger.info(f"{'='*50}")
        
        # If fine-tuning, use the extended parameter grid
        if fine_tune and clf_name in config.FINE_TUNE_GRIDS:
            param_grid = config.FINE_TUNE_GRIDS[clf_name]
            logger.info(f"Using fine-tuning grid for {clf_name}")
        
        best_accuracy = 0
        best_ngram = None
        best_params = None
        
        for ngram_name, ngram_range in ngrams.items():
            logger.info(f"\nTraining with {ngram_name}: {ngram_range}")
            
            # Extract features with the specified n-gram range and SVD settings
            X_train_vec, X_test_vec, feature_pipeline = extract_features(
                X_train=X_train,
                X_test=X_test,
                config=config,
                ngram_range=ngram_range,
                use_svd=use_svd,
                n_components=n_components
            )
            
            # Log feature extraction details
            logger.info("\n" + "="*40)
            logger.info(f"Feature Extraction - {ngram_name} ({ngram_range})")
            logger.info("="*40)
            logger.info(f"Training shape: {X_train_vec.shape}")
            logger.info(f"Test shape: {X_test_vec.shape}")
            if use_svd and hasattr(feature_pipeline.named_steps.get('svd'), 'explained_variance_ratio_'):
                explained_variance = np.sum(feature_pipeline.named_steps['svd'].explained_variance_ratio_)
                logger.info(f"Explained variance: {explained_variance:.2%}")
            
            # Log classifier and parameter grid
            logger.info("\n" + "="*40)
            logger.info(f"Training {clf_name} with {ngram_name}")
            logger.info("="*40)
            logger.info(f"Parameter grid:\n{json.dumps(param_grid, indent=2, default=str)}")
            
            # Use RandomizedSearchCV for faster search
            search = RandomizedSearchCV(
                estimator=clf,
                param_distributions=param_grid,
                n_iter=30 if fine_tune else 10,
                scoring='f1_weighted',
                cv=5,
                n_jobs=-1,
                verbose=2,  # Increased verbosity
                random_state=config.RANDOM_STATE,
                return_train_score=True
            )
            
            search.fit(X_train_vec, y_train)
            
            # Get best model and predictions
            best_clf = search.best_estimator_
            y_pred = best_clf.predict(X_test_vec)
            
            # Calculate metrics
            test_accuracy = accuracy_score(y_test, y_pred)
            test_precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            test_recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            test_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # Log detailed results
            logger.info("\n" + "-"*40)
            logger.info(f"Best parameters for {clf_name} ({ngram_name}):")
            for param, value in search.best_params_.items():
                logger.info(f"  {param}: {value}")
            logger.info("\nTest Set Performance:")
            logger.info(f"  Accuracy:  {test_accuracy:.4f}")
            logger.info(f"  Precision: {test_precision:.4f}")
            logger.info(f"  Recall:    {test_recall:.4f}")
            logger.info(f"  F1 Score:  {test_f1:.4f}")
            logger.info("-"*40 + "\n")
            
            metrics = {
                'classifier': clf_name,
                'ngram': ngram_name,
                'ngram_range': str(ngram_range),
                'use_svd': use_svd,
                'svd_components': n_components if use_svd else None,
                'best_params': search.best_params_,
                'cv_best_score': search.best_score_,
                'accuracy': test_accuracy,
                'precision': test_precision,
                'recall': test_recall,
                'f1': test_f1,
                'vocab_size': X_train_vec.shape[1],
                'feature_pipeline': feature_pipeline,
                'cv_results': search.cv_results_
            }
            
            results.append(metrics)
            logger.info(f"{ngram_name} - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
            
            # Save confusion matrix
            if hasattr(best_clf, 'predict'):
                plt.figure(figsize=(8, 6))
                ConfusionMatrixDisplay.from_estimator(best_clf, X_test_vec, y_test)
                plt.title(f"{clf_name} - {ngram_name}\nAccuracy: {metrics['accuracy']:.4f}")
                plt.tight_layout()
                plt.savefig(os.path.join(results_dir, f"confusion_{clf_name}_{ngram_name}.png"))
                plt.close()
            
            # Track best n-gram for this classifier
            if metrics['accuracy'] > best_accuracy:
                best_accuracy = metrics['accuracy']
                best_ngram = ngram_name
                best_params = search.best_params_
                
                # Save the best model and feature pipeline
                model_dir = os.path.join(results_dir, f"{clf_name.lower()}_{ngram_name}")
                if use_svd:
                    model_dir += f"_svd{n_components}"
                os.makedirs(model_dir, exist_ok=True)
                
                model_filename = os.path.join(model_dir, f"best_model.joblib")
                joblib.dump(best_clf, model_filename)
                
                pipeline_filename = os.path.join(model_dir, "feature_pipeline.joblib")
                joblib.dump(feature_pipeline, pipeline_filename)
                
                logger.info(f"Saved model and pipeline to {model_dir}")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results to CSV
    results_df.to_csv(os.path.join(results_dir, 'optimization_results.csv'), index=False)
    
    # Plot results
    if not results_df.empty:
        plot_optimization_results(results_df, results_dir)
    
    return results_df

def plot_optimization_results(results_df: pd.DataFrame, save_dir: str):
    """Plot optimization results."""
    plt.figure(figsize=(14, 8))
    sns.set_theme(style="whitegrid")
    
    # Plot accuracy
    plt.subplot(2, 1, 1)
    sns.barplot(x='classifier', y='accuracy', hue='ngram', data=results_df)
    plt.title("Accuracy Across Classifiers and N-grams")
    plt.xticks(rotation=45)
    plt.legend(title='N-gram', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot F1 score
    plt.subplot(2, 1, 2)
    sns.barplot(x='classifier', y='f1', hue='ngram', data=results_df)
    plt.title("F1 Score Across Classifiers and N-grams")
    plt.xticks(rotation=45)
    plt.legend(title='N-gram', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'optimization_results.png'), bbox_inches='tight')
    plt.close()

def get_best_model(results_df: pd.DataFrame, results_dir: str) -> Tuple[Any, Any, Dict]:
    """
    Get the best model based on F1 score.
    
    Args:
        results_df: DataFrame containing optimization results
        results_dir: Directory where models are saved
        
    Returns:
        Tuple of (best_model, feature_pipeline, best_result_dict)
    """
    if results_df.empty:
        raise ValueError("No results available in the provided DataFrame")
    
    # Find the best model based on F1 score
    best_idx = results_df['f1'].idxmax()
    best_result = results_df.loc[best_idx].to_dict()
    
    # Construct model directory name
    model_dir = f"{best_result['classifier'].lower()}_{best_result['ngram'].lower().replace(' ', '_')}"
    if best_result['use_svd']:
        model_dir += f"_svd{best_result['svd_components']}"
    
    model_path = os.path.join(results_dir, model_dir, "best_model.joblib")
    pipeline_path = os.path.join(results_dir, model_dir, "feature_pipeline.joblib")
    
    # Load the best model and feature pipeline
    if not os.path.exists(model_path) or not os.path.exists(pipeline_path):
        raise FileNotFoundError(
            f"Model or pipeline not found. Expected files:\n"
            f"- {model_path}\n"
            f"- {pipeline_path}"
        )
    
    logger.info("\n" + "="*80)
    logger.info("Best Model Summary")
    logger.info("="*80)
    logger.info(f"Classifier: {best_result['classifier']}")
    logger.info(f"N-gram: {best_result['ngram']} ({best_result['ngram_range']})")
    if best_result['use_svd']:
        logger.info(f"SVD Components: {best_result['svd_components']}")
    logger.info(f"Test F1 Score: {best_result['f1']:.4f}")
    logger.info(f"Test Accuracy: {best_result['accuracy']:.4f}")
    logger.info("\nBest Parameters:")
    for param, value in best_result['best_params'].items():
        logger.info(f"  {param}: {value}")
    logger.info("\n" + "="*80 + "\n")
    
    model = joblib.load(model_path)
    feature_pipeline = joblib.load(pipeline_path)
    
    return model, feature_pipeline, best_result
