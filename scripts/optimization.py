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
from typing import Dict, Tuple, Any, List
from .config import Config

logger = logging.getLogger(__name__)

# Initialize config to access parameter grids
config = Config()

def optimize_ngram_for_classifier_with_hyperparams(
    X_train: List[str],
    y_train: List[str],
    X_test: List[str],
    y_test: List[str],
    results_dir: str = 'results',
    fine_tune: bool = False
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
    
    Returns:
        DataFrame containing results for all experiments
    """
    """
    Optimize n-gram ranges and hyperparameters for multiple classifiers.
    
    Args:
        classifiers: Dictionary of classifiers and their parameter grids
        X_train: Training text data
        y_train: Training labels
        X_test: Test text data
        y_test: Test labels
        ngrams: Dictionary of n-gram configurations
        results_dir: Directory to save results and plots
        fine_tune: Whether to perform fine-tuning with extended parameter grid
    
    Returns:
        DataFrame containing results for all experiments
    """
    os.makedirs(results_dir, exist_ok=True)
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
            
            # Vectorize text
            tfidf = TfidfVectorizer(
                ngram_range=ngram_range,
                max_features=20000,
                min_df=3,
                max_df=0.9,
                analyzer='word'
            )
            
            X_train_vec = tfidf.fit_transform(X_train)
            X_test_vec = tfidf.transform(X_test)
            
            # Use RandomizedSearchCV for faster search
            search = RandomizedSearchCV(
                clf,
                param_distributions=param_grid,
                n_iter=30 if fine_tune else 10,
                scoring='f1_weighted',
                cv=5,
                n_jobs=-1,
                verbose=1,
                random_state=42
            )
            
            search.fit(X_train_vec, y_train)
            
            # Get best model and predictions
            best_clf = search.best_estimator_
            y_pred = best_clf.predict(X_test_vec)
            
            # Calculate metrics
            metrics = {
                'classifier': clf_name,
                'ngram': ngram_name,
                'ngram_range': ngram_range,
                'best_params': search.best_params_,
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
                'cv_best_score': search.best_score_,
                'vocab_size': X_train_vec.shape[1]
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
                
                # Save best model for this classifier
                model_path = os.path.join(results_dir, f"best_{clf_name}.pkl")
                joblib.dump({
                    'model': best_clf,
                    'vectorizer': tfidf,
                    'metrics': metrics,
                    'params': best_params
                }, model_path)
    
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

def get_best_model(results_df: pd.DataFrame, results_dir: str) -> Tuple[Any, Dict]:
    """Get the best model based on F1 score."""
    if results_df.empty:
        raise ValueError("No results available")
    
    best_idx = results_df['f1'].idxmax()
    best_result = results_df.loc[best_idx]
    
    # Load the best model
    model_path = os.path.join(results_dir, f"best_{best_result['classifier']}.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model_data = joblib.load(model_path)
    return model_data['model'], model_data['vectorizer'], best_result.to_dict()
