"""
Model definitions and grid search for sentiment analysis.
"""
import logging
import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import get_scorer
from scripts.utils import log_step

def train_and_select_model(X_train_vec, y_train, config):
    """
    Train and select the best model using grid search.
    
    Args:
        X_train_vec: Vectorized training features
        y_train: Training labels
        config: Configuration object with model parameters
        
    Returns:
        tuple: (best_model, best_params) - Best trained model and its parameters
    """
    logger = logging.getLogger(__name__)
    
    @log_step("Initialize models")
    def _initialize_models():
        models = {
            'SVM (Linear)': (SVC, config.GRID_SEARCH_PARAMS['svm_linear']),
            'SVM (RBF)': (SVC, config.GRID_SEARCH_PARAMS['svm_rbf']),
            'Naive Bayes': (MultinomialNB, config.GRID_SEARCH_PARAMS['nb']),
            'Logistic Regression': (LogisticRegression, config.GRID_SEARCH_PARAMS['logreg'])
        }
        logger.info(f"Initialized {len(models)} models for training")
        return models
    
    @log_step("Train model")
    def _train_model(name, clf_class, params):
        logger.info(f"Training {name} with parameters: {params}")
        
        # Get the scoring function for logging
        scorer = get_scorer(config.SCORING)
        
        # Define scoring metrics based on configuration
        use_multi_metric = config.SCORING != 'f1_weighted'  # Only use multi-metric if not using default
        
        # Define default scoring metrics that will be used in both single and multi-metric modes
        scoring_metrics = {
            'accuracy': 'accuracy',
            'precision_weighted': 'precision_weighted',
            'recall_weighted': 'recall_weighted',
            'f1_weighted': 'f1_weighted',
            'roc_auc_ovr': 'roc_auc_ovr' if hasattr(clf_class(), 'predict_proba') else None
        }
        
        # Remove None values
        scoring_metrics = {k: v for k, v in scoring_metrics.items() if v is not None}
        
        if use_multi_metric:
            # For multi-metric scoring, refit must be one of the keys in scoring_metrics
            refit_metric = 'f1_weighted'  # Default refit metric
            
            gs = GridSearchCV(
                clf_class(), 
                params, 
                scoring=scoring_metrics,
                refit=refit_metric,
                n_jobs=-1, 
                cv=5, 
                verbose=1,
                return_train_score=True
            )
        else:
            # For single metric scoring, use the standard approach
            gs = GridSearchCV(
                clf_class(),
                params,
                scoring=config.SCORING,
                refit=True,
                n_jobs=-1,
                cv=5,
                verbose=1,
                return_train_score=True
            )
        
        gs.fit(X_train_vec, y_train)
        
        # Log detailed training results with colors
        logger.info("\n" + "="*70)
        logger.info(f"{name} - Training Results")
        logger.info("="*70)
        
        # Always log the best parameters
        logger.info(f"Best parameters: {gs.best_params_}")
        
        # Log metrics based on single or multi-metric mode
        if use_multi_metric:
            # For multi-metric scoring, log all metrics
            logger.info("\nCross-validated metrics (mean ± std):")
            for metric_name in scoring_metrics:
                if f'mean_test_{metric_name}' in gs.cv_results_:
                    mean_score = gs.cv_results_[f'mean_test_{metric_name}'][gs.best_index_]
                    std_score = gs.cv_results_[f'std_test_{metric_name}'][gs.best_index_]
                    # Color code based on metric value (higher is better)
                    if metric_name in ['accuracy', 'f1_weighted', 'roc_auc_ovr']:
                        if mean_score > 0.9:
                            log_func = logger.success
                        elif mean_score > 0.7:
                            log_func = logger.info
                        else:
                            log_func = logger.warning
                    else:
                        log_func = logger.info
                        
                    log_func(f"  %(IMPORTANT)s{metric_name.upper()}: %(reset)s{mean_score:.4f} (±{std_score:.4f})")
            
            # Get the main score used for refitting
            main_metric = refit_metric if use_multi_metric else config.SCORING
            best_score = gs.best_score_ if not use_multi_metric else gs.cv_results_[f'mean_test_{main_metric}'][gs.best_index_]
            
            # Color code the best score
            if best_score > 0.9:
                log_func = logger.success
            elif best_score > 0.7:
                log_func = logger.info
            else:
                log_func = logger.warning
                
            log_func(f"\n%(GREEN)sBest %(BOLD)s{main_metric}%(RESET)s%(GREEN)s: %(BOLD){best_score:.4f}%(RESET)s")
            
            # Log training vs test scores for the main metric with color coding
            if f'mean_train_{main_metric}' in gs.cv_results_:
                train_score = gs.cv_results_[f'mean_train_{main_metric}'][gs.best_index_]
                test_score = gs.cv_results_[f'mean_test_{main_metric}'][gs.best_index_]
                score_diff = abs(train_score - test_score)
                
                # Choose log level based on overfitting risk
                if score_diff > 0.15:  # High risk of overfitting
                    log_func = logger.error
                elif score_diff > 0.1:  # Moderate risk
                    log_func = logger.warning
                else:  # Low risk
                    log_func = logger.info
                
                log_msg = f"Train vs Test {main_metric}: %(CYAN){train_score:.4f}%(reset)s (train) vs "
                log_msg += f"%(GREEN){test_score:.4f}%(reset)s (test)"
                log_msg += f" %(YELLOW)[Δ={score_diff:.4f}]%(reset)s"
                
                log_func(log_msg)
                if score_diff > 0.1:  # Arbitrary threshold
                    logger.warning("%(YELLOW)Possible overfitting detected - large gap between train and test scores%(reset)s")
        else:
            # For single metric scoring
            logger.info(f"Best {config.SCORING}: {gs.best_score_:.4f}")
            logger.info(f"Best parameters: {gs.best_params_}")
            
            # Log training vs test scores
            if 'mean_train_score' in gs.cv_results_:
                train_score = gs.cv_results_['mean_train_score'][gs.best_index_]
                test_score = gs.cv_results_['mean_test_score'][gs.best_index_]
                logger.info(f"\nTrain vs Test score: {train_score:.4f} (train) vs {test_score:.4f} (test)")
                if abs(train_score - test_score) > 0.1:  # Arbitrary threshold
                    logger.warning("Possible overfitting detected - large gap between train and test scores")
        
        # Log all parameter combinations tried
        logger.info("\nAll parameter combinations tried:")
        for i, (params, mean_score, std_score) in enumerate(zip(
            gs.cv_results_['params'],
            gs.cv_results_['mean_test_score'],
            gs.cv_results_['std_test_score']
        )):
            logger.info(f"{i+1}. Params: {params}")
            logger.info(f"   {config.SCORING}: {mean_score:.4f} (±{std_score:.4f})")
            
            # Log all metrics for this parameter combination
            for metric in scoring_metrics:
                if f'mean_test_{metric}' in gs.cv_results_:
                    m = gs.cv_results_[f'mean_test_{metric}'][i]
                    s = gs.cv_results_[f'std_test_{metric}'][i]
                    logger.info(f"   {metric.upper()}: {m:.4f} (±{s:.4f})")
            
            # Log training score if available
            if 'mean_train_score' in gs.cv_results_:
                train = gs.cv_results_['mean_train_score'][i]
                logger.info(f"   Train score: {train:.4f}")
        
        logger.info("="*50 + "\n")
        
        return gs.best_estimator_, gs.best_score_, gs.best_params_
    
    try:
        models = _initialize_models()
        best_model = None
        best_score = -1
        best_params = None
        best_model_name = None
        
        # Train all models and keep track of the best one
        for name, (clf_class, params) in models.items():
            try:
                model, score, params = _train_model(name, clf_class, params)
                if score > best_score:
                    best_score = score
                    best_model = model
                    best_params = params
                    best_model_name = name
            except Exception as e:
                logger.error(f"Error training {name}: {str(e)}", exc_info=True)
                continue
        
        if best_model is None:
            raise ValueError("All models failed to train")
        
        # Log best model info
        logger.info(
            f"Best model: {best_model_name} with {config.SCORING}: {best_score:.4f}\n"
            f"Parameters: {best_params}"
        )
        
        # Save the best model
        joblib.dump(best_model, config.MODEL_PATH)
        logger.info(f"Best model saved to {config.MODEL_PATH}")
        
        return best_model, best_params
        
    except Exception as e:
        logger.error(f"Error in model training: {str(e)}", exc_info=True)
        raise
