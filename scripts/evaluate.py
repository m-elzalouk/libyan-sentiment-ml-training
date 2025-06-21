"""
Evaluation and reporting for sentiment analysis models.
"""
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score
)
import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scripts.utils import log_step

def evaluate_and_save_results(model, X_test_vec, y_test, vectorizer, config, best_params):
    """
    Evaluate the model and save detailed results and visualizations.
    
    Args:
        model: Trained model
        X_test_vec: Vectorized test features
        y_test: Test labels
        vectorizer: Fitted vectorizer
        config: Configuration object
        best_params: Best parameters from grid search
    """
    logger = logging.getLogger(__name__)
    
    @log_step("Generate predictions")
    def _generate_predictions():
        logger.info("Generating predictions on test set")
        y_pred = model.predict(X_test_vec)
        
        # Get prediction probabilities if available
        y_proba = None
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test_vec)[:, 1]
            logger.debug("Prediction probabilities generated")
        else:
            logger.info("Model does not support probability predictions")
            
        return y_pred, y_proba
    
    @log_step("Calculate metrics")
    def _calculate_metrics(y_true, y_pred, y_proba):
        logger.info("Calculating evaluation metrics")
        
        # Basic metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted'),
            'best_params': best_params
        }
        
        # Add ROC-AUC if probabilities are available
        if y_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
            metrics['avg_precision'] = average_precision_score(y_true, y_proba)
            logger.info(f"ROC-AUC: {metrics['roc_auc']:.4f}, Avg Precision: {metrics['avg_precision']:.4f}")
        
        # Detailed classification report
        metrics['classification_report'] = classification_report(
            y_true, y_pred, 
            output_dict=True,
            target_names=['Negative', 'Positive']
        )
        
        logger.info(
            f"Accuracy: {metrics['accuracy']:.4f}, "
            f"Precision: {metrics['precision']:.4f}, "
            f"Recall: {metrics['recall']:.4f}, "
            f"F1: {metrics['f1']:.4f}"
        )
        
        return metrics
    
    @log_step("Generate confusion matrix")
    def _plot_confusion_matrix(y_true, y_pred, output_path):
        logger.info("Generating confusion matrix")
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues', 
            xticklabels=['Negative', 'Positive'], 
            yticklabels=['Negative', 'Positive']
        )
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Confusion matrix saved to {output_path}")
        return cm
    
    @log_step("Generate ROC and PR curves")
    def _plot_roc_pr_curves(y_true, y_proba, output_prefix):
        if y_proba is None:
            logger.info("Skipping ROC/PR curves - no probability predictions available")
            return None, None
            
        logger.info("Generating ROC and PR curves")
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = roc_auc_score(y_true, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        roc_path = f"{output_prefix}_roc.png"
        plt.savefig(roc_path)
        plt.close()
        
        # PR Curve
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        avg_precision = average_precision_score(y_true, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.step(recall, precision, where='post', label=f'AP={avg_precision:.2f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall Curve')
        plt.legend(loc='lower left')
        pr_path = f"{output_prefix}_pr.png"
        plt.savefig(pr_path)
        plt.close()
        
        logger.info(f"ROC and PR curves saved to {roc_path} and {pr_path}")
        return roc_auc, avg_precision
    
    @log_step("Save evaluation results")
    def _save_results(metrics, output_path):
        logger.info(f"Saving evaluation results to {output_path}")
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
                return int(obj)
            elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_numpy_types(x) for x in obj]
            return obj
        
        # Save JSON results
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(convert_numpy_types(metrics), f, indent=2, ensure_ascii=False)
        
        # Print formatted results to console
        logger.info("\n" + "="*50)
        logger.info("EVALUATION RESULTS")
        logger.info("="*50)
        for key, value in metrics.items():
            if key != 'classification_report':
                logger.info(f"{key.replace('_', ' ').title()}: {value}")
        logger.info("\n" + "Classification Report:")
        logger.info(classification_report(
            y_test, y_pred, 
            target_names=['Negative', 'Positive'],
            digits=4
        ))
    
    try:
        # Ensure output directories exist
        os.makedirs(os.path.dirname(config.RESULTS_PATH) or '.', exist_ok=True)
        os.makedirs(os.path.dirname(config.CONFUSION_MATRIX_PATH) or '.', exist_ok=True)
        
        # Generate predictions and metrics
        y_pred, y_proba = _generate_predictions()
        metrics = _calculate_metrics(y_test, y_pred, y_proba)
        
        # Generate visualizations
        _plot_confusion_matrix(y_test, y_pred, config.CONFUSION_MATRIX_PATH)
        roc_auc, avg_precision = _plot_roc_pr_curves(
            y_test, 
            y_proba, 
            os.path.splitext(config.ROC_CURVE_PATH)[0]
        )
        
        # Update metrics with visualization paths
        metrics['visualizations'] = {
            'confusion_matrix': config.CONFUSION_MATRIX_PATH,
            'roc_curve': f"{os.path.splitext(config.ROC_CURVE_PATH)[0]}_roc.png",
            'pr_curve': f"{os.path.splitext(config.ROC_CURVE_PATH)[0]}_pr.png"
        }
        
        # Save all results
        _save_results(metrics, config.RESULTS_PATH)
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}", exc_info=True)
        raise
