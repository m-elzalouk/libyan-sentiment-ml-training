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
import matplotlib
matplotlib.use('Agg')  # Set the backend to 'Agg' to avoid GUI-related issues
import matplotlib.pyplot as plt
import seaborn as sns
from scripts.utils import log_step
from matplotlib.ticker import MaxNLocator  # For better y-axis ticks

# Set the style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

def evaluate_and_save_results(model, X_test_combined, y_test, results_dir, config, best_params=None, ngram_config=None):
    """
    Evaluate the model and save detailed results and visualizations.
    
    Args:
        model: Trained model
        X_test_combined: Combined test features (TF-IDF + numeric features)
        y_test: Test labels
        results_dir: Directory to save results
        config: Configuration object
        best_params: Best parameters from grid search (optional)
        ngram_config: Dictionary containing n-gram configuration (name, range, vocab_size)
    """
    os.makedirs(results_dir, exist_ok=True)
    logger = logging.getLogger(__name__)
    
    # Ensure y_test is numpy array
    y_test = np.array(y_test) if not isinstance(y_test, np.ndarray) else y_test
    
    @log_step("Generate predictions")
    def _generate_predictions():
        logger.info("Generating predictions on test set")
        logger.info(f"Input shape for prediction: {X_test_combined.shape}")
        
        # Generate predictions
        y_pred = model.predict(X_test_combined)
        
        # Get prediction probabilities if available
        y_proba = None
        if hasattr(model, "predict_proba"):
            try:
                y_proba = model.predict_proba(X_test_combined)[:, 1]
                logger.debug("Prediction probabilities generated")
            except IndexError:
                # Handle case where predict_proba returns only one column
                y_proba = model.predict_proba(X_test_combined)
                if y_proba.shape[1] == 1:
                    y_proba = y_proba.ravel()
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
            'model_params': best_params or {},
            'ngram_name': ngram_config.get('name', 'unknown') if ngram_config else 'unknown',
            'ngram_range': ngram_config.get('range', 'unknown') if ngram_config else 'unknown',
            'vocab_size': ngram_config.get('vocab_size', 0) if ngram_config else 0,
            'best_params': best_params or {}
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
    
    @log_step("Generate classifier comparison plots")
    def _plot_classifier_comparison(metrics, output_prefix):
        """Generate comparison plots for different classifiers and n-grams."""
        # This is a placeholder for actual multi-classifier comparison
        # In a real scenario, you would compare multiple models
        model_name = metrics.get('model_name', 'Current Model')
        accuracy = metrics['accuracy']
        
        # Create a simple comparison plot (can be enhanced with actual multiple model results)
        plt.figure(figsize=(10, 6))
        sns.barplot(
            x=['Current Model'],
            y=[accuracy],
            hue=['Accuracy'],
            palette='viridis'
        )
        plt.title(f'Model Performance: {model_name}')
        plt.ylim(0, 1.1)
        plt.ylabel('Score')
        plt.xticks(rotation=45)
        comparison_path = f"{output_prefix}_comparison.png"
        plt.tight_layout()
        plt.savefig(comparison_path)
        plt.close()
        logger.info(f"Classifier comparison plot saved to {comparison_path}")
        
        return comparison_path

    @log_step("Generate F1-score per class")
    def _plot_f1_per_class(classification_report_dict, output_prefix):
        """Generate a bar plot of F1-scores for each class."""
        # Extract class-wise metrics (excluding 'accuracy', 'macro avg', 'weighted avg')
        class_metrics = {k: v for k, v in classification_report_dict.items() 
                         if isinstance(v, dict) and 'f1-score' in v}
        
        if not class_metrics:
            logger.warning("No class-wise metrics found in classification report")
            return None
            
        # Create DataFrame for easier plotting
        df = pd.DataFrame(class_metrics).transpose()
        
        # Plot
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(
            x=list(df.index),
            y=list(df['f1-score']),
            hue=list(df.index),  # Add hue to avoid deprecation warning
            palette='viridis',
            legend=False  # Don't show legend since we're using hue just for colors
        )
        ax.edgecolor='black'
        
        # Add value labels on top of bars
        for p in ax.patches:
            ax.annotate(
                f"{p.get_height():.2f}",
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center',
                xytext=(0, 5),
                textcoords='offset points'
            )
        
        plt.title('F1-Score per Class', pad=20)
        plt.xlabel('Class')
        plt.ylabel('F1-Score')
        plt.ylim(0, 1.1)
        plt.xticks(rotation=45, ha='right')
        
        f1_path = f"{output_prefix}_f1_per_class.png"
        plt.tight_layout()
        plt.savefig(f1_path)
        plt.close()
        logger.info(f"F1-score per class plot saved to {f1_path}")
        return f1_path

    @log_step("Generate detailed classification report")
    def _save_detailed_report(classification_report_dict, output_prefix):
        """Save detailed classification report as CSV and HTML.
        
        Args:
            classification_report_dict: Dictionary containing classification metrics
            output_prefix: Prefix for output file paths
            
        Returns:
            tuple: Paths to the saved CSV and HTML files
        """
        # Convert to DataFrame for CSV output
        report_df = pd.DataFrame(classification_report_dict).transpose()
        
        # Save as CSV
        csv_path = f"{output_prefix}_classification_report.csv"
        report_df.to_csv(csv_path)
        
        # Save as HTML with better formatting (simplified to avoid jinja2 dependency)
        html_path = f"{output_prefix}_classification_report.html"
        
        # Create a simple HTML table without pandas styling
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Classification Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h2 { color: #333; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: center; }
                th { background-color: #f2f2f2; font-weight: bold; }
                tr:nth-child(even) { background-color: #f9f9f9; }
                .metric { font-weight: bold; }
            </style>
        </head>
        <body>
            <h2>Detailed Classification Report</h2>
            <table>
                <thead>
                    <tr>
                        <th>Class</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>F1-Score</th>
                        <th>Support</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        try:
            # Check if the classification report is in the expected format
            if not isinstance(classification_report_dict, dict):
                logger.warning(f"Unexpected classification report format: {type(classification_report_dict)}")
                raise ValueError("Classification report must be a dictionary")
            
            # Add rows for each class
            for class_name, metrics in classification_report_dict.items():
                if class_name in ['accuracy', 'macro avg', 'weighted avg']:
                    continue
                
                if not isinstance(metrics, dict):
                    logger.warning(f"Unexpected metrics format for class {class_name}: {metrics}")
                    continue
                    
                html_content += f"""
                        <tr>
                            <td class="metric">{class_name}</td>
                            <td>{metrics.get('precision', 0):.2f}</td>
                            <td>{metrics.get('recall', 0):.2f}</td>
                            <td>{metrics.get('f1-score', 0):.2f}</td>
                            <td>{metrics.get('support', 0):.0f}</td>
                        </tr>
                """
            
            # Add summary rows
            for class_name in ['accuracy', 'macro avg', 'weighted avg']:
                if class_name in classification_report_dict:
                    metrics = classification_report_dict[class_name]
                    if class_name == 'accuracy':
                        html_content += f"""
                            <tr>
                                <td class="metric">Accuracy</td>
                                <td colspan="3"></td>
                                <td>{metrics:.2f}</td>
                            </tr>
                        """
                    elif isinstance(metrics, dict):
                        html_content += f"""
                            <tr>
                                <td class="metric">{class_name}</td>
                                <td>{metrics.get('precision', 0):.2f}</td>
                                <td>{metrics.get('recall', 0):.2f}</td>
                                <td>{metrics.get('f1-score', 0):.2f}</td>
                                <td>{metrics.get('support', 0):.0f}</td>
                            </tr>
                        """
        except Exception as e:
            logger.error(f"Error generating HTML report: {str(e)}", exc_info=True)
            # Fallback to simple table with just the DataFrame
            html_content += f"""
                <tr>
                    <td colspan="5">
                        <p>Error generating detailed report. See CSV for data.</p>
                        <pre>{report_df.to_html()}</pre>
                    </td>
                </tr>
            """
        
        # Close HTML
        html_content += """
                </tbody>
            </table>
        </body>
        </html>
        """
        
        # Save HTML file
        try:
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            logger.info(f"Detailed classification reports saved to {csv_path} and {html_path}")
        except Exception as e:
            logger.error(f"Error saving HTML report: {str(e)}", exc_info=True)
            # If HTML save fails, just return the CSV path
            return csv_path, None
            
        return csv_path, html_path

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

    from datetime import datetime
    
    @log_step("Save evaluation results")
    def _save_results(metrics, output_path):
        logger.info(f"Saving evaluation results to {output_path}")
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        
        # Convert numpy types to native Python types for JSON serialization
        # Compatible with both NumPy 1.x and 2.x
        def convert_numpy_types(obj):
            # Handle numpy numeric types
            if hasattr(np, 'number') and isinstance(obj, np.number):
                return float(obj) if isinstance(obj, np.floating) else int(obj)
                
            # Handle numpy arrays
            if hasattr(np, 'ndarray') and isinstance(obj, np.ndarray):
                return obj.tolist()
                
            # Handle dictionaries and lists recursively
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_numpy_types(x) for x in obj]
                
            # Return the object as is if no conversion is needed
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
        
        # Format and log classification report
        report_df = pd.DataFrame(metrics['classification_report']).transpose()
        logger.info("\n" + "="*50)
        logger.info("DETAILED CLASSIFICATION REPORT")
        logger.info("="*50)
        logger.info("\n" + report_df.to_string())
        
        return metrics
    
    try:
        # Generate base filenames for all outputs
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{model.__class__.__name__.lower()}_{timestamp}"
        
        # Set up paths for all outputs
        results_path = os.path.join(results_dir, f"{base_filename}_results.json")
        confusion_matrix_path = os.path.join(results_dir, f"{base_filename}_confusion_matrix.png")
        output_prefix = os.path.join(results_dir, base_filename)
        
        # Generate predictions and metrics
        y_pred, y_proba = _generate_predictions()
        metrics = _calculate_metrics(y_test, y_pred, y_proba)
        
        # Add model name and parameters to metrics
        metrics.update({
            'model_name': model.__class__.__name__,
            'model_params': best_params,
            'timestamp': datetime.now().isoformat(),
            'ngram_range': getattr(model, 'feature_importances_', None) is not None and \
                          hasattr(model, 'feature_names_in_') and \
                          hasattr(model, 'get_params') and \
                          model.get_params().get('ngram_range', None) or 'unknown'
        })
        
        # Generate visualizations
        _plot_confusion_matrix(y_test, y_pred, confusion_matrix_path)
        _plot_roc_pr_curves(y_test, y_proba, output_prefix)
        
        # Generate additional visualizations
        _plot_classifier_comparison(metrics, output_prefix)
        _plot_f1_per_class(metrics['classification_report'], output_prefix)
        _save_detailed_report(metrics['classification_report'], output_prefix)
        
        # Save results (including paths to all generated files)
        metrics['visualizations'] = {
            'confusion_matrix': confusion_matrix_path,
            'roc_curve': f"{output_prefix}_roc.png",
            'pr_curve': f"{output_prefix}_pr.png",
            'classifier_comparison': f"{output_prefix}_comparison.png",
            'f1_per_class': f"{output_prefix}_f1_per_class.png",
            'classification_report_csv': f"{output_prefix}_classification_report.csv",
            'classification_report_html': f"{output_prefix}_classification_report.html"
        }
        
        # Save the metrics
        _save_results(metrics, results_path)
        
        # Log completion with all output paths
        logger.info("\n" + "="*50)
        logger.info("EVALUATION COMPLETE")
        logger.info("="*50)
        logger.info(f"All results and visualizations have been saved to: {results_dir}")
        logger.info(f"- Results: {results_path}")
        logger.info(f"- Confusion Matrix: {confusion_matrix_path}")
        logger.info(f"- ROC Curve: {output_prefix}_roc.png")
        logger.info(f"- PR Curve: {output_prefix}_pr.png")
        logger.info(f"- Classifier Comparison: {output_prefix}_comparison.png")
        logger.info(f"- F1 per Class: {output_prefix}_f1_per_class.png")
        logger.info(f"- Detailed Reports: {output_prefix}_classification_report.{{csv,html}}")
        logger.info("="*50)
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}", exc_info=True)
        raise
    
@log_step("Save evaluation results")
def _save_results(metrics, output_path):
    logger.info(f"Saving evaluation results to {output_path}")
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)