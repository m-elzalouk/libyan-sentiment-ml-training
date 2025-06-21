"""
Configuration and hyperparameters for the sentiment analysis pipeline.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env at the start
load_dotenv()

class Config:
    # Directory paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.getenv("DATA_PATH", "data/dataset_cleaned-positive-negative-v2.csv")
    RESULTS_DIR = os.getenv("RESULTS_DIR", "results")
    MODELS_DIR = os.getenv("MODELS_DIR", "models")
    LOGS_DIR = os.getenv("LOGS_DIR", "logs")
    
    # Logging configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = os.path.join(LOGS_DIR, "training.log")
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    RESULTS_PATH = os.path.join(RESULTS_DIR, "scores.json")
    MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
    CONFUSION_MATRIX_PATH = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    ROC_CURVE_PATH = os.path.join(RESULTS_DIR, "roc_curve.png")
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    USE_SMOTE = False  # Set True to enable SMOTE
    VECTORIZER_PARAMS = {
        'max_features': 20000,
        'ngram_range': (1, 3),
        'min_df': 3,
        'max_df': 0.9,
        'analyzer': 'word',
    }
    GRID_SEARCH_PARAMS = {
        'svm_linear': {
            'C': [0.1, 1, 10],
            'kernel': ['linear'],
            'class_weight': ['balanced']
        },
        'svm_rbf': {
            'C': [0.1, 1, 10],
            'kernel': ['rbf'],
            'class_weight': ['balanced']
        },
        'logreg': {
            'solver': ['saga', 'liblinear'],
            'penalty': ['l2'],
            'C': [0.1, 1, 10]
        },
        'nb': {
            'alpha': [0.1, 1.0, 5.0]
        }
    }
    SCORING = 'f1_weighted'
