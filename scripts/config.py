"""
Configuration and hyperparameters for the sentiment analysis pipeline.
"""
import os
from dotenv import load_dotenv
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.linear_model import LogisticRegression

# Load environment variables from .env at the start
load_dotenv()

class Config:
    # Directory paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.getenv("DATA_PATH", "data/dataset_cleaned-positive-negative-v2.csv")
    Test_DATA_PATH = os.getenv("Test_DATA_PATH", "data/test.csv")
    RESULTS_DIR = os.getenv("RESULTS_DIR", "results")
    MODELS_DIR = os.getenv("MODELS_DIR", "models")
    LOGS_DIR = os.getenv("LOGS_DIR", "logs")
    
    # Logging configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    LOG_FILE_FORMAT = os.getenv("LOG_FILE_FORMAT", "training_%Y%m%d_%H%M%S.log")
    
    # Console logging format with colors
    LOG_COLORS = {
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    }
    
    # File logging format (no colors)
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Colored console format
    CONSOLE_FORMAT = "%(log_color)s%(levelname)-8s%(reset)s %(asctime)s - %(name)s - %(message)s"
    
    # Secondary log colors (for highlighting)
    SECONDARY_LOG_COLORS = {
        'message': {
            'SUCCESS': 'green',
            'IMPORTANT': 'yellow',
            'ERROR': 'red',
        }
    }
    
    @property
    def LOG_FILE(self):
        """Generate log file path with timestamp."""
        from datetime import datetime
        os.makedirs(self.LOGS_DIR, exist_ok=True)
        return os.path.join(
            self.LOGS_DIR,
            datetime.now().strftime(self.LOG_FILE_FORMAT)
        )
    RESULTS_PATH = os.path.join(RESULTS_DIR, "scores.json")
    MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
    CONFUSION_MATRIX_PATH = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    ROC_CURVE_PATH = os.path.join(RESULTS_DIR, "roc_curve.png")
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    USE_SMOTE = False  # Set True to enable SMOTE
    # N-gram configurations for optimization
    NGRAMS = {
        'Unigrams': (1, 1),
        'Bigrams': (2, 2),
        'Trigrams': (3, 3),
        'Unigrams & Bigrams': (1, 2),
        'Unigrams & Trigrams': (1, 3),
        'Bigrams & Trigrams': (2, 3),
    }
    # Vectorizer parameters
    VECTORIZER_PARAMS = {
        'max_features': 80000,
        'ngram_range': (1, 3),
        'min_df': 5,
        'max_df': 0.99,
        'analyzer': 'word',
    }
    
    # Hyperparameter grids for grid search
    GRID_SEARCH_PARAMS = {
        'svm_linear': {
            'C': [0.01, 0.1, 1, 10],  # Smaller range of C values
            'kernel': ['linear','rbf'],
            'gamma': ['scale','auto'],
            'class_weight': ['balanced'],  # Only use balanced to handle class imbalance
            'probability': [True],
            'random_state': [42,43,44,45],
            'max_iter': [5000,10000,15000],
            'tol': [1e-2],  # Higher tolerance for faster convergence
            'cache_size': [1500]  # Increase cache size
        },
        'svm_rbf': {
            'C': [0.01, 0.1, 1],  # Smaller range of C values
            'kernel': ['rbf'],
            'gamma': ['scale'],  # Start with just 'scale'
            'class_weight': ['balanced'],  # Only use balanced
            'probability': [True],
            'random_state': [42,43,44,45],
            'max_iter': [10000],
            'tol': [1e-2],  # Higher tolerance for faster convergence
            'cache_size': [1000]  # Increase cache size
        },
        # Parameters for Naive Bayes (Gaussian for SVD, Multinomial otherwise)
        'nb': {
            'var_smoothing': [1e-9, 1e-8, 1e-7]  # For GaussianNB when using SVD
        },
        'nb_multinomial': {
            'alpha': [0.1, 0.5, 1.0],  # For MultinomialNB when not using SVD
            'fit_prior': [True, False]
        },
        'logreg': [
            # Configuration for l1/l2 penalties
            {
                'penalty': ['l1', 'l2'],
                'C': [0.01, 0.1, 1, 10],
                'solver': ['liblinear', 'saga'],
                'class_weight': [None, 'balanced'],
                'random_state': [42,43,44,45],
                'max_iter': [10000]
            },
            # Configuration for elasticnet penalty (includes l1_ratio)
            {
                'penalty': ['elasticnet'],
                'C': [0.01, 0.1, 1, 10],
                'solver': ['saga'],  # Only saga supports elasticnet
                'l1_ratio': [0.1, 0.5, 0.9],
                'class_weight': [None, 'balanced'],
                'random_state': [42,43,44,45],
                'max_iter': [10000]
            },
            # Configuration for no penalty
            {
                'penalty': [None],  # Changed from 'none' to None
                'solver': ['lbfgs', 'newton-cg', 'sag', 'saga'],
                'class_weight': [None, 'balanced'],
                'random_state': [42,43,44,45],
                'max_iter': [10000]
            }
        ]
    }
    
    # Classifier configurations for optimization
    CLASSIFIERS = {
        'SVM': (SVC(probability=True, random_state=42, max_iter=10000), {
            'kernel': ['linear', 'rbf'],
            'C': [0.01, 0.1, 1, 10],
            'gamma': ['scale', 'auto', 0.01, 0.1],
            'class_weight': [None, 'balanced']
        }),
        'Naive Bayes': (MultinomialNB(), {
            'alpha': [0.1, 0.5, 1.0]
        }),
        'GaussianNB': (GaussianNB(), {
            'var_smoothing': [1e-9, 1e-8, 1e-7]
        }),
        'Logistic Regression': (LogisticRegression(random_state=42, max_iter=10000), {
            'penalty': ['l1', 'l2', 'elasticnet', None],
            'C': [0.01, 0.1, 1, 10],
            'solver': ['liblinear', 'saga'],
            'l1_ratio': [0.1, 0.5, 0.9],  # For elasticnet
            'class_weight': [None, 'balanced']
        }),
    }

    # Fine-tuning parameter grids
    FINE_TUNE_GRIDS = {
        'SVM': {
            'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
            'gamma': ['scale', 'auto', 0.0001, 0.001, 0.01, 0.1, 1],
            'class_weight': [None, 'balanced'],
            'degree': [2, 3, 4],  # For poly kernel
            'coef0': [0.0, 0.1, 1.0],  # For poly/sigmoid
            'shrinking': [True, False],
            'max_iter': [10000, 20000],
            'decision_function_shape': ['ovr', 'ovo']
        },
        'Naive Bayes': {
            'alpha': [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
            'fit_prior': [True, False],
            'class_prior': [None, [0.3, 0.7], [0.4, 0.6]]
        },
        'Logistic Regression': {
            'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'penalty': ['l1', 'l2', 'elasticnet', None],
            'solver': ['liblinear', 'saga', 'sag', 'lbfgs', 'newton-cg'],
            'class_weight': [None, 'balanced'],
            'max_iter': [1000, 5000, 10000],
            'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],  # For elasticnet
            'fit_intercept': [True, False],
            'warm_start': [True, False],
            'multi_class': ['auto', 'ovr', 'multinomial']
        }
    }
    SCORING = 'f1_weighted'
