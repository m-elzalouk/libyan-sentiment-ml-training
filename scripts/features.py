"""
Feature extraction using TfidfVectorizer.
"""
import logging
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scripts.utils import log_step

def extract_features(X_train, X_test, config):
    """
    Extract features from text data using TF-IDF vectorization.
    
    Args:
        X_train: Training text data
        X_test: Test text data
        config: Configuration object with vectorizer parameters
        
    Returns:
        tuple: (X_train_vec, X_test_vec, vectorizer) - Vectorized features and the vectorizer
    """
    logger = logging.getLogger(__name__)
    
    @log_step("Initialize TF-IDF Vectorizer")
    def _initialize_vectorizer():
        logger.info(f"Initializing TF-IDF Vectorizer with params: {config.VECTORIZER_PARAMS}")
        return TfidfVectorizer(**config.VECTORIZER_PARAMS)
    
    @log_step("Fit and transform training data")
    def _fit_transform_train(vectorizer, X_train):
        X_train_vec = vectorizer.fit_transform(X_train)
        logger.info(
            f"TF-IDF Vectorizer fitted. Vocabulary size: {len(vectorizer.vocabulary_)}. "
            f"Training matrix shape: {X_train_vec.shape}"
        )
        return X_train_vec
    
    @log_step("Transform test data")
    def _transform_test(vectorizer, X_test):
        X_test_vec = vectorizer.transform(X_test)
        logger.info(f"Test data transformed. Test matrix shape: {X_test_vec.shape}")
        
        # Log sparsity
        train_nnz = X_test_vec.nnz
        train_size = X_test_vec.shape[0] * X_test_vec.shape[1]
        sparsity = 100.0 * (1.0 - train_nnz / (train_size + 1e-10))
        logger.info(f"Test matrix sparsity: {sparsity:.2f}%")
        
        return X_test_vec
    
    try:
        vectorizer = _initialize_vectorizer()
        X_train_vec = _fit_transform_train(vectorizer, X_train)
        X_test_vec = _transform_test(vectorizer, X_test)
        
        # Log most important features (top 10)
        if hasattr(vectorizer, 'get_feature_names_out'):
            feature_names = vectorizer.get_feature_names_out()
            mean_tfidf = np.asarray(X_train_vec.mean(axis=0)).ravel()
            top_indices = mean_tfidf.argsort()[-10:][::-1]
            top_features = [(feature_names[i], mean_tfidf[i]) for i in top_indices]
            
            logger.info("Top 10 most important features (by mean TF-IDF):")
            for feature, score in top_features:
                logger.info(f"  - {feature}: {score:.4f}")
        
        return X_train_vec, X_test_vec, vectorizer
        
    except Exception as e:
        logger.error(f"Error during feature extraction: {str(e)}", exc_info=True)
        raise
