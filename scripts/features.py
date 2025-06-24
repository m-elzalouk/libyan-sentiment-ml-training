"""
Feature extraction using TfidfVectorizer with optional TruncatedSVD for dimensionality reduction.
"""
import logging
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from scripts.utils import log_step

def extract_features(X_train, X_test, config, ngram_range=None, use_svd=True, n_components=300):
    """
    Extract features from text data using TF-IDF vectorization with optional SVD reduction.
    
    Args:
        X_train: Training text data
        X_test: Test text data
        config: Configuration object with vectorizer parameters
        ngram_range: Optional tuple (min_n, max_n) to override the n-gram range
        use_svd: Whether to apply TruncatedSVD for dimensionality reduction
        n_components: Number of components for TruncatedSVD if use_svd is True
        
    Returns:
        tuple: (X_train_vec, X_test_vec, pipeline) - Transformed features and the feature extraction pipeline
    """
    logger = logging.getLogger(__name__)
    
    @log_step("Initialize feature extraction pipeline")
    def _initialize_pipeline():
        # Create a copy of vectorizer params to avoid modifying the original
        vectorizer_params = config.VECTORIZER_PARAMS.copy()
        
        # Override ngram_range if provided
        if ngram_range is not None:
            vectorizer_params['ngram_range'] = ngram_range
        
        # Create the vectorizer
        vectorizer = TfidfVectorizer(**vectorizer_params)
        
        # Create the pipeline
        pipeline_steps = [
            ('tfidf', vectorizer)
        ]
        
        # Add SVD if enabled and we have enough features
        if use_svd:
            # Make a copy of vectorizer params to modify for the test fit
            test_vectorizer_params = vectorizer_params.copy()
            
            # Try with the current parameters first
            try:
                test_vectorizer = TfidfVectorizer(**test_vectorizer_params)
                X_sample = test_vectorizer.fit_transform(X_train[:1000])  # Use a sample for efficiency
                n_features = X_sample.shape[1]
                
                # If we get here, the vectorizer worked with current params
                max_possible = min(n_components, test_vectorizer_params.get('max_features', 10000) - 1)
                
                if n_features >= 2:
                    n_components_actual = min(max_possible, n_features - 1)
                    n_components_actual = max(2, n_components_actual)  # Ensure at least 2 components
                    
                    svd = TruncatedSVD(
                        n_components=n_components_actual,
                        n_iter=7,
                        random_state=config.RANDOM_STATE
                    )
                    pipeline_steps.append(('svd', svd))
                    logger.info(f"Added TruncatedSVD with {n_components_actual} components "
                              f"(input features: {n_features})")
                else:
                    logger.warning(
                        f"Not enough features ({n_features}) for SVD. "
                        "Skipping dimensionality reduction."
                    )
                    
            except ValueError as e:
                if "no terms remain" in str(e):
                    logger.warning(
                        f"No terms remain with current parameters. Adjusting min_df and max_df. Error: {e}"
                    )
                    # Try more lenient parameters
                    test_vectorizer_params['min_df'] = 1  # Minimum document frequency
                    test_vectorizer_params['max_df'] = 1.0  # Maximum document frequency
                    
                    try:
                        test_vectorizer = TfidfVectorizer(**test_vectorizer_params)
                        X_sample = test_vectorizer.fit_transform(X_train[:1000])
                        n_features = X_sample.shape[1]
                        
                        # Update the main vectorizer params
                        vectorizer_params.update(test_vectorizer_params)
                        
                        if n_features >= 2:
                            n_components_actual = min(n_components, n_features - 1)
                            n_components_actual = max(2, n_components_actual)
                            
                            svd = TruncatedSVD(
                                n_components=n_components_actual,
                                n_iter=7,
                                random_state=config.RANDOM_STATE
                            )
                            pipeline_steps = [
                                ('tfidf', TfidfVectorizer(**vectorizer_params)),
                                ('svd', svd)
                            ]
                            logger.info(f"Added TruncatedSVD with {n_components_actual} components "
                                      f"after adjusting vectorizer parameters (features: {n_features})")
                        else:
                            logger.warning(
                                "Still not enough features after adjusting parameters. "
                                "Skipping SVD."
                            )
                            
                    except Exception as e2:
                        logger.error(f"Failed to initialize vectorizer even after adjustment: {e2}")
                        raise
                else:
                    # Re-raise if it's a different error
                    raise
        
        return Pipeline(pipeline_steps)
    
    @log_step("Fit and transform training data")
    def _fit_transform_train(pipeline, X_train):
        X_train_vec = pipeline.fit_transform(X_train)
        
        # Log pipeline steps and dimensions
        steps_info = " → ".join([name for name, _ in pipeline.steps])
        logger.info(f"Pipeline: {steps_info}")
        
        # Get vocabulary size from the vectorizer
        vectorizer = pipeline.named_steps.get('tfidf')
        if vectorizer is not None and hasattr(vectorizer, 'vocabulary_'):
            logger.info(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
        
        # Get explained variance if SVD is used
        if 'svd' in pipeline.named_steps:
            svd = pipeline.named_steps['svd']
            try:
                # Only access explained_variance_ratio_ if it exists
                if hasattr(svd, 'explained_variance_ratio_'):
                    explained_variance = np.sum(svd.explained_variance_ratio_)
                    logger.info(
                        f"SVD reduced to {svd.n_components} components, "
                        f"explaining {explained_variance:.1%} of variance"
                    )
                else:
                    logger.info(
                        f"SVD initialized with {svd.n_components} components "
                        "(variance not yet calculated)"
                    )
            except Exception as e:
                logger.warning(f"Could not log SVD variance: {str(e)}")
        
        logger.info(f"Training matrix shape: {X_train_vec.shape}")
        return X_train_vec
    
    @log_step("Transform test data")
    def _transform_test(pipeline, X_test):
        X_test_vec = pipeline.transform(X_test)
        logger.info(f"Test data transformed. Test matrix shape: {X_test_vec.shape}")
        
        # Log sparsity for sparse matrices
        if hasattr(X_test_vec, 'nnz'):
            logger.info(
                f"Sparsity: {100 * (1 - X_test_vec.nnz / (X_test_vec.shape[0] * X_test_vec.shape[1])):.2f}% "
                f"({X_test_vec.nnz} non-zero elements of {X_test_vec.shape[0] * X_test_vec.shape[1]} total)"
            )
        
        return X_test_vec
    
    try:
        # Initialize the pipeline
        pipeline = _initialize_pipeline()
        
        # Fit and transform the training data
        X_train_vec = _fit_transform_train(pipeline, X_train)
        
        # Transform the test data
        X_test_vec = _transform_test(pipeline, X_test)
        
        # Log most important features (top 10)
        if hasattr(pipeline.named_steps['tfidf'], 'get_feature_names_out'):
            feature_names = pipeline.named_steps['tfidf'].get_feature_names_out()
            mean_tfidf = np.asarray(X_train_vec.mean(axis=0)).ravel()
            top_indices = mean_tfidf.argsort()[-10:][::-1]
            top_features = [(feature_names[i], mean_tfidf[i]) for i in top_indices]
            
            logger.info("Top 10 most important features (by mean TF-IDF):")
            for feature, score in top_features:
                logger.info(f"  - {feature}: {score:.4f}")
        
        return X_train_vec, X_test_vec, pipeline
        
    except Exception as e:
        logger.error(f"Error during feature extraction: {str(e)}", exc_info=True)
        raise
