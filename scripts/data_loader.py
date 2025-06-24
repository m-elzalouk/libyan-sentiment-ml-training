"""
Data loading and preprocessing for Libyan Dialect Sentiment Analysis.
"""
import os
import logging
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from scripts.utils import log_step
from dotenv import load_dotenv
from typing import Tuple, Union, List

# Load environment variables from .env at the start
load_dotenv()

def load_and_prepare_data(config, return_dataframe=False):
    """
    Load and preprocess the dataset.
    
    Args:
        config: Configuration object containing parameters
        return_dataframe: If True, returns the full DataFrame instead of split data
        
    Returns:
        If return_dataframe is False (default):
            tuple: (X_train, X_test, y_train, y_test) - Split and preprocessed data
        If return_dataframe is True:
            pd.DataFrame: Cleaned and preprocessed DataFrame
    """
    logger = logging.getLogger(__name__)
    
    @log_step("Load dataset")
    def _load_data():
        if not os.path.exists(config.Test_DATA_PATH):
            logger.error(f"Data file not found at {config.Test_DATA_PATH}")
            raise FileNotFoundError(f"Data file not found at {config.Test_DATA_PATH}")
        
        logger.info(f"Loading data from {config.Test_DATA_PATH}")
        return pd.read_csv(config.Test_DATA_PATH)
    
    @log_step("Clean and preprocess data")
    def _clean_data(df):
        initial_count = len(df)
        df = df.dropna(subset=["Processed_Text_base_ai", "Sentiment"])
        df = df[df["Sentiment"].isin(["positive", "negative"])]
        
        cleaned_count = len(df)
        if initial_count > cleaned_count:
            logger.warning(
                f"Dropped {initial_count - cleaned_count} rows with missing or invalid data. "
                f"Kept {cleaned_count} rows."
            )
        
        return df
    
    def _extract_text_features(text):
        """Extract basic text features."""
        text = str(text)  # Ensure text is string
        text_length = len(text)
        word_count = len(text.split())
        
        return {
            'text_length': text_length,
            'word_count': word_count
        }

    @log_step("Prepare features and labels")
    def _prepare_features(df: pd.DataFrame) -> Tuple[Union[pd.DataFrame, pd.Series], np.ndarray]:
        """
        Prepare features including text and engineered features.
        
        Args:
            df: Input DataFrame with text data
            
        Returns:
            Tuple containing features (X) and labels (y)
        """
        # Map sentiment to numeric labels
        map_sentiment_to_numeric_labels = os.getenv("MAP_SENTIMENT_TO_NUMERIC_LABELS", "True").lower() == "true"
        
        # Log class distribution
        class_dist = df["Sentiment"].value_counts().to_dict()
        logger.info(f"Class distribution: {class_dist}")
        
        # Extract text features
        logger.info("Extracting text features...")
        text_features = df["Processed_Text_base_ai"].astype(str).apply(_extract_text_features)
        
        # Convert features to DataFrame
        features_df = pd.DataFrame(list(text_features))
        
        # Add original text
        features_df['text'] = df["Processed_Text_base_ai"].astype(str)
        
        # Convert labels if needed
        y = (df["Sentiment"].map({'positive': 1, 'negative': 0}) 
             if map_sentiment_to_numeric_labels 
             else df["Sentiment"].astype(str))
        
        return features_df, y.values

    @log_step("Split data into train/test sets")
    def _split_data(X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=config.TEST_SIZE, 
            random_state=config.RANDOM_STATE, 
            stratify=y
        )
        
        logger.info(
            f"Split data into training ({len(X_train)} samples) and "
            f"test ({len(X_test)} samples) sets"
        )
        
        # Log class distribution in train/test
        train_dist = np.bincount(y_train)
        test_dist = np.bincount(y_test)
        logger.info(f"Train set class distribution: {dict(enumerate(train_dist))}")
        logger.info(f"Test set class distribution: {dict(enumerate(test_dist))}")
        
        return X_train, X_test, y_train, y_test
    
    try:
        # Execute the pipeline
        df = _load_data()
        df = _clean_data(df)
        
        if return_dataframe:
            # Return the cleaned DataFrame directly
            logger.info(f"Returning DataFrame with shape: {df.shape}")
            return df
            
        # Otherwise, prepare features and split the data
        X, y = _prepare_features(df)
        return _split_data(X, y)
        
    except Exception as e:
        logger.error(f"Data loading and preparation failed with error: {str(e)}")
        raise
