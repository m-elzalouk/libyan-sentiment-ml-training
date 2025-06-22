"""
Data loading and preprocessing for Libyan Dialect Sentiment Analysis.
"""
import os
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scripts.utils import log_step
from dotenv import load_dotenv

# Load environment variables from .env at the start
load_dotenv()

def load_and_prepare_data(config):
    """
    Load and preprocess the dataset.
    
    Args:
        config: Configuration object containing parameters
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test) - Split and preprocessed data
    """
    logger = logging.getLogger(__name__)
    
    @log_step("Load dataset")
    def _load_data():
        if not os.path.exists(config.DATA_PATH):
            logger.error(f"Data file not found at {config.DATA_PATH}")
            raise FileNotFoundError(f"Data file not found at {config.DATA_PATH}")
        
        logger.info(f"Loading data from {config.DATA_PATH}")
        return pd.read_csv(config.DATA_PATH)
    
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
    
    @log_step("Prepare features and labels")
    def _prepare_features(df):
        # Map sentiment to numeric labels
        map_sentiment_to_numeric_labels = os.getenv("MAP_SENTIMENT_TO_NUMERIC_LABELS", True)
        if map_sentiment_to_numeric_labels:
            logger.info(f"Mapping sentiment to numeric labels")
        #     label_mapping = {"positive": 1, "negative": 0}
        #     df["label"] = df["Sentiment"].map(label_mapping)
        df['label'] = df['Sentiment']
        # Log class distribution
        class_dist = df["label"].value_counts().to_dict()
        logger.info(f"Class distribution: {class_dist}")
        return df["Processed_Text_base_ai"].astype(str), (
            df["label"].map({'positive': 1, 'negative': 0}) if map_sentiment_to_numeric_labels else df["label"].astype(str)
        )

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
        X, y = _prepare_features(df)
        return _split_data(X, y)
        
    except Exception as e:
        logger.error(f"Error in data loading/preparation: {str(e)}", exc_info=True)
        raise
