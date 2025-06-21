"""
Utility functions for logging, timing, and SMOTE support.
"""
import os
import sys
import time
import logging
from logging.handlers import RotatingFileHandler
from contextlib import contextmanager

try:
    import colorlog
    from colorlog import ColoredFormatter
    COLOR_AVAILABLE = True
except ImportError:
    COLOR_AVAILABLE = False

from scripts.config import Config

def setup_logging(logger_name=None):
    """
    Set up logging with both file and console handlers.
    
    Args:
        logger_name: Optional name for the logger. If None, returns the root logger.
        
    Returns:
        logging.Logger: Configured logger instance
    """
    config = Config()
    
    # Create logs directory if it doesn't exist
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    
    # Get or create the logger
    logger = logging.getLogger(logger_name) if logger_name else logging.getLogger()
    logger.setLevel(config.LOG_LEVEL)
    
    # Clear any existing handlers to avoid duplicate logs
    if logger.handlers:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
    
    # Don't propagate to parent loggers to avoid duplicate logs
    logger.propagate = False
    
    # Create formatters
    file_formatter = logging.Formatter(
        fmt=config.LOG_FORMAT,
        datefmt=config.LOG_DATE_FORMAT
    )
    
    # Set up colored formatter if available
    if COLOR_AVAILABLE and sys.stderr.isatty():
        console_formatter = ColoredFormatter(
            '%(log_color)s%(levelname)-8s%(reset)s %(asctime)s - %(name)s - %(message)s',
            datefmt=config.LOG_DATE_FORMAT,
            reset=True,
            log_colors=config.LOG_COLORS,
            secondary_log_colors=config.SECONDARY_LOG_COLORS,
            style='%'
        )
    else:
        console_formatter = file_formatter
    
    try:
        # File handler (rotates when file reaches 5MB, keeps 3 backup files)
        file_handler = RotatingFileHandler(
            config.LOG_FILE,
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3,
            encoding='utf-8'
        )
        file_handler.setLevel(config.LOG_LEVEL)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    except (IOError, OSError) as e:
        print(f"Warning: Could not set up file logging: {e}")
        print("Falling back to console logging only.")
    
    # Console handler (always add console handler)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(config.LOG_LEVEL)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Add custom log levels for better visual distinction
    if not hasattr(logging, 'SUCCESS'):
        logging.SUCCESS = 25  # Between WARNING and INFO
        logging.addLevelName(logging.SUCCESS, 'SUCCESS')
        
        def success(self, message, *args, **kws):
            if self.isEnabledFor(logging.SUCCESS):
                self._log(logging.SUCCESS, message, args, **kws)
        
        logging.Logger.success = success
        
        def success_root(message, *args, **kwargs):
            logging.log(logging.SUCCESS, message, *args, **kwargs)
            
        logging.success = success_root
    
    # Log the logging configuration
    logger.debug("Logging initialized")
    logger.debug(f"Log level set to: {config.LOG_LEVEL}")
    try:
        logger.debug(f"Log file: {os.path.abspath(config.LOG_FILE)}")
    except Exception:
        logger.debug("Log file: Not available (console only)")
    
    return logger
    
    # Log the start of the application
    logger.info("=" * 50)
    logger.info("Starting training pipeline")
    logger.info("=" * 50)
    
    return logger

@contextmanager
def log_duration(operation_name="Operation"):
    """Context manager to log the duration of an operation."""
    logger = logging.getLogger(__name__)
    start_time = time.time()
    logger.info(f"{operation_name} started")
    
    try:
        yield
    except Exception as e:
        logger.error(f"{operation_name} failed with error: {str(e)}", exc_info=True)
        raise
    finally:
        duration = time.time() - start_time
        logger.info(f"{operation_name} completed in {duration:.2f} seconds")

def log_step(step_name):
    """Decorator to log the start and end of a function."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = logging.getLogger(__name__)
            logger.info(f"Starting: {step_name}")
            try:
                result = func(*args, **kwargs)
                logger.info(f"Completed: {step_name}")
                return result
            except Exception as e:
                logger.error(f"Failed: {step_name} - {str(e)}", exc_info=True)
                raise
        return wrapper
    return decorator

# Optionally, add SMOTE support here if needed
# from imblearn.over_sampling import SMOTE
# def apply_smote(X, y):
#     sm = SMOTE(random_state=42)
#     X_res, y_res = sm.fit_resample(X, y)
#     return X_res, y_res
