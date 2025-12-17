import logging
import os
from datetime import datetime

def setup_logger(name, log_file=None, level=logging.INFO):
    """
    Configure and return a logger.
    
    Args:
        name: Logger name
        log_file: Log file path; if None, log only to console
        level: Log level
    
    Returns:
        logging.Logger: Configured logger
    """
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid adding duplicate handlers
    if logger.handlers:
        logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        # Ensure log directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# Create default logger
default_logger = setup_logger('rf_scene_generator')

def log_info(message):
    """Record info log"""
    default_logger.info(message)

def log_error(message):
    """Record error log"""
    default_logger.error(message)

def log_warning(message):
    """Record warning log"""
    default_logger.warning(message)

def log_debug(message):
    """Record debug log"""
    default_logger.debug(message)

def set_log_file(log_file_path):
    """Set log file output path"""
    global default_logger
    default_logger = setup_logger('rf_scene_generator', log_file_path)