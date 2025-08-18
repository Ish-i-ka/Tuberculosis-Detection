import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from datetime import datetime

from constant import training_pipeline as CONSTANT
import config

# A global logger instance
logger = logging.getLogger('tb_project_logger')
logger.setLevel(logging.INFO)

# A flag to prevent adding multiple handlers if setup_logger is called multiple times
_logger_configured = False

def setup_logger(log_dir: str, log_file_prefix: str):
    """
    Sets up a custom logger that logs to both console and a rotating file.
    The log file will have a timestamp in its name.
    Args:
        log_dir (str): The directory where log files should be saved.
        log_file_prefix (str): Prefix for the log file name (e.g., 'pipeline_log').
    """
    global _logger_configured

    if _logger_configured:
        logger.setLevel(logging.INFO) # Ensure level is correct if already configured
        return

    os.makedirs(log_dir, exist_ok=True) # Ensure logs directory exists

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_name = f"{log_file_prefix}_{timestamp}.log"
    log_file_path = os.path.join(log_dir, log_file_name)

    file_handler = RotatingFileHandler(log_file_path, maxBytes=10*1024*1024, backupCount=5)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    logger.info("Logging initialized: Console and timestamped file handlers configured.")
    _logger_configured = True