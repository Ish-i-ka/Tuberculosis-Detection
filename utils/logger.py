# tb_detection_project/utils/logger.py

import logging
import os
import sys
from logging.handlers import RotatingFileHandler # Still using RotatingFileHandler for robustness
from datetime import datetime

# Import config.py (need to adjust sys.path as it's outside utils)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..') # Go up from utils/ to project root
sys.path.append(project_root)
import config

# --- Configure the single, global logger instance ---
logger = logging.getLogger('tb_project_logger') # Get the logger by a unique name
logger.setLevel(logging.INFO) # Default minimum level for logs to be processed

# Prevent adding multiple handlers if the logger is already configured
if not logger.handlers:
    # Ensure logs directory exists - this is where the automatic creation happens!
    os.makedirs(config.LOGS_DIR, exist_ok=True)

    # Generate a timestamped log file name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_name = f"project_log_{timestamp}.log"
    log_file_path = os.path.join(config.LOGS_DIR, log_file_name)

    # File Handler for detailed logs (rotates logs when file size reaches 10MB)
    file_handler = RotatingFileHandler(
        log_file_path,
        maxBytes=10*1024*1024, # 10 MB
        backupCount=5
    )
    # Detailed formatter for file logs (timestamp, logger name, level, message)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console Handler for real-time feedback
    console_handler = logging.StreamHandler(sys.stdout)
    # Simpler formatter for console logs (just level and message)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    logger.info("Logging initialized: Console and timestamped file handlers configured.")
