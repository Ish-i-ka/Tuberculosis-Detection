# tb_detection_project/utils/common.py

import os
import shutil
import sys
# Import the global logger instance and exceptions from their packages
from utils.logger import logger
from utils.exception import FileSystemError, TBDetectionError

def create_and_clear_directory(path: str, description: str = "directory"):
    try:
        if os.path.exists(path):
            logger.info(f"Removing existing {description}: {path}")
            shutil.rmtree(path)
        os.makedirs(path)
        logger.info(f"Created {description}: {path}")
    except OSError as e:
        raise FileSystemError(f"Failed to create/clear {description} at {path}: {e}", sys.exc_info()) from e

def exit_on_critical_error(e: Exception, message: str = "An unrecoverable error occurred."):
    log_message = str(e) if isinstance(e, TBDetectionError) else f"{message}: {e}"
    logger.critical(f"SCRIPT HALTED: {log_message}", exc_info=True)
    sys.exit(1)