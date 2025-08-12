# tb_detection_project/utils/common.py

import os
import shutil
import sys
# Import the global logger instance
from utils.logger import logger
# Import custom exceptions
from utils.exceptions import FileSystemError, TBDetectionError # Import base for catch-all

def create_and_clear_directory(path: str, description: str = "directory"):
    """
    Creates a directory, clearing its contents if it already exists.
    Logs actions and handles OSError gracefully by re-raising a custom exception.
    Args:
        path (str): The path to the directory to create/clear.
        description (str): A descriptive name for the directory (for logging).
    Raises:
        FileSystemError: If an OS error occurs during directory operations.
    """
    try:
        if os.path.exists(path):
            logger.info(f"Removing existing {description}: {path}")
            shutil.rmtree(path)
        os.makedirs(path)
        logger.info(f"Created {description}: {path}")
    except OSError as e:
        # Re-raise with custom FileSystemError, passing sys.exc_info() for traceback details
        raise FileSystemError(f"Failed to create/clear {description} at {path}: {e}", sys.exc_info()) from e

def exit_on_critical_error(e: Exception, message: str = "An unrecoverable error occurred."):
    """
    Logs a critical error with full traceback and exits the script.
    It intelligently logs the exception message.
    Args:
        e (Exception): The exception object caught.
        message (str): A custom message to prepend to the log.
    """
    # If it's one of our custom exceptions, its __str__ method already formats the message nicely.
    # Otherwise, use the generic Exception's string representation.
    log_message = str(e) if isinstance(e, TBDetectionError) else f"{message}: {e}"
    
    # Log the error at CRITICAL level, including the full traceback (exc_info=True)
    # We pass exc_info=True to logger.critical so it captures the original traceback
    # even if 'e' is our custom exception.
    logger.critical(f"SCRIPT HALTED: {log_message}", exc_info=True)
    sys.exit(1) # Exit with an error code, indicating failure