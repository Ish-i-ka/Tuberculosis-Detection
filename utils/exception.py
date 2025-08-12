# tb_detection_project/utils/exceptions.py

import sys

class TBDetectionError(Exception):
    """
    Base class for custom exceptions in the Tuberculosis Detection project.
    Captures error details including file name and line number.
    """
    def __init__(self, message: str, error_details: sys.exc_info = None):
        super().__init__(message) # Initialize base Exception with the main message
        self.message = message
        self.lineno = None
        self.file_name = None

        if error_details:
            # error_details should be the tuple returned by sys.exc_info()
            # (exc_type, exc_value, exc_traceback)
            exc_type, exc_value, exc_tb = error_details
            if exc_tb: # Ensure traceback exists
                self.lineno = exc_tb.tb_lineno
                self.file_name = exc_tb.tb_frame.f_code.co_filename
        
    def __str__(self):
        # Format the error message to include file and line number if available
        if self.file_name and self.lineno is not None:
            return f"Error in [{self.file_name}] at line [{self.lineno}]: {self.message}"
        else:
            return f"Error: {self.message}"

# --- Specific Custom Exception Classes ---

class DataError(TBDetectionError):
    """Exception raised for errors related to data loading, processing, or structure."""
    def __init__(self, message="A data-related error occurred.", error_details: sys.exc_info = None):
        super().__init__(message, error_details)

class FileSystemError(TBDetectionError):
    """Exception raised for errors during file system operations (e.g., creating/copying files)."""
    def __init__(self, message="A file system error occurred.", error_details: sys.exc_info = None):
        super().__init__(message, error_details)

class ConfigurationError(TBDetectionError):
    """Exception raised for errors related to incorrect or missing configurations."""
    def __init__(self, message="A configuration error occurred.", error_details: sys.exc_info = None):
        super().__init__(message, error_details)

class ModelError(TBDetectionError):
    """Exception raised for errors during model building, training, or prediction."""
    def __init__(self, message="A model-related error occurred.", error_details: sys.exc_info = None):
        super().__init__(message, error_details)

# --- Example Usage (for testing the exception itself, not to be part of main script flow) ---
if __name__ == '__main__':
    # This block is for testing just the exception's behavior
    try:
        # Simulate an error
        value = 1 / 0
    except ZeroDivisionError as e:
        # Create a custom exception, passing the original error message and sys.exc_info()
        custom_exception = TBDetectionError("Division by zero occurred.", sys.exc_info())
        print(custom_exception) # This will call __str__
        # For a more specific error:
        custom_data_error = DataError("Input data was invalid.", sys.exc_info())
        print(custom_data_error)
        
    try:
        raise ConfigurationError("Missing essential API key.", sys.exc_info())
    except TBDetectionError as e:
        print(e)