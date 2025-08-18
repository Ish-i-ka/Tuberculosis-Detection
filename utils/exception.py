import sys

class TBDetectionError(Exception):
    def __init__(self, message: str, error_details: sys.exc_info = None):
        super().__init__(message)
        self.message = message
        self.lineno = None
        self.file_name = None

        if error_details:
            exc_type, exc_value, exc_tb = error_details
            if exc_tb:
                self.lineno = exc_tb.tb_lineno
                self.file_name = exc_tb.tb_frame.f_code.co_filename
        
    def __str__(self):
        if self.file_name and self.lineno is not None:
            return f"Error in [{self.file_name}] at line [{self.lineno}]: {self.message}"
        else:
            return f"Error: {self.message}"

class DataError(TBDetectionError): pass
class FileSystemError(TBDetectionError): pass
class ConfigurationError(TBDetectionError): pass
class ModelError(TBDetectionError): pass