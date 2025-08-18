# tb_detection_project/utils/mlflow_utils.py

import os
import sys
import mlflow
import numpy as np
from dotenv import load_dotenv
from mlflow.models.signature import infer_signature
from contextlib import contextmanager
from typing import Dict, Any, Optional, Generator, Union
from utils.logger import logger
from utils.exception import TBDetectionError

def setup_mlflow() -> None:
    try:
        load_dotenv() 
        
        mlflow_tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
        dagshub_token = os.getenv('DAGSHUB_TOKEN')
        
        if not mlflow_tracking_uri:
            logger.warning("MLFLOW_TRACKING_URI not found in environment variables. Using local tracking.")
            mlflow_tracking_uri = "file:./mlruns"
        
        if dagshub_token:
            os.environ['MLFLOW_TRACKING_USERNAME'] = 'Ish-i-ka'
            os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_token
        
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        logger.info(f"MLflow tracking URI set to: {mlflow_tracking_uri}")
        
        experiment_name = os.getenv('MLFLOW_EXPERIMENT_NAME', 'TB-Detection')
        mlflow.set_experiment(experiment_name)
        
    except Exception as e:
        raise TBDetectionError(f"Error setting up MLflow tracking: {str(e)}", sys.exc_info())

@contextmanager
def start_run(run_name: Optional[str] = None, nested: bool = False) -> Generator:
    """
    Context manager for MLflow run.
    
    Args:
        run_name: Optional name for the run
        nested: Whether this is a nested run
        
    Yields:
        mlflow.ActiveRun: The active MLflow run
        
    Raises:
        TBDetectionError: If there's an error during the run
    """
    try:
        with mlflow.start_run(run_name=run_name, nested=nested) as run:
            # Log system info and dependencies
            try:
                mlflow.set_tag("python_version", sys.version)
                mlflow.set_tag("mlflow_version", mlflow.__version__)
               
            except Exception as e:
                logger.warning(f"Failed to log environment tags: {str(e)}")
                
            yield run
            
    except Exception as e:
        raise TBDetectionError(f"Error in MLflow run: {str(e)}", sys.exc_info())

def log_params(params: Dict[str, Any]) -> None:
    """
    Log parameters to MLflow.
    
    Args:
        params: Dictionary of parameters to log
        
    Raises:
        TBDetectionError: If there's an error logging parameters
    """
    try:
        # Convert any non-string values to strings
        formatted_params = {
            k: str(v) if not isinstance(v, (int, float, str, bool)) else v
            for k, v in params.items()
        }
        mlflow.log_params(formatted_params)
    except Exception as e:
        raise TBDetectionError(f"Error logging parameters: {str(e)}", sys.exc_info())

def log_metrics(metrics: Dict[str, float], step: Optional[int] = None) -> None:
    """
    Log metrics to MLflow.
    
    Args:
        metrics: Dictionary of metrics to log
        step: Optional step number for the metrics
        
    Raises:
        TBDetectionError: If there's an error logging metrics
    """
    try:
        mlflow.log_metrics(metrics, step=step)
    except Exception as e:
        raise TBDetectionError(f"Error logging metrics: {str(e)}", sys.exc_info())

def log_artifact(local_path: str) -> None:
    """
    Log an artifact (file) to MLflow.
    
    Args:
        local_path: Path to the file to log
        
    Raises:
        TBDetectionError: If there's an error logging the artifact
    """
    try:
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Artifact file not found: {local_path}")
        mlflow.log_artifact(local_path)
    except Exception as e:
        raise TBDetectionError(f"Error logging artifact: {str(e)}", sys.exc_info())

def log_model(model: Any, artifact_path: str, input_example: Optional[np.ndarray] = None) -> None:
    """
    Log a model to MLflow with signature and input example.
    
    Args:
        model: The model to log
        artifact_path: Path where to log the model
        input_example: Optional example input for model signature
        
    Raises:
        TBDetectionError: If there's an error logging the model
    """
    try:
        signature = None
        if input_example is not None:
            try:
                prediction = model.predict(input_example)
                signature = infer_signature(input_example, prediction)
            except Exception as e:
                logger.warning(f"Failed to create model signature: {str(e)}")
        
        # Log the model with TensorFlow flavor
        mlflow.tensorflow.log_model(
            model,
            artifact_path,
            signature=signature,
            input_example=input_example
        )
        
        # Log additional model artifacts
        try:
            # Save and log model summary
            from io import StringIO
            summary_file = StringIO()
            model.summary(print_fn=lambda x: summary_file.write(x + '\n'))
            summary_content = summary_file.getvalue()
            
            temp_path = "model_summary.txt"
            with open(temp_path, "w") as f:
                f.write(summary_content)
            mlflow.log_artifact(temp_path)
            os.remove(temp_path)
            
        except Exception as e:
            logger.warning(f"Failed to log model summary: {str(e)}")
            
    except Exception as e:
        raise TBDetectionError(f"Error logging model: {str(e)}", sys.exc_info())

def end_run() -> None:
    try:
        mlflow.end_run()
    except Exception as e:
        raise TBDetectionError(f"Error ending MLflow run: {str(e)}", sys.exc_info())
