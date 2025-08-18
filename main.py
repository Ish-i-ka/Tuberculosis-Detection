# tuberculosis_detection/main.py

import os
import sys
from datetime import datetime

project_root_abs_path = os.path.abspath(os.path.dirname(__file__))

if project_root_abs_path not in sys.path:
    sys.path.insert(0, project_root_abs_path)


from constant import training_pipeline as CONSTANT
from entity.config_entity import TrainingPipelineConfig, DataSplittingConfig, ModelTrainerConfig, ModelEvaluationConfig
from utils.logger import logger
from utils.exception import TBDetectionError
from utils.common import exit_on_critical_error, create_and_clear_directory
from scripts.split_data import split_data
from scripts.train_model import train_model
from scripts.evaluate_model import run_evaluation
from utils.logger import setup_logger
from utils.mlflow_utils import setup_mlflow, start_run, log_params, log_metrics, log_artifact, log_model

setup_logger(log_dir="logs", log_file_prefix="pipeline_log")
import config 
import tensorflow as tf

def main():
    """
    Orchestrates the entire TB Detection workflow:
    1. Initializes pipeline configuration for a unique run.
    2. Executes Data Splitting and Preparation stage.
    3. Executes Model Training stage.
    4. Executes Model Evaluation and Interpretation stage.
    5. Tracks experiments with MLflow
    """

    setup_mlflow()
    
    logger.info("--- Starting End-to-End TB Detection Workflow ---")
    
    try:
        #Initialize the overall training pipeline configuration
        training_pipeline_config = TrainingPipelineConfig(timestamp=datetime.now())
    
        create_and_clear_directory(
            os.path.join(config.BASE_DIR, training_pipeline_config.artifact_dir),
            "pipeline artifact root directory"
        )
        logger.info(f"Artifacts for this run will be stored in: {os.path.join(config.BASE_DIR, training_pipeline_config.artifact_dir)}")

        # Start MLflow run
        with start_run(run_name=f"tb_detection_{training_pipeline_config.timestamp_str}"):
            # Phase 1: Data Splitting and Preparation
            logger.info("\n--- Executing Stage 1: Data Splitting and Preparation ---")
            data_splitting_config = DataSplittingConfig(training_pipeline_config=training_pipeline_config)
            data_splitting_artifact = split_data(config=data_splitting_config)
            logger.info("Stage 1: Data Splitting and Preparation Completed Successfully.")
            logger.info(f"Data Distribution: {data_splitting_artifact.class_distribution}")
            
            # Log data splitting parameters and metrics
            log_params({"data_split": data_splitting_artifact.class_distribution})

            # Phase 2: Model Training
            logger.info("\n--- Executing Stage 2: Model Training ---")
            model_trainer_config = ModelTrainerConfig(training_pipeline_config=training_pipeline_config)
            model_trainer_artifact = train_model(config=model_trainer_config)
            logger.info("Stage 2: Model Training Completed Successfully.")
            
            # Log training metrics
            training_metrics = {
                "training_accuracy": model_trainer_artifact.training_accuracy,
                "training_loss": model_trainer_artifact.training_loss,
                "validation_accuracy": model_trainer_artifact.validation_accuracy,
                "validation_loss": model_trainer_artifact.validation_loss
            }
            log_metrics(training_metrics)
            
            # Log model parameters
            log_params(model_trainer_artifact.model_params)
            
            # Log training history plot
            log_artifact(model_trainer_artifact.training_history_plot)

            # Phase 3: Model Evaluation and Interpretation
            logger.info("\n--- Executing Stage 3: Model Evaluation and Interpretation ---")
            model_evaluation_config = ModelEvaluationConfig(training_pipeline_config=training_pipeline_config)
            evaluation_artifact = run_evaluation(config=model_evaluation_config)
            logger.info("Stage 3: Model Evaluation and Interpretation Completed Successfully.")
            
            # Log evaluation metrics
            evaluation_metrics = {
                "test_accuracy": evaluation_artifact.test_accuracy,
                "auc_score": evaluation_artifact.auc_score
            }
            log_metrics(evaluation_metrics)
            
            # Log evaluation artifacts
            log_artifact(evaluation_artifact.confusion_matrix_path)
            log_artifact(evaluation_artifact.roc_curve_path)
            
            # Log the model (load it first)
            model = tf.keras.models.load_model(model_trainer_artifact.trained_model_path)
            log_model(model, "model")

        logger.info("\n--- End-to-End TB Detection Workflow Completed Successfully! ---")

    except TBDetectionError as e:
        exit_on_critical_error(e, "A project-specific error occurred during the main workflow execution.")
    except Exception as e:
        exit_on_critical_error(e, "An unexpected error occurred during the main workflow execution.")

if __name__ == "__main__":
    main()