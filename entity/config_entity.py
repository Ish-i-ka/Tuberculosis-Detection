from datetime import datetime
import os
from constant import training_pipeline as CONSTANT
import config 

class TrainingPipelineConfig:
    """Overall configuration for a single pipeline run, including timestamped artifact directory."""
    def __init__(self, timestamp: datetime = datetime.now()):
        self.timestamp_str: str = timestamp.strftime("%Y%m%d_%H%M%S")
        self.pipeline_name: str = CONSTANT.PIPELINE_NAME
        
        # Artifacts/YYYYMMDD_HHMMSS (main artifact directory for this run)
        self.artifact_dir: str = os.path.join(CONSTANT.ARTIFACT_DIR_NAME, self.timestamp_str)
        
        # Paths for logs within this specific artifact run
        self.logs_dir: str = os.path.join(self.artifact_dir, CONSTANT.LOGS_DIR_NAME)
        
        # Path for the final model (saved outside the timestamped artifact_dir for easy access)
        self.final_model_save_path: str = os.path.join(
            config.BASE_DIR, CONSTANT.FINAL_MODEL_SAVE_DIR, CONSTANT.MODEL_FILE_NAME
        )
        self.final_class_indices_save_path: str = os.path.join(
            config.BASE_DIR, CONSTANT.FINAL_MODEL_SAVE_DIR, CONSTANT.CLASS_INDICES_FILE_NAME
        )


class DataSplittingConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        # Artifacts/timestamp/data_splitting
        self.data_splitting_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, CONSTANT.DATA_SPLITTING_DIR_NAME
        )
        # Store processed data in data/processed at project root
        self.processed_data_dir: str = os.path.join(
            config.BASE_DIR, "data", "processed"
        )
        # Full path to the raw data based on project root and config
        self.raw_data_path: str = os.path.join(
            config.BASE_DIR, "data", "raw", config.RAW_DATA_FOLDER_NAME
        )
        self.classes: list = ["Normal", "Tuberculosis"] # Hardcoded classes
        self.split_ratios: dict = CONSTANT.TRAIN_VAL_TEST_SPLIT_RATIOS


class ModelTrainerConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        # Artifacts/timestamp/model_trainer
        self.model_trainer_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, CONSTANT.MODEL_TRAINER_DIR_NAME
        )
        # Artifacts/timestamp/model_trainer/trained_model
        self.trained_model_dir: str = os.path.join(
            self.model_trainer_dir, CONSTANT.MODEL_TRAINER_TRAINED_MODEL_SUBDIR
        )
        # These will be the actual paths where the trained model and class indices are saved
        self.trained_model_file_path: str = os.path.join(
            self.trained_model_dir, CONSTANT.MODEL_FILE_NAME
        )
        self.class_indices_file_path: str = os.path.join(
            self.trained_model_dir, CONSTANT.CLASS_INDICES_FILE_NAME
        )
        
        self.img_height: int = CONSTANT.IMG_HEIGHT
        self.img_width: int = CONSTANT.IMG_WIDTH
        self.batch_size: int = CONSTANT.BATCH_SIZE
        
        # Default HPs 
        self.dense_units: int = CONSTANT.HP_DENSE_UNITS
        self.dropout_rate: float = CONSTANT.HP_DROPOUT_RATE
        self.lr_phase1: float = CONSTANT.LR_PHASE1_DEFAULT
        self.lr_phase2: float = CONSTANT.LR_PHASE2_DEFAULT
        self.epochs_phase1: int = CONSTANT.EPOCHS_PHASE1_DEFAULT
        self.epochs_phase2: int = CONSTANT.EPOCHS_PHASE2_DEFAULT
        self.training_history_plot_name: str = CONSTANT.TRAINING_HISTORY_PLOT_NAME


class ModelEvaluationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        # Artifacts/timestamp/model_evaluation
        self.model_evaluation_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, CONSTANT.MODEL_EVALUATION_DIR_NAME
        )
        # Artifacts/timestamp/model_evaluation/plots
        self.evaluation_plots_dir: str = os.path.join(
            self.model_evaluation_dir, CONSTANT.EVALUATION_PLOTS_SUBDIR
        )
        # Artifacts/timestamp/model_evaluation/grad_cam_examples
        self.grad_cam_examples_dir: str = os.path.join(
            self.model_evaluation_dir, CONSTANT.GRAD_CAM_EXAMPLES_SUBDIR
        )
        self.img_height: int = CONSTANT.IMG_HEIGHT
        self.img_width: int = CONSTANT.IMG_WIDTH
        self.batch_size: int = CONSTANT.BATCH_SIZE
        
        # Paths to input model and data from previous stages
        self.trained_model_path: str = os.path.join(
            training_pipeline_config.artifact_dir, 
            CONSTANT.MODEL_TRAINER_DIR_NAME,
            CONSTANT.MODEL_TRAINER_TRAINED_MODEL_SUBDIR,
            CONSTANT.MODEL_FILE_NAME
        )
        self.class_indices_path: str = os.path.join(
            training_pipeline_config.artifact_dir, 
            CONSTANT.MODEL_TRAINER_DIR_NAME,
            CONSTANT.MODEL_TRAINER_TRAINED_MODEL_SUBDIR,
            CONSTANT.CLASS_INDICES_FILE_NAME
        )
        self.processed_data_path: str = os.path.join(
            training_pipeline_config.artifact_dir, 
            CONSTANT.DATA_SPLITTING_DIR_NAME,
            CONSTANT.PROCESSED_DATA_SUBDIR_NAME
        )
        