import os

PIPELINE_NAME: str = "TBDetectionPipeline"
ARTIFACT_DIR_NAME: str = "Artifacts" # Main directory for all pipeline outputs (e.g., Artifacts/YYYYMMDD_HHMMSS)
FINAL_MODEL_SAVE_DIR: str = "final_model" # Top-level final model directory relative to project root

MODEL_FILE_NAME: str = "tb_detection_resnet50_best.h5" # Name of the saved Keras model file
CLASS_INDICES_FILE_NAME: str = "class_indices.txt" # Name of the class indices file
TRAINING_HISTORY_PLOT_NAME: str = "training_history.png" # Name of the training plot

# --- Data Splitting and Preprocessing Constants ---
DATA_SPLITTING_DIR_NAME: str = "data_splitting" # Directory for data splitting stage artifacts
PROCESSED_DATA_SUBDIR_NAME: str = "processed_data" # Subdir within data_splitting_dir for train/val/test splits

# --- Model Training Constants ---
MODEL_TRAINER_DIR_NAME: str = "model_trainer" # Directory for model trainer stage artifacts
MODEL_TRAINER_TRAINED_MODEL_SUBDIR: str = "trained_model" # Subdir within model_trainer_dir

# --- Model Evaluation Constants ---
MODEL_EVALUATION_DIR_NAME: str = "model_evaluation" # Directory for evaluation stage artifacts
EVALUATION_PLOTS_SUBDIR: str = "plots" # Subdir for confusion matrix, ROC curve
GRAD_CAM_EXAMPLES_SUBDIR: str = "grad_cam_examples" # Subdir for Grad-CAM images

LOGS_DIR_NAME: str = "logs" # Base directory for log files (under artifact_dir)
LOG_FILE_PREFIX: str = "pipeline_log" # Prefix for timestamped log files

IMG_HEIGHT: int = 224
IMG_WIDTH: int = 224
BATCH_SIZE: int = 32

HP_DENSE_UNITS: int = 256 # Default dense layer units
HP_DROPOUT_RATE: float = 0.5 # Default dropout rate
LR_PHASE1_DEFAULT: float = 0.001
LR_PHASE2_DEFAULT: float = 0.00001
EPOCHS_PHASE1_DEFAULT: int = 10
EPOCHS_PHASE2_DEFAULT: int = 20

TRAIN_VAL_TEST_SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}