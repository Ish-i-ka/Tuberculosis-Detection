import os

# --- Base Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Project root
RAW_DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'TB_Chest_Radiography_Database')
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')
GRAD_CAM_DIR = os.path.join(RESULTS_DIR, 'grad_cam_examples')
LOGS_DIR = os.path.join(BASE_DIR, 'logs') # Directory for log files
LOG_FILENAME = 'project_log.log' 

# --- Data Splitting Parameters ---
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
CLASSES = ['Normal', 'Tuberculosis']

# --- Model Training Parameters ---
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
EPOCHS_PHASE1 = 10 # Train custom head
EPOCHS_PHASE2 = 20 # Fine-tune
LEARNING_RATE_PHASE1 = 0.001
LEARNING_RATE_PHASE2 = 0.00001 # Much smaller for fine-tuning


# --- Model Specifics --- 
MODEL_NAME = 'resnet50'
MODEL_FILENAME = f'tb_detection_{MODEL_NAME}_best.h5'

# --- Hyperparameter Tuning Parameters ---
ENABLE_TUNING = True # Set to True to run tuning, False to use defaults
TUNER_PROJECT_NAME = 'tb_detection_tuning'
TUNER_DIRECTORY = os.path.join(BASE_DIR, 'tuner_results') # Where Keras Tuner logs its trials
MAX_TRIALS = 15 # Max number of hyperparameter combinations to try
EXECUTIONS_PER_TRIAL = 1 # Number of models to train per trial (for robustness, but increases time)
OVERWRITE_TUNER_RESULTS = True # Set to True to start a fresh tuning run each time

# Hyperparameter Search Space (for Keras Tuner)
HP_DENSE_UNITS_CHOICES = [128, 256, 512] # Options for the dense layer in the custom head
HP_DROPOUT_RATE_MIN = 0.2
HP_DROPOUT_RATE_MAX = 0.6
HP_DROPOUT_RATE_STEP = 0.1
HP_LR_PH1_CHOICES = [1e-3, 5e-4] # Learning rates for Phase 1
HP_LR_PH2_CHOICES = [1e-5, 5e-6, 1e-6] # Learning rates for Phase 2 (for fine-tuning)

# Path to save/load best hyperparameters
BEST_HPS_PATH = os.path.join(MODELS_DIR, 'best_hps.json')