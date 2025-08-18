import os

# --- Base Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Project root
RAW_DATA_FOLDER_NAME = 'TB_Chest_Radiography_Database'
RAW_DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', RAW_DATA_FOLDER_NAME)
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