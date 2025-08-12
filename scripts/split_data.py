# tb_detection_project/scripts/split_data.py

import os
import shutil
import random
from glob import glob
from sklearn.model_selection import train_test_split
from tqdm import tqdm # For progress bar during file copying

# --- Import Configuration ---
import sys
# Add project root directory to sys.path to import config.py
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.join(current_script_dir, '..')
sys.path.append(project_root_dir)
import config

# --- Configuration (using variables from config.py) ---
RAW_DATA_PATH = config.RAW_DATA_PATH
PROCESSED_DATA_PATH = config.PROCESSED_DATA_PATH
TRAIN_SPLIT = config.TRAIN_SPLIT
VAL_SPLIT = config.VAL_SPLIT
TEST_SPLIT = config.TEST_SPLIT
CLASSES = config.CLASSES

# --- Helper function to create/recreate directories ---
# This function will ensure that data/processed/ and its subfolders are clean each run
def create_and_clear_directory(path):
    if os.path.exists(path):
        print(f"Removing existing directory: {path}")
        shutil.rmtree(path)
    os.makedirs(path)
    print(f"Created directory: {path}")

# --- Main data splitting logic ---
def split_data():
    print(f"--- Starting Data Splitting Process ---")
    print(f"Source Raw Data Path: {RAW_DATA_PATH}")
    print(f"Destination Processed Data Path: {PROCESSED_DATA_PATH}")

    # --- Step 1: Prepare Destination Directories ---
    create_and_clear_directory(PROCESSED_DATA_PATH)
    for subset in ['train', 'val', 'test']:
        for cls in CLASSES:
            os.makedirs(os.path.join(PROCESSED_DATA_PATH, subset, cls))
    print("Destination directories prepared.")

    # --- Step 2: Collect All Image Paths from Raw Data ---
    all_image_paths = []
    all_labels = []

    if not os.path.exists(RAW_DATA_PATH):
        print(f"\nCRITICAL ERROR: Raw data directory not found at: {RAW_DATA_PATH}")
        print("Please ensure your Kaggle dataset is unzipped into this location.")
        sys.exit(1) # Exit the script if raw data path is invalid

    print(f"\nScanning for images in raw data...")
    for cls_name in CLASSES:
        cls_path = os.path.join(RAW_DATA_PATH, cls_name)

        if not os.path.exists(cls_path):
            print(f"  WARNING: Class folder '{cls_name}' not found at: {cls_path}. Skipping.")
            continue

        unique_images_in_class = set()
        for ext in ['*.png', '*.PNG', '*.jpeg', '*.JPEG', '*.jpg', '*.JPG']:
            unique_images_in_class.update(glob(os.path.join(cls_path, ext)))
        
        images_for_current_class = list(unique_images_in_class) # Convert back to list
        
        if not images_for_current_class:
            print(f"  WARNING: No images found with common extensions in '{cls_name}' folder: {cls_path}. Skipping.")
            continue

        all_image_paths.extend(images_for_current_class)
        all_labels.extend([cls_name] * len(images_for_current_class))
    
    if not all_image_paths:
        print(f"\nCRITICAL ERROR: No images were found across any specified classes ({CLASSES}) in {RAW_DATA_PATH}.")
        print("Please verify the dataset structure and file extensions.")
        sys.exit(1) # Exit if no images found at all

    print(f"Found a total of {len(all_image_paths)} images for splitting.")

    # --- Step 3: Split Data into Train, Validation, and Test Sets ---
    # Stratified split to maintain class proportions
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        all_image_paths, all_labels,
        test_size=(VAL_SPLIT + TEST_SPLIT),
        stratify=all_labels,
        random_state=42 # For reproducibility
    )

    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels,
        test_size=(TEST_SPLIT / (VAL_SPLIT + TEST_SPLIT)), # Calculate test_size relative to temp_paths
        stratify=temp_labels,
        random_state=42 # For reproducibility
    )

    datasets = {
        'train': list(zip(train_paths, train_labels)),
        'val': list(zip(val_paths, val_labels)),
        'test': list(zip(test_paths, test_labels))
    }
    print("Data split into training, validation, and test sets.")

    # --- Step 4: Copy Files to Processed Directories ---
    for subset, data_list in datasets.items():
        print(f"\nCopying {subset} images...")
        for img_path, label in tqdm(data_list, desc=f"Copying {subset}"):
            dest_dir = os.path.join(PROCESSED_DATA_PATH, subset, label)
            shutil.copy(img_path, dest_dir)
    
    print("\n--- Data Splitting Complete. Final Counts: ---")
    for subset in ['train', 'val', 'test']:
        print(f"--- {subset.upper()} ---")
        for cls in CLASSES:
            count = len(os.listdir(os.path.join(PROCESSED_DATA_PATH, subset, cls)))
            print(f"  {cls}: {count} images")

if __name__ == "__main__":
    split_data()