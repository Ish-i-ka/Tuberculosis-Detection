# tb_detection_project/scripts/split_data.py

import os
import shutil
import random
from glob import glob
from sklearn.model_selection import train_test_split
from tqdm import tqdm # For progress bar during file copying

# --- Import Configuration ---
import sys

import config
from utils.logger import logger # Import the global logger
from utils.exception import TBDetectionError,DataError, FileSystemError # Import custom exceptions
from utils.common import create_and_clear_directory, exit_on_critical_error 

# --- Configuration (using variables from config.py) ---
RAW_DATA_PATH = config.RAW_DATA_PATH
PROCESSED_DATA_PATH = config.PROCESSED_DATA_PATH
TRAIN_SPLIT = config.TRAIN_SPLIT
VAL_SPLIT = config.VAL_SPLIT
TEST_SPLIT = config.TEST_SPLIT
CLASSES = config.CLASSES

def split_data():
    logger.info(f"--- Starting Data Splitting Process ---")
    logger.info(f"Source Raw Data Path: {RAW_DATA_PATH}")
    logger.info(f"Destination Processed Data Path: {PROCESSED_DATA_PATH}")

    try:
        # This function now raises FileSystemError if there's an issue
        create_and_clear_directory(PROCESSED_DATA_PATH, "processed data directory")
        for subset in ['train', 'val', 'test']:
            for cls in CLASSES:
                # os.makedirs will raise an OSError if it fails, caught by FileSystemError
                os.makedirs(os.path.join(PROCESSED_DATA_PATH, subset, cls), exist_ok=True)
        logger.info("Destination directories prepared.")

        # Collect All Image Paths from Raw Data 
        all_image_paths = []
        all_labels = []

        if not os.path.exists(RAW_DATA_PATH):
            # Raise a custom DataError, passing sys.exc_info()
            raise DataError(f"Raw data directory not found at: {RAW_DATA_PATH}", sys.exc_info())

        logger.info(f"\nScanning for images in raw data...")
        for cls_name in CLASSES:
            cls_path = os.path.join(RAW_DATA_PATH, cls_name)

            if not os.path.exists(cls_path):
                logger.warning(f"Class folder '{cls_name}' not found at: {cls_path}. Skipping.")
                continue

            unique_images_in_class = set()
            for ext in ['*.png', '*.PNG', '*.jpeg', '*.JPEG', '*.jpg', '*.JPG']:
                unique_images_in_class.update(glob(os.path.join(cls_path, ext)))
            
            images_for_current_class = list(unique_images_in_class)

            if not images_for_current_class:
                logger.warning(f"No images found with common extensions in '{cls_name}' folder: {cls_path}. Skipping.")
                continue

            all_image_paths.extend(images_for_current_class)
            all_labels.extend([cls_name] * len(images_for_current_class))
        
        if not all_image_paths:
            # Raise a custom DataError if no images are found at all
            raise DataError("No unique images were found across any specified classes.", sys.exc_info())

        logger.info(f"Found a total of {len(all_image_paths)} unique images for splitting.")

        #Split Data into Train, Validation, and Test Sets
        train_paths, temp_paths, train_labels, temp_labels = train_test_split(
            all_image_paths, all_labels,
            test_size=(VAL_SPLIT + TEST_SPLIT),
            stratify=all_labels,
            random_state=42
        )

        val_paths, test_paths, val_labels, test_labels = train_test_split(
            temp_paths, temp_labels,
            test_size=(TEST_SPLIT / (VAL_SPLIT + TEST_SPLIT)),
            stratify=temp_labels,
            random_state=42
        )

        datasets = {
            'train': list(zip(train_paths, train_labels)),
            'val': list(zip(val_paths, val_labels)),
            'test': list(zip(test_paths, test_labels))
        }
        logger.info("Data split into training, validation, and test sets.")

        #Copy Files to Processed Directories
        for subset, data_list in datasets.items():
            logger.info(f"\nCopying {subset} images...")
            for img_path, label in tqdm(data_list, desc=f"Copying {subset}"):
                dest_dir = os.path.join(PROCESSED_DATA_PATH, subset, label)
                try:
                    shutil.copy(img_path, dest_dir)
                except Exception as copy_e:
                    # Catch copy errors and raise a FileSystemError
                    raise FileSystemError(f"Failed to copy '{os.path.basename(img_path)}' to '{dest_dir}': {copy_e}", sys.exc_info()) from copy_e
        
        logger.info("\n--- Data Splitting Complete. Final Counts: ---")
        for subset in ['train', 'val', 'test']:
            logger.info(f"--- {subset.upper()} ---")
            for cls in CLASSES:
                count = len(os.listdir(os.path.join(PROCESSED_DATA_PATH, subset, cls)))
                logger.info(f"  {cls}: {count} images")

    # Catch custom exceptions or any other generic exceptions
    except TBDetectionError as e:
        exit_on_critical_error(e, "A project-specific error occurred during data splitting.")
    except Exception as e:
        exit_on_critical_error(e, "An unexpected error occurred during data splitting.")

if __name__ == "__main__":
    split_data()