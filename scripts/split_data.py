import os
import shutil
import random
from glob import glob
from sklearn.model_selection import train_test_split
from tqdm import tqdm # For progress bar during file copying

# Ensure project root is in sys.path for imports
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import sys
import config
from utils.logger import logger
from utils.exception import DataError, FileSystemError, TBDetectionError
from utils.common import create_and_clear_directory, exit_on_critical_error
from entity.config_entity import DataSplittingConfig
from entity.artifact_entity import DataSplittingArtifact
from constant import training_pipeline as CONSTANT


def split_data(config: DataSplittingConfig) -> DataSplittingArtifact:
    raw_data_path = config.raw_data_path
    processed_data_dir = config.processed_data_dir
    classes = config.classes
    split_ratios = config.split_ratios
    data_splitting_dir = config.data_splitting_dir
    
    # Initialize class distribution dictionary
    class_distribution = {
        'train': {cls: 0 for cls in classes},
        'val': {cls: 0 for cls in classes},
        'test': {cls: 0 for cls in classes}
    }
    
    logger.info(f"--- Starting Data Splitting Process ---")
    logger.info(f"Source Raw Data Path: {raw_data_path}")
    logger.info(f"Destination Processed Data Path: {processed_data_dir}")

    try:
        create_and_clear_directory(data_splitting_dir, "data splitting directory")
        create_and_clear_directory(processed_data_dir, "processed data directory")
        for subset in ['train', 'val', 'test']:
            for cls in classes:
                # os.makedirs will raise an OSError if it fails, caught by FileSystemError
                os.makedirs(os.path.join(processed_data_dir, subset, cls), exist_ok=True)
        logger.info("Destination directories prepared.")

        # Collect All Image Paths from Raw Data 
        all_image_paths = []
        all_labels = []

        if not os.path.exists(raw_data_path):
            raise DataError(f"Raw data directory not found at: {raw_data_path}", sys.exc_info())

        logger.info(f"\nScanning for images in raw data...")
        for cls_name in classes:
            cls_path = os.path.join(raw_data_path, cls_name)

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
            raise DataError("No unique images were found across any specified classes.", sys.exc_info())

        logger.info(f"Found a total of {len(all_image_paths)} unique images for splitting.")

        train_ratio = split_ratios["train"]
        val_ratio = split_ratios["val"]
        test_ratio = split_ratios["test"]

        #Split Data into Train, Validation, and Test Sets
        train_paths, temp_paths, train_labels, temp_labels = train_test_split(
            all_image_paths, all_labels,
            test_size=(val_ratio + test_ratio),
            stratify=all_labels,
            random_state=42
        )

        val_paths, test_paths, val_labels, test_labels = train_test_split(
            temp_paths, temp_labels,
            test_size=(test_ratio / (val_ratio + test_ratio)),
            stratify=temp_labels,
            random_state=42
        )

        datasets = {
            'train': list(zip(train_paths, train_labels)),
            'val': list(zip(val_paths, val_labels)),
            'test': list(zip(test_paths, test_labels))
        }
        logger.info("Data split into training, validation, and test sets.")

        # Copy Files to both data/processed and artifacts directories
        artifacts_processed_dir = os.path.join(data_splitting_dir, CONSTANT.PROCESSED_DATA_SUBDIR_NAME)
        create_and_clear_directory(artifacts_processed_dir, "artifacts processed data directory")

        # Create required directories in artifacts
        for subset in ['train', 'val', 'test']:
            for cls in classes:
                os.makedirs(os.path.join(artifacts_processed_dir, subset, cls), exist_ok=True)

        #Copy Files to both Processed Directories
        for subset, data_list in datasets.items():
            logger.info(f"\nCopying {subset} images...")
            for img_path, label in tqdm(data_list, desc=f"Copying {subset}"):
                # Copy to data/processed
                dest_dir_data = os.path.join(processed_data_dir, subset, label)
                # Copy to artifacts
                dest_dir_artifacts = os.path.join(artifacts_processed_dir, subset, label)
                try:
                    shutil.copy(img_path, dest_dir_data)
                    shutil.copy(img_path, dest_dir_artifacts)
                except Exception as copy_e:
                    # Catch copy errors and raise a FileSystemError
                    raise FileSystemError(f"Failed to copy '{os.path.basename(img_path)}': {copy_e}", sys.exc_info()) from copy_e
        
        logger.info("\n--- Data Splitting Complete. Final Counts: ---")
        # Update class distribution and log counts
        for subset in ['train', 'val', 'test']:
            logger.info(f"--- {subset.upper()} ---")
            for cls in classes:
                count = len(os.listdir(os.path.join(processed_data_dir, subset, cls)))
                class_distribution[subset][cls] = count
                logger.info(f"  {cls}: {count} images")
        
        # Create and return DataSplittingArtifact
        artifact = DataSplittingArtifact(
            data_splitting_dir=data_splitting_dir,
            processed_data_dir=processed_data_dir,
            train_dir=os.path.join(processed_data_dir, 'train'),
            val_dir=os.path.join(processed_data_dir, 'val'),
            test_dir=os.path.join(processed_data_dir, 'test'),
            class_distribution=class_distribution
        )
        
        return artifact
    except TBDetectionError as e:
        exit_on_critical_error(e, "A project-specific error occurred during data splitting.")
    except Exception as e:
        exit_on_critical_error(e, "An unexpected error occurred during data splitting.")

"""if __name__ == "__main__":
    from datetime import datetime
    from entity.config_entity import TrainingPipelineConfig, DataSplittingConfig
    import config as project_root_config
    
    try:
        current_timestamp = datetime.now() 
        training_config = TrainingPipelineConfig(timestamp=current_timestamp)
        project_root_abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) 
        base_artifact_dir_full_path = os.path.join(project_root_abs_path, CONSTANT.ARTIFACT_DIR_NAME, training_config.timestamp_str)
        
        create_and_clear_directory(base_artifact_dir_full_path, "pipeline artifact root directory for individual run")
        logger.info(f"Artifacts for this individual run will be stored in: {base_artifact_dir_full_path}")
        logger.info(f"Please use this timestamp for subsequent individual script runs: {training_config.timestamp_str}")
        
        data_splitting_config = DataSplittingConfig(training_pipeline_config=training_config)
        split_data(config=data_splitting_config)
    except Exception as e:
        exit_on_critical_error(e, "Error during individual split_data.py execution.")"""