# tb_detection_project/entity/artifact_entity.py

from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class DataSplittingArtifact:
    """
    Artifact class for data splitting stage outputs.
    
    Attributes:
        data_splitting_dir: Base directory for data splitting artifacts
        processed_data_dir: Directory containing processed and split data
        train_dir: Directory containing training data
        val_dir: Directory containing validation data
        test_dir: Directory containing test data
        class_distribution: Dictionary containing count of images per class in each split
    """
    data_splitting_dir: str
    processed_data_dir: str
    train_dir: str
    val_dir: str
    test_dir: str
    class_distribution: Dict[str, Dict[str, int]]  # {'train': {'Normal': N, 'TB': M}, ...}

@dataclass
class ModelTrainerArtifact:
    """
    Artifact class for model training stage outputs.
    
    Attributes:
        model_trainer_dir: Base directory for model training artifacts
        trained_model_dir: Directory containing the trained model files
        trained_model_path: Path to the saved model file
        class_indices_path: Path to the class indices mapping file
        training_accuracy: Final training accuracy
        validation_accuracy: Final validation accuracy
        training_loss: Final training loss
        validation_loss: Final validation loss
        model_params: Dictionary containing model parameters (epochs, learning rates, etc.)
        training_history_plot: Path to the training history plot
    """
    model_trainer_dir: str
    trained_model_dir: str
    trained_model_path: str
    class_indices_path: str
    training_accuracy: float
    validation_accuracy: float
    training_loss: float
    validation_loss: float
    model_params: Dict[str, any]
    training_history_plot: str

@dataclass
class EvaluationArtifact:
    """
    Artifact class for model evaluation stage outputs.
    
    Attributes:
        evaluation_dir: Base directory for evaluation artifacts
        plots_dir: Directory containing evaluation plots
        grad_cam_dir: Directory containing Grad-CAM visualizations
        test_accuracy: Model accuracy on test set
        confusion_matrix_path: Path to confusion matrix plot
        roc_curve_path: Path to ROC curve plot
        auc_score: Area Under the ROC Curve score
        classification_report: Dictionary containing precision, recall, f1-score
        grad_cam_examples: Dictionary mapping image paths to their Grad-CAM visualization paths
    """
    evaluation_dir: str
    plots_dir: str
    grad_cam_dir: str
    test_accuracy: float
    confusion_matrix_path: str
    roc_curve_path: str
    auc_score: float
    classification_report: Dict[str, Dict[str, float]]
    grad_cam_examples: Dict[str, str]  # {original_img_path: grad_cam_path}
