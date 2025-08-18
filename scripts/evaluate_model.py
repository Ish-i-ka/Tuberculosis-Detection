import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, ConfusionMatrixDisplay, RocCurveDisplay
from PIL import Image
from tensorflow.keras.models import load_model
import cv2 # For image processing (resizing, overlaying)
import matplotlib.cm as cm # For colormaps
import json # To load class_indices.txt
import sys 
import random

import config 
from utils.logger import logger
from utils.exception import DataError, ModelError, FileSystemError, ConfigurationError, TBDetectionError
from utils.common import create_and_clear_directory, exit_on_critical_error
from entity.config_entity import ModelEvaluationConfig
from entity.artifact_entity import EvaluationArtifact
from constant import training_pipeline as CONSTANT 

# Import for Grad-CAM 
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore


#Func for grad cam generation
def generate_and_save_gradcam(model, img_path, original_img_pil, prediction_class_idx, class_name, save_path, img_height, img_width):
    """Generates and overlays a Grad-CAM heatmap
        on the original image and saves it.
        - original_img_pil: the img loaded as Pillow(PIL) image object 
        """
    
    #Image preprocessing for model and Grad-cam
    img_resized = original_img_pil.resize((img_height, img_width))        #resizing img as required by cnn model
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)  #resized img converted to arr
    img_array = np.expand_dims(img_array, axis=0)                       #extra dim added so that input becomes a batch of one img to the model
    
    img_array /= 255.0      #normalizing pixel values
    last_conv_layer_name = 'conv5_block3_out'   #this is chosen as last layer as it contains high spatial info. and high semantic features
    
    try:
        gradcam = Gradcam(model, model_modifier=ReplaceToLinear(), clone=True)
        #this is an obj of Gradcam where the original model is cloned and in that clone temporarily the act. fn. of o/p layer is converted to linear
        
        score = CategoricalScore([prediction_class_idx])    #stores the categorical score of the pred class as the pred class index is passe(0: Normal, 1:Tb)
        cam = gradcam(score, img_array, penultimate_layer=last_conv_layer_name)     #generate Class Activation Map(CAM), penultimate layer is the layer from which CAM is generated
        heatmap = cam[0]
        
        heatmap = np.uint8(255 * heatmap/ np.max(heatmap))    #de-normalizing the pixels to 0-255 and convert it to unsigned int-8bit
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  #Jet colormap applied on heatmap to make it visually interpretable 

        original_img_cv2 = cv2.cvtColor(np.array(original_img_pil), cv2.COLOR_RGB2BGR)
        #convert original pil img to array nd then rgb to bgr channel order for usage by open cv
        
        heatmap = cv2.resize(heatmap, (original_img_cv2.shape[1], original_img_cv2.shape[0]))
        #resize heatmap according to img dim.
        
        overlay = cv2.addWeighted(original_img_cv2, 0.6, heatmap, 0.4, 0)       #(base_image,alpha val for opaqueness of image,heatmap, alpha for opaqueness of heatmap, gamma val added to each pxl val)
        #overlaying heatmap on original image
        
        # Save the combined image using Matplotlib
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        plt.imshow(original_img_pil)
        plt.title(f'Original Image\nPredicted: {class_name}')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)) # Convert back to RGB for matplotlib
        plt.title(f'Grad-CAM Overlay ({class_name})')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
                
    except Exception as e:
        logger.error(f"Failed to generate Grad-CAM for {save_path}: {e}", exc_info=True)
    
def run_evaluation(config: ModelEvaluationConfig) -> EvaluationArtifact:
    logger.info("--- Starting Model Evaluation and Interpretation ---")
    # Initialize dictionary to store grad-cam examples
    grad_cam_paths = {}
    
    model_evaluation_dir = config.model_evaluation_dir
    evaluation_plots_dir = config.evaluation_plots_dir
    grad_cam_examples_dir = config.grad_cam_examples_dir
    img_height = config.img_height
    img_width = config.img_width
    batch_size = config.batch_size
    trained_model_path = config.trained_model_path
    class_indices_path = config.class_indices_path
    processed_data_path = config.processed_data_path
    
    try:
        create_and_clear_directory(model_evaluation_dir, "model evaluation directory")
        create_and_clear_directory(evaluation_plots_dir, "evaluation plots directory")
        create_and_clear_directory(grad_cam_examples_dir, "Grad-CAM examples directory")
        #load trained model
        if not os.path.exists(trained_model_path):
            raise ModelError(f"Trained model not found at {trained_model_path}", sys.exc_info())
        
        model = tf.keras.models.load_model(trained_model_path)
        logger.info("Model loaded succesfully for evaluation")
        

        if not os.path.exists(class_indices_path):
                raise ConfigurationError(f"Class indices file not found at {class_indices_path}.", sys.exc_info())
        
        with open(class_indices_path, 'r') as file:
            class_indices = json.load(file)      #class indices becomes a dict
        class_names = sorted(class_indices.keys(), key = lambda x:class_indices[x])
        logger.info(f"Loaded class indices: {class_indices}. Ordered class names: {class_names}")
        
        #Prepare Test data
        test_datagen = ImageDataGenerator(rescale = 1./255)      #Normalizing the pxl values of test data
        processed_data_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")
        test_generator = test_datagen.flow_from_directory(
            os.path.join(processed_data_root,'test'),
            target_size = (img_height, img_width),
            batch_size = batch_size,
            class_mode = 'binary',
            color_mode = 'rgb',
            shuffle = False
        )
        if test_generator.samples == 0:
            raise DataError(f"No test images found in {os.path.join(processed_data_path,'test')}.", sys.exc_info())
        
        y_true = test_generator.classes      #stores the true class of each test data yielded by test generator
        y_pred_proba = model.predict(test_generator, steps=test_generator.samples // batch_size + (test_generator.samples % batch_size > 0))
        #model predicts the prob of TB class for each test img and "steps" ensures that even if the last batch is partially filled then also the prediction is done for that batch
        
        y_pred = (y_pred_proba>0.5).astype(int)      #y_pred = [[0], [1], [0], [1], ..., [0]]
        #checks into pred proba array, if model is more than 50% confident then False else True, then false convert to 0 and true to 1
        logger.info("Predictions generated on test set.")
        
        logger.info("\n--- Model Evaluation Results ---")
        
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        logger.info(f"\nClassification Report:\n{json.dumps(report, indent=4)}")
        
        cm = confusion_matrix(y_true,y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        
        fig,ax = plt.subplots(figsize=(8,8))
        disp.plot(cmap=plt.cm.Blues, ax=ax)
        plt.title("Confusion Matrix")
        cm_save_path = os.path.join(evaluation_plots_dir, 'confusion_matrix.png')
        plt.savefig(cm_save_path)
        plt.close(fig)
        logger.info(f"Confusion Matrix saved to: {cm_save_path}")
        
        
        #ROC-AUC curve
        fpr, tpr, thresholds = roc_curve(y_true,y_pred_proba)
        roc_auc = auc(fpr, tpr)      #calculate area under ROC curve
        disp = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='ResNet50 Classifier')
        fig, ax = plt.subplots(figsize=(8,8))
        disp.plot(ax=ax)
        plt.title("ROC Curve")
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Guess') # Plot dummy model curve line
        roc_save_path = os.path.join(evaluation_plots_dir, 'roc_curve.png')
        plt.savefig(roc_save_path)
        plt.close(fig)
        logger.info(f"ROC Curve saved to: {roc_save_path}")
        logger.info(f"AUC Score: {roc_auc:.4f}")
        
        #Grad CAM for Model Interpretation
        logger.info("\n--- Generating Grad-CAM Explanations ---")
        
        correct_pred_indices = np.where(y_true==y_pred.flatten())[0]
        #stores a lst of indices where predictions are correct
        
        test_filepaths = test_generator.filepaths        #get all test file paths
        normal_indices = [i for i in correct_pred_indices if y_true[i] == class_indices['Normal']]      #stores indices of normal cls that r pred correctly
        tb_indices = [i for i in correct_pred_indices if y_true[i] == class_indices['Tuberculosis']]    #stores indices of tb cls that r pred correctly
            
        sample_normal_paths = random.sample([test_filepaths[i] for i in normal_indices], min(3, len(normal_indices)))       #lst of sample img paths that are correctly pred as normal
        sample_tb_paths = random.sample([test_filepaths[i] for i in tb_indices], min(3, len(tb_indices)))

        logger.info(f"Generating Grad-CAM for {len(sample_normal_paths)} Normal samples and {len(sample_tb_paths)} TB samples...")

        normal_idx_val = class_indices['Normal']        #takes the idx val of normal
        tb_idx_val = class_indices['Tuberculosis']
        
        # Generate Grad-CAM for Normal samples
        for i,img_path in enumerate(sample_normal_paths):
            original_img_pil = Image.open(img_path).convert('RGB')
            generate_and_save_gradcam(model, img_path, original_img_pil, normal_idx_val, 'Normal', os.path.join(grad_cam_examples_dir, f'normal_gradcam_{i+1}.png'), img_height, img_width)
            
            logger.info(f"  Generated Grad-CAM for Normal example {i+1}")
        
        # Generate Grad-CAM for TB samples
        for i,img_path in enumerate(sample_tb_paths):
            original_img_pil = Image.open(img_path).convert('RGB')
            generate_and_save_gradcam(model, img_path, original_img_pil, tb_idx_val, 'Tuberculosis', os.path.join(grad_cam_examples_dir, f'tb_gradcam_{i+1}.png'), img_height, img_width)
            
            grad_cam_paths[img_path] = os.path.join(grad_cam_examples_dir, f'tb_gradcam_{i+1}.png')
            logger.info(f"  Generated Grad-CAM for Tuberculosis example {i+1}")
            
        # Create evaluation artifact
        artifact = EvaluationArtifact(
            evaluation_dir=model_evaluation_dir,
            plots_dir=evaluation_plots_dir,
            grad_cam_dir=grad_cam_examples_dir,
            test_accuracy=report['accuracy'],
            confusion_matrix_path=cm_save_path,
            roc_curve_path=roc_save_path,
            auc_score=roc_auc,
            classification_report=report,
            grad_cam_examples=grad_cam_paths
        )
            
        logger.info("\nModel Evaluation and Interpretation Complete")
        return artifact
         
    except TBDetectionError as e:
        exit_on_critical_error(e, "A project-specific error occurred during model evaluation.")
    except Exception as e:
        exit_on_critical_error(e, "An unexpected error occurred during model evaluation.")
        
# ... (all imports and function definitions for evaluate_model.py go here) ...

if __name__ == "__main__":
    # --- For Individual Execution / Testing ---
    # To maintain consistency, the timestamp here MUST match the one generated by split_data.py
    # and used by train_model.py.
    # Manually update 'fixed_timestamp_str' with the output from split_data.py's log.

    from datetime import datetime
    # Import necessary config entities for local setup
    from entity.config_entity import TrainingPipelineConfig, ModelEvaluationConfig
    # Import project-level config for BASE_DIR
    import config as project_root_config
    
    fixed_timestamp_str = "YYYYMMDD_HHMMSS" # <-- !!! YOU MUST CHANGE THIS !!!

    try:
        # Initialize TrainingPipelineConfig with the fixed timestamp
        training_config = TrainingPipelineConfig(timestamp=datetime.strptime(fixed_timestamp_str, "%Y%m%d_%H%M%S")) 

        # Ensure the logger is aware of the current run's artifact directory
        logger.info(f"Evaluating using artifacts from: {os.path.join(project_root_config.BASE_DIR, training_config.artifact_dir)}")
        
        model_eval_config = ModelEvaluationConfig(training_pipeline_config=training_config)
        run_evaluation(config=model_eval_config)
    except Exception as e:
        exit_on_critical_error(e, "Error during individual evaluate_model.py execution.")