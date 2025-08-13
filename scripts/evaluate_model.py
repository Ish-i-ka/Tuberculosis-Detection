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
from utils.exception import DataError, ModelError, FileSystemError, TBDetectionError,ConfigurationError 
from utils.common import exit_on_critical_error 

# Import for Grad-CAM 
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore

PROCESSED_DATA_PATH = config.PROCESSED_DATA_PATH
MODELS_DIR = config.MODELS_DIR
RESULTS_DIR = config.RESULTS_DIR
PLOTS_DIR = config.path.join(RESULTS_DIR, 'plots') 
GRAD_CAM_DIR = config.path.join(RESULTS_DIR, 'grad_cam_examples')

IMG_HEIGHT, IMG_WIDTH = config.IMG_HEIGHT, config.IMG_WIDTH
BATCH_SIZE = config.BATCH_SIZE
MODEL_FILENAME = config.MODEL_FILENAME

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(GRAD_CAM_DIR, exist_ok=True)

#Func for grad cam generation
def generate_and_save_gradcam(model, img_path, original_img_pil, prediction_class_idx, class_name, save_path):
    """Generates and overlays a Grad-CAM heatmap
        on the original image and saves it.
        - original_img_pil: the img loaded as Pillow(PIL) image object 
        """
    
    #Image preprocessing for model and Grad-cam
    img_resized = original_img_pil.resize(IMG_HEIGHT, IMG_WIDTH)        #resizing img as required by cnn model
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
    
def run_evaluation():
    logger.info("--- Starting Model Evaluation and Interpretation ---")
   
    try:
        #load trained model
        model_path = os.path.join(MODELS_DIR, MODEL_FILENAME)
        if not os.path.exists(model_path):
            raise ModelError(f"Trained model not found at {model_path}", sys.exc_info())
        
        model = tf.keras.models.load_model(model_path)
        logger.info("Model loaded succesfully for evaluation")
        
        class_indices_path = os.path.join(MODELS_DIR, "class_indices.txt")
        if not os.path.exists(class_indices_path):
                raise ConfigurationError(f"Class indices file not found at {class_indices_path}.", sys.exc_info())
        
        with open(class_indices_path, 'r') as file:
            class_indices = json.load(file)      #class indices becomes a dict
        class_names = sorted(class_indices.keys(), key = lambda x:class_indices[x])
        logger.info(f"Loaded class indices: {class_indices}. Ordered class names: {class_names}")
        
        #Prepare Test data
        test_datagen = ImageDataGenerator(rescale = 1./255)      #Normalizing the pxl values of test data
        test_generator = test_datagen.flow_from_directory(
            os.path.join(PROCESSED_DATA_PATH,'test'),
            target_size = (IMG_HEIGHT, IMG_WIDTH),
            batch_size = BATCH_SIZE,
            class_mode = 'binary',
            color_mode = 'rgb',
            shuffle = False
        )
        if test_generator.samples == 0:
            raise DataError(f"No test images found in {os.path.join(PROCESSED_DATA_PATH,'test')}.", sys.exc_info())
        
        y_true = test_generator.classes      #stores the true class of each test data yielded by test generator
        y_pred_proba = model.predict(test_generator, steps=test_generator.samples // BATCH_SIZE + (test_generator.samples % BATCH_SIZE > 0))
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
        cm_save_path = os.path.join(PLOTS_DIR, 'confusion_matrix.png')
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
        roc_save_path = os.path.join(PLOTS_DIR, 'roc_curve.png')
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
            generate_and_save_gradcam(model, img_path, original_img_pil, normal_idx_val, 'Normal', os.path.join(GRAD_CAM_DIR, f'normal_gradcam_{i+1}.png'))
            
            logger.info(f"  Generated Grad-CAM for Normal example {i+1}")
        
        # Generate Grad-CAM for TB samples
        for i,img_path in enumerate(sample_tb_paths):
            original_img_pil = Image.open(img_path).convert('RGB')
            generate_and_save_gradcam(model, img_path, original_img_pil, tb_idx_val, 'Tuberculosis', os.path.join(GRAD_CAM_DIR, f'tb_gradcam_{i+1}.png'))
            
            logger.info(f"  Generated Grad-CAM for Tuberculosis example {i+1}")
            
        logger.info("\nModel Evaluation and Interpretation Complete")
         
    except TBDetectionError as e:
        exit_on_critical_error(e, "A project-specific error occurred during model evaluation.")
    except Exception as e:
        exit_on_critical_error(e, "An unexpected error occurred during model evaluation.")
        
if __name__ == "__main__":
    run_evaluation() 