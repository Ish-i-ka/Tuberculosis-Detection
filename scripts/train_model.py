import os 
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator     #Data augmentation to improve model generalization
from tensorflow.keras.applications import ResNet50                      # Pre-trained CNN model for transfer learning with 50 layers
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau        #ModelCheckpoint to save the best model, ReduceLROnPlateau to reduce learning rate when validation loss plateaus
from tensorflow.keras.metrics import AUC, Precision, Recall
import matplotlib.pyplot as plt
import numpy as np
import json 


import sys
# Add project root directory to sys.path to import config.py
current_script_dir = os.path.dirname(os.path.abspath(__file__))     #/scripts
project_root_dir = os.path.join(current_script_dir, '..')           #/scripts/..
sys.path.append(project_root_dir)
import config
from utils.logger import logger # Import the global logger
from utils.exception import DataError, ModelError, ConfigurationError, TBDetectionError # Import custom exceptions
from utils.common import exit_on_critical_error

PROCESSED_DATA_PATH = config.PROCESSED_DATA_PATH
MODELS_DIR = config.MODELS_DIR
PLOTS_DIR = config.PLOTS_DIR         # For saving training plots

IMG_HEIGHT, IMG_WIDTH = config.IMG_HEIGHT, config.IMG_WIDTH
BATCH_SIZE = config.BATCH_SIZE
EPOCHS_PHASE1 = config.EPOCHS_PHASE1
EPOCHS_PHASE2 = config.EPOCHS_PHASE2
LEARNING_RATE_PHASE1 = config.LEARNING_RATE_PHASE1
LEARNING_RATE_PHASE2 = config.LEARNING_RATE_PHASE2
MODEL_FILENAME = config.MODEL_FILENAME      #tb_detection_resnet50_best.h5

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

try:
    try:
        #Train-Data Augmentation and Normalization
        train_datagen = ImageDataGenerator(
            rescale=1./255,         # Normalize pixel values to 0-1 by div with 255
            rotation_range=20, 
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'     #fills new pixels after transformations with nearest pixel value
        )

        #Val-Data Normalization
        val_datagen = ImageDataGenerator(rescale=1./255)  # Only rescale, no augmentation coz real world data shud be used for validation

        train_generator = train_datagen.flow_from_directory(
            os.path.join(PROCESSED_DATA_PATH, 'train'),
            target_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=BATCH_SIZE,
            class_mode='binary', # For 2 classes ('Normal', 'Tuberculosis')
            color_mode='rgb', # ResNet expects 3 channels (will convert grayscale to 3 channels)
            shuffle=True # Shuffle training data
        )
        validation_generator = val_datagen.flow_from_directory(
            os.path.join(PROCESSED_DATA_PATH, 'val'),
            target_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=BATCH_SIZE,
            class_mode='binary',
            color_mode='rgb',
            shuffle=False       #No shuffling 
        )
    

    except Exception as e:
        # Raise DataError if data generators cannot be created
        raise DataError(f"Failed to create data generators from {PROCESSED_DATA_PATH}. "
                        "Ensure split_data.py has run successfully.", sys.exc_info()) from e

    logger.info("Data generators for training created.")
    logger.info(f"Class indices: {train_generator.class_indices}")
    with open(os.path.join(MODELS_DIR, 'class_indices.txt'), 'w') as f:
        json.dump(train_generator.class_indices, f)
    logger.info(f"Class indices saved to {os.path.join(MODELS_DIR, 'class_indices.txt')}")
        
    #Model building
    def build_model(base_model_trainable=False, learning_rate=0.001):
        try:
            base_model = ResNet50(weights = 'imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))      #exclude the top layer for transfer learning
        except Exception as e:
            raise ModelError(f"Failed to load ResNet50 base model or ImageNet weights: {e}", sys.exc_info()) from e
        
        for layer in base_model.layers:
            layer.trainable = base_model_trainable  #weights of the base model layers will not be updated since false is passed
            
        x = base_model.output               #holds the f-map produced by base ResNet
        x = GlobalAveragePooling2D()(x)     #Avgpooling to reduce f-map to a single vector per image
        x = Dense(256, activation='relu')(x) #Fully connected layer with 256 neurones
        x = Dropout(0.5)(x)                 #Dropout 50% inputs to prevent overfitting
        
        predictions = Dense(1, activation='sigmoid')(x)  # Sigmoid for binary classification
        
        model = Model(inputs=base_model.input, outputs=predictions)  
        
        #Model compilation
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',  
            metrics=['accuracy', Precision(name='precision'), Recall(name='recall'), AUC(name='auc')]
        )
        
        return model, base_model 

    #Callbacks for Training
    checkpoint_path=os.path.join(MODELS_DIR, MODEL_FILENAME)
    model_checkpoint=ModelCheckpoint(
        checkpoint_path,
        monitor='val_loss',
        save_best_only=True,        #saves the best model only
        mode='min',                 #when val_loss is minimum
        verbose=1
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience = 5,
        restore_best_weights=True,  # Restore weights from the best epoch
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor = 0.2,                           # Reduce learning rate by a factor of 0.2
        patience = 3,
        min_lr = config.LEARNING_RATE_PHASE2 / 10,  # Minimum learning rate to prevent too low values
        verbose = 1
    )

    callbacks = [model_checkpoint, early_stopping, reduce_lr]

    #Model Training Phase 1
    print(f"--- Starting Phase 1 Training for {EPOCHS_PHASE1} epochs ---")
    logger.info(f"\nPhase 1: Training custom head for {config.EPOCHS_PHASE1} epochs (Base model frozen) ---")

    model, base_model = build_model(base_model_trainable=False, learning_rate=LEARNING_RATE_PHASE1)
    model.summary(print_fn=logger.info) # Print model summary to check architecture

    try:
        #history is a dictionary that contains training and validation loss and accuracy for each epoch
        history_phase1 = model.fit(
            train_generator,                                        #normalized train data passed
            steps_per_epoch=train_generator.samples//BATCH_SIZE,    # Number of iterations per epoch
            epochs=EPOCHS_PHASE1,           #Model goes thru the train data for these many times
            validation_data = validation_generator,
            validation_steps = validation_generator.samples//BATCH_SIZE,
            callbacks=callbacks
        ) 
    except Exception as e:
        raise ModelError(f"Error during Phase 1 model training: {e}", sys.exc_info()) from e
    logger.info("Phase 1 training complete.")
    
    try:
        model = tf.keras.models.load_model(checkpoint_path)     #Loads best model from phase1
    except Exception as e:
        raise ModelError(f"Failed to load best model from Phase 1 for fine-tuning: {e}", sys.exc_info()) from e

    #Fine Tuning Phase 2
    print(f"\n--- Starting Phase 2 Fine-Tuning for {EPOCHS_PHASE2} epochs ---")
    logger.info(f"\n--- Phase 2: Fine-tuning ResNet50 (unfrozen layers) for {config.EPOCHS_PHASE2} epochs ---")
    
    #Selectively unfreezing layers of base model
    for layer in base_model.layers:
        if 'conv5_block' in layer.name or 'conv4_block' in layer.name:
            layer.trainable = True      #unfreeze last layers as they learn high level features
        else:
            layer.trainable = False  #freeze early layers to retain learned features


    model.compile(
        optimizer = Adam(learning_rate=LEARNING_RATE_PHASE2),
        loss = 'binary_crossentropy',
        metrics = ['accuracy', Precision(name='precision'), Recall(name='recall'), AUC(name='auc')]
    )

    model.summary(print_fn=logger.info) # Print model summary of fine tuned model

    try:
        history_phase2 = model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // BATCH_SIZE,
            epochs=EPOCHS_PHASE2,   
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // BATCH_SIZE,
            callbacks=callbacks
        )
    except Exception as e:
        raise ModelError(f"Error during Phase 2 model fine-tuning: {e}", sys.exc_info()) from e
    logger.info("Phase 2 fine-tuning complete.")

    logger.info(f"\nModel training complete. Best model saved to: {checkpoint_path}")
    print(f"\n--- Training Completed ---")

    #Func to plot training history
    def plot_training_history(hist1,hist2,plot_filename):
        hist = {}
        #hist1.history is a dict where for each metric there is a list of val wid each val belonging to diff epoch
        for key in hist1.history.keys():
            hist[key] = hist1.history[key] + hist2.history[key]     #adding the metrics of models of both phases
            
        epochs_range = range(len(hist['accuracy']))
        
        plt.figure(figsize=(12,8))
        
        #Accuracy plotting
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, hist['accuracy'], label='Training Accuracy')
        plt.plot(epochs_range, hist['val_accuracy'], label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')
        plt.grid(True)
        
        #Loss plotting
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, hist['loss'], label='Training Loss')
        plt.plot(epochs_range, hist['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')
        plt.grid(True)

        plt.tight_layout()
        plot_save_path = os.path.join(PLOTS_DIR, plot_filename)
        plt.savefig(plot_save_path)
        plt.show()
        print(f"Training history plot saved to: {plot_save_path}")

    plot_training_history(history_phase1, history_phase2, 'training_history.png')
    
    logger.info("\n--- Model Training and Tuning Completed ---")
    
except TBDetectionError as e:
    exit_on_critical_error(e, "A project-specific error occurred during model training.")
except Exception as e:
    exit_on_critical_error(e, "An unexpected error occurred during model training.")