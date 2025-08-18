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

import config 
from utils.logger import logger
from utils.exception import DataError, ModelError, TBDetectionError
from utils.common import create_and_clear_directory, exit_on_critical_error
from entity.config_entity import ModelTrainerConfig
from entity.artifact_entity import ModelTrainerArtifact
from constant import training_pipeline as CONSTANT

def train_model(config: ModelTrainerConfig) -> ModelTrainerArtifact:
    model_trainer_dir = config.model_trainer_dir
    trained_model_dir = config.trained_model_dir
    trained_model_file_path = config.trained_model_file_path
    class_indices_file_path = config.class_indices_file_path

    final_metrics = {
        'training_accuracy': 0.0,
        'validation_accuracy': 0.0,
        'training_loss': 0.0,
        'validation_loss': 0.0
    }
    img_height = config.img_height
    img_width = config.img_width
    batch_size = config.batch_size
    dense_units = config.dense_units
    dropout_rate = config.dropout_rate
    lr_phase1 = config.lr_phase1
    lr_phase2 = config.lr_phase2
    epochs_phase1 = config.epochs_phase1
    epochs_phase2 = config.epochs_phase2
    training_history_plot_name = config.training_history_plot_name

    logger.info("--- Starting Model Training ---")
    
    create_and_clear_directory(model_trainer_dir, "model trainer directory")
    create_and_clear_directory(trained_model_dir, "trained model output directory")
    
    try:
        processed_data_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")
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
                os.path.join(processed_data_root, 'train'),
                target_size=(img_height, img_width),
                batch_size=batch_size,
                class_mode='binary', # For 2 classes ('Normal', 'Tuberculosis')
                color_mode='rgb', # ResNet expects 3 channels (will convert grayscale to 3 channels)
                shuffle=True # Shuffle training data
            )
            validation_generator = val_datagen.flow_from_directory(
                os.path.join(processed_data_root, 'val'),
                target_size=(img_height, img_width),
                batch_size=batch_size,
                class_mode='binary',
                color_mode='rgb',
                shuffle=False       #No shuffling 
            )
        

        except Exception as e:
            # Raise DataError if data generators cannot be created
            raise DataError(f"Failed to create data generators from {processed_data_root}. "
                            "Ensure split_data.py has run successfully.", sys.exc_info()) from e

        logger.info("Data generators for training created.")
        logger.info(f"Class indices: {train_generator.class_indices}")
        
        with open(class_indices_file_path, 'w') as f:
            json.dump(train_generator.class_indices, f)
        logger.info(f"Class indices saved to {class_indices_file_path}")
            
        #Model building
        def build_model(base_model_trainable=False, learning_rate=lr_phase1,dense_units_param=dense_units, dropout_rate_param=dropout_rate):
            try:
                base_model = ResNet50(weights = 'imagenet', include_top=False, input_shape=(img_height, img_width, 3))      #exclude the top layer for transfer learning
            except Exception as e:
                raise ModelError(f"Failed to load ResNet50 base model or ImageNet weights: {e}", sys.exc_info()) from e
            
            for layer in base_model.layers:
                layer.trainable = base_model_trainable  #weights of the base model layers will not be updated since false is passed
                
            x = base_model.output               #holds the f-map produced by base ResNet
            x = GlobalAveragePooling2D()(x)     #Avgpooling to reduce f-map to a single vector per image
            x = Dense(dense_units_param, activation='relu')(x) #Fully connected layer with 256 neurones
            x = Dropout(dropout_rate_param)(x)                 #Dropout 50% inputs to prevent overfitting
            
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
        model_checkpoint=ModelCheckpoint(
            trained_model_file_path,
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
            min_lr = lr_phase2 / 10,  # Minimum learning rate to prevent too low values
            verbose = 1
        )

        callbacks = [model_checkpoint, early_stopping, reduce_lr]

        #Model Training Phase 1
        print(f"--- Starting Phase 1 Training for {epochs_phase1} epochs ---")
        logger.info(f"\nPhase 1: Training custom head for {epochs_phase1} epochs (Base model frozen) ---")

        model, base_model = build_model(base_model_trainable=False, learning_rate=lr_phase1)
        
        print("Model Summary of Phase1:\n")
        model.summary(print_fn=logger.info) # print_fn directs summary to logger

        try:
            #history is a dictionary that contains training and validation loss and accuracy for each epoch
            history_phase1 = model.fit(
                train_generator,                                        #normalized train data passed
                steps_per_epoch=train_generator.samples//batch_size,    # Number of iterations per epoch
                epochs=epochs_phase1,           #Model goes thru the train data for these many times
                validation_data = validation_generator,
                validation_steps = validation_generator.samples//batch_size,
                callbacks=callbacks
            ) 
        except Exception as e:
            raise ModelError(f"Error during Phase 1 model training: {e}", sys.exc_info()) from e
        logger.info("Phase 1 training complete.")
        
        try:
            model = tf.keras.models.load_model(trained_model_file_path)     #Loads best model from phase1
        except Exception as e:
            raise ModelError(f"Failed to load best model from Phase 1 for fine-tuning: {e}", sys.exc_info()) from e

        #Fine Tuning Phase 2
        print(f"\n--- Starting Phase 2 Fine-Tuning for {epochs_phase2} epochs ---")
        logger.info(f"\n--- Phase 2: Fine-tuning ResNet50 (unfrozen layers) for {epochs_phase2} epochs ---")
        
        #Selectively unfreezing layers of base model
        for layer in base_model.layers:
            if 'conv5_block' in layer.name or 'conv4_block' in layer.name:
                layer.trainable = True      #unfreeze last layers as they learn high level features
            else:
                layer.trainable = False  #freeze early layers to retain learned features


        model.compile(
            optimizer = Adam(learning_rate=lr_phase2),
            loss = 'binary_crossentropy',
            metrics = ['accuracy', Precision(name='precision'), Recall(name='recall'), AUC(name='auc')]
        )

        model.summary(print_fn=logger.info) # Print model summary of fine tuned model

        try:
            history_phase2 = model.fit(
                train_generator,
                steps_per_epoch=train_generator.samples // batch_size,
                epochs=epochs_phase2,   
                validation_data=validation_generator,
                validation_steps=validation_generator.samples // batch_size,
                callbacks=callbacks
            )
        except Exception as e:
            raise ModelError(f"Error during Phase 2 model fine-tuning: {e}", sys.exc_info()) from e
        logger.info("Phase 2 fine-tuning complete.")

        logger.info(f"\nModel training complete. Best model saved to: {trained_model_file_path}")
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
            plot_save_path = os.path.join(os.path.dirname(trained_model_dir),
                                          CONSTANT.MODEL_EVALUATION_DIR_NAME,
                                          CONSTANT.EVALUATION_PLOTS_SUBDIR,
                                          plot_filename)
            os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)
            plt.savefig(plot_save_path)
            plt.show()
            logger.info(f"Training history plot saved to: {plot_save_path}")

        history_plot_path = os.path.join(os.path.dirname(trained_model_dir),
                                          CONSTANT.MODEL_EVALUATION_DIR_NAME,
                                          CONSTANT.EVALUATION_PLOTS_SUBDIR,
                                          'training_history.png')
        plot_training_history(history_phase1, history_phase2, 'training_history.png')
        
        # Update final metrics
        final_metrics['training_accuracy'] = history_phase2.history['accuracy'][-1]
        final_metrics['validation_accuracy'] = history_phase2.history['val_accuracy'][-1]
        final_metrics['training_loss'] = history_phase2.history['loss'][-1]
        final_metrics['validation_loss'] = history_phase2.history['val_loss'][-1]
        
        # Create model parameters dictionary
        model_params = {
            'epochs_phase1': epochs_phase1,
            'epochs_phase2': epochs_phase2,
            'learning_rate_phase1': lr_phase1,
            'learning_rate_phase2': lr_phase2,
            'dense_units': dense_units,
            'dropout_rate': dropout_rate,
            'batch_size': batch_size,
            'img_height': img_height,
            'img_width': img_width
        }
        
        # Create and return ModelTrainerArtifact
        artifact = ModelTrainerArtifact(
            model_trainer_dir=model_trainer_dir,
            trained_model_dir=trained_model_dir,
            trained_model_path=trained_model_file_path,
            class_indices_path=class_indices_file_path,
            training_accuracy=final_metrics['training_accuracy'],
            validation_accuracy=final_metrics['validation_accuracy'],
            training_loss=final_metrics['training_loss'],
            validation_loss=final_metrics['validation_loss'],
            model_params=model_params,
            training_history_plot=history_plot_path
        )
        
        logger.info("\n--- Model Training and Tuning Completed ---")
        return artifact
        
    except TBDetectionError as e:
        exit_on_critical_error(e, "A project-specific error occurred during model training.")
    except Exception as e:
        exit_on_critical_error(e, "An unexpected error occurred during model training.")
        

"""if __name__ == "__main__":
    from datetime import datetime
    from entity.config_entity import TrainingPipelineConfig, ModelTrainerConfig
    # Import project-level config for BASE_DIR
    import config as project_root_config
    
    fixed_timestamp_str = "YYYYMMDD_HHMMSS" 

    try:
        training_config = TrainingPipelineConfig(timestamp=datetime.strptime(fixed_timestamp_str, "%Y%m%d_%H%M%S")) 
        logger.info(f"Training using artifacts from: {os.path.join(project_root_config.BASE_DIR, training_config.artifact_dir)}")
        
        model_trainer_config = ModelTrainerConfig(training_pipeline_config=training_config)
        train_model(config=model_trainer_config)
    except Exception as e:
        exit_on_critical_error(e, "Error during individual train_model.py execution.")"""