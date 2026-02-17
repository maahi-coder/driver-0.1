import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from config import settings
except ImportError:
    # Fallback to defaults or local config if run directly
    pass

class EyeStateModel:
    def __init__(self, input_shape=(224, 224, 3)):
        self.input_shape = input_shape
        self.model = self._build_model()

    def _build_model(self):
        """
        Builds the MobileNetV2 model with transfer learning.
        """
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=self.input_shape)
        
        # Freeze the base model layers
        for layer in base_model.layers:
            layer.trainable = False
            
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(1, activation='sigmoid')(x)  # Binary classification: Open vs Closed
        
        model = Model(inputs=base_model.input, outputs=predictions)
        model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
        
        return model

    def train(self, data_dir, batch_size=32, epochs=10):
        """
        Trains the model on the provided dataset.
        Args:
            data_dir: Directory containing 'open' and 'closed' subdirectories.
        """
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=0.2
        )

        train_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='binary',
            subset='training'
        )

        validation_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=self.input_shape[:2],
            batch_size=batch_size,
            class_mode='binary',
            subset='validation'
        )

        if train_generator.samples == 0:
            print("Error: No training images found in data directory.")
            return

        self.model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // batch_size,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // batch_size,
            epochs=epochs
        )
        
        # Save the model
        if not os.path.exists(settings.MODELS_DIR):
            os.makedirs(settings.MODELS_DIR)
            
        self.model.save(settings.MODEL_PATH)
        print(f"Model saved to {settings.MODEL_PATH}")

if __name__ == '__main__':
    # Example usage:
    # python models/train_model.py
    
    # We need a dataset to run this. The user should provide one.
    # For now, we'll just check if the directory exists.
    data_path = os.path.join(settings.DATA_DIR, 'dataset_B_Eye_Images') # Standard dataset name often used
    
    if os.path.exists(data_path):
        print(f"Training on data from {data_path}...")
        model = EyeStateModel()
        model.train(data_path)
    else:
        print(f"Dataset not found at {data_path}. Please place 'open' and 'closed' eye images in subfolders there.")
        print("Creating a dummy model for testing purposes...")
        
        # Create a dummy model and save it so run.py doesn't crash on start
        model = EyeStateModel()
        if not os.path.exists(settings.MODELS_DIR):
             os.makedirs(settings.MODELS_DIR)
        model.model.save(settings.MODEL_PATH)
        print(f"Dummy model saved to {settings.MODEL_PATH} (WARNING: This model is untrained)")
