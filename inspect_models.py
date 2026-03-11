import os
import tensorflow as tf
from tensorflow import keras

# Enable unsafe deserialization for Lambda layers
keras.config.enable_unsafe_deserialization()

models = [
    'c:/THESIS/server/v4/best_asl_grouped_model_final.keras',
    'c:/THESIS/server/v4/best_asl_grouped_model.keras'
]

for model_path in models:
    if os.path.exists(model_path):
        print(f"\nInspecting {model_path}...")
        try:
            model = keras.models.load_model(model_path, safe_mode=False)
            print(f"Loaded successfully.")
            print(f"Input shape: {model.input_shape}")
            print(f"Output shape: {model.output_shape}")
        except Exception as e:
            print(f"Error loading {model_path}: {e}")
    else:
        print(f"File not found: {model_path}")
