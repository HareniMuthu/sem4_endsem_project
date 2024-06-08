import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.models import Model
from tqdm import tqdm
import time

# Enable GPU usage
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Load ResNet101 model
base_model = ResNet101(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model = Model(inputs=base_model.input, outputs=base_model.output)

# Directory containing the dataset
data_dir = r'D:\B-TECH\SEM_4\h\ml\fnal\dataset'
output_csv = r'C:\h_dataset\features.csv'

# Create a CSV file and write the header
# Update the feature dimension based on ResNet101 output shape
num_features = base_model.output_shape[1] * base_model.output_shape[2] * base_model.output_shape[3]
with open(output_csv, 'w') as f:
    header = ','.join([f'feature_{i}' for i in range(num_features)]) + ',label\n'
    f.write(header)

# Iterate through each folder (label) in the dataset directory
folders = [os.path.join(data_dir, folder) for folder in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, folder))]

total_images = sum([len(files) for r, d, files in os.walk(data_dir)])
processed_images = 0

start_time = time.time()

with tqdm(total=total_images, desc="Processing images", unit="image") as pbar:
    for folder in folders:
        label = os.path.basename(folder)
        for img_name in os.listdir(folder):
            img_path = os.path.join(folder, img_name)
            try:
                # Load and preprocess the image
                img = image.load_img(img_path, target_size=(224, 224))  # ResNet101 expects 224x224 images
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = preprocess_input(img_array)

                # Extract features using ResNet101
                features = model.predict(img_array)
                features_flat = features.flatten()

                # Append features and label to the CSV file
                with open(output_csv, 'a') as f:
                    feature_str = ','.join(map(str, features_flat))
                    f.write(f"{feature_str},{label}\n")

            except Exception as e:
                print(f"Error processing image {img_path}: {e}")

            processed_images += 1
            elapsed_time = time.time() - start_time
            estimated_total_time = (elapsed_time / processed_images) * total_images
            remaining_time = estimated_total_time - elapsed_time
            pbar.set_postfix(remaining=f"{remaining_time:.2f}s")
            pbar.update(1)

print(f"Feature extraction completed. Saved to {output_csv}")
