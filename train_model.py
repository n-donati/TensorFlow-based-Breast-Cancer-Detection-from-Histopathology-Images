import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detection.model.cancer_detector import CancerDetector, load_and_preprocess_data

# Set the path to your dataset
data_dir = 'path/to/dataset'

# Load and preprocess the data
print("Loading and preprocessing data...")
images, labels = load_and_preprocess_data(data_dir)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Create and train the model
print("Creating and training the model...")
detector = CancerDetector()
history = detector.train(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32)

# Save the trained model
model_save_path = os.path.join('detection', 'model', 'breast_cancer_model.h5')
print(f"Saving the model to {model_save_path}...")
detector.save_model(model_save_path)

print("Model training complete!")

# Optional: Convert to OpenVINO
print("Converting model to OpenVINO format...")
detector.convert_to_openvino(model_save_path)
print("Conversion complete!")

# Test the model
import cv2

test_image_path = 'path/to/test/image.jpg'
test_image = cv2.imread(test_image_path)
result = detector.predict(test_image)
print(f"Test prediction (TensorFlow): Cancer probability: {result}")

result_openvino = detector.predict_openvino(test_image)
print(f"Test prediction (OpenVINO): Cancer probability: {result_openvino}")