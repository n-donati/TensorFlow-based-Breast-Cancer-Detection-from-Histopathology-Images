import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
import os
import cv2
from openvino.runtime import Core
import subprocess

class CancerDetector:
    def __init__(self, model_path=None):
        if model_path and os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
        else:
            self.model = self._build_model()
        
        self.ie = Core()
        self.compiled_model = None

    def _build_model(self):
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(50, 50, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model

    def train(self, train_data, train_labels, validation_data=None, epochs=50, batch_size=32):
        if validation_data is None:
            validation_split = 0.2
        else:
            validation_split = None

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )

        history = self.model.fit(
            train_data, train_labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            validation_data=validation_data,
            callbacks=[early_stopping]
        )
        return history

    def save_model(self, path):
        self.model.save(path)

    def load_model(self, path):
        self.model = tf.keras.models.load_model(path)

    def predict(self, image):
        img = cv2.resize(image, (50, 50))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        return self.model.predict(img)[0][0]

    def convert_to_openvino(self, model_path):
        # Use mo command-line tool to convert the model
        output_dir = os.path.dirname(model_path)
        output_name = "breast_cancer_model_openvino"
        command = f"mo --saved_model_dir {model_path} --output_dir {output_dir} --model_name {output_name}"
        
        try:
            subprocess.run(command, shell=True, check=True)
            print(f"Model converted and saved as {output_name}.xml")
            
            # Load the converted model
            xml_path = os.path.join(output_dir, f"{output_name}.xml")
            self.compiled_model = self.ie.compile_model(xml_path, "CPU")
        except subprocess.CalledProcessError as e:
            print(f"Error converting model: {e}")
        except Exception as e:
            print(f"Error loading converted model: {e}")

    def predict_openvino(self, image):
        if self.compiled_model is None:
            raise ValueError("OpenVINO model not loaded. Call convert_to_openvino first.")
        
        img = cv2.resize(image, (50, 50))
        img = img / 255.0
        img = np.expand_dims(img, axis=0).astype(np.float32)
        results = self.compiled_model(img)
        output_key = list(results.keys())[0]
        return results[output_key][0][0]

def load_and_preprocess_data(data_dir):
    images = []
    labels = []
    for label in ['cancerous', 'non_cancerous']:
        path = os.path.join(data_dir, label)
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (50, 50))
            img = img / 255.0  # Normalize the image
            images.append(img)
            labels.append(1 if label == 'cancerous' else 0)
    return np.array(images), np.array(labels)