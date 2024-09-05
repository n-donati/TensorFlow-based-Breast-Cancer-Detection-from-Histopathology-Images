# Breast Cancer Detection Project

## Table of Contents
1. [Project Overview](#project-overview)
2. [How It Works](#how-it-works)
3. [Project Structure](#project-structure)
4. [Setup and Installation](#setup-and-installation)
5. [Preparing the Dataset](#preparing-the-dataset)
6. [Training the Model](#training-the-model)
7. [Running the Web Application](#running-the-web-application)
8. [Creating Your Own Model](#creating-your-own-model)

## Project Overview

This project is a web application that uses machine learning to detect breast cancer from histopathology images. It consists of a Django web interface and a TensorFlow/Keras backend, with OpenVINO optimization for inference.

## How It Works

1. **Data Preparation**: The system uses a dataset of breast histopathology images, categorized into 'cancerous' and 'non-cancerous'.

2. **Model Architecture**: We use a Convolutional Neural Network (CNN) implemented in TensorFlow/Keras. The model takes 50x50 pixel RGB images as input and outputs a probability of the image containing cancerous cells.

3. **Training**: The model is trained on the prepared dataset, using binary cross-entropy loss and the Adam optimizer.

4. **Optimization**: After training, the model is converted to OpenVINO format for optimized inference.

5. **Web Interface**: A Django web application allows users to upload images for prediction.

6. **Prediction**: When an image is uploaded, it's preprocessed (resized to 50x50 pixels and normalized) and then fed into the model for prediction.

## Project Structure

```
breast_cancer_detection/
├── breast_cancer_detection/  # Main Django project directory
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── detection/  # Django app directory
│   ├── model/
│   │   ├── cancer_detector.py  # Contains the CancerDetector class
│   │   └── breast_cancer_model.h5  # Saved model file (after training)
│   ├── templates/
│   │   └── detection/
│   │       ├── home.html
│   │       └── result.html
│   ├── views.py
│   └── urls.py
├── dataset/  # Directory for training data (not in repo)
│   ├── cancerous/
│   └── non_cancerous/
├── manage.py
├── requirements.txt
└── train_model.py  # Script to train the model
```

## Setup and Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/breast_cancer_detection.git
   cd breast_cancer_detection
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Preparing the Dataset

1. Create a `dataset` directory in the project root.
2. Inside `dataset`, create two subdirectories: `cancerous` and `non_cancerous`.
3. Place your histopathology images in the appropriate directories based on their classification.

Your dataset structure should look like this:
```
dataset/
├── cancerous/
│   ├── image1.png
│   ├── image2.png
│   └── ...
└── non_cancerous/
    ├── image1.png
    ├── image2.png
    └── ...
```

## Training the Model

1. Open `train_model.py` and update the `data_dir` variable to point to your dataset:
   ```python
   data_dir = 'path/to/your/dataset'
   ```

2. Run the training script:
   ```
   python train_model.py
   ```

3. The script will train the model and save it as `detection/model/breast_cancer_model.h5`.

## Running the Web Application

1. Make sure you're in the project root directory.

2. Apply database migrations:
   ```
   python manage.py migrate
   ```

3. Start the Django development server:
   ```
   python manage.py runserver
   ```

4. Open a web browser and go to `http://127.0.0.1:8000/`.

5. Use the web interface to upload an image and get a prediction.

## Creating Your Own Model

To create your own model, you can modify the `_build_model` method in the `CancerDetector` class (`detection/model/cancer_detector.py`):

1. Open `detection/model/cancer_detector.py`.

2. Locate the `_build_model` method in the `CancerDetector` class.

3. Modify the model architecture as desired. For example:

   ```python
   def _build_model(self):
       model = models.Sequential([
           layers.Conv2D(64, (3, 3), activation='relu', input_shape=(50, 50, 3)),
           layers.MaxPooling2D((2, 2)),
           layers.Conv2D(128, (3, 3), activation='relu'),
           layers.MaxPooling2D((2, 2)),
           layers.Conv2D(128, (3, 3), activation='relu'),
           layers.Flatten(),
           layers.Dense(128, activation='relu'),
           layers.Dropout(0.5),
           layers.Dense(1, activation='sigmoid')
       ])
       model.compile(optimizer='adam',
                     loss='binary_crossentropy',
                     metrics=['accuracy'])
       return model
   ```

4. Save the changes and re-run the training script to train your new model.