from django.shortcuts import render
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from .model.cancer_detector import CancerDetector
import cv2
import numpy as np
import os
import tempfile

# Initialize the model
model_path = os.path.join(os.path.dirname(__file__), 'model', 'breast_cancer_model.h5')
detector = CancerDetector(model_path)

try:
    detector.convert_to_openvino(model_path)
except Exception as e:
    print(f"Error converting model to OpenVINO: {e}")
    print("Falling back to TensorFlow model")

def home(request):
    if request.method == 'POST' and request.FILES['image']:
        image_file = request.FILES['image']
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            for chunk in image_file.chunks():
                temp_file.write(chunk)
        
        try:
            # Read the image using OpenCV
            image = cv2.imread(temp_file.name)
            
            try:
                result = detector.predict_openvino(image)
            except Exception as e:
                print(f"Error using OpenVINO model: {e}")
                print("Falling back to TensorFlow model")
                result = detector.predict(image)
            
            return render(request, 'detection/result.html', {'result': result})
        
        finally:
            # Ensure the temporary file is deleted
            os.unlink(temp_file.name)
    
    return render(request, 'detection/home.html')