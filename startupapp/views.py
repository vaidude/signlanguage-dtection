from django.shortcuts import render

# Create your views here.
def index(request):
    return render(request, 'index.html')

from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import tensorflow as tf
import cv2
import numpy as np
import base64

# Load the model once when the server starts
MODEL_PATH = 'C:\\Users\\vaiva\\OneDrive\\Desktop\\office files\\signlang\\sign_language_mnist_model.h5'
model = tf.keras.models.load_model(MODEL_PATH)

# Mapping from label indices to letters (A-Y, no J=9 or Z=25)
label_to_letter = {i: chr(65 + i + (1 if i >= 9 else 0)) for i in range(25)}

@csrf_exempt
def sign_language_view(request):
    """Handle both page rendering and prediction."""
    if request.method == 'POST':
        try:
            # Get the image data from POST request (base64 encoded)
            image_data = request.POST.get('image')
            image_data = image_data.split(',')[1]  # Remove 'data:image/jpeg;base64,' prefix
            image_bytes = base64.b64decode(image_data)
            
            # Convert to OpenCV image
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            
            # Preprocess the image (28x28)
            img = cv2.resize(img, (28, 28))
            img = img / 255.0  # Normalize
            img = img.reshape(1, 28, 28, 1)  # Reshape for model
            
            # Make prediction
            prediction = model.predict(img)
            predicted_label = np.argmax(prediction, axis=1)[0]
            predicted_letter = label_to_letter[predicted_label]
            
            return JsonResponse({'letter': predicted_letter})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    # If GET request, render the page
    return render(request, 'signindex.html')