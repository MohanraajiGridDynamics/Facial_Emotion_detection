import cv2
import numpy as np
import base64
import json
import keras
from django.http import JsonResponse
from django.shortcuts import render
from io import BytesIO
from PIL import Image
import os
from tensorflow.keras.models import load_model
#from tensorflow.keras.models import load_model

# try:
#     model = load_model('emotion/emotion_model.hdf5', compile=False)  # Add compile=False
#     print("✅ Model loaded successfully!")
# except Exception as e:
#     print(f"❌ Error loading model: {e}")

from tensorflow import keras

# model = keras.models.load_model('emotion/emotion_model.hdf5', compile=False)
# model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5), loss='categorical_crossentropy')

model = keras.models.load_model("emotion/emotion_model_fixed.hdf5")



# Load Pretrained Emotion Detection Model (Example: FER)
# model = keras.models.load_model('emotion/emotion_model.hdf5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# MODEL_PATH = "emotion/emotion_model.hdf5"  # Make sure this is correct

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  
# MODEL_PATH = os.path.join(BASE_DIR, "emotion", "emotion_model.hdf5")
MODEL_PATH = os.path.join("E:/Django/Face/emotion_recognition/emotion/emotion_model.hdf5")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model file not found: {MODEL_PATH}")

try:
    model = load_model(MODEL_PATH, compile=False)  # Load pre-trained model
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")

try:
    model = load_model(MODEL_PATH, compile=False)  # Use compile=False to avoid issues
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

def home(request):
    return render(request, 'index.html')

def detect_emotion(request):
    if request.method == "POST":
        data = json.loads(request.body)
        image_data = data.get("image").split(",")[1]
        image_bytes = base64.b64decode(image_data)
        
        img = Image.open(BytesIO(image_bytes)).convert('L')  # Convert to grayscale
        img = img.resize((48, 48))  # Resize for the model
        img = np.array(img) / 255.0  # Normalize
        img = np.expand_dims(img, axis=0)
        img = np.expand_dims(img, axis=-1)

        prediction = model.predict(img)
        emotion = emotion_labels[np.argmax(prediction)]

        return JsonResponse({"emotion": emotion})

