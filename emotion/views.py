import cv2
import numpy as np
import os
from django.http import JsonResponse, StreamingHttpResponse
from django.shortcuts import render
from tensorflow.keras.models import load_model
from tensorflow import keras

# Define the base directory and model path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Use the fixed model you saved (update the filename if needed)
MODEL_PATH = os.path.join(BASE_DIR, "emotion_model_fixed.hdf5")

# Ensure the model file exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model file not found: {MODEL_PATH}")

# Load the pre-trained emotion detection model
try:
    model = load_model(MODEL_PATH, compile=False)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# Define emotion labels corresponding to your model's outputs
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Initialize the Haar cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def home(request):
    """
    Render the main page.
    """
    return render(request, 'index.html')

def detect_emotion(request):
    """
    Capture an image from the webcam, detect faces, and predict the emotion on the first face.
    Returns the emotion as JSON.
    """
    print("📸 Capturing image for emotion detection...")
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("❌ Failed to capture image")
        return JsonResponse({'error': 'Failed to capture image'})

    # Convert the image to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        print("⚠️ No face detected")
        return JsonResponse({'emotion': 'No face detected'})

    # Process only the first detected face (you can modify to loop over all faces)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi_gray = roi_gray / 255.0  # Normalize
        roi_gray = np.expand_dims(roi_gray, axis=0)
        roi_gray = np.expand_dims(roi_gray, axis=-1)

        # Predict the emotion
        prediction = model.predict(roi_gray)
        emotion = emotion_labels[np.argmax(prediction)]
        print(f"🎭 Predicted Emotion: {emotion}")
        return JsonResponse({'emotion': emotion})

    return JsonResponse({'emotion': 'Unknown'})

def count_faces(request):
    """
    Capture an image from the webcam, detect faces, and return the count.
    """
    print("📸 Capturing image for face count...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return JsonResponse({'error': 'Camera not accessible'})

    ret, frame = cap.read()
    cap.release()

    if not ret:
        return JsonResponse({'error': 'Failed to capture image'})

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    face_count = len(faces)
    print(f"🔢 Detected {face_count} face(s)")
    return JsonResponse({'face_count': face_count})
