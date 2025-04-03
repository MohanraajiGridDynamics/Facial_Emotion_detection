import cv2
import numpy as np
import time
import base64
import json
import keras
from django.http import JsonResponse, StreamingHttpResponse
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
#emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# MODEL_PATH = "emotion/emotion_model.hdf5"  # Make sure this is correct

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  
MODEL_PATH = os.path.join(BASE_DIR, "emotion_model.hdf5")
#MODEL_PATH = os.path.join("E:/Django/Face/emotion_recognition/emotion/emotion_model.hdf5")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

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


def detect_emotion_live():
    cap = cv2.VideoCapture(0)  # Open the webcam
    last_prediction_time = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale for better detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        current_time = time.time()

        if current_time - last_prediction_time >= 2:  # Process every 2 seconds
            for (x, y, w, h) in faces:
                face_roi = gray[y:y + h, x:x + w]
                face_roi = cv2.resize(face_roi, (48, 48))  # Resize for the model
                face_roi = np.expand_dims(face_roi, axis=0)
                face_roi = np.expand_dims(face_roi, axis=-1)
                face_roi = face_roi / 255.0  # Normalize

                prediction = model.predict(face_roi)
                emotion = emotion_labels[np.argmax(prediction)]
                print(f"Detected Emotion: {emotion}")

            last_prediction_time = current_time

        # Show the live feed
        cv2.imshow("Real-Time Emotion Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            break

    cap.release()
    cv2.destroyAllWindows()


# Django view to serve the video stream
def video_feed(request):
    return StreamingHttpResponse(detect_emotion_live(), content_type="multipart/x-mixed-replace;boundary=frame")


def detect_emotion(request):
    print("📸 Capturing image...")

    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("❌ Failed to capture image")
        return JsonResponse({'error': 'Failed to capture image'})

    # Convert image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        print("⚠️ No face detected")
        return JsonResponse({'emotion': 'No face detected'})

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48))  # Resize for model
        roi_gray = roi_gray / 255.0  # Normalize
        roi_gray = np.expand_dims(roi_gray, axis=0)
        roi_gray = np.expand_dims(roi_gray, axis=-1)

        # Predict Emotion
        prediction = model.predict(roi_gray)
        emotion_label = emotions[np.argmax(prediction)]

        print(f"🎭 Predicted Emotion: {emotion_label}")
        return JsonResponse({'emotion': emotion_label})

    print("⚠️ No valid face found in loop")
    return JsonResponse({'emotion': 'Unknown'})


# def detect_emotion(request):
#     cap = cv2.VideoCapture(0)
#     ret, frame = cap.read()
#     cap.release()

#     if not ret:
#         return JsonResponse({'error': 'Failed to capture image'})

#     # Convert image to grayscale
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

#     if len(faces) == 0:
#         return JsonResponse({'emotion': 'No face detected'})

#     for (x, y, w, h) in faces:
#         roi_gray = gray[y:y + h, x:x + w]
#         roi_gray = cv2.resize(roi_gray, (48, 48))  # Resize to match model input
#         roi_gray = roi_gray / 255.0  # Normalize
#         roi_gray = np.expand_dims(roi_gray, axis=0)
#         roi_gray = np.expand_dims(roi_gray, axis=-1)

#         # Predict Emotion
#         prediction = model.predict(roi_gray)
#         emotion_label = emotions[np.argmax(prediction)]

#         return JsonResponse({'emotion': emotion_label})

# def detect_emotion(request):
#     if request.method == "POST":
#         data = json.loads(request.body)
#         image_data = data.get("image").split(",")[1]
#         image_bytes = base64.b64decode(image_data)
        
#         img = Image.open(BytesIO(image_bytes)).convert('L')  # Convert to grayscale
#         img = img.resize((48, 48))  # Resize for the model
#         img = np.array(img) / 255.0  # Normalize
#         img = np.expand_dims(img, axis=0)
#         img = np.expand_dims(img, axis=-1)

#         prediction = model.predict(img)
#         emotion = emotion_labels[np.argmax(prediction)]

#         return JsonResponse({"emotion": emotion})

