import cv2
import numpy as np
import os
from django.http import JsonResponse, StreamingHttpResponse
from django.shortcuts import render
from tensorflow.keras.models import load_model
from tensorflow import keras
import time

# Define base directory and model path for your emotion model (unchanged)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "emotion_model_fixed.hdf5")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model file not found: {MODEL_PATH}")

try:
    model = load_model(MODEL_PATH, compile=False)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Initialize Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')


def home(request):
    """
    Render the main page.
    """
    return render(request, 'index.html')


def detect_emotion(request):
    """
    Capture an image and detect emotion on the first detected face.
    """
    print("📸 Capturing image for emotion detection...")
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("❌ Failed to capture image")
        return JsonResponse({'error': 'Failed to capture image'})

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        print("⚠️ No face detected")
        return JsonResponse({'emotion': 'No face detected'})

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi_gray = roi_gray / 255.0  # Normalize
        roi_gray = np.expand_dims(roi_gray, axis=0)
        roi_gray = np.expand_dims(roi_gray, axis=-1)

        prediction = model.predict(roi_gray)
        emotion = emotion_labels[np.argmax(prediction)]
        print(f"🎭 Predicted Emotion: {emotion}")
        return JsonResponse({'emotion': emotion})

    return JsonResponse({'emotion': 'Unknown'})


def count_faces(request):
    """
    Capture an image and count the number of detected faces.
    """
    print("📸 Capturing image for face count...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return JsonResponse({'error': 'Camera not accessible'})

    ret, frame = cap.read()
    cap.release()

    if not ret:
        return JsonResponse({'error': 'Failed to capture image'})

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    face_count = len(faces)
    print(f"🔢 Detected {face_count} face(s)")
    return JsonResponse({'face_count': face_count})


def monitor_head_eye_movement(request):
    """
    Capture an image and monitor head movement (left, right, top, bottom, center)
    and eye movement (within the face region) using Haar cascades.
    """
    print("📸 Capturing image for head & eye movement analysis...")
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return JsonResponse({'error': 'Failed to capture image'})

    # Get frame dimensions and compute frame center
    frame_height, frame_width = frame.shape[:2]
    frame_center = (frame_width // 2, frame_height // 2)

    # Detect faces
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    head_direction = "No face detected"
    eye_directions = []

    if len(faces) > 0:
        # For simplicity, use the first detected face
        (x, y, w, h) = faces[0]
        face_center = (x + w // 2, y + h // 2)

        # Determine head movement relative to frame center using a threshold
        thresh_x = frame_width * 0.1  # 10% of frame width
        thresh_y = frame_height * 0.1  # 10% of frame height

        if face_center[0] < frame_center[0] - thresh_x:
            head_horizontal = "Left"
        elif face_center[0] > frame_center[0] + thresh_x:
            head_horizontal = "Right"
        else:
            head_horizontal = "Center"

        if face_center[1] < frame_center[1] - thresh_y:
            head_vertical = "Top"
        elif face_center[1] > frame_center[1] + thresh_y:
            head_vertical = "Bottom"
        else:
            head_vertical = "Center"

        if head_horizontal == "Center" and head_vertical == "Center":
            head_direction = "Center"
        else:
            head_direction = f"{head_vertical} {head_horizontal}" if head_vertical != "Center" else head_horizontal

        # Detect eyes within the face region
        face_roi_gray = gray[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(face_roi_gray, scaleFactor=1.1, minNeighbors=5)

        for (ex, ey, ew, eh) in eyes:
            eye_center = (ex + ew // 2, ey + eh // 2)
            # Determine eye movement relative to face center using thresholds
            thresh_ex = w * 0.1
            thresh_ey = h * 0.1

            if eye_center[0] < w // 2 - thresh_ex:
                eye_horizontal = "Left"
            elif eye_center[0] > w // 2 + thresh_ex:
                eye_horizontal = "Right"
            else:
                eye_horizontal = "Center"

            if eye_center[1] < h // 2 - thresh_ey:
                eye_vertical = "Top"
            elif eye_center[1] > h // 2 + thresh_ey:
                eye_vertical = "Bottom"
            else:
                eye_vertical = "Center"

            if eye_horizontal == "Center" and eye_vertical == "Center":
                eye_direction = "Center"
            else:
                eye_direction = f"{eye_vertical} {eye_horizontal}" if eye_vertical != "Center" else eye_horizontal

            eye_directions.append(eye_direction)

    result = {
        "head_direction": head_direction,
        "eye_directions": eye_directions
    }
    print("🔍 Head Movement:", head_direction)
    print("🔍 Eye Movement:", eye_directions)
    return JsonResponse(result)
