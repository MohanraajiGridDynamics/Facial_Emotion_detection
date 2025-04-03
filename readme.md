Dependencies to install:

Dependencies List
Django – The web framework to build and serve the application.

OpenCV-Python – For computer vision tasks like capturing video, face, and eye detection.

TensorFlow – For loading and running the pre‑trained emotion detection model.

Keras – Usually included with TensorFlow (if you’re using TensorFlow 2.x, Keras is integrated).

NumPy – For numerical operations and image processing.

Pillow – For image processing and handling (used by Django templates or if you work with images).

h5py – For reading and writing HDF5 model files.

Installation Command
Run the following command in your terminal or virtual environment:

sh
Copy
Edit
pip install django opencv-python tensorflow numpy pillow h5py
Note:

If you're on a system without GPU support or prefer a lighter installation, consider installing tensorflow-cpu instead of tensorflow:

sh
Copy
Edit
pip install tensorflow-cpu
Make sure you’re in your project’s virtual environment (e.g., activate your .venv or similar).

Verifying Installations
After installing, you can verify by running:

sh
Copy
Edit
python -c "import django; print('Django', django.get_version())"
python -c "import cv2; print('OpenCV', cv2.__version__)"
python -c "import tensorflow as tf; print('TensorFlow', tf.__version__)"
python -c "import numpy; print('NumPy', numpy.__version__)"
python -c "import PIL; print('Pillow installed')"
python -c "import h5py; print('h5py', h5py.__version__)"
This ensures all libraries are correctly installed and available for your project.

Once all dependencies are installed, you can proceed to run your Django project.



Project Overview



The project leverages the following technologies and components:

Django:
The Python web framework used to build and serve the web application.

OpenCV:
A library for computer vision tasks (face, head, and eye detection).

Keras/TensorFlow:
Used for loading a pre‑trained emotion detection model that predicts emotions from face images.

Haar Cascades:
Pre‑trained classifiers in OpenCV for face and eye detection.

Frontend (HTML/JavaScript):
A simple webpage to display a live video stream, detected emotion, face count, and head/eye movement information. JavaScript fetches data from Django endpoints at regular intervals (every 2 seconds).

Project Structure
A typical project structure might look like this:

bash
Copy
Edit
Face_Emotion_recognition/
├── emotion_recognition/         # Main Django project folder
│   ├── __init__.py
│   ├── settings.py              # Django settings (includes app registrations, static files config, etc.)
│   ├── urls.py                  # Main URL configuration that includes the app’s URLs
│   └── wsgi.py
├── emotion/                     # Django app folder
│   ├── __init__.py
│   ├── views.py                 # All view functions: emotion detection, face count, head & eye movement monitoring
│   ├── urls.py                  # URL patterns for the app – maps routes to views
│   ├── templates/
│   │   └── index.html           # Main HTML template that displays video stream and data via JavaScript
│   └── migrations/              # Django migrations folder (auto‑generated)
├── emotion_model_fixed.hdf5     # Pre‑trained Keras model file (saved in the app folder)
└── manage.py                    # Django management script (used to run the server, manage commands, etc.)
Detailed File Explanations
1. manage.py
Purpose:
Entry point for Django commands. You run this file to start the server, create migrations, or perform other management tasks.

Usage:
Running python manage.py runserver launches the development server.

2. emotion_recognition/settings.py
Purpose:
Contains all configuration settings for the Django project such as installed apps, middleware, database settings, static file paths, etc.

Key Points:

Ensure that the app emotion is included in the INSTALLED_APPS list.

Configure static and template directories as needed.

3. emotion_recognition/urls.py
Purpose:
The main URL configuration file. It includes URL patterns for the entire project.

Key Points:

It uses Django’s include function to delegate URL routing to the app-level urls.py.

Example:

python
Copy
Edit
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('emotion.urls')),  # Includes all URLs defined in the emotion app.
]
4. emotion/urls.py
Purpose:
Contains URL patterns specific to the emotion app.

Key Routes:

/ – Renders the homepage (index.html).

/detect_emotion/ – Captures an image and runs emotion detection.

/count_faces/ – Captures an image and returns the number of detected faces.

/monitor_movement/ – Captures an image and analyzes head and eye movement.

/video_feed/ – (Optional) Streams a live video feed with detection annotations.

Example:

python
Copy
Edit
from django.urls import path
from .views import home, detect_emotion, count_faces, monitor_head_eye_movement

urlpatterns = [
    path('', home, name='home'),
    path('detect_emotion/', detect_emotion, name='detect_emotion'),
    path('count_faces/', count_faces, name='count_faces'),
    path('monitor_movement/', monitor_head_eye_movement, name='monitor_movement'),
    # path('video_feed/', video_feed, name='video_feed'),  # if using live streaming
]
5. emotion/views.py
Purpose:
Contains all the view functions that handle requests and perform computer vision tasks.

Key Functions:

home(request)
Renders the homepage (index.html).

detect_emotion(request)

Captures an image from the webcam.

Converts the frame to grayscale.

Uses Haar cascades to detect a face.

Preprocesses the face region (resizes, normalizes) to feed into the Keras model.

Predicts the emotion from the face and returns the result as JSON.

count_faces(request)

Captures an image.

Converts it to grayscale.

Uses Haar cascades to detect all faces.

Counts and returns the number of detected faces as JSON.

monitor_head_eye_movement(request)

Captures an image.

Detects the face and eyes using Haar cascades.

Determines head movement by comparing the face’s center to the frame’s center.

Determines eye movement relative to the face region.

Returns the computed directions (head and eyes) as JSON.

video_feed(request) (optional)

Streams a live video feed where detected faces are outlined. Useful for debugging or visual feedback.

Example Code Snippet from views.py:

python
Copy
Edit
import cv2
import numpy as np
import os
from django.http import JsonResponse, StreamingHttpResponse
from django.shortcuts import render
from tensorflow.keras.models import load_model
from tensorflow import keras
import time

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

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def home(request):
    return render(request, 'index.html')

def detect_emotion(request):
    # ... (code to capture image, detect face, predict emotion)
    # Returns JSON with the detected emotion
    pass

def count_faces(request):
    # ... (code to capture image, count faces)
    # Returns JSON with face_count
    pass

def monitor_head_eye_movement(request):
    # ... (code to capture image, detect face and eyes, determine head and eye movement)
    # Returns JSON with head_direction and eye_directions
    pass

def video_feed(request):
    # ... (optional live streaming code)
    pass
(The complete code in our previous explanation includes all the details for these functions.)

6. emotion/templates/index.html
Purpose:
The main webpage that the user sees.

Key Components:

Video Stream/Image Display:
An <img> tag (or <video> element) to display the live video feed if using streaming.

Data Display:
Elements (like <span> tags) to display detected emotion, face count, head movement, and eye movement.

JavaScript:
Uses the Fetch API (or AJAX) to periodically call Django endpoints (e.g., /detect_emotion/, /count_faces/, /monitor_movement/) every 2 seconds. The returned JSON data is then used to update the HTML dynamically.

Example:

html
Copy
Edit
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Real-Time Face & Emotion Detection</title>
    <style>
        body { font-family: Arial, sans-serif; text-align: center; }
        #video-feed { width: 60%; margin-top: 20px; border: 2px solid #333; }
    </style>
</head>
<body>
    <h1>Real-Time Emotion and Movement Detection</h1>
    <!-- Display live video (if implemented) -->
    <img id="video-feed" src="{% url 'video_feed' %}" alt="Live Video Stream">

    <!-- Display detected emotion -->
    <h2>Detected Emotion: <span id="emotion">N/A</span></h2>
    
    <!-- Display face count -->
    <h2>Number of Faces: <span id="face-count">0</span></h2>
    
    <!-- Display head & eye movement -->
    <h2>Head Movement: <span id="head-direction">N/A</span></h2>
    <h2>Eye Movements: <span id="eye-directions">N/A</span></h2>
    
    <script>
        // Function to update detected emotion
        function updateEmotion() {
            fetch("{% url 'detect_emotion' %}")
            .then(response => response.json())
            .then(data => {
                document.getElementById("emotion").innerText = data.emotion;
            })
            .catch(error => console.error("Error fetching emotion:", error));
        }

        // Function to update face count
        function updateFaceCount() {
            fetch("{% url 'count_faces' %}")
            .then(response => response.json())
            .then(data => {
                document.getElementById("face-count").innerText = data.face_count;
            })
            .catch(error => console.error("Error fetching face count:", error));
        }

        // Function to update head and eye movement
        function updateMovement() {
            fetch("{% url 'monitor_movement' %}")
            .then(response => response.json())
            .then(data => {
                document.getElementById("head-direction").innerText = data.head_direction;
                document.getElementById("eye-directions").innerText = data.eye_directions.join(", ");
            })
            .catch(error => console.error("Error fetching movement:", error));
        }

        // Update all values every 2 seconds
        setInterval(() => {
            updateEmotion();
            updateFaceCount();
            updateMovement();
        }, 2000);
    </script>
</body>
</html>
How the Project Works Together
User Visits the Homepage:
The Django view home renders index.html.

Live Data Updates:
JavaScript in index.html makes periodic requests (every 2 seconds) to:

/detect_emotion/ to update the detected emotion.

/count_faces/ to update the number of faces detected.

/monitor_movement/ to update head and eye movement directions.

Processing on the Server:
Each Django view:

Opens the webcam.

Captures a frame.

Uses OpenCV with Haar cascades to detect faces and eyes.

(For emotion) Preprocesses the face image and runs it through the pre‑trained Keras model.

Computes the required information (emotion, face count, head/eye movement) and returns it as JSON.

Data Display:
The JSON responses update the webpage elements so that the user sees real‑time data.

Conclusion
This project combines computer vision (using OpenCV) and deep learning (using a Keras model) within a Django framework to create a real‑time system that:

Detects emotions from faces.

Counts how many faces appear.

Monitors head and eye movement (with simple heuristics).

Each file plays a key role—from managing URL routing and views in Django to dynamically updating the user interface with HTML and JavaScript. You can extend or refine these components as needed (for example, by using more advanced head pose estimation or eye tracking libraries).

Let me know if you have any further questions or need additional clarifications!







You said:
Great, now explain me the views.py alone


ChatGPT said:
Reasoned for 4 seconds
Below is a detailed explanation of the views.py file for your project. This file contains all the server‑side logic for handling requests related to emotion detection, face counting, and monitoring head & eye movement. Let’s break it down section by section.

1. Import Statements and Model Setup
python
Copy
Edit
import cv2
import numpy as np
import os
from django.http import JsonResponse, StreamingHttpResponse
from django.shortcuts import render
from tensorflow.keras.models import load_model
from tensorflow import keras
import time
Purpose:
These imports bring in required libraries:

OpenCV (cv2): For image processing and face/eye detection.

NumPy: For numerical operations.

OS: To work with file paths.

Django utilities: JsonResponse, StreamingHttpResponse, and render to handle HTTP responses and render templates.

TensorFlow/Keras: To load and use your pre-trained emotion detection model.

Time: For handling timing in live processing.

2. Setting Up the Model and Haar Cascades
python
Copy
Edit
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

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
Model Path Setup:

The code determines the current directory of the views.py file and sets MODEL_PATH to the fixed model file.

If the model file does not exist, it raises an error.

Model Loading:

The model is loaded with compile=False (since you don’t need to recompile it for inference).

A list emotion_labels maps the model’s output indices to human‑readable emotion names.

Haar Cascades:

Haar cascades for face and eye detection are loaded from OpenCV’s default directory. These classifiers will be used to detect faces and eyes in images.

3. The home View
python
Copy
Edit
def home(request):
    """
    Render the main page.
    """
    return render(request, 'index.html')
Purpose:
The simplest view, which renders the homepage template (index.html). This page typically contains the UI that displays the video feed, emotion, face count, and movement information.

4. The detect_emotion View
python
Copy
Edit
def detect_emotion(request):
    """
    Capture an image from the webcam, detect a face, preprocess it, run the model, and return the predicted emotion.
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
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi_gray = roi_gray / 255.0  # Normalize pixel values
        roi_gray = np.expand_dims(roi_gray, axis=0)
        roi_gray = np.expand_dims(roi_gray, axis=-1)

        prediction = model.predict(roi_gray)
        emotion = emotion_labels[np.argmax(prediction)]
        print(f"🎭 Predicted Emotion: {emotion}")
        return JsonResponse({'emotion': emotion})

    return JsonResponse({'emotion': 'Unknown'})
Purpose:

This view captures a single frame from the webcam.

Converts the frame to grayscale (which is generally more effective for face detection).

Uses the face cascade to detect faces.

If a face is detected, the face region (ROI) is resized to 48×48 (the input size expected by the model), normalized, and expanded in dimensions so it can be processed by the Keras model.

The model predicts the emotion, which is then returned as a JSON response.

Flow:

Capture frame → Check if captured successfully → Convert to grayscale → Detect faces → For the first face, process and predict emotion → Return result.

5. The count_faces View
python
Copy
Edit
def count_faces(request):
    """
    Capture an image from the webcam, detect faces, count them, and return the count.
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
Purpose:

This view also captures an image from the webcam and converts it to grayscale.

It uses the face cascade to detect all faces and then counts them.

The total face count is returned as JSON.

Use Case:
Useful for scenarios where you want to display how many faces are present in the frame.

6. The monitor_head_eye_movement View
python
Copy
Edit
def monitor_head_eye_movement(request):
    """
    Capture an image, detect the face and eyes, and determine the head and eye movement directions.
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
        (x, y, w, h) = faces[0]
        face_center = (x + w//2, y + h//2)
        
        # Thresholds to determine movement
        thresh_x = frame_width * 0.1
        thresh_y = frame_height * 0.1

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
        face_roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(face_roi_gray, scaleFactor=1.1, minNeighbors=5)
        
        for (ex, ey, ew, eh) in eyes:
            eye_center = (ex + ew//2, ey + eh//2)
            thresh_ex = w * 0.1
            thresh_ey = h * 0.1

            if eye_center[0] < w//2 - thresh_ex:
                eye_horizontal = "Left"
            elif eye_center[0] > w//2 + thresh_ex:
                eye_horizontal = "Right"
            else:
                eye_horizontal = "Center"

            if eye_center[1] < h//2 - thresh_ey:
                eye_vertical = "Top"
            elif eye_center[1] > h//2 + thresh_ey:
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
Purpose:

This view captures a frame, determines the center of the entire frame, and then detects the face.

It compares the center of the face to the frame center to decide if the head is positioned to the left, right, top, bottom, or center.

Within the detected face region, it then detects eyes and calculates each eye’s relative position (left, right, top, bottom, or center) compared to the face center.

Return Value:
The computed head movement and a list of eye directions are returned as JSON. This allows the frontend to display these details in real time.

Summary
Imports & Setup:
Libraries are imported, the model is loaded, and Haar cascades are initialized.

Views:

home: Renders the main page.

detect_emotion: Captures a frame, detects a face, and predicts the emotion using the pre-trained model.

count_faces: Captures a frame and counts the number of faces.

monitor_head_eye_movement: Captures a frame and analyzes head and eye positions relative to the frame and face region.

Each function uses OpenCV for image capture and processing, and then returns the results in JSON format so that the frontend can periodically update the displayed information.

This is how the views.py file orchestrates the core functionalities of your project. Let me know if you need further details or clarifications!