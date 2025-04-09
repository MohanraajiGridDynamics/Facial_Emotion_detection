import cv2
import face_recognition
import numpy as np
from django.http import StreamingHttpResponse, JsonResponse
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.conf import settings
import os

from django.views.decorators.csrf import csrf_exempt

reference_encoding = None
face_count = 0
latest_directions = {"face": "N/A", "eye": "N/A"}  # ✅ global, don't overwrite inside gen_frames


def upload_and_run(request):
    global reference_encoding

    if request.method == 'POST' and request.FILES.get('image'):
        image = request.FILES['image']
        fs = FileSystemStorage()
        filename = fs.save(image.name, image)
        file_path = os.path.join(settings.MEDIA_ROOT, filename)

        reference_image = cv2.imread(file_path)
        rgb = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb)

        if encodings:
            reference_encoding = encodings[0]
        else:
            reference_encoding = None
            print("⚠️ No face detected in the uploaded image.")

    return render(request, 'face_app/index.html')


def gen_frames():
    global reference_encoding, face_count, latest_directions

    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        face_count = len(face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            match = False
            if reference_encoding is not None:
                match = face_recognition.compare_faces([reference_encoding], face_encoding)[0]

            label = "MATCH" if match else "NO MATCH"
            color = (0, 255, 0) if match else (0, 0, 255)

            face_dir = "Center"
            eye_dir = "Center"
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                face_center_x = (left + right) // 2
                frame_center_x = frame.shape[1] // 2

                if face_center_x < frame_center_x - 40:
                    face_dir = "Left"
                elif face_center_x > frame_center_x + 40:
                    face_dir = "Right"

                eyes = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
                detected_eyes = eyes.detectMultiScale(gray[top:bottom, left:right])

                for (ex, ey, ew, eh) in detected_eyes:
                    eye_center_x = ex + ew // 2
                    if eye_center_x < ew * 0.4:
                        eye_dir = "Left"
                    elif eye_center_x > ew * 0.6:
                        eye_dir = "Right"
                    else:
                        eye_dir = "Center"
                    break  # Only consider the first detected eye
            except Exception as e:
                print("Error while detecting direction:", e)

            # ✅ Update global variable, not local
            latest_directions["face"] = face_dir
            latest_directions["eye"] = eye_dir

            # Draw visuals
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, label, (left, top - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, f"{face_dir}, {eye_dir}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


def face_count_view(request):
    global face_count
    return JsonResponse({'count': face_count})


def video_feed(request):
    return StreamingHttpResponse(gen_frames(), content_type='multipart/x-mixed-replace; boundary=frame')


@csrf_exempt
def direction_view(request):
    global latest_directions
    return JsonResponse(latest_directions)


def get_directions(request):
    global latest_directions
    return JsonResponse({
        "face_direction": latest_directions.get("face", "N/A"),
        "eye_direction": latest_directions.get("eye", "N/A")
    })
