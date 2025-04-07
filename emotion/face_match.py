import face_recognition
import cv2
import numpy as np

def run_face_match(reference_image_path):
    print("[INFO] Loading and encoding reference image...")
    reference_image = cv2.imread(reference_image_path)

    if reference_image is None:
        print("[ERROR] Could not load the reference image.")
        return

    reference_rgb = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)
    reference_face_locations = face_recognition.face_locations(reference_rgb)
    reference_face_encodings = face_recognition.face_encodings(reference_rgb, reference_face_locations)

    if len(reference_face_encodings) == 0:
        print("[ERROR] No face found in the reference image.")
        return

    reference_encoding = reference_face_encodings[0]
    print("[INFO] Reference face encoding loaded.")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    print("[INFO] Webcam started. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            match = face_recognition.compare_faces([reference_encoding], face_encoding)[0]
            color = (0, 255, 0) if match else (0, 0, 255)
            label = "MATCH" if match else "NO MATCH"

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.imshow("Face Match", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()
