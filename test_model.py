import numpy as np
import cv2
from tensorflow import keras

MODEL_PATH = "E:/Django/Face/emotion_recognition/emotion/emotion_model.hdf5"
model = keras.models.load_model(MODEL_PATH)

# Define emotion labels
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Create a dummy input to test the model
test_input = np.random.rand(1, 48, 48, 1)  # Random grayscale image
prediction = model.predict(test_input)
print("Test Prediction Output:", prediction)
print("Predicted Emotion:", emotions[np.argmax(prediction)])
