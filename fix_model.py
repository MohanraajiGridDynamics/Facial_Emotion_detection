from tensorflow import keras

# Load the existing model
model = keras.models.load_model("E:/Django/Face/emotion_recognition/emotion/emotion_model.hdf5", compile=False)

# Recompile the model with a corrected optimizer
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5), loss='categorical_crossentropy')

# Save the corrected model
model.save("E:/Django/Face/emotion_recognition/emotion/emotion_model_fixed.hdf5")
