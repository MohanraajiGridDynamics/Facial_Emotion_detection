import h5py

filepath = "emotion/emotion_model.hdf5"  # Adjust this if needed

try:
    with h5py.File(filepath, "r") as f:
        print("✅ HDF5 model file is valid!")
except Exception as e:
    print("❌ Corrupt file:", e)
