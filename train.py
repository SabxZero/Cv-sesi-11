import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# Fungsi untuk memuat dataset wajah
def load_faces_and_labels(dataset_path):
    faces = []
    labels = []

    for file in os.listdir(dataset_path):
        if file.endswith(".jpg"):
            img_path = os.path.join(dataset_path, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"File gambar tidak valid: {img_path}")
                continue
            try:
                img = cv2.resize(img, (64, 64))  # Resize gambar menjadi 64x64
                faces.append(img)
                labels.append("Maximillian")  # Label untuk semua foto, bisa disesuaikan
            except Exception as e:
                print(f"Error saat memproses file {img_path}: {e}")
                continue

    if len(faces) == 0:
        raise ValueError("Dataset kosong. Pastikan folder dataset berisi gambar .jpg yang valid.")

    return faces, labels

# Path ke dataset
dataset_path = "D:\Dataset"  # Ganti dengan path dataset Anda

# Memuat dataset
faces, labels = load_faces_and_labels(dataset_path)

# Melakukan encoding label
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)  # Semua label akan menjadi '0'
encoded_labels = to_categorical(encoded_labels)       # Ubah ke one-hot encoding

# Membagi dataset menjadi data latih dan data uji
faces_train, faces_test, labels_train, labels_test = train_test_split(
    faces, encoded_labels, test_size=0.2, random_state=42
)

# Mengubah data latih dan data uji ke dalam format numpy array
faces_train = np.array(faces_train).reshape(-1, 64, 64, 1) / 255.0  # Normalisasi pixel ke [0, 1]
faces_test = np.array(faces_test).reshape(-1, 64, 64, 1) / 255.0

# Membuat model CNN sederhana
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 1)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation="relu"),
    Dense(len(label_encoder.classes_), activation="softmax")
])

# Kompilasi model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Melatih model
model.fit(
    faces_train,
    labels_train,
    validation_data=(faces_test, labels_test),
    epochs=10,
    verbose=1
)

# Evaluasi model
loss, accuracy = model.evaluate(faces_test, labels_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Simpan model dan label encoder
model.save("face_recognition_model.h5")
np.save("label_encoder.npy", label_encoder.classes_)

print("Model dan label encoder berhasil disimpan.")
