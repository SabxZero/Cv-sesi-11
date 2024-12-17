import cv2
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# Load model pengenalan wajah
model = load_model("face_recognition_model.h5")
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load("label_encoder.npy")

# Buka kamera
camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Deteksi wajah menggunakan Haar Cascades
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (64, 64))

        # Normalisasi dan prediksi menggunakan model
        normalized_face = face_roi / 255.0
        reshaped_face = np.reshape(normalized_face, (1, 64, 64, 1))
        result = model.predict(reshaped_face)
        label = label_encoder.classes_[np.argmax(result)]

        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow("Face Recognition", frame)

    # Keluar dari loop dengan menekan tombol 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Tutup kamera dan jendela
camera.release()
cv2.destroyAllWindows()