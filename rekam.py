import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load HAAR face classifier
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load functions
def face_extractor(img):
    # Function detects faces and returns the cropped face
    # If no face detected, it returns None
    faces = face_classifier.detectMultiScale(img, 1.3, 5)
    
    if len(faces) == 0:
        return None
    
    # Crop the first face found
    for (x, y, w, h) in faces:
        x = max(x - 10, 0)
        y = max(y - 10, 0)
        cropped_face = img[y:y + h + 50, x:x + w + 50]
        return cropped_face

# Initialize Webcam
cap = cv2.VideoCapture(0)
count = 0

# Collect 100 samples of your face from webcam input
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    face = face_extractor(frame)
    if face is not None:
        count += 1
        face = cv2.resize(face, (400, 400))  # Resize the face

        # Save file in the specified directory with a unique name
        file_name_path = f'D:/Dataset/Images.{count}.jpg'
        cv2.imwrite(file_name_path, face)q

        # Display the face in Jupyter Notebook
        plt.imshow(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title(f'Captured Face {count}')
        plt.show()

    else:
        print("Face not found")
        pass

    if count == 100:  # Stop after collecting 100 samples
        break

cap.release()
print("Collecting Samples Complete")