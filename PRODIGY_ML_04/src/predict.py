import cv2
import numpy as np
import pickle
from tensorflow.keras.model import load_model


IMG_SIZE = 32
MODEL_PATH = "archive/gesture_cnn.h5"


model = load_model(MODEL_PATH)


gesture_labels = [
    "palm",
    "fist",
    "thumb",
    "index",
    "ok",
    "c"
]


def predict_gesture(image_path):
    img = cv2.imread(image_path)

    if img is None:
        print("Image not found")
        return

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)

    print("Predicted Gesture:", gesture_labels[predicted_class])


predict_gesture("test_image.jpg")
