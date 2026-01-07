import os
import cv2
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


DATASET_PATH = "data/leapGestRecog"
IMG_SIZE = 64


X = []
y = []


for subject in os.listdir(DATASET_PATH):
    subject_path = os.path.join(DATASET_PATH, subject)

    if not os.path.isdir(subject_path):
        continue

    for gesture in os.listdir(subject_path):
        gesture_path = os.path.join(subject_path, gesture)

        if not os.path.isdir(gesture_path):
            continue

        for img_name in os.listdir(gesture_path):
            img_path = os.path.join(gesture_path, img_name)

            img = cv2.imread(img_path)

            if img is None:
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img / 255.0

            X.append(img)
            y.append(gesture)


X = np.array(X)
y = np.array(y)


encoder = LabelEncoder()
y = encoder.fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = Sequential()

model.add(
    Conv2D(
        32,
        (3, 3),
        activation="relu",
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
)

model.add(MaxPooling2D((2, 2)))

model.add(
    Conv2D(
        64,
        (3, 3),
        activation="relu"
    )
)

model.add(MaxPooling2D((2, 2)))

model.add(Flatten())

model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))

model.add(Dense(len(np.unique(y)), activation="softmax"))


model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


model.summary()


history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test)
)


test_loss, test_accuracy = model.evaluate(X_test, y_test)

print("Test Accuracy:", test_accuracy)
