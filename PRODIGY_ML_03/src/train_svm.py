import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from load_data import load_image_data
from sklearn.utils import shuffle



DATA_DIR = "./data/train/train"
IMG_SIZE = 32


print("Loading image data...")
X, y = load_image_data(DATA_DIR, IMG_SIZE)


X, y = shuffle(X, y, random_state=42)


print("X shape:", X.shape)
print("y shape:", y.shape)

MAX_SAMPLES = 1000

X = X[:MAX_SAMPLES]
y = y[:MAX_SAMPLES]


print("Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training SVM model...")
svm_model = SVC(
    kernel="linear",
    C=1.0
)


svm_model.fit(X_train, y_train)

print("Evaluating model...")
y_pred = svm_model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
