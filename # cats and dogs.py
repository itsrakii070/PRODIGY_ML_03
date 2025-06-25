import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from skimage.feature import hog

# === Config ===
DATA_DIR = 'train'  # Set to your dataset path
IMG_SIZE = 64       # Resize image to 64x64
LIMIT = 2000        # Max images to load (for speed)

# === Load Data ===
def load_images(data_dir, limit=LIMIT):
    X, y = [], []
    count = 0
    for file in os.listdir(data_dir):
        if file.endswith('.jpg'):
            label = 1 if 'dog' in file else 0
            path = os.path.join(data_dir, file)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            X.append(img)
            y.append(label)
            count += 1
            if count >= limit:
                break
    return np.array(X), np.array(y)

print("Loading images...")
X, y = load_images(DATA_DIR)

# === Feature Extraction (HOG) ===
def extract_hog_features(images):
    features = []
    for img in images:
        hog_feat = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
        features.append(hog_feat)
    return np.array(features)

print("Extracting HOG features...")
X_hog = extract_hog_features(X)

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X_hog, y, test_size=0.2, random_state=42)

# === Train SVM Model ===
print("Training SVM...")
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# === Evaluate ===
print("Evaluating...")
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Cat", "Dog"]))
