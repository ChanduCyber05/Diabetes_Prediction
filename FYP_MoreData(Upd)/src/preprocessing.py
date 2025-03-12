import os
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Define dataset paths
DATA_DIR = "C:/Users/harsh/diabetes detection/data"
DIABETIC_DIR = os.path.join(DATA_DIR, "diabetic dataset")
NON_DIABETIC_DIR = os.path.join(DATA_DIR, "nondiabetic dataset")

# Ensure dataset directories exist
if not os.path.exists(DIABETIC_DIR) or not os.path.exists(NON_DIABETIC_DIR):
    raise FileNotFoundError("❌ Dataset folders not found! Run preprocessing with correct paths.")

# Image augmentation setup based on the QIRT paper
data_gen = ImageDataGenerator(
    rotation_range=40,
    shear_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.5, 1.5]
)

def load_images(folder, label):
    images, labels = [], []
    for patient in os.listdir(folder):
        patient_path = os.path.join(folder, patient)
        if os.path.isdir(patient_path):
            for img_file in os.listdir(patient_path):
                img_path = os.path.join(patient_path, img_file)
                img = cv2.imread(img_path)
                img = cv2.resize(img, (224, 224))  # Resize as per the paper
                img = img / 255.0  # Normalize
                images.append(img)
                labels.append(label)
    return images, labels

# Load and preprocess images
X_d, y_d = load_images(DIABETIC_DIR, 1)
X_nd, y_nd = load_images(NON_DIABETIC_DIR, 0)

# Convert to NumPy arrays
X = np.array(X_d + X_nd, dtype=np.float32)
y = np.array(y_d + y_nd, dtype=np.int32)

# Train-test-validation split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.33, stratify=y_temp, random_state=42)

# Save numpy arrays
np.save("data/train/X_train.npy", X_train)
np.save("data/train/y_train.npy", y_train)
np.save("data/test/X_test.npy", X_test)
np.save("data/test/y_test.npy", y_test)
np.save("data/val/X_val.npy", X_val)
np.save("data/val/y_val.npy", y_val)

print("✅ Data preprocessing completed successfully! Datasets saved in 'data/train', 'data/test', 'data/val'.")
