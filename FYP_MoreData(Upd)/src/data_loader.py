import os
import numpy as np

# Define dataset paths
DATA_DIR = "data"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
VAL_DIR = os.path.join(DATA_DIR, "val")

# Function to load dataset
def load_data():
    try:
        print("🔍 Loading preprocessed dataset...")

        # Load training data
        X_train = np.load(os.path.join(TRAIN_DIR, "X_train.npy"))
        y_train = np.load(os.path.join(TRAIN_DIR, "y_train.npy"))

        # Load testing data
        X_test = np.load(os.path.join(TEST_DIR, "X_test.npy"))
        y_test = np.load(os.path.join(TEST_DIR, "y_test.npy"))

        # Load validation data
        X_val = np.load(os.path.join(VAL_DIR, "X_val.npy"))
        y_val = np.load(os.path.join(VAL_DIR, "y_val.npy"))

        print(f"✅ Dataset Loaded Successfully!")
        print(f"📊 Training Data: {X_train.shape}, Labels: {y_train.shape}")
        print(f"📊 Testing Data: {X_test.shape}, Labels: {y_test.shape}")
        print(f"📊 Validation Data: {X_val.shape}, Labels: {y_val.shape}")

        return X_train, y_train, X_test, y_test, X_val, y_val

    except FileNotFoundError as e:
        print(f"❌ ERROR: {e}")
        print("🔍 Ensure that preprocessing.py has been run and files exist in 'data/train', 'data/test', 'data/val'.")
        exit()

# Run the function if the script is executed directly
if __name__ == "__main__":
    load_data()
