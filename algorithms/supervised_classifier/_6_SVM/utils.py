
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
DATA_FILE = os.path.join(BASE_DIR, "training_data", "SVM", "svm_data.txt")

def load_svm_dataset():
    X = []
    y = []

    with open(DATA_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            line = line.replace("(", "").replace(")", "")
            parts = line.split(",")

            x1 = float(parts[0])
            x2 = float(parts[1])
            label = int(float(parts[2]))   # <--- ⭐ FIX HERE

            X.append([x1, x2])
            y.append(label)

    return np.array(X), np.array(y)
