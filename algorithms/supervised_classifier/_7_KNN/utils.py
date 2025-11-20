
import os
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
DATA_FILE = os.path.join(BASE, "training_data", "KNN", "knn_data.txt")

def load_knn_dataset():
    X = []
    y = []

    with open(DATA_FILE, "r") as f:
        for line in f:
            line = line.strip().replace("(", "").replace(")", "")
            x1, x2, lab = line.split(",")
            X.append([float(x1), float(x2)])
            y.append(int(float(lab)))

    return np.array(X), np.array(y)
