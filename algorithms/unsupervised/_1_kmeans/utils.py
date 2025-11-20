import os
import numpy as np

def load_kmeans_data():
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

    path = os.path.join(base, "training_data", "kmeans", "kmeans_data.txt")

    X = []
    with open(path, "r") as f:
        for line in f:
            a, b = line.strip().split(",")
            X.append([float(a), float(b)])

    return np.array(X)
