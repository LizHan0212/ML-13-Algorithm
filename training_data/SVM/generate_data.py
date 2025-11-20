import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
OUT_DIR = os.path.join(BASE_DIR, "training_data", "SVM")

os.makedirs(OUT_DIR, exist_ok=True)

OUT_FILE = os.path.join(OUT_DIR, "svm_data.txt")

def generate_svm_data():
    np.random.seed(42)

    n = 150

    # Two clusters
    X0 = np.random.normal(loc=[2, 2], scale=1.0, size=(n//2, 2))
    X1 = np.random.normal(loc=[7, 7], scale=1.0, size=(n//2, 2))

    # Make sure values are positive
    X0 = np.clip(X0, 0, None)
    X1 = np.clip(X1, 0, None)

    y0 = np.zeros((n//2, 1), dtype=int)
    y1 = np.ones((n//2, 1), dtype=int)

    X = np.vstack([X0, X1])
    y = np.vstack([y0, y1])

    data = np.hstack([X, y])

    with open(OUT_FILE, "w") as f:
        for x1, x2, label in data:
            f.write(f"({x1:.4f},{x2:.4f},{label})\n")

    print(f"Generated SVM training data → {OUT_FILE}")

if __name__ == "__main__":
    generate_svm_data()
