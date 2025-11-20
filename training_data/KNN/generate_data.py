
import os
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
OUT_DIR = os.path.join(BASE, "training_data", "KNN")
os.makedirs(OUT_DIR, exist_ok=True)

OUT_FILE = os.path.join(OUT_DIR, "knn_data.txt")

def generate_knn_data():
    np.random.seed(42)

    n = 100

    # ----------------------------
    # HEAVY MIXED / OVERLAPPING DATA
    # ----------------------------
    # Both classes drawn from nearly identical distributions,
    # but shifted slightly so not 100% identical.

    # Class 0
    X0 = np.random.normal(loc=[5, 5], scale=2.0, size=(n//2, 2))

    # Class 1 (slightly shifted)
    X1 = np.random.normal(loc=[5.5, 4.5], scale=2.0, size=(n//2, 2))

    # Clip negatives just to avoid weird plot scaling
    X0 = np.clip(X0, 0, None)
    X1 = np.clip(X1, 0, None)

    # Labels
    y0 = np.zeros((n//2, 1), dtype=int)
    y1 = np.ones((n//2, 1), dtype=int)

    # Combine
    X = np.vstack([X0, X1])
    y = np.vstack([y0, y1])
    data = np.hstack([X, y])

    # Write file
    with open(OUT_FILE, "w") as f:
        for x1, x2, label in data:
            f.write(f"({x1:.4f},{x2:.4f},{int(label)})\n")

    print("Generated HEAVY-OVERLAP KNN dataset →", OUT_FILE)


if __name__ == "__main__":
    generate_knn_data()
