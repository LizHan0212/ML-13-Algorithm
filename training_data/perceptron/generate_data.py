import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
OUT_DIR = os.path.join(BASE_DIR, "training_data", "perceptron")

os.makedirs(OUT_DIR, exist_ok=True)

def generate_perceptron_dataset():
    n = 10000
    w = 2.5
    b = -0.5

    X = np.random.uniform(-5, 5, size=(n, 1))
    y = (X[:, 0] * w + b >= 0).astype(int)

    # Add some noise
    flip_idx = np.random.choice(n, size=n // 20, replace=False)
    y[flip_idx] = 1 - y[flip_idx]

    path = os.path.join(OUT_DIR, "perceptron_1f_10k.txt")

    with open(path, "w") as f:
        for i in range(n):
            f.write(f"{X[i,0]:.5f},{y[i]}\n")

    print("Generated:", path)


if __name__ == "__main__":
    generate_perceptron_dataset()
