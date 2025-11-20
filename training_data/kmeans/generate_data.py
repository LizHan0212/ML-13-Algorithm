import os
import numpy as np

def generate_kmeans_data():
    # Resolve project root
    base = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    out_dir = os.path.join(base, "training_data", "kmeans")
    os.makedirs(out_dir, exist_ok=True)

    path = os.path.join(out_dir, "kmeans_data.txt")

    # Reproducibility
    np.random.seed(42)

    # Three loose clusters (but will be mixed during training)
    centers = [(2, 2), (6, 6), (2, 7)]
    data = []

    for (cx, cy) in centers:
        for _ in range(20):
            x = np.random.normal(cx, 0.8)   # noise
            y = np.random.normal(cy, 0.8)
            data.append((x, y))

    # Only take first 50 points
    data = np.array(data[:50])

    # Save to TXT
    with open(path, "w") as f:
        for x, y in data:
            f.write(f"{x},{y}\n")

    print("Generated:", path)


if __name__ == "__main__":
    generate_kmeans_data()
