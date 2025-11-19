import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
OUT_DIR = os.path.join(BASE_DIR, "training_data", "centroid_classifier")

os.makedirs(OUT_DIR, exist_ok=True)

def generate():
    num_classes = 3
    means = [-3, 0, 3]   # centers for 3 classes

    for f in [1, 3]:
        all_X = []
        all_y = []

        for c in range(num_classes):
            Xc = np.random.normal(loc=means[c], scale=1.0, size=(100, f))
            yc = np.full((100, 1), c)
            all_X.append(Xc)
            all_y.append(yc)

        X = np.vstack(all_X)
        y = np.vstack(all_y)
        data = np.hstack([X, y])

        fname = f"centroid_{f}f_100.txt"
        path = os.path.join(OUT_DIR, fname)

        with open(path, "w") as ftxt:
            for row in data:
                feats = ",".join([f"{v:.4f}" for v in row[:-1]])
                ftxt.write(f"({feats},{int(row[-1])})\n")

if __name__ == "__main__":
    generate()
    print("Multi-class centroid training data generated.")
