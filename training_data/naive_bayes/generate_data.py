import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
OUT_DIR = os.path.join(BASE_DIR, "training_data", "naive_bayes")

os.makedirs(OUT_DIR, exist_ok=True)

def generate():
    np.random.seed(1)

    num_features = 3
    num_classes = 3
    samples_per_class = 200

    all_rows = []

    # Define different distributions for each class
    class_centers = [1, 3, 4]

    for c in range(num_classes):
        X = np.random.normal(loc=class_centers[c], scale=0.8, size=(samples_per_class, num_features))
        X = np.clip(X.round().astype(int), 0, 4)  # discretize + clamp to 0-4
        y = np.full((samples_per_class, 1), c)

        data = np.hstack([X, y])
        all_rows.append(data)

    final = np.vstack(all_rows)

    out_path = os.path.join(OUT_DIR, "nb_3f_600.txt")
    with open(out_path, "w") as f:
        for row in final:
            features = ",".join(str(int(v)) for v in row[:-1])
            f.write(f"({features},{int(row[-1])})\n")

    print("Naive Bayes dataset generated!")

if __name__ == "__main__":
    generate()
