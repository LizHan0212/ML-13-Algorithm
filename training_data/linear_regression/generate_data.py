import numpy as np
import os

# Directory where THIS script lives
THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def save_dataset(filename, data):
    """Save dataset as text in (x1,x2,...,y) format."""
    path = os.path.join(THIS_DIR, filename)
    with open(path, "w") as f:
        for row in data:
            f.write("(" + ",".join(f"{v:.4f}" for v in row) + ")\n")
    print("Saved:", path)


# ============================================================
#                  DATASET GENERATORS
# ============================================================

def generate_1f(n):
    """
    1 feature → 1 label
    More realistic data:
      - strong noise
      - slight non-linearity
    """
    x = np.random.uniform(-10, 10, size=(n, 1))

    # Stronger noise
    noise = np.random.normal(scale=8.0, size=(n, 1))

    # Linear + slight nonlinear term
    y = 3 * x + 5 + 0.2 * (x ** 2) + noise

    return np.hstack([x, y])


def generate_3f(n):
    """
    3 features → 1 label
    More realistic:
      - stronger noise
    """
    X = np.random.uniform(-5, 5, size=(n, 3))

    # More noise
    noise = np.random.normal(scale=5.0, size=(n, 1))

    y = (
        1.5 * X[:, [0]]
        - 2.3 * X[:, [1]]
        + 0.7 * X[:, [2]]
        + 4
        + noise
    )

    return np.hstack([X, y])


def generate_5f(n):
    """
    5 features → 1 label
    Realistic + larger noise.
    """
    X = np.random.uniform(-3, 3, size=(n, 5))

    # Even more noise
    noise = np.random.normal(scale=10.0, size=(n, 1))

    y = (
        0.5 * X[:, [0]] ** 2 +
        1.2 * X[:, [1]] -
        0.8 * X[:, [2]] +
        0.3 * X[:, [3]] +
        2.0 * X[:, [4]] +
        noise
    )

    return np.hstack([X, y])


# ============================================================
#               GENERATE ALL DATASETS
# ============================================================

if __name__ == "__main__":
    sizes = {
        "10k": 10000,
        "50k": 50000,
        "100k": 100000
    }

    print("\nGenerating Linear Regression Datasets...\n")

    for label, n in sizes.items():
        save_dataset(f"lr_1f_{n}.txt", generate_1f(n))
        save_dataset(f"lr_3f_{n}.txt", generate_3f(n))
        save_dataset(f"lr_5f_{n}.txt", generate_5f(n))

    print("\nAll linear regression datasets generated successfully!\n")
