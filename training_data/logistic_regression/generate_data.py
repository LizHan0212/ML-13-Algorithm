import numpy as np
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def save_dataset(filename, data):
    path = os.path.join(THIS_DIR, filename)
    with open(path, "w") as f:
        for row in data:
            f.write("(" + ",".join(f"{v:.4f}" for v in row) + ")\n")
    print("Saved:", filename)


# ======================================================
#                2-FEATURE (HEAVY NOISE)
# ======================================================

def generate_2f(n):
    # Base features
    X = np.random.uniform(-6, 6, size=(n, 2))

    # True weights (used to create underlying "truth")
    true_w = np.array([1.4, -1.1])
    true_b = 0.3

    # Logit
    z = X @ true_w + true_b

    # Sigmoid base probability
    base_prob = 1 / (1 + np.exp(-z))

    # Add strong noise
    noise = (
        np.random.normal(scale=0.35, size=n)       # heavy Gaussian
        + np.random.uniform(-0.25, 0.25, size=n)   # uniform noise
    )

    # Mixed probability
    probs = np.clip(base_prob + noise, 0, 1)

    # Labels
    y = (probs >= 0.5).astype(int).reshape(-1, 1)

    return np.hstack([X, y])


# ======================================================
#                3-FEATURE (HEAVY NOISE)
# ======================================================

def generate_3f(n):
    X = np.random.uniform(-4.5, 4.5, size=(n, 3))

    # Nonlinear underlying relation
    z = (
        1.8 * X[:, 0]
        - 1.2 * X[:, 1]**2
        + 1.5 * np.sin(X[:, 2])
        + 0.7
    )

    base_prob = 1 / (1 + np.exp(-z))

    # Stronger noise than 2f
    noise = (
        np.random.normal(scale=0.45, size=n) +
        np.random.uniform(-0.30, 0.30, size=n)
    )

    probs = np.clip(base_prob + noise, 0, 1)

    y = (probs >= 0.5).astype(int).reshape(-1, 1)

    return np.hstack([X, y])


# ======================================================
#                5-FEATURE (STRONGEST NOISE)
# ======================================================

def generate_5f(n):
    X = np.random.uniform(-3.5, 3.5, size=(n, 5))

    # Complex nonlinear logit
    z = (
        1.0 * X[:, 0]
        - 1.6 * np.cos(X[:, 1])
        + 0.9 * X[:, 2] * X[:, 3]
        - 0.8 * X[:, 4]**2
        + 0.5
    )

    base_prob = 1 / (1 + np.exp(-z))

    # Very heavy noise for difficult classification
    noise = (
        np.random.normal(scale=0.55, size=n) +
        np.random.uniform(-0.35, 0.35, size=n)
    )

    probs = np.clip(base_prob + noise, 0, 1)

    y = (probs >= 0.5).astype(int).reshape(-1, 1)

    return np.hstack([X, y])


# ======================================================
#                GENERATE ALL DATASETS
# ======================================================

if __name__ == "__main__":
    sizes = {
        "10k": 10000,
        "50k": 50000,
        "100k": 100000
    }

    print("\nGenerating noisy logistic regression datasets...\n")

    for label, n in sizes.items():
        save_dataset(f"lg_2f_{n}.txt", generate_2f(n))
        save_dataset(f"lg_3f_{n}.txt", generate_3f(n))
        save_dataset(f"lg_5f_{n}.txt", generate_5f(n))

    print("\nAll logistic regression datasets generated with heavy noise!\n")
