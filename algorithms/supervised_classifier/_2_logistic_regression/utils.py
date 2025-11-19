import os
import numpy as np

# Base path for dataset directory
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
    )
)

TRAIN_DIR = os.path.join(BASE_DIR, "training_data", "logistic_regression")


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def logistic_loss(y, p):
    """
    Binary cross-entropy loss.
    """
    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def get_dataset_path(feature_count, dataset_size):
    size_map = {
        "10k": "10000",
        "50k": "50000",
        "100k": "100000"
    }
    filename = f"lg_{feature_count}f_{size_map[dataset_size]}.txt"
    return os.path.join(TRAIN_DIR, filename)
