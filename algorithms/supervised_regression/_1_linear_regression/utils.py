import os
import numpy as np

# Determine base path of project
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
    )
)

# Training data directory
TRAIN_DIR = os.path.join(BASE_DIR, "training_data", "linear_regression")


def add_bias(X):
    n = X.shape[0]
    return np.hstack([np.ones((n, 1)), X])


def mse_loss(y, y_pred):
    return np.mean((y - y_pred) ** 2)


def get_dataset_path(feature_count: int, dataset_size: str):
    """
    Builds the correct dataset filename based on feature count and size.
    Ex:
        feature_count = 3
        dataset_size = "50k"
        -> lr_3f_50000.txt
    """
    size_map = {
        "10k": "10000",
        "50k": "50000",
        "100k": "100000"
    }

    filename = f"lr_{feature_count}f_{size_map[dataset_size]}.txt"
    return os.path.join(TRAIN_DIR, filename)
