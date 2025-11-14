import os
import numpy as np
from utils.dataset_readers import load_xy_txt_file


# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
TRAINING_DIR = os.path.join(BASE_DIR, "training_data")

DATASETS = {
    "Dataset 1 (clean linear)": os.path.join(TRAINING_DIR, "linear_regression_training_data_1.txt"),
    "Dataset 2 (noisy linear)": os.path.join(TRAINING_DIR, "linear_regression_training_data_2.txt"),
    "Dataset 3 (non-linear)": os.path.join(TRAINING_DIR, "linear_regression_training_data_3.txt"),
}


class ManualLinearRegression:
    def __init__(self):
        self.coef_ = None      # slope
        self.intercept_ = None # intercept

    def fit(self, X, y):
        """
        Perform linear regression by solving the Normal Equation:
            β = (X^T X)^(-1) X^T y
        """
        X = np.array(X)
        y = np.array(y)

        # Add column of ones for intercept
        X_b = np.c_[np.ones((X.shape[0], 1)), X]   # shape (n,2)

        # Normal Equation
        beta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

        self.intercept_ = beta[0]
        self.coef_ = beta[1]

    def predict(self, X):
        """
        Compute predictions:
            y = intercept + slope * x
        """
        X = np.array(X).reshape(-1)
        return self.intercept_ + self.coef_ * X


def train_linear_regression_from_file(filepath: str):
    X, y = load_xy_txt_file(filepath)

    model = ManualLinearRegression()
    model.fit(X, y)

    return model, X, y
