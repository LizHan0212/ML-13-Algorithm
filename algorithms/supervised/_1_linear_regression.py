import os
import numpy as np
from utils.dataset_readers import load_multi_feature_txt_file


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
TRAIN_DIR = os.path.join(BASE_DIR, "training_data", "linear_regression")


# ============================================================
#                HELPER: SELECT DATASET FILE
# ============================================================

def get_dataset_path(feature_count, dataset_size):
    size_str = {
        "10k": "10000",
        "50k": "50000",
        "100k": "100000"
    }[dataset_size]

    feature_str = f"{feature_count}f"

    fname = f"lr_{feature_str}_{size_str}.txt"

    return os.path.join(TRAIN_DIR, fname)


# ============================================================
#           MODEL CLASS (Unified for all training types)
# ============================================================

class LRModel:
    def __init__(self, coef, intercept):
        self.coef_ = np.array(coef)      # shape (d,)
        self.intercept_ = float(intercept)

    def predict(self, X):
        """
        X shape: (n, d)
        """
        X = np.array(X)
        return X @ self.coef_ + self.intercept_


# ============================================================
#           NORMAL EQUATION (Closed-form solution)
# ============================================================

def normal_equation_train(X, y):
    n = X.shape[0]

    # Add bias column
    Xb = np.hstack([np.ones((n, 1)), X])  # (n, d+1)

    # β = (X^T X)^(-1) X^T y
    beta = np.linalg.pinv(Xb.T @ Xb) @ Xb.T @ y

    intercept = beta[0]
    coef = beta[1:]

    return LRModel(coef, intercept)


# ============================================================
#                   BATCH GRADIENT DESCENT
# ============================================================

def batch_gd_train(X, y, lr=0.001, iters=200):
    n, d = X.shape

    m = np.zeros(d)
    b = 0.0

    for _ in range(iters):
        y_pred = X @ m + b

        dm = (-2/n) * (X.T @ (y - y_pred))
        db = (-2/n) * np.sum(y - y_pred)

        m -= lr * dm
        b -= lr * db

    return LRModel(m, b)


# ============================================================
#                   STOCHASTIC GRADIENT DESCENT
# ============================================================

def sgd_train(X, y, lr=0.001, iters=1):
    n, d = X.shape

    m = np.zeros(d)
    b = 0.0

    for _ in range(iters):
        for i in range(n):
            x_i = X[i]
            y_i = y[i]

            y_pred = m @ x_i + b

            dm = -2 * x_i * (y_i - y_pred)
            db = -2 * (y_i - y_pred)

            m -= lr * dm
            b -= lr * db

    return LRModel(m, b)


# ============================================================
#                    MINI-BATCH GRADIENT DESCENT
# ============================================================

def mini_batch_train(X, y, batch_size=64, lr=0.001, epochs=10):
    n, d = X.shape

    m = np.zeros(d)
    b = 0.0

    for _ in range(epochs):

        # shuffle each epoch
        idx = np.random.permutation(n)
        X = X[idx]
        y = y[idx]

        for i in range(0, n, batch_size):
            Xb = X[i:i+batch_size]
            yb = y[i:i+batch_size]

            y_pred = Xb @ m + b

            dm = (-2/len(Xb)) * (Xb.T @ (yb - y_pred))
            db = (-2/len(Xb)) * np.sum(yb - y_pred)

            m -= lr * dm
            b -= lr * db

    return LRModel(m, b)


# ============================================================
#                      MASTER TRAIN FUNCTION
# ============================================================

def load_and_train_model(feature_count, dataset_size, algorithm):
    path = get_dataset_path(feature_count, dataset_size)

    X, y = load_multi_feature_txt_file(path)

    if algorithm == "Normal Equation":
        return normal_equation_train(X, y)

    elif algorithm == "BGD":
        return batch_gd_train(X, y, lr=0.001, iters=400)

    elif algorithm == "SGD":
        return sgd_train(X, y, lr=0.0001, iters=1)

    elif algorithm == "Mini-Batch (64)":
        return mini_batch_train(X, y, batch_size=64, lr=0.001, epochs=10)

    else:
        raise ValueError("Unknown training method.")
