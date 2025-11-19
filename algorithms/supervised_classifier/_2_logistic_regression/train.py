import numpy as np
from .model import LogisticRegressionModel
from .utils import sigmoid


# ======================================================
#                 BATCH GRADIENT DESCENT
# ======================================================

def train_bgd(X, y, lr=0.01, iters=200):
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0

    for _ in range(iters):
        z = X @ w + b
        p = sigmoid(z)

        dw = (1/n) * (X.T @ (p - y))
        db = (1/n) * np.sum(p - y)

        w -= lr * dw
        b -= lr * db

    return LogisticRegressionModel(w, b)


# ======================================================
#              STOCHASTIC GRADIENT DESCENT
# ======================================================

def train_sgd(X, y, lr=0.01, epochs=1):
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0

    for _ in range(epochs):
        for i in range(n):
            xi = X[i]
            yi = y[i]

            p = sigmoid(w @ xi + b)

            dw = (p - yi) * xi
            db = (p - yi)

            w -= lr * dw
            b -= lr * db

    return LogisticRegressionModel(w, b)


# ======================================================
#              MINI-BATCH GRADIENT DESCENT
# ======================================================

def train_minibatch(X, y, lr=0.01, batch_size=64, epochs=10):
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0

    for _ in range(epochs):
        idx = np.random.permutation(n)
        X = X[idx]
        y = y[idx]

        for i in range(0, n, batch_size):
            Xb = X[i:i+batch_size]
            yb = y[i:i+batch_size]

            p = sigmoid(Xb @ w + b)

            dw = (1/len(Xb)) * (Xb.T @ (p - yb))
            db = (1/len(Xb)) * np.sum(p - yb)

            w -= lr * dw
            b -= lr * db

    return LogisticRegressionModel(w, b)


# ======================================================
#            MASTER TRAIN FUNCTION (NO NEWTON)
# ======================================================

def train_logistic_regression(X, y, algorithm):
    if algorithm == "BGD":
        return train_bgd(X, y)

    if algorithm == "SGD":
        return train_sgd(X, y)

    if algorithm == "Mini-Batch (64)":
        return train_minibatch(X, y)

    raise ValueError("Unknown training method.")
