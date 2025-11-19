import numpy as np
from .model import LinearRegressionModel
from .utils import add_bias


# ============================================================
#                NORMAL EQUATION
# ============================================================

def normal_equation_train(X, y):
    Xb = add_bias(X)
    beta = np.linalg.pinv(Xb.T @ Xb) @ Xb.T @ y
    intercept = beta[0]
    coef = beta[1:]
    return LinearRegressionModel(coef, intercept)


# ============================================================
#                BATCH GRADIENT DESCENT
# ============================================================

def batch_gd_train(X, y, lr=0.001, iters=300):
    n, d = X.shape
    m = np.zeros(d)
    b = 0.0

    for _ in range(iters):
        y_pred = X @ m + b

        dm = (-2 / n) * (X.T @ (y - y_pred))
        db = (-2 / n) * np.sum(y - y_pred)

        m -= lr * dm
        b -= lr * db

    return LinearRegressionModel(m, b)


# ============================================================
#                STOCHASTIC GRADIENT DESCENT
# ============================================================

def sgd_train(X, y, lr=0.0001, iters=1):
    n, d = X.shape
    m = np.zeros(d)
    b = 0.0

    for _ in range(iters):
        for i in range(n):
            xi = X[i]
            yi = y[i]

            y_pred = m @ xi + b

            dm = -2 * xi * (yi - y_pred)
            db = -2 * (yi - y_pred)

            m -= lr * dm
            b -= lr * db

    return LinearRegressionModel(m, b)


# ============================================================
#            MINI-BATCH GRADIENT DESCENT
# ============================================================

def minibatch_train(X, y, batch_size=64, lr=0.001, epochs=10):
    n, d = X.shape
    m = np.zeros(d)
    b = 0.0

    for _ in range(epochs):
        idx = np.random.permutation(n)
        X = X[idx]
        y = y[idx]

        for i in range(0, n, batch_size):
            Xb = X[i:i + batch_size]
            yb = y[i:i + batch_size]

            y_pred = Xb @ m + b

            dm = (-2 / len(Xb)) * (Xb.T @ (yb - y_pred))
            db = (-2 / len(Xb)) * np.sum(yb - y_pred)

            m -= lr * dm
            b -= lr * db

    return LinearRegressionModel(m, b)


# ============================================================
#            MASTER TRAIN FUNCTION
# ============================================================

def train_linear_regression(X, y, algorithm):
    if algorithm == "Normal Equation":
        return normal_equation_train(X, y)

    if algorithm == "BGD":
        return batch_gd_train(X, y)

    if algorithm == "SGD":
        return sgd_train(X, y)

    if algorithm == "Mini-Batch (64)":
        return minibatch_train(X, y)

    raise ValueError("Unknown algorithm: " + algorithm)
