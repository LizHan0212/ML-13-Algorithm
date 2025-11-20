
import numpy as np
from .model import SVMModel
from .utils import load_svm_dataset


def train_svm(C=1.0, lr=0.001, epochs=200):
    """
    Soft-margin linear SVM using sub-gradient descent.
    Implements:

        J(w,b) = 1/2 ||w||^2 + C * sum( max(0, 1 − y_i (w·x_i + b)) )

    y must be in {-1, +1}.
    """

    X, y = load_svm_dataset()
    y2 = np.where(y == 1, 1, -1)  # convert {0,1} to {-1,+1}

    n, d = X.shape

    w = np.zeros(d)
    b = 0.0

    for _ in range(epochs):
        for i in range(n):
            margin = y2[i] * (np.dot(w, X[i]) + b)

            if margin >= 1:
                # Only regularization term contributes
                dw = w
                db = 0
            else:
                # Hinge is active
                dw = w - C * y2[i] * X[i]
                db = -C * y2[i]

            # Update parameters
            w -= lr * dw
            b -= lr * db

    return SVMModel(w, b)
