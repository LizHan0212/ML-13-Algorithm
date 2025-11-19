# algorithms/supervised_classifier/_3_perceptron/train.py

import numpy as np
from .model import PerceptronModel


def train_perceptron(X, y, lr=1.0, epochs=10):
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0

    # Perceptron Learning Rule
    for _ in range(epochs):
        for i in range(n):
            x_i = X[i]
            y_i = y[i]
            pred = 1 if (w @ x_i + b) >= 0 else 0
            if pred != y_i:
                update = lr * (y_i - pred)
                w += update * x_i
                b += update

    return PerceptronModel(w, b)
