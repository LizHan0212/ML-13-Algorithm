import matplotlib.pyplot as plt
import numpy as np


def plot_linear_regression(model, X, y, title="Linear Regression"):
    plt.figure(figsize=(6, 4))

    # Data points
    plt.scatter(X, y, color="blue", label="Data")

    # Regression line
    x_line = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
    y_pred = model.predict(x_line)
    plt.plot(x_line, y_pred, color="red", label="Fit line")

    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.tight_layout()
    return plt