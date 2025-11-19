import numpy as np


class LinearRegressionModel:
    def __init__(self, coef, intercept):
        """
        coef: shape (d,)
        intercept: scalar
        """
        self.coef_ = np.array(coef, dtype=float)
        self.intercept_ = float(intercept)

    def predict(self, X):
        """
        X shape: (n, d)
        """
        X = np.array(X, dtype=float)
        return X @ self.coef_ + self.intercept_
