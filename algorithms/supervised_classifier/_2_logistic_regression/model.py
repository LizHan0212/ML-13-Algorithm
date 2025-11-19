import numpy as np


class LogisticRegressionModel:
    def __init__(self, coef, intercept):
        """
        coef: shape (d,)
        intercept: scalar
        """
        self.coef_ = np.array(coef, dtype=float)
        self.intercept_ = float(intercept)

    def predict_proba(self, X):
        """
        Returns probability P(y = 1 | x)
        X shape: (n, d)
        """
        X = np.array(X, dtype=float)
        z = X @ self.coef_ + self.intercept_
        return 1 / (1 + np.exp(-z))

    def predict(self, X, threshold=0.5):
        """ Returns 0/1 prediction. """
        p = self.predict_proba(X)
        return (p >= threshold).astype(int)
