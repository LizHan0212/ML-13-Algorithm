# algorithms/supervised_classifier/_3_perceptron/model.py

import numpy as np

class PerceptronModel:
    def __init__(self, coef, intercept):
        self.coef_ = np.array(coef)
        self.intercept_ = float(intercept)

    def predict(self, X):
        X = np.array(X)
        return (X @ self.coef_ + self.intercept_ >= 0).astype(int)
