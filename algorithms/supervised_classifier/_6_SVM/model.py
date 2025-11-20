
import numpy as np

class SVMModel:
    def __init__(self, w, b):
        self.w = w
        self.b = b

    def hypothesis(self, X):
        return np.dot(X, self.w) + self.b

    def predict(self, X):
        scores = self.hypothesis(X)
        return (scores >= 0).astype(int)
