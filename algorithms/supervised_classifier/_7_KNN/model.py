
import numpy as np

class KNNModel:
    def __init__(self, X, y, k):
        self.X = X
        self.y = y
        self.k = k

    def predict(self, Xquery):
        preds = []
        for q in Xquery:
            dist = np.linalg.norm(self.X - q, axis=1)
            idx = np.argsort(dist)[:self.k]
            labels = self.y[idx]
            pred = np.bincount(labels).argmax()
            preds.append(pred)
        return np.array(preds)
