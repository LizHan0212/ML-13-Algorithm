import numpy as np

class KMeansModel:
    def __init__(self, centroids, assignments, errors):
        self.centroids = centroids
        self.assignments = assignments
        self.errors = errors  # SSE per iteration

    def predict(self, X):
        # returns index of nearest centroid
        dists = np.linalg.norm(X - self.centroids[:, None], axis=2)
        return np.argmin(dists, axis=0)
