import numpy as np
from algorithms.unsupervised._1_kmeans.model import KMeansModel
from algorithms.unsupervised._1_kmeans.utils import load_kmeans_data

def train_kmeans(k, iters=10):
    X = load_kmeans_data()
    n = len(X)

    # Randomly choose K initial centroids
    np.random.seed(42)
    centroids = X[np.random.choice(n, k, replace=False)]

    errors = []

    for _ in range(iters):
        # Compute distances (K x N)
        dists = np.linalg.norm(X - centroids[:, None], axis=2)

        # Assign clusters
        assignments = np.argmin(dists, axis=0)

        # Update centroids
        for j in range(k):
            pts = X[assignments == j]
            if len(pts) > 0:
                centroids[j] = pts.mean(axis=0)

        # Compute SSE
        sse = 0
        for j in range(k):
            pts = X[assignments == j]
            sse += np.sum((pts - centroids[j])**2)
        errors.append(sse)

    return KMeansModel(centroids, assignments, errors)
