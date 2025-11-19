# algorithms/supervised_classifier/_4_centroid_classifier/train.py

import numpy as np
from utils.dataset_readers import load_multi_feature_txt_file
from .utils import get_dataset_path

class CentroidModel:
    def __init__(self, centroids):
        self.centroids = centroids   # shape (K, d)

    def predict(self, X):
        X = np.array(X)
        # compute distances to all K centroids
        dists = np.linalg.norm(X[:, None, :] - self.centroids[None, :, :], axis=2)
        # return argmin
        return np.argmin(dists, axis=1)


def load_and_train_centroid(feature_count):
    path = get_dataset_path(feature_count)
    X, y = load_multi_feature_txt_file(path)
    classes = np.unique(y)

    centroids = []
    for c in classes:
        centroids.append(X[y.flatten() == c].mean(axis=0))

    centroids = np.array(centroids)

    return CentroidModel(centroids), X, y
