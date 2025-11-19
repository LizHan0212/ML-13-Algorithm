# algorithms/supervised_classifier/_5_naive_bayes/train.py

import numpy as np
from utils.dataset_readers import load_multi_feature_txt_file
from .utils import get_dataset_path
from .model import NaiveBayesModel

def train_naive_bayes():
    path = get_dataset_path()
    X, y = load_multi_feature_txt_file(path)

    # Convert features + labels to integers (CRITICAL FIX)
    X = X.astype(int)
    y = y.astype(int)

    classes = np.unique(y)
    K = len(classes)
    N, d = X.shape

    num_bins = int(X.max()) + 1

    priors = np.zeros(K)
    likelihoods = np.zeros((K, d, num_bins))

    for c in classes:
        Xc = X[y.flatten() == c]
        priors[c] = len(Xc) / N

        for j in range(d):
            counts = np.bincount(Xc[:, j], minlength=num_bins) + 1
            likelihoods[c, j, :] = counts / counts.sum()

    return NaiveBayesModel(priors, likelihoods), X, y
