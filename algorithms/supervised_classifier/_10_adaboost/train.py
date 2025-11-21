
import numpy as np
from .utils import load_dataset


def build_stump(X, y, weights):
    """
    One-level decision stump.
    """
    X = np.array(X)
    y = np.array(y)

    best_err = 999
    best_stump = None

    n_samples, n_features = X.shape

    for f in range(n_features):
        values = [0, 1]     # because categorical with 2 values each

        pred_map = {}
        for v in values:
            pred_map[v] = None

        # try assigning each value → 0 or 1
        # brute force tiny space: 2^2 = 4 combos
        from itertools import product
        for combo in product([0, 1], repeat=len(values)):
            pred_map = {val: combo[i] for i, val in enumerate(values)}

            preds = np.array([pred_map[row[f]] for row in X])
            miss = preds != y
            err = np.sum(weights[miss])

            if err < best_err:
                best_err = err
                best_stump = {
                    "feature": f,
                    "pred_map": pred_map
                }

    return best_stump, best_err


def train_adaboost(T=5):
    X, y = load_dataset()
    X = np.array(X)
    y = np.array(y)

    # convert labels 0→-1, 1→+1
    y2 = np.where(y == 1, 1, -1)

    n = len(X)
    weights = np.ones(n) / n

    stumps = []
    alphas = []

    for t in range(T):
        stump, err = build_stump(X, y, weights)

        # avoid division by zero or degenerate stump
        err = max(err, 1e-6)
        alpha = 0.5 * np.log((1 - err) / err)

        # predict with stump
        preds = np.array([stump["pred_map"][row[stump["feature"]]] for row in X])
        preds2 = np.where(preds == 1, 1, -1)

        # update weights
        weights *= np.exp(-alpha * y2 * preds2)
        weights /= np.sum(weights)

        stumps.append(stump)
        alphas.append(alpha)

    return stumps, alphas, X, y
