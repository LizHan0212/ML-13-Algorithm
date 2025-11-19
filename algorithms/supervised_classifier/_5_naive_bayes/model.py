
import numpy as np

class NaiveBayesModel:
    def __init__(self, priors, likelihoods):
        self.priors = priors            # shape: (K,)
        self.likelihoods = likelihoods  # shape: (K, d, num_bins)

    def predict(self, X):
        X = np.array(X)
        preds = []

        for x in X:
            log_probs = []
            for c in range(len(self.priors)):
                lp = np.log(self.priors[c])
                for j, val in enumerate(x):
                    lp += np.log(self.likelihoods[c, j, val])
                log_probs.append(lp)

            preds.append(np.argmax(log_probs))

        return np.array(preds)
