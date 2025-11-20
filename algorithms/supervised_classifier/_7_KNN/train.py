
from .model import KNNModel
from .utils import load_knn_dataset

def train_knn(k):
    X, y = load_knn_dataset()
    return KNNModel(X, y, k)
