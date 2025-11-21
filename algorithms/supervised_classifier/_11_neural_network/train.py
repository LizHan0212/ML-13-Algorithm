import numpy as np
from .utils import load_mnist
from .model import SimpleNN

def train_mnist(epochs=3):
    X, y = load_mnist()

    # use 10k samples
    X = X[:10000]
    y = y[:10000]

    X = X.reshape(len(X), -1).astype(float) / 255.0

    model = SimpleNN(input_size=784, hidden=128, output=10)

    for e in range(epochs):
        z1, a1, a2 = model.forward(X)

        loss = -np.mean(np.log(a2[range(len(y)), y] + 1e-8))
        print(f"Epoch {e+1}, Loss = {loss:.4f}")

        model.backward(X, y, z1, a1, a2, lr=0.01)

    return model
