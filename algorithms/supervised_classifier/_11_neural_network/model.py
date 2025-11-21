import numpy as np

class SimpleNN:
    def __init__(self, input_size=784, hidden=128, output=10):
        self.W1 = np.random.randn(input_size, hidden) * 0.01
        self.b1 = np.zeros((1, hidden))
        self.W2 = np.random.randn(hidden, output) * 0.01
        self.b2 = np.zeros((1, output))

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_deriv(x):
        return (x > 0).astype(float)

    @staticmethod
    def softmax(z):
        exp = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)

    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = self.relu(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = self.softmax(z2)
        return z1, a1, a2

    def backward(self, X, y, z1, a1, a2, lr=0.01):
        m = X.shape[0]

        dz2 = a2.copy()
        dz2[range(m), y] -= 1
        dz2 /= m

        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = dz2 @ self.W2.T
        dz1 = da1 * self.relu_deriv(z1)

        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2

    def predict(self, X):
        _, _, out = self.forward(X)
        return np.argmax(out, axis=1)
