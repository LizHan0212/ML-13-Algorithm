def create_linear_data(n=50):
    # Generate a simple linear dataset (y = 2x + noise) with n samples.
    import numpy as np
    X = np.linspace(0, 1, n).reshape(-1, 1)
    y = 2 * X[:, 0] + 0.3 * np.random.randn(n)
    return X, y