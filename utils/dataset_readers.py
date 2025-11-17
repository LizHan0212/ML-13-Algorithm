import re
import numpy as np


def load_multi_feature_txt_file(path: str):
    """
    Reads a dataset where each line is like:
        (x1, x2, ..., xN, y)

    Returns:
        X: shape (n, d)   -- features
        y: shape (n,)     -- label
    """
    X_list = []
    y_list = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Extract numbers inside parentheses
            nums = re.findall(r"[+-]?\d*\.?\d+", line)

            if not nums:
                continue

            nums = [float(v) for v in nums]

            # Last value is label (y)
            *features, y = nums

            X_list.append(features)
            y_list.append(y)

    X = np.array(X_list)
    y = np.array(y_list)

    return X, y
