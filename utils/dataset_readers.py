import re
import numpy as np


def load_xy_txt_file(path: str):
    """
    Reads a text file containing coordinates in the format:
        (x,y),(x,y),(x,y)
    Returns:
        X: numpy array shape (n,1)
        y: numpy array shape (n,)
    """
    with open(path, "r") as f:
        content = f.read().strip()

    # Find all (x,y) inside entire text
    pairs = re.findall(r"\(\s*([+-]?\d*\.?\d+)\s*,\s*([+-]?\d*\.?\d+)\s*\)", content)

    if not pairs:
        raise ValueError("No valid (x,y) pairs found in file")

    xs = []
    ys = []
    for x, y in pairs:
        xs.append(float(x))
        ys.append(float(y))

    X = np.array(xs).reshape(-1, 1)
    y = np.array(ys)

    return X, y