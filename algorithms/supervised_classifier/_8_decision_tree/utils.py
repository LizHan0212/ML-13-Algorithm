import numpy as np

def load_tree_dataset():
    X = []
    y = []
    path = "training_data/decision_tree/tree_data.txt"

    with open(path, "r") as f:
        for line in f:
            weather, workload, candy, mood = line.strip().split(",")
            X.append([weather, workload, candy])
            y.append(mood)

    return np.array(X), np.array(y)
