import os
import random

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "training_data", "decision_tree", "tree_data.txt")

# Encoding maps
weather_map = {"sunny": 0, "rain": 1}
inv_weather = {v: k for k, v in weather_map.items()}

workload_map = {"light": 0, "heavy": 1}
inv_workload = {v: k for k, v in workload_map.items()}

candy_map = {"yes": 0, "no": 1}
inv_candy = {v: k for k, v in candy_map.items()}

mood_map = {"happy": 0, "sad": 1}
inv_mood = {v: k for k, v in mood_map.items()}


def load_dataset():
    X = []
    y = []
    with open(DATA_PATH, "r") as f:
        for line in f:
            w, wl, c, m = line.strip().split(",")

            X.append([
                weather_map[w],
                workload_map[wl],
                candy_map[c]
            ])
            y.append(mood_map[m])

    return X, y


def majority(arr):
    return max(set(arr), key=arr.count)


def bootstrap_sample(X, y):
    n = len(X)
    idx = [random.randrange(n) for _ in range(n)]
    return [X[i] for i in idx], [y[i] for i in idx]


def pick_feature_subset():
    """Pick 2 out of 3 features."""
    feats = [0, 1, 2]
    random.shuffle(feats)
    return feats[:2]
