
import os

DATA_PATH = os.path.join(
    "training_data",
    "decision_tree",
    "tree_data.txt"
)

weather_map  = {"sunny": 0, "rain": 1}
work_map     = {"light": 0, "heavy": 1}
candy_map    = {"yes": 0, "no": 1}
mood_map     = {"happy": 0, "sad": 1}

def load_dataset():
    X = []
    y = []

    with open(DATA_PATH, "r") as f:
        for line in f:
            parts = line.strip().split(",")

            w = weather_map[parts[0]]
            wk = work_map[parts[1]]
            c = candy_map[parts[2]]
            label = mood_map[parts[3]]

            X.append([w, wk, c])
            y.append(label)

    return X, y
