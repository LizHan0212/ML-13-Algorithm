# algorithms/supervised_classifier/_3_perceptron/utils.py

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
TRAIN_DIR = os.path.join(BASE_DIR, "training_data", "perceptron")

def get_dataset_path():
    return os.path.join(TRAIN_DIR, "perceptron_1f_10k.txt")
