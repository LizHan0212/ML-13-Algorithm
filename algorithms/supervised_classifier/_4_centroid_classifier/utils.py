# algorithms/supervised_classifier/_4_centroid_classifier/utils.py

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
TRAIN_DIR = os.path.join(BASE_DIR, "training_data", "centroid_classifier")

def get_dataset_path(feature_count):
    fname = f"centroid_{feature_count}f_100.txt"
    return os.path.join(TRAIN_DIR, fname)
