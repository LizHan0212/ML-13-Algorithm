import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
TRAIN_DIR = os.path.join(BASE_DIR, "training_data", "naive_bayes")

def get_dataset_path():
    return os.path.join(TRAIN_DIR, "nb_3f_600.txt")