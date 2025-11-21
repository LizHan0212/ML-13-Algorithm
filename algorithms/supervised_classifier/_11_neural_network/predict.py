import numpy as np
from .model import SimpleNN

def predict_digit(model, img28):
    x = img28.reshape(1, 784) / 255.0
    return int(model.predict(x)[0])
