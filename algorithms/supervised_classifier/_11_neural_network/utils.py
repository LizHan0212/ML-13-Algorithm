import os
import urllib.request
import numpy as np
from PIL import Image
from PIL import ImageGrab

MNIST_DIR = "training_data/mnist"

URLS = {
    "train_images": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
}

FILES = {
    "train_images": "train-images-idx3-ubyte",
    "train_labels": "train-labels-idx1-ubyte",
}

def download_mnist():
    os.makedirs(MNIST_DIR, exist_ok=True)

    for key in URLS:
        gz_path = os.path.join(MNIST_DIR, FILES[key] + ".gz")
        raw_path = os.path.join(MNIST_DIR, FILES[key])

        if not os.path.exists(raw_path):
            print(f"Downloading {key}...")
            urllib.request.urlretrieve(URLS[key], gz_path)

            import gzip
            with gzip.open(gz_path, "rb") as f_in, open(raw_path, "wb") as f_out:
                f_out.write(f_in.read())

def load_mnist():
    download_mnist()

    def load_images(path):
        with open(path, "rb") as f:
            f.read(16)
            data = np.frombuffer(f.read(), dtype=np.uint8)
            return data.reshape(-1, 28, 28)

    def load_labels(path):
        with open(path, "rb") as f:
            f.read(8)
            return np.frombuffer(f.read(), dtype=np.uint8)

    X = load_images(os.path.join(MNIST_DIR, FILES["train_images"]))
    y = load_labels(os.path.join(MNIST_DIR, FILES["train_labels"]))

    return X, y

def preprocess_canvas_image(canvas):
    # Get canvas position on screen
    canvas.update()
    x = canvas.winfo_rootx()
    y = canvas.winfo_rooty()
    w = x + canvas.winfo_width()
    h = y + canvas.winfo_height()

    # Capture the screen region
    from PIL import ImageGrab
    img = ImageGrab.grab(bbox=(x, y, w, h))

    # Convert to grayscale
    img = img.convert("L")

    # Resize to 28x28
    img = img.resize((28, 28))

    # Invert (white background → black strokes)
    from PIL import ImageOps
    img = ImageOps.invert(img)

    # Normalize
    arr = np.array(img).astype(float) / 255.0

    return arr.reshape(1, 784)
