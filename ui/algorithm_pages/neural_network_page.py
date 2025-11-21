import tkinter as tk
from tkinter import ttk
import numpy as np
from PIL import Image, ImageDraw

from algorithms.supervised_classifier._11_neural_network.train import train_mnist
from algorithms.supervised_classifier._11_neural_network.utils import preprocess_canvas_image

def load_neural_network_page(main_frame, content_frame):

    main_frame.pack_forget()
    content_frame.pack(expand=True, fill="both")

    for w in content_frame.winfo_children():
        w.destroy()

    frame = ttk.Frame(content_frame, padding=20)
    frame.pack(expand=True, fill="both")

    left = ttk.Frame(frame)
    left.grid(row=0, column=0, sticky="nw", padx=20)

    draw_panel = ttk.Frame(frame)
    draw_panel.grid(row=0, column=1, sticky="nsew")
    frame.columnconfigure(1, weight=1)

    ttk.Button(left, text="← Back",
               command=lambda: back()).pack(anchor="w", pady=5)

    ttk.Label(left, text="Neural Network (Digit Recognition)", font=("Arial", 16, "bold")).pack(pady=10)

    status = ttk.Label(left, text="")
    status.pack(pady=5)

    model_holder = {"model": None}

    def train_model():
        status.config(text="Training... (3 epochs, 10k samples)", foreground="orange")
        frame.update_idletasks()

        model = train_mnist()
        model_holder["model"] = model

        status.config(text="Training complete!", foreground="green")

    ttk.Button(left, text="Train Model", command=train_model).pack(pady=10)

    ttk.Label(left, text="-"*40).pack(pady=10)

    ttk.Label(left, text="Draw a digit (0–9)", font=("Arial", 14, "bold")).pack(pady=5)

    canvas = tk.Canvas(draw_panel, width=280, height=280, bg="white")
    canvas.pack()

    img = Image.new("L", (280, 280), 255)
    draw = ImageDraw.Draw(img)

    def paint(event):
        x, y = event.x, event.y
        r = 10
        canvas.create_oval(x-r, y-r, x+r, y+r, fill="black")
        draw.ellipse([x-r, y-r, x+r, y+r], fill=0)

    canvas.bind("<B1-Motion>", paint)

    result_label = ttk.Label(left, text="Prediction: -", font=("Arial", 14))
    result_label.pack(pady=5)

    def predict():
        if model_holder["model"] is None:
            result_label.config(text="Train first!", foreground="red")
            return

        arr = preprocess_canvas_image(canvas)
        pred = model_holder["model"].predict(arr)[0]

        result_label.config(text=f"Prediction: {pred}", foreground="black")

    ttk.Button(left, text="Predict", command=predict).pack(pady=5)

    def back():
        content_frame.pack_forget()
        main_frame.pack()
