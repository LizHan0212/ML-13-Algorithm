# ui/algorithm_pages/perceptron_page.py

import tkinter as tk
from tkinter import ttk
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from utils.dataset_readers import load_multi_feature_txt_file
from algorithms.supervised_classifier._3_perceptron.utils import get_dataset_path
from algorithms.supervised_classifier._3_perceptron.train import train_perceptron


def load_perceptron_page(main_frame, content_frame):
    main_frame.pack_forget()
    content_frame.pack(expand=True, fill="both")
    for w in content_frame.winfo_children():
        w.destroy()

    f = ttk.Frame(content_frame)
    f.pack(expand=True, fill="both")

    left = ttk.Frame(f)
    left.grid(row=0, column=0, padx=15, sticky="nw")

    right = ttk.Frame(f)
    right.grid(row=0, column=1, sticky="nsew")
    f.columnconfigure(1, weight=1)

    ttk.Button(left, text="← Back", command=lambda: _back(main_frame, content_frame)).pack()

    ttk.Label(left, text="Perceptron", font=("Arial", 14, "bold")).pack(pady=5)
    ttk.Label(left, text="One Feature • 10k Samples").pack(pady=2)
    ttk.Label(left, text="Optimization Algorithm: Perceptron Learning Rule").pack(pady=2)

    status = ttk.Label(left, text="")
    status.pack()

    model = {"obj": None}
    X_cache = None
    y_cache = None

    def train():
        status.config(text="Training...", foreground="gold")
        path = get_dataset_path()
        X, y = load_multi_feature_txt_file(path)
        nonlocal X_cache, y_cache
        X_cache, y_cache = X, y
        model["obj"] = train_perceptron(X, y)
        status.config(text="Done", foreground="green")
        update_plot()

    ttk.Button(left, text="Train", command=train).pack(pady=5)

    # ---- Separator ----
    ttk.Separator(left, orient="horizontal").pack(fill="x", pady=10)

    fig = Figure(figsize=(4, 4))
    ax = fig.add_subplot(111)
    canvas = FigureCanvasTkAgg(fig, master=right)
    canvas.get_tk_widget().pack(expand=True, fill="both")

    def update_plot():
        ax.clear()
        m = model["obj"]
        if m is None:
            canvas.draw()
            return

        Xs = X_cache[:200, 0]
        ys = y_cache[:200]
        ax.scatter(Xs, ys, c=ys, cmap="bwr", s=15)

        w = m.coef_[0]
        b = m.intercept_
        xs = np.linspace(min(Xs), max(Xs), 100)
        ax.plot(xs, -(w * xs + b), "k--")

        canvas.draw()

    ttk.Label(left, text="Prediction").pack(pady=5)
    entry = ttk.Entry(left, width=10)
    entry.pack()

    pred_label = ttk.Label(left, text="")
    pred_label.pack()

    def predict():
        if model["obj"] is None:
            pred_label.config(text="Train first", foreground="red")
            return
        x = float(entry.get())
        pred = model["obj"].predict([[x]])[0]
        pred_label.config(text=f"class = {pred}")

    ttk.Button(left, text="Predict", command=predict).pack(pady=5)


def _back(main_frame, content_frame):
    content_frame.pack_forget()
    main_frame.pack()
