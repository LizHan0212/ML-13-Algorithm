# ui/algorithm_pages/naive_bayes_page.py

import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from algorithms.supervised_classifier._5_naive_bayes.train import train_naive_bayes


def load_naive_bayes_page(main_frame, content_frame):

    main_frame.pack_forget()
    content_frame.pack(expand=True, fill="both")

    for w in content_frame.winfo_children():
        w.destroy()

    frame = ttk.Frame(content_frame, padding=20)
    frame.pack(expand=True, fill="both")

    # Left + plot
    left = ttk.Frame(frame)
    left.grid(row=0, column=0, sticky="nw", padx=20)

    plot_panel = ttk.Frame(frame)
    plot_panel.grid(row=0, column=1, sticky="nsew")
    frame.columnconfigure(1, weight=1)

    # Back button
    ttk.Button(left, text="← Back", command=lambda: back()).pack(anchor="w", pady=5)

    ttk.Label(left, text="Naive Bayes Classifier", font=("Arial", 16, "bold")).pack(pady=10)
    status = ttk.Label(left, text="")
    status.pack(pady=5)

    ttk.Label(
        left,
        text="Dataset: 3 features, 100 samples",
        font=("Arial", 10, "italic")
    ).pack(pady=(0, 10))

    model_holder = {"model": None, "X": None, "y": None}

    # Matplotlib
    fig = plt.Figure(figsize=(4, 4))
    ax = fig.add_subplot(111)
    canvas = FigureCanvasTkAgg(fig, master=plot_panel)
    canvas.get_tk_widget().pack(expand=True, fill="both")

    # Train button
    def train():
        status.config(text="Training...", foreground="orange")
        model, X, y = train_naive_bayes()
        model_holder["model"] = model
        model_holder["X"] = X
        model_holder["y"] = y

        status.config(text="Training complete.", foreground="green")
        update_plot()

    ttk.Button(left, text="Train Model", command=train).pack(pady=10)

    ttk.Label(left, text="-"*40).pack(pady=10)

    # Prediction UI
    ttk.Label(left, text="Prediction", font=("Arial", 14, "bold")).pack(pady=5)

    entries = []
    entry_frame = ttk.Frame(left)
    entry_frame.pack()

    d = 3  # always 3 features in our dataset
    ttk.Label(entry_frame, text="Enter 3 discrete values (0–4):").pack()
    for _ in range(d):
        e = ttk.Entry(entry_frame, width=10)
        e.pack(pady=1)
        entries.append(e)

    result_label = ttk.Label(left, text="Prediction: -")
    result_label.pack(pady=5)

    def predict():
        model = model_holder["model"]
        if model is None:
            result_label.config(text="Train first!", foreground="red")
            return
        try:
            vals = [int(e.get()) for e in entries]
            y = model.predict([vals])[0]
            result_label.config(text=f"Prediction: {y}", foreground="black")
        except:
            result_label.config(text="Invalid input", foreground="red")

    ttk.Button(left, text="Predict", command=predict).pack(pady=5)

    # ------- Plot -------
    def update_plot():
        ax.clear()
        X = model_holder["X"]
        y = model_holder["y"]

        # Only show feature1 vs feature2
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap="tab10", s=20, alpha=0.6)
        ax.set_title("Naive Bayes (Feature1 vs Feature2)")
        canvas.draw()

    def back():
        content_frame.pack_forget()
        main_frame.pack()
