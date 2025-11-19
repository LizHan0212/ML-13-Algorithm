# ui/algorithm_pages/centroid_classifier_page.py

import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from algorithms.supervised_classifier._4_centroid_classifier.train import load_and_train_centroid


def load_centroid_classifier_page(main_frame, content_frame):

    # Switch UI
    main_frame.pack_forget()
    content_frame.pack(expand=True, fill="both")

    for w in content_frame.winfo_children():
        w.destroy()

    frame = ttk.Frame(content_frame, padding=20)
    frame.pack(expand=True, fill="both")

    left = ttk.Frame(frame)
    left.grid(row=0, column=0, sticky="nw", padx=20)

    plot_panel = ttk.Frame(frame)
    plot_panel.grid(row=0, column=1, sticky="nsew")
    frame.columnconfigure(1, weight=1)

    # Back button
    ttk.Button(left, text="← Back", command=lambda: back()).pack(anchor="w", pady=5)

    ttk.Label(left, text="Centroid Classifier", font=("Arial", 16, "bold")).pack(pady=10)

    # Feature count (only 1 or 3)
    ttk.Label(left, text="Number of Features:").pack(anchor="w")
    f_var = tk.StringVar(value="1")
    ttk.Combobox(left, textvariable=f_var, values=["1", "3"], width=10, state="readonly").pack(anchor="w")

    status = ttk.Label(left, text="")
    status.pack(pady=5)

    model_holder = {"model": None, "X": None, "y": None}

    # Matplotlib
    fig = plt.Figure(figsize=(4, 4))
    ax = fig.add_subplot(111)
    canvas = FigureCanvasTkAgg(fig, master=plot_panel)
    canvas.get_tk_widget().pack(expand=True, fill="both")

    # Train button
    def train():
        status.config(text="Training...", foreground="orange")
        fcount = int(f_var.get())

        model, X, y = load_and_train_centroid(fcount)
        model_holder["model"] = model
        model_holder["X"] = X
        model_holder["y"] = y

        status.config(text="Training complete.", foreground="green")
        update_plot()

    ttk.Button(left, text="Train Model", command=train).pack(pady=10)

    ttk.Label(left, text="-"*40).pack(pady=10)

    # Prediction
    ttk.Label(left, text="Prediction", font=("Arial", 14, "bold")).pack()

    entry_frame = ttk.Frame(left)
    entry_frame.pack()

    entries = []

    def rebuild_entries(*_):
        for w in entry_frame.winfo_children():
            w.destroy()
        entries.clear()

        count = int(f_var.get())
        ttk.Label(entry_frame, text=f"Enter {count} values:").pack()
        for _ in range(count):
            e = ttk.Entry(entry_frame, width=12)
            e.pack(pady=1)
            entries.append(e)

    f_var.trace_add("write", rebuild_entries)
    rebuild_entries()

    result_label = ttk.Label(left, text="Prediction: -")
    result_label.pack(pady=5)

    def predict():
        model = model_holder["model"]
        if model is None:
            result_label.config(text="Train first!", foreground="red")
            return

        try:
            vals = [float(e.get()) for e in entries]
            y = model.predict([vals])[0]
            result_label.config(text=f"Prediction: {y}", foreground="black")
        except:
            result_label.config(text="Invalid input", foreground="red")

    ttk.Button(left, text="Predict", command=predict).pack(pady=5)

    # Plot update
    def update_plot():
        ax.clear()
        model = model_holder["model"]
        X = model_holder["X"]
        y = model_holder["y"]

        if model is None:
            ax.set_title("Train model to visualize.")
            canvas.draw()
            return

        fcount = int(f_var.get())

        # -------- Multi-Class 1D Visualization --------
        if fcount == 1:
            idx = np.random.choice(len(X), min(150, len(X)), replace=False)
            ax.scatter(X[idx, 0], y[idx], c=y[idx], cmap="tab10", s=20)
            ax.set_title("Centroid Classification (Multi-Class 1D)")

        # -------- Multi-Class (no plot for >1 feature) --------
        else:
            txt = "Class Centroids:\n"
            for i, centroid in enumerate(model.centroids):
                txt += f"  Class {i}: {centroid}\n"
            ax.text(0.1, 0.5, txt, fontsize=12)
            ax.set_axis_off()

        canvas.draw()

    # Back logic
    def back():
        content_frame.pack_forget()
        main_frame.pack()
