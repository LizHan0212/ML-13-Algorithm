# ui/algorithm_pages/knn_page.py

import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from algorithms.supervised_classifier._7_KNN.train import train_knn
from algorithms.supervised_classifier._7_KNN.utils import load_knn_dataset


def load_knn_page(main_frame, content_frame):

    main_frame.pack_forget()
    content_frame.pack(expand=True, fill="both")

    for w in content_frame.winfo_children():
        w.destroy()

    frame = ttk.Frame(content_frame, padding=20)
    frame.pack(expand=True, fill="both")

    # Left panel
    left = ttk.Frame(frame)
    left.grid(row=0, column=0, sticky="nw", padx=20)

    # Plot panel
    plot_panel = ttk.Frame(frame)
    plot_panel.grid(row=0, column=1, sticky="nsew")
    frame.columnconfigure(1, weight=1)

    # Back button
    ttk.Button(left, text="← Back",
               command=lambda: back()).pack(anchor="w", pady=5)

    ttk.Label(left, text="K-Nearest Neighbors", font=("Arial", 16, "bold")).pack(pady=10)
    ttk.Label(left, text="Dataset: 2 features, 100 samples").pack(pady=2)

    status = ttk.Label(left, text="")
    status.pack(pady=5)

    # K selection
    ttk.Label(left, text="Choose K:").pack()
    k_var = tk.StringVar(value="3")
    ttk.Combobox(left, textvariable=k_var,
                 values=["1", "3", "5"], width=10).pack(pady=4)

    # Matplotlib setup
    fig = plt.Figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    canvas = FigureCanvasTkAgg(fig, master=plot_panel)
    canvas.get_tk_widget().pack(expand=True, fill="both")

    model_holder = {"model": None, "X": None, "y": None}

    # Train / Load KNN model (lazy model)
    def train():
        status.config(text="Loading data...", foreground="orange")
        k = int(k_var.get())

        model = train_knn(k)
        X, y = load_knn_dataset()

        model_holder["model"] = model
        model_holder["X"] = X
        model_holder["y"] = y

        status.config(text="Ready.", foreground="green")
        update_plot()

    ttk.Button(left, text="Apply KNN", command=train).pack(pady=10)

    ttk.Label(left, text="-" * 40).pack(pady=10)

    # Prediction UI
    ttk.Label(left, text="Prediction", font=("Arial", 14, "bold")).pack(pady=5)

    entry_frame = ttk.Frame(left)
    entry_frame.pack()

    ttk.Label(entry_frame, text="Enter 2 numbers:").pack()
    e1 = ttk.Entry(entry_frame, width=10); e1.pack(pady=2)
    e2 = ttk.Entry(entry_frame, width=10); e2.pack(pady=2)

    result_label = ttk.Label(left, text="Prediction: -")
    result_label.pack(pady=5)

    # -----------------------------
    # UPDATED PREDICT FUNCTION
    # -----------------------------
    def predict():
        model = model_holder["model"]
        if model is None:
            result_label.config(text="Train first!", foreground="red")
            return

        try:
            x1 = float(e1.get())
            x2 = float(e2.get())
            q = np.array([[x1, x2]])

            # Compute distances for visualization
            X = model_holder["X"]
            dists = np.linalg.norm(X - q, axis=1)
            k = int(k_var.get())

            # K nearest points
            neighbor_idx = np.argsort(dists)[:k]

            # Prediction
            pred = model.predict(q)[0]
            symbol = "X" if pred == 0 else "O"

            result_label.config(text=f"Prediction: {symbol}", foreground="black")

            # Update the graph with query + highlighted neighbors
            update_plot(query_point=(x1, x2),
                        neighbor_indices=neighbor_idx)

        except:
            result_label.config(text="Invalid", foreground="red")

    ttk.Button(left, text="Predict", command=predict).pack(pady=5)

    # -----------------------------
    # UPDATED PLOT FUNCTION
    # -----------------------------
    def update_plot(query_point=None, neighbor_indices=None):
        ax.clear()

        X = model_holder["X"]
        y = model_holder["y"]

        # Draw dataset points
        for i in range(len(X)):
            if y[i] == 0:
                ax.scatter(X[i, 0], X[i, 1], marker="x", color="red")
            else:
                ax.scatter(X[i, 0], X[i, 1], marker="o", color="blue")

        # If user prediction point exists, draw it
        if query_point is not None:
            qx, qy = query_point

            ax.scatter(qx, qy,
                       s=150,
                       facecolors="black",
                       edgecolors="yellow",
                       marker="o",
                       linewidths=2)

            # Draw lines to K nearest neighbors
            if neighbor_indices is not None:
                for idx in neighbor_indices:
                    nx, ny = X[idx]
                    ax.plot([qx, nx], [qy, ny],
                            color="gray", linewidth=1)

                    # Highlight the neighbor
                    ax.scatter(nx, ny,
                               s=200,
                               facecolors='none',
                               edgecolors='black',
                               linewidths=2)

        ax.set_title("KNN (K = " + k_var.get() + ")")
        canvas.draw()

    def back():
        content_frame.pack_forget()
        main_frame.pack()
