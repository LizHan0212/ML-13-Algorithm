import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from algorithms.unsupervised._1_kmeans.train import train_kmeans
from algorithms.unsupervised._1_kmeans.utils import load_kmeans_data


def load_kmeans_page(main_frame, content_frame):

    main_frame.pack_forget()
    content_frame.pack(expand=True, fill="both")
    for w in content_frame.winfo_children():
        w.destroy()

    frame = ttk.Frame(content_frame, padding=20)
    frame.pack(expand=True, fill="both")

    left = ttk.Frame(frame)
    left.grid(row=0, column=0, sticky="nw", padx=20)

    right = ttk.Frame(frame)
    right.grid(row=0, column=1, sticky="nsew")
    frame.columnconfigure(1, weight=1)

    ttk.Button(left, text="← Back",
               command=lambda: back()).pack(anchor="w", pady=5)

    ttk.Label(left, text="K-Means Clustering", font=("Arial", 16, "bold")).pack(pady=10)
    ttk.Label(left, text="Dataset: 50 samples, 2 features").pack(pady=2)

    status = ttk.Label(left, text="")
    status.pack(pady=5)

    # K selection
    ttk.Label(left, text="Choose K:").pack()
    k_var = tk.StringVar(value="3")
    ttk.Combobox(left, textvariable=k_var,
                 values=["2", "3", "4"], width=10).pack(pady=4)

    # Matplotlib figure (two subplots)
    fig = plt.Figure(figsize=(7, 5))
    ax_clusters = fig.add_subplot(121)
    ax_error = fig.add_subplot(122)

    canvas = FigureCanvasTkAgg(fig, master=right)
    canvas.get_tk_widget().pack(expand=True, fill="both")

    model_holder = {"model": None}

    # TRAIN BUTTON
    def train():
        status.config(text="Training...", foreground="orange")
        k = int(k_var.get())
        model = train_kmeans(k)
        model_holder["model"] = model
        status.config(text="Done.", foreground="green")
        update_plot()

    ttk.Button(left, text="Run K-Means", command=train).pack(pady=10)

    # PLOTTING
    def update_plot():
        ax_clusters.clear()
        ax_error.clear()

        X = load_kmeans_data()
        model = model_holder["model"]

        # Plot clusters
        for j in range(len(model.centroids)):
            pts = X[model.assignments == j]
            ax_clusters.scatter(pts[:,0], pts[:,1], s=40, alpha=0.7)
            cx, cy = model.centroids[j]
            ax_clusters.scatter(cx, cy, s=130, marker="X", color="black")

        ax_clusters.set_title("Clusters")

        # Plot error curve
        ax_error.plot(model.errors, marker="o")
        ax_error.set_title("Error (SSE) across 10 iterations")
        ax_error.set_xlabel("Iteration")
        ax_error.set_ylabel("SSE")

        canvas.draw()

    def back():
        content_frame.pack_forget()
        main_frame.pack()
