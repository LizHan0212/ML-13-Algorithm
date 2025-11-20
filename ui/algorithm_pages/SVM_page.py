# ui/algorithm_pages/svm_page.py

import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from algorithms.supervised_classifier._6_SVM.train import train_svm
from algorithms.supervised_classifier._6_SVM.utils import load_svm_dataset


def load_svm_page(main_frame, content_frame):

    main_frame.pack_forget()
    content_frame.pack(expand=True, fill="both")

    for w in content_frame.winfo_children():
        w.destroy()

    frame = ttk.Frame(content_frame, padding=20)
    frame.pack(expand=True, fill="both")

    # Left panel
    left = ttk.Frame(frame)
    left.grid(row=0, column=0, sticky="nw", padx=20)

    # Right panel (plot)
    plot_panel = ttk.Frame(frame)
    plot_panel.grid(row=0, column=1, sticky="nsew")
    frame.columnconfigure(1, weight=1)

    # Back button
    ttk.Button(left, text="← Back", command=lambda: back()).pack(anchor="w", pady=5)

    ttk.Label(left, text="Support Vector Machine", font=("Arial", 16, "bold")).pack(pady=10)
    ttk.Label(left, text="Dataset: 2 features, 150 samples", font=("Arial", 10)).pack(pady=2)

    status = ttk.Label(left, text="")
    status.pack(pady=5)

    model_holder = {"model": None, "X": None, "y": None}

    # Matplotlib figure
    fig = plt.Figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    canvas = FigureCanvasTkAgg(fig, master=plot_panel)
    canvas.get_tk_widget().pack(expand=True, fill="both")

    # Train model
    def train():
        status.config(text="Training...", foreground="orange")
        model = train_svm(C=1.0, lr=0.001, epochs=300)

        X, y = load_svm_dataset()
        model_holder["model"] = model
        model_holder["X"] = X
        model_holder["y"] = y

        status.config(text="Training complete.", foreground="green")
        update_plot()

    ttk.Button(left, text="Train Model", command=train).pack(pady=10)

    ttk.Label(left, text="-" * 40).pack(pady=10)

    # Prediction section
    ttk.Label(left, text="Prediction", font=("Arial", 14, "bold")).pack(pady=5)

    entry_frame = ttk.Frame(left)
    entry_frame.pack()

    ttk.Label(entry_frame, text="Enter 2 numerical values:").pack()
    e1 = ttk.Entry(entry_frame, width=10)
    e2 = ttk.Entry(entry_frame, width=10)
    e1.pack(pady=2)
    e2.pack(pady=2)

    result_label = ttk.Label(left, text="Prediction: -")
    result_label.pack(pady=5)

    def predict():
        model = model_holder["model"]
        if model is None:
            result_label.config(text="Train first!", foreground="red")
            return

        try:
            x1 = float(e1.get())
            x2 = float(e2.get())
            point = np.array([[x1, x2]])

            pred = model.predict(point)[0]

            # ---- convert to X / O ----
            symbol = "X" if pred == 0 else "O"
            result_label.config(text=f"Prediction: {symbol}", foreground="black")

            # update graph with predicted point drawn
            update_plot(query_point=(x1, x2))

        except:
            result_label.config(text="Invalid input", foreground="red")

    ttk.Button(left, text="Predict", command=predict).pack(pady=5)

    # Plotting
    def update_plot(query_point=None):
        ax.clear()

        X = model_holder["X"]
        y = model_holder["y"]
        model = model_holder["model"]

        # Plot data points (X and O)
        for i in range(len(X)):
            if y[i] == 0:
                ax.scatter(X[i, 0], X[i, 1], marker="x", color="red")
            else:
                ax.scatter(X[i, 0], X[i, 1], marker="o", color="blue")

        # Draw user prediction point if provided
        if query_point is not None:
            px, py = query_point
            ax.scatter(
                px, py,
                s=150,
                marker="o",
                facecolors="black",
                edgecolors="yellow",
                linewidths=2,
                label="Query"
            )

        # Draw SVM boundaries
        w = model.w
        b = model.b

        x_vals = np.linspace(0, 10, 200)
        if w[1] != 0:
            # decision line
            y_vals = -(w[0] * x_vals + b) / w[1]
            # margins
            y_vals_m1 = -(w[0] * x_vals + b - 1) / w[1]
            y_vals_p1 = -(w[0] * x_vals + b + 1) / w[1]

            ax.plot(x_vals, y_vals, "k-", label="Decision Boundary")
            ax.plot(x_vals, y_vals_m1, "k--", linewidth=1)
            ax.plot(x_vals, y_vals_p1, "k--", linewidth=1)

        # Shaded decision regions
        xx, yy = np.meshgrid(np.linspace(0, 10, 200), np.linspace(0, 10, 200))
        grid = np.c_[xx.ravel(), yy.ravel()]
        Z = model.predict(grid).reshape(xx.shape)

        ax.contourf(xx, yy, Z, alpha=0.15, cmap="bwr")

        ax.set_title("SVM Decision Boundary + Margins")
        canvas.draw()

    def back():
        content_frame.pack_forget()
        main_frame.pack()
