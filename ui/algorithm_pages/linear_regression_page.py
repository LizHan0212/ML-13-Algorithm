import tkinter as tk
from tkinter import ttk
import numpy as np

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from algorithms.supervised._1_linear_regression import (
    load_and_train_model,
    get_dataset_path
)
from utils.dataset_readers import load_multi_feature_txt_file


def load_linear_regression_page(left_panel, right_panel):

    # Holder for model + feature count
    class Holder:
        model = None
        feature_count = 1

    holder = Holder()

    # ============================================================
    #                TRAINING CONFIG SECTION (LEFT TOP)
    # ============================================================

    ttk.Label(left_panel, text="Training Configuration",
              font=("Arial", 13, "bold")).pack(pady=(5, 5))

    # ----- Number of Features -----
    ttk.Label(left_panel, text="Number of Features:",
              font=("Arial", 11)).pack(anchor="w")

    feature_var = tk.StringVar(value="1")
    ttk.Combobox(
        left_panel, textvariable=feature_var,
        values=["1", "3", "5"], state="readonly", width=10
    ).pack(anchor="w", pady=2)

    # ----- Dataset Size -----
    ttk.Label(left_panel, text="Dataset Size:",
              font=("Arial", 11)).pack(anchor="w")

    dataset_var = tk.StringVar(value="10k")
    ttk.Combobox(
        left_panel, textvariable=dataset_var,
        values=["10k", "50k", "100k"], state="readonly", width=10
    ).pack(anchor="w", pady=2)

    # ----- Optimization Algorithm -----
    ttk.Label(left_panel, text="Optimization Algorithm:",
              font=("Arial", 11)).pack(anchor="w", pady=(10, 0))

    algo_var = tk.StringVar(value="BGD")
    ttk.Combobox(
        left_panel, textvariable=algo_var,
        values=["BGD", "SGD", "Mini-Batch (64)", "Normal Equation"],
        state="readonly", width=18
    ).pack(anchor="w", pady=2)

    # ----- Training Status -----
    status_label = ttk.Label(left_panel, text="", font=("Arial", 10))
    status_label.pack(anchor="w", pady=(0, 5))

    # ============================================================
    #                RIGHT PANEL VISUALIZATION
    # ============================================================

    def update_visualization():
        # Clear right panel
        for widget in right_panel.winfo_children():
            widget.destroy()

        model = holder.model
        if model is None:
            tk.Label(right_panel, text="No model trained yet.",
                     font=("Arial", 12)).pack()
            return

        d = holder.feature_count

        # ===============================================
        #       CASE 1: 1 FEATURE — PLOT GRAPH
        # ===============================================
        if d == 1:
            path = get_dataset_path(1, dataset_var.get())
            X, y = load_multi_feature_txt_file(path)

            # --- SAMPLE POINTS FOR VISUALIZATION ---
            if len(X) > 500:
                idx = np.random.choice(len(X), 500, replace=False)
                X_plot = X[idx]
                y_plot = y[idx]
            else:
                X_plot = X
                y_plot = y

            fig = Figure(figsize=(5, 4), dpi=100)
            ax = fig.add_subplot(111)
            ax.set_title("Linear Regression Fit (1 Feature)")
            ax.set_xlabel("X")
            ax.set_ylabel("y")

            # Scatter (sampled)
            ax.scatter(X_plot[:, 0], y_plot, s=6, alpha=0.6)

            # Prediction line
            x_sorted = np.sort(X[:, 0])
            y_pred = model.predict(x_sorted.reshape(-1, 1))
            ax.plot(x_sorted, y_pred, color="red", linewidth=2)

            canvas = FigureCanvasTkAgg(fig, master=right_panel)
            canvas.draw()
            canvas.get_tk_widget().pack(expand=True, fill="both")
            return

        # ===============================================
        #       CASE 2: MULTI-FEATURE → COEFFICIENT TABLE
        # ===============================================
        coef_frame = ttk.Frame(right_panel)
        coef_frame.pack(pady=10)

        ttk.Label(coef_frame, text="Model Coefficients",
                  font=("Arial", 14, "bold")).grid(row=0, column=0, columnspan=2, pady=5)

        ttk.Label(coef_frame, text="Feature",
                  font=("Arial", 12, "underline")).grid(row=1, column=0, padx=10)
        ttk.Label(coef_frame, text="Coefficient",
                  font=("Arial", 12, "underline")).grid(row=1, column=1, padx=10)

        # Fill values
        for i, c in enumerate(model.coef_):
            ttk.Label(coef_frame, text=f"X{i+1}", font=("Arial", 11)).grid(
                row=i + 2, column=0, padx=10)
            ttk.Label(coef_frame, text=f"{c:.4f}", font=("Arial", 11)).grid(
                row=i + 2, column=1, padx=10)

        ttk.Label(coef_frame, text="Intercept",
                  font=("Arial", 11)).grid(row=len(model.coef_) + 2, column=0, pady=(10, 0))
        ttk.Label(coef_frame, text=f"{model.intercept_:.4f}",
                  font=("Arial", 11)).grid(row=len(model.coef_) + 2, column=1, pady=(10, 0))

    # ============================================================
    #               TRAIN MODEL BUTTON + LOGIC
    # ============================================================

    def train_model():
        status_label.config(text="Training in progress...",
                            foreground="goldenrod")
        left_panel.update_idletasks()

        try:
            model = load_and_train_model(
                feature_count=int(feature_var.get()),
                dataset_size=dataset_var.get(),
                algorithm=algo_var.get()
            )

            holder.model = model
            holder.feature_count = int(feature_var.get())

            status_label.config(text="Training complete!",
                                foreground="green")

            update_visualization()

        except Exception as e:
            status_label.config(text=f"Training failed: {e}",
                                foreground="red")
            raise

    ttk.Button(left_panel, text="Train Model",
               command=train_model).pack(pady=5)

    # ============================================================
    #                PREDICTION SECTION (LEFT BOTTOM)
    # ============================================================

    ttk.Label(left_panel, text="Prediction",
              font=("Arial", 13, "bold")).pack(pady=(20, 5))

    ttk.Separator(left_panel, orient="horizontal").pack(fill="x", pady=(0, 10))

    entry_frame = ttk.Frame(left_panel)
    entry_frame.pack(anchor="w")

    entry_boxes = []

    # Dynamically update prediction input fields
    def update_entries(*args):
        # Clear ALL widgets from this frame
        for widget in entry_frame.winfo_children():
            widget.destroy()

        entry_boxes.clear()

        count = int(feature_var.get())

        ttk.Label(
            entry_frame,
            text=f"Enter {count} feature values:",
            font=("Arial", 11)
        ).pack(anchor="w")

        for _ in range(count):
            ent = ttk.Entry(entry_frame, width=12)
            ent.pack(anchor="w", pady=1)
            entry_boxes.append(ent)

    feature_var.trace_add("write", update_entries)
    update_entries()

    prediction_label = ttk.Label(left_panel, text="Prediction: -",
                                 font=("Arial", 11))
    prediction_label.pack(pady=5)

    def predict_value():
        model = holder.model
        if model is None:
            prediction_label.config(text="Train model first!",
                                    foreground="red")
            return

        try:
            values = [float(e.get()) for e in entry_boxes]
            y = model.predict([values])[0]
            prediction_label.config(text=f"Prediction: {y:.2f}",
                                    foreground="black")
        except:
            prediction_label.config(text="Invalid input",
                                    foreground="red")

    ttk.Button(left_panel, text="Predict Value",
               command=predict_value).pack(pady=5)
