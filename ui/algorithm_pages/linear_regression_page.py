import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


def load_linear_regression_page(left_panel, right_panel):

    from algorithms.supervised._1_linear_regression import (
        DATASETS,
        train_linear_regression_from_file
    )

    # Use a holder object so IDE stops complaining
    class ModelHolder:
        model = None

    holder = ModelHolder()

    # LEFT PANEL: Controls + Input----------------------------------
    ttk.Label(left_panel, text="Choose Dataset:",
              font=("Arial", 12, "bold")).pack(pady=10)

    dataset_var = tk.StringVar(value=list(DATASETS.keys())[0])

    for label in DATASETS:
        ttk.Radiobutton(
            left_panel,
            text=label,
            variable=dataset_var,
            value=label
        ).pack(anchor="w", padx=10)

    # ERROR / STATUS LABEL-------------------------------------------------------
    status_label = ttk.Label(left_panel, text="", font=("Arial", 10))
    status_label.pack(pady=5)

    # RIGHT PANEL: Blank Graph Area---------------------------------------------
    fig = Figure(figsize=(5, 4), dpi=100)
    ax = fig.add_subplot(111)

    ax.set_title("Graph will appear here after training")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    canvas = FigureCanvasTkAgg(fig, master=right_panel)
    canvas.draw()
    canvas.get_tk_widget().pack(expand=True, fill="both")

    # Training BUTTON------------------------------------------------------------
    def run_training():
        dataset_name = dataset_var.get()
        filepath = DATASETS[dataset_name]

        try:
            model, X, y = train_linear_regression_from_file(filepath)
            holder.model = model   # store model

            ax.clear()
            ax.scatter(X, y, color="blue", label="data")

            # Fit line
            import numpy as np
            x_line = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
            y_pred = model.predict(x_line)

            ax.plot(x_line, y_pred, color="red", label="fit")
            ax.set_title(dataset_name)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.legend()

            canvas.draw()
            status_label.config(text="Training complete!", foreground="green")

        except Exception as e:
            status_label.config(text=str(e), foreground="red")

    ttk.Button(left_panel, text="Train Model",
               command=run_training).pack(pady=20)

    # PREDICTION INPUT------------------------------------------------------------------
    ttk.Label(left_panel, text="Enter X value:", font=("Arial", 12)).pack(pady=5)

    x_entry = ttk.Entry(left_panel, width=15)
    x_entry.pack(pady=5)

    prediction_label = ttk.Label(left_panel, text="Prediction: -", font=("Arial", 11))
    prediction_label.pack(pady=10)

    # PREDICT BUTTON-----------------------------------------------------------------
    def predict_value():
        model = holder.model
        if model is None:
            prediction_label.config(text="Train model first!", foreground="red")
            return

        try:
            x_val = float(x_entry.get())
            import numpy as np
            y_val = model.predict(np.array([[x_val]]))[0]
            prediction_label.config(text=f"Prediction: y = {y_val:.2f}",
                                    foreground="black")
        except Exception as e:
            prediction_label.config(text=f"Error: {str(e)}",
                                    foreground="red")

    ttk.Button(left_panel, text="Predict Value",
               command=predict_value).pack(pady=10)