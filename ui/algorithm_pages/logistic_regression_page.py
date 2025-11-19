import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from utils.dataset_readers import load_multi_feature_txt_file
from algorithms.supervised_classifier._2_logistic_regression.utils import get_dataset_path
from algorithms.supervised_classifier._2_logistic_regression.train import train_logistic_regression


def load_logistic_regression_page(main_frame, content_frame):
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

    ttk.Label(left, text="Logistic Regression", font=("Arial", 14, "bold")).pack(pady=5)

    ttk.Label(left, text="Features:").pack(anchor="w")
    feature_var = tk.StringVar(value="2")
    ttk.Combobox(left, textvariable=feature_var,
                 values=["2", "3", "5"], width=6, state="readonly").pack()

    ttk.Label(left, text="Dataset Size:").pack(anchor="w")
    size_var = tk.StringVar(value="10k")
    ttk.Combobox(left, textvariable=size_var,
                 values=["10k", "50k", "100k"], width=6, state="readonly").pack()

    ttk.Label(left, text="Algorithm:").pack(anchor="w")
    algo_var = tk.StringVar(value="BGD")
    ttk.Combobox(left, textvariable=algo_var,
                 values=["BGD", "SGD", "Mini-Batch (64)"], width=16, state="readonly").pack()

    status = ttk.Label(left, text="")
    status.pack()

    model = {"obj": None}
    X_cache = None
    y_cache = None

    def train():
        status.config(text="Training...", foreground="gold")
        path = get_dataset_path(int(feature_var.get()), size_var.get())
        X, y = load_multi_feature_txt_file(path)
        nonlocal X_cache, y_cache
        X_cache, y_cache = X, y
        model["obj"] = train_logistic_regression(X, y, algo_var.get())
        status.config(text="Done", foreground="green")
        update_plot()

    ttk.Button(left, text="Train", command=train).pack(pady=5)

    # -------- Separator --------
    ttk.Separator(left, orient="horizontal").pack(fill="x", pady=10)

    fig = plt.Figure(figsize=(4, 4))
    ax = fig.add_subplot(111)
    canvas = FigureCanvasTkAgg(fig, master=right)
    canvas.get_tk_widget().pack(expand=True, fill="both")

    def update_plot():
        ax.clear()
        m = model["obj"]
        if m is None:
            canvas.draw()
            return

        fcount = int(feature_var.get())

        if fcount == 2:
            idx = np.random.choice(len(X_cache), min(200, len(X_cache)), replace=False)
            Xs = X_cache[idx]
            ys = y_cache[idx]
            ax.scatter(Xs[:, 0], Xs[:, 1], c=ys, cmap="bwr", s=15, alpha=0.5)

            w1, w2 = m.coef_
            b = m.intercept_
            xs = np.linspace(-5, 5, 100)
            ax.plot(xs, (-(w1 * xs + b) / w2), "k--")
        else:
            txt = "Coefs:\n"
            for i, c in enumerate(m.coef_):
                txt += f"w{i+1} = {c:.3f}\n"
            txt += f"b = {m.intercept_:.3f}"
            ax.text(0.1, 0.5, txt)

        canvas.draw()

    ttk.Label(left, text="Prediction").pack(pady=5)
    inputs = []

    def build_inputs(*_):
        for w in inputs:
            w.destroy()
        inputs.clear()
        count = int(feature_var.get())
        for _ in range(count):
            e = ttk.Entry(left, width=10)
            e.pack()
            inputs.append(e)

    feature_var.trace_add("write", build_inputs)
    build_inputs()

    pred_label = ttk.Label(left, text="")
    pred_label.pack()

    def predict():
        if model["obj"] is None:
            pred_label.config(text="Train first", foreground="red")
            return
        vals = [float(e.get()) for e in inputs]
        X_in = np.array(vals).reshape(1, -1)
        prob = model["obj"].predict_proba(X_in)[0]
        pred = model["obj"].predict(X_in)[0]
        pred_label.config(text=f"P={prob:.3f}, Class={pred}")

    ttk.Button(left, text="Predict", command=predict).pack(pady=3)


def _back(main_frame, content_frame):
    content_frame.pack_forget()
    main_frame.pack()
