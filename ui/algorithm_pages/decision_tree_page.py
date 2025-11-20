import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from algorithms.supervised_classifier._8_decision_tree.train import train_decision_tree
from algorithms.supervised_classifier._8_decision_tree.utils import load_tree_dataset


def load_decision_tree_page(main_frame, content_frame):

    main_frame.pack_forget()
    content_frame.pack(expand=True, fill="both")

    for w in content_frame.winfo_children():
        w.destroy()

    frame = ttk.Frame(content_frame, padding=20)
    frame.pack(expand=True, fill="both")

    # Left panel
    left = ttk.Frame(frame)
    left.grid(row=0, column=0, sticky="nw", padx=20)

    # Right panel (table + tree plot)
    right = ttk.Frame(frame)
    right.grid(row=0, column=1, sticky="nsew")
    frame.columnconfigure(1, weight=1)

    # Back button
    ttk.Button(left, text="← Back", command=lambda: back()).pack(anchor="w", pady=5)

    ttk.Label(left, text="Decision Tree Classifier", font=("Arial", 16, "bold")).pack(pady=10)
    ttk.Label(left, text="Dataset: Weather + Workload + Candy → Mood").pack(pady=2)

    status = ttk.Label(left, text="")
    status.pack(pady=4)

    model_holder = {"model": None}

    # Train button
    def train():
        X, y = load_tree_dataset()
        model = train_decision_tree(X, y)
        model_holder["model"] = model
        status.config(text="Training complete!", foreground="green")
        show_table()
        show_tree_plot(model)

    ttk.Button(left, text="Train Model", command=train).pack(pady=10)

    ttk.Label(left, text="-" * 40).pack(pady=10)

    # Prediction
    ttk.Label(left, text="Prediction", font=("Arial", 14, "bold")).pack(pady=5)

    pred_frame = ttk.Frame(left)
    pred_frame.pack()

    ttk.Label(pred_frame, text="Weather (sunny/rain):").pack()
    e_weather = ttk.Entry(pred_frame, width=12); e_weather.pack(pady=1)

    ttk.Label(pred_frame, text="Workload (light/heavy):").pack()
    e_work = ttk.Entry(pred_frame, width=12); e_work.pack(pady=1)

    ttk.Label(pred_frame, text="Candy (yes/no):").pack()
    e_candy = ttk.Entry(pred_frame, width=12); e_candy.pack(pady=1)

    result_label = ttk.Label(left, text="Prediction: -")
    result_label.pack(pady=6)

    def predict():
        model = model_holder["model"]
        if model is None:
            result_label.config(text="Train first!", foreground="red")
            return

        w = e_weather.get().strip()
        wk = e_work.get().strip()
        c = e_candy.get().strip()

        try:
            pred = model.predict([[w, wk, c]])[0]
            symbol = "happy" if pred == "happy" else "sad"
            result_label.config(text=f"Prediction: {symbol}", foreground="black")
        except:
            result_label.config(text="Invalid Input", foreground="red")

    ttk.Button(left, text="Predict", command=predict).pack(pady=5)

    # Right panel content ----------------------------------------------------

    # Table
    table_frame = ttk.Frame(right)
    table_frame.pack(fill="x", pady=10)

    cols = ("weather", "workload", "candy", "mood")
    table = ttk.Treeview(table_frame, columns=cols, show="headings", height=8)

    for col in cols:
        table.heading(col, text=col.capitalize())
        table.column(col, width=120)

    table.pack(fill="x")

    def show_table():
        table.delete(*table.get_children())
        X, y = load_tree_dataset()
        for i in range(len(X)):
            weather, workload, candy = X[i]
            mood = y[i]
            table.insert("", "end", values=(weather, workload, candy, mood))

    # Tree plot
    fig = plt.Figure(figsize=(5, 6))
    ax = fig.add_subplot(111)
    canvas = FigureCanvasTkAgg(fig, master=right)
    canvas.get_tk_widget().pack(expand=True, fill="both")

    def show_tree_plot(model):
        ax.clear()
        ax.set_axis_off()

        # simple recursive text drawing
        def draw_node(node, x, y, dx):
            if node["type"] == "leaf":
                ax.text(x, y, f"Leaf: {node['class']}", ha="center",
                        fontsize=12, bbox=dict(boxstyle="round", fc="lightgreen"))
                return

            # internal node
            ax.text(x, y, f"{node['feature']} = ?", ha="center",
                    fontsize=12, bbox=dict(boxstyle="round", fc="lightblue"))

            # left child
            ax.plot([x, x - dx], [y - 0.1, y - 0.3], "k-")
            ax.text(x - dx, y - 0.12, f"{node['values'][0]}", fontsize=10)
            draw_node(node["children"][0], x - dx, y - 0.4, dx * 0.5)

            # right child
            ax.plot([x, x + dx], [y - 0.1, y - 0.3], "k-")
            ax.text(x + dx, y - 0.12, f"{node['values'][1]}", fontsize=10)
            draw_node(node["children"][1], x + dx, y - 0.4, dx * 0.5)

        draw_node(model.tree, x=0.5, y=0.95, dx=0.25)
        canvas.draw()

    # Back
    def back():
        content_frame.pack_forget()
        main_frame.pack()
