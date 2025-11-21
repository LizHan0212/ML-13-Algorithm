# ui/algorithm_pages/adaboost_page.py

import tkinter as tk
from tkinter import ttk
import numpy as np

from algorithms.supervised_classifier._10_adaboost.train import train_adaboost
from algorithms.supervised_classifier._10_adaboost.model import adaboost_predict

# Category maps
weather_list = ["sunny", "rain"]
work_list = ["light", "heavy"]
candy_list = ["yes", "no"]

weather_map = {"sunny": 0, "rain": 1}
work_map = {"light": 0, "heavy": 1}
candy_map = {"yes": 0, "no": 1}
mood_inv = {0: "happy", 1: "sad"}


def load_adaboost_page(main_frame, content_frame):
    # Switch screens
    main_frame.pack_forget()
    content_frame.pack(expand=True, fill="both")

    for w in content_frame.winfo_children():
        w.destroy()

    frame = ttk.Frame(content_frame, padding=20)
    frame.pack(expand=True, fill="both")

    # Left
    left = ttk.Frame(frame)
    left.grid(row=0, column=0, sticky="nw", padx=20)

    # Right (scrollable)
    right = ttk.Frame(frame)
    right.grid(row=0, column=1, sticky="nsew")
    frame.columnconfigure(1, weight=1)

    # Back button
    ttk.Button(left, text="← Back",
               command=lambda: back()).pack(anchor="w", pady=5)

    ttk.Label(left, text="AdaBoost (Decision Stumps)",
              font=("Arial", 16, "bold")).pack(pady=10)

    ttk.Label(left,
              text="Dataset: 15 samples, 3 categorical features\nWeak learners (T): 5",
              font=("Arial", 10)).pack(pady=2)

    status = ttk.Label(left, text="")
    status.pack(pady=5)

    # Scrollable text box for dataset + stumps
    text_frame = ttk.Frame(right)
    text_frame.pack(expand=True, fill="both")

    text_box = tk.Text(text_frame, wrap="word", height=30)
    text_scroll = ttk.Scrollbar(text_frame,
                                orient="vertical",
                                command=text_box.yview)
    text_box.configure(yscrollcommand=text_scroll.set)

    text_box.pack(side="left", expand=True, fill="both")
    text_scroll.pack(side="right", fill="y")

    model_holder = {
        "stumps": None,
        "alphas": None,
        "X": None,
        "y": None
    }

    # Train button
    def train():
        status.config(text="Training...", foreground="orange")
        stumps, alphas, X, y = train_adaboost(T=5)

        model_holder["stumps"] = stumps
        model_holder["alphas"] = alphas
        model_holder["X"] = X
        model_holder["y"] = y

        status.config(text="Training complete.", foreground="green")
        update_display()

    ttk.Button(left, text="Train Model", command=train).pack(pady=10)

    ttk.Label(left, text="-" * 40).pack(pady=10)

    # ========== Prediction UI ==========
    ttk.Label(left, text="Prediction",
              font=("Arial", 14, "bold")).pack(pady=5)

    # Dropdowns for 3 features
    ttk.Label(left, text="Weather:").pack()
    weather_var = tk.StringVar(value="sunny")
    ttk.Combobox(left, textvariable=weather_var,
                 values=weather_list, width=10).pack()

    ttk.Label(left, text="Workload:").pack()
    work_var = tk.StringVar(value="light")
    ttk.Combobox(left, textvariable=work_var,
                 values=work_list, width=10).pack()

    ttk.Label(left, text="Candy:").pack()
    candy_var = tk.StringVar(value="yes")
    ttk.Combobox(left, textvariable=candy_var,
                 values=candy_list, width=10).pack()

    result_label = ttk.Label(left, text="Prediction: -")
    result_label.pack(pady=5)

    def predict():
        stumps = model_holder["stumps"]
        alphas = model_holder["alphas"]

        if stumps is None:
            result_label.config(text="Train first!", foreground="red")
            return

        try:
            row = [
                weather_map[weather_var.get()],
                work_map[work_var.get()],
                candy_map[candy_var.get()]
            ]

            pred_num = adaboost_predict(stumps, alphas, row)
            pred_word = mood_inv[pred_num]

            result_label.config(
                text=f"Prediction: {pred_word}",
                foreground="black"
            )

            update_display(pred_info=(row, pred_word))

        except Exception as e:
            result_label.config(text="Invalid input", foreground="red")
            print("Prediction error:", e)

    ttk.Button(left, text="Predict", command=predict).pack(pady=5)

    # ========== Update right-side window ==========
    def update_display(pred_info=None):
        text_box.delete("1.0", tk.END)

        stumps = model_holder["stumps"]
        alphas = model_holder["alphas"]
        X = model_holder["X"]
        y = model_holder["y"]

        if stumps is None:
            return

        # === Dataset Table ===
        text_box.insert(tk.END, "=== TRAINING DATA (15 rows) ===\n")
        text_box.insert(tk.END, "weather | workload | candy | mood\n")
        text_box.insert(tk.END, "-" * 45 + "\n")

        for i in range(len(X)):
            w = weather_list[X[i][0]]
            wk = work_list[X[i][1]]
            c = candy_list[X[i][2]]
            m = mood_inv[y[i]]
            text_box.insert(tk.END, f"{w:6s} | {wk:7s} | {c:5s} → {m}\n")

        # === Stumps ===
        text_box.insert(tk.END, "\n\n=== WEAK LEARNERS (T = 5) ===\n")

        for i, stump in enumerate(stumps):
            f = stump["feature"]
            fmap = ["weather", "workload", "candy"][f]

            text_box.insert(tk.END, f"\n-- Stump {i+1} --\n")
            text_box.insert(tk.END, f"Feature used: {fmap}\n")
            text_box.insert(tk.END, f"α weight = {alphas[i]:.3f}\n")
            text_box.insert(tk.END, "Prediction rules:\n")

            for val, out in stump["pred_map"].items():
                vname = (weather_list if f == 0 else
                         work_list if f == 1 else
                         candy_list)[val]
                text_box.insert(tk.END, f"  If {fmap} = {vname} → {mood_inv[out]}\n")

        # === Prediction Explanation ===
        if pred_info is not None:
            row, pred_word = pred_info
            text_box.insert(tk.END, "\n\n=== FINAL PREDICTION EXPLANATION ===\n")

            text_box.insert(tk.END,
                            f"Input: weather={weather_list[row[0]]}, "
                            f"workload={work_list[row[1]]}, "
                            f"candy={candy_list[row[2]]}\n\n")

            text_box.insert(tk.END, "Weighted stump votes (α * h(x)):\n")

            weighted_sum = 0
            for i, stump in enumerate(stumps):
                f = stump["feature"]
                pred = stump["pred_map"][row[f]]  # 0 or 1
                pred_word_i = mood_inv[pred]
                alpha_i = alphas[i]
                contrib = alpha_i * (1 if pred == 1 else -1)  # map happy/sad → -1/+1
                weighted_sum += contrib

                text_box.insert(
                    tk.END,
                    f"Stump {i+1}: predicts {pred_word_i:<5s} → "
                    f"{alpha_i:.3f} * ({1 if pred==1 else -1}) = {contrib:.3f}\n"
                )

            text_box.insert(tk.END, f"\nTotal weighted sum = {weighted_sum:.3f}\n")
            text_box.insert(tk.END, f"Final Output = {pred_word}\n")

        text_box.see("end")

    def back():
        content_frame.pack_forget()
        main_frame.pack()
