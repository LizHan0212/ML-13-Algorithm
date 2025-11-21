import tkinter as tk
from tkinter import ttk

from algorithms.supervised_classifier._9_random_forest.train import train_random_forest


def load_random_forest_page(main_frame, content_frame):

    main_frame.pack_forget()
    content_frame.pack(expand=True, fill="both")

    for w in content_frame.winfo_children():
        w.destroy()

    frame = ttk.Frame(content_frame, padding=20)
    frame.pack(expand=True, fill="both")

    # Left side
    left = ttk.Frame(frame)
    left.grid(row=0, column=0, sticky="nw", padx=20)

    # Right side (scrollable)
    right_wrapper = ttk.Frame(frame)
    right_wrapper.grid(row=0, column=1, sticky="nsew")
    frame.columnconfigure(1, weight=1)

    # Scrollable area
    canvas = tk.Canvas(right_wrapper)
    scrollbar = ttk.Scrollbar(right_wrapper, orient="vertical", command=canvas.yview)
    scroll_frame = ttk.Frame(canvas)

    scroll_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    # Title & Controls
    ttk.Button(left, text="← Back", command=lambda: back()).pack(anchor="w", pady=5)
    ttk.Label(left, text="Random Forest", font=("Arial", 16, "bold")).pack(pady=10)
    ttk.Label(left, text="Dataset: 15 samples, 3 categorical features").pack()

    ttk.Label(left, text="Number of Trees:").pack(pady=(10, 3))
    n_var = tk.StringVar(value="5")
    ttk.Combobox(left, textvariable=n_var, values=["3","5","7","10"], width=10).pack()

    status = ttk.Label(left, text="")
    status.pack(pady=5)

    model_holder = {"forest": None, "feats": None, "X": None, "y": None}

    def train():
        status.config(text="Training...", foreground="orange")
        n = int(n_var.get())
        forest, feats, X, y = train_random_forest(n)
        model_holder["forest"] = forest
        model_holder["feats"] = feats
        model_holder["X"] = X
        model_holder["y"] = y
        status.config(text="Training complete.", foreground="green")
        update_display()

    ttk.Button(left, text="Train Model", command=train).pack(pady=10)

    ttk.Label(left, text="-"*40).pack(pady=10)

    # --- Prediction ---
    ttk.Label(left, text="Predict", font=("Arial", 14, "bold")).pack()
    ttk.Label(left, text="weather (sunny/rain)").pack()
    e1 = ttk.Entry(left, width=12); e1.pack(pady=2)
    ttk.Label(left, text="workload (light/heavy)").pack()
    e2 = ttk.Entry(left, width=12); e2.pack(pady=2)
    ttk.Label(left, text="candy (yes/no)").pack()
    e3 = ttk.Entry(left, width=12); e3.pack(pady=2)

    result = ttk.Label(left, text="Prediction: -")
    result.pack(pady=5)

    tree_vote_box = tk.Text(left, width=40, height=10)
    tree_vote_box.pack(pady=10)

    def predict():
        forest = model_holder["forest"]
        feats = model_holder["feats"]

        if forest is None:
            result.config(text="Train first!", foreground="red")
            return

        weather = e1.get().strip().lower()
        work = e2.get().strip().lower()
        candy = e3.get().strip().lower()

        mapping = {
            "sunny": 0, "rain": 1,
            "light": 0, "heavy": 1,
            "yes": 0, "no": 1
        }

        # Validate user input
        try:
            row = [mapping[weather], mapping[work], mapping[candy]]
        except:
            result.config(text="Invalid input", foreground="red")
            return

        # ---- COLLECT VOTES FROM TREES ----
        from algorithms.supervised_classifier._9_random_forest.model import random_forest_predict

        votes = []
        tree_vote_box.delete("1.0", tk.END)

        for i, (tree, fset) in enumerate(zip(forest, feats)):
            pred = random_forest_predict([tree], [fset], row)  # predict with 1 tree
            votes.append(pred)

            mood = "happy" if pred == 0 else "sad"
            tree_vote_box.insert(tk.END, f"Tree {i + 1}: {mood}\n")

        # ---- MAJORITY VOTE ----
        final = 0 if votes.count(0) > votes.count(1) else 1
        final_label = "happy" if final == 0 else "sad"

        result.config(text=f"Final Prediction: {final_label}", foreground="black")

    ttk.Button(left, text="Predict", command=predict).pack(pady=5)

    # ---- Display trees & table ----
    def tree_to_string(node, feats, indent=""):
        if "leaf" in node:
            return f"{indent}→ { 'happy' if node['leaf']==0 else 'sad' }\n"

        s = f"{indent}Feature {feats[node['feature']]}\n"
        for val, child in node["children"].items():
            vname = ["sunny/rain", "light/heavy", "yes/no"][feats[node["feature"]]]
            sval = ["sunny","rain","light","heavy","yes","no"][val]
            s += f"{indent} {sval}:\n"
            s += tree_to_string(child, feats, indent + "  ")
        return s

    def update_display():
        for w in scroll_frame.winfo_children():
            w.destroy()

        X = model_holder["X"]
        y = model_holder["y"]

        header = ttk.Label(scroll_frame, text="Training Data:", font=("Arial", 12, "bold"))
        header.pack(anchor="w")

        table = tk.Text(scroll_frame, height=10, width=50)
        table.pack(pady=5)

        for i in range(len(X)):
            wv = "sunny" if X[i][0]==0 else "rain"
            wl = "light" if X[i][1]==0 else "heavy"
            cd = "yes" if X[i][2]==0 else "no"
            mood = "happy" if y[i]==0 else "sad"
            table.insert(tk.END, f"{wv:6s} | {wl:6s} | {cd:3s} → {mood}\n")

        ttk.Label(scroll_frame, text="Random Forest Trees:", font=("Arial", 12, "bold")).pack(anchor="w", pady=10)

        forest = model_holder["forest"]
        feats = model_holder["feats"]

        for i, tree in enumerate(forest):
            ttk.Label(scroll_frame, text=f"-- Tree {i+1} --", font=("Arial", 10, "bold")).pack(anchor="w")
            txt = tk.Text(scroll_frame, height=7, width=60)
            txt.pack(pady=5)
            txt.insert(tk.END, tree_to_string(tree, feats[i]))

    def back():
        content_frame.pack_forget()
        main_frame.pack()
