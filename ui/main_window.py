import tkinter as tk
from tkinter import ttk
from ui.algo_windows import load_algorithm_page


def launch_main_ui():
    root = tk.Tk()
    root.title("ML Algorithm Demonstrator")
    root.geometry("800x600")

    # Main menu frame
    main_frame = ttk.Frame(root, padding=20)
    main_frame.pack(expand=True, fill="both")

    # Content frame (algorithm pages)
    content_frame = ttk.Frame(root, padding=20)

    # Title
    ttk.Label(main_frame, text="Machine Learning Algorithms",
              font=("Arial", 18, "bold")).pack(pady=20)

    # Two-column layout
    container = ttk.Frame(main_frame)
    container.pack()

    supervised_frame = ttk.Frame(container)
    unsupervised_frame = ttk.Frame(container)

    supervised_frame.grid(row=0, column=0, padx=40, sticky="n")
    unsupervised_frame.grid(row=0, column=1, padx=40, sticky="n")

    ttk.Label(supervised_frame, text="Supervised",
              font=("Arial", 14, "bold")).pack(pady=10)

    ttk.Label(unsupervised_frame, text="Unsupervised",
              font=("Arial", 14, "bold")).pack(pady=10)

    # List of algorithms
    supervised_algos = [
        "1. Linear Regression",
        "2. Logistic Regression",
        "3. KNN",
        "4. SVM",
        "5. Naive Bayes",
        "6. Decision Trees",
        "7. Bagging",
        "8. Random Forest",
        "9. Boosting",
        "10. Neural Networks"
    ]

    unsupervised_algos = [
        "11. K-Means",
        "12. Dimensionality Reduction",
        "13. PCA"
    ]

    # Button builder
    def create_buttons(frame, algo_list):
        for algo in algo_list:
            ttk.Button(
                frame,
                text=algo,
                width=25,
                command=lambda a=algo:
                    load_algorithm_page(a, main_frame, content_frame)
            ).pack(pady=5)

    create_buttons(supervised_frame, supervised_algos)
    create_buttons(unsupervised_frame, unsupervised_algos)

    root.mainloop()
