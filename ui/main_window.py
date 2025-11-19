import tkinter as tk
from tkinter import ttk

from ui.algorithm_pages.linear_regression_page import load_linear_regression_page
from ui.algorithm_pages.logistic_regression_page import load_logistic_regression_page
from ui.algorithm_pages.perceptron_page import load_perceptron_page


def launch_main_ui():
    root = tk.Tk()
    root.title("ML Algorithm Demonstrator")
    root.geometry("1400x900")

    # Main menu frame (top-level)
    main_frame = ttk.Frame(root, padding=20)
    main_frame.pack(expand=True, fill="both")

    # Content frame for algorithm pages
    content_frame = ttk.Frame(root, padding=20)

    # -----------------------------------------------------------
    #                     Master Title
    # -----------------------------------------------------------
    ttk.Label(
        main_frame,
        text="Machine Learning Algorithms",
        font=("Arial", 28, "bold")
    ).pack(pady=20)

    # -----------------------------------------------------------
    #             Main Container: Supervised | separator | Unsupervised
    # -----------------------------------------------------------
    container = ttk.Frame(main_frame)
    container.pack(pady=10, expand=True)

    # LEFT SIDE = SUPERVISED BLOCK
    supervised_block = ttk.Frame(container)
    supervised_block.grid(row=0, column=0, padx=40, sticky="n")

    # VERTICAL SEPARATOR
    vert_sep = ttk.Separator(container, orient="vertical")
    vert_sep.grid(row=0, column=1, sticky="ns", padx=30)

    # RIGHT SIDE = UNSUPERVISED BLOCK
    unsupervised_block = ttk.Frame(container)
    unsupervised_block.grid(row=0, column=2, padx=40, sticky="n")

    # ===============================================================
    #                         SUPERVISED
    # ===============================================================
    ttk.Label(
        supervised_block,
        text="Supervised Learning",
        font=("Arial", 20, "bold")
    ).pack(pady=(0, 20))

    # Two subcolumns: Regression (left) and Classification (right)
    sup_subcontainer = ttk.Frame(supervised_block)
    sup_subcontainer.pack()

    regression_frame = ttk.Frame(sup_subcontainer)
    classification_frame = ttk.Frame(sup_subcontainer)

    regression_frame.grid(row=0, column=0, padx=40, sticky="n")
    classification_frame.grid(row=0, column=1, padx=40, sticky="n")

    # ------------------- Regression -------------------
    ttk.Label(
        regression_frame,
        text="Regression",
        font=("Arial", 16, "underline")
    ).pack(pady=(0, 10))

    regression_algos = [
        ("1. Linear Regression", load_linear_regression_page),
    ]

    # ------------------- Classification -------------------
    ttk.Label(
        classification_frame,
        text="Classification",
        font=("Arial", 16, "underline")
    ).pack(pady=(0, 10))

    classification_algos = [
        ("2. Logistic Regression", load_logistic_regression_page),
        ("3. Perceptron", load_perceptron_page),
        ("4. KNN", None),
        ("5. SVM", None),
        ("6. Naive Bayes", None),
        ("7. Decision Trees", None),
        ("8. Bagging", None),
        ("9. Random Forest", None),
        ("10. Boosting", None),
        ("11. Neural Networks", None),
    ]

    # ===============================================================
    #                         UNSUPERVISED
    # ===============================================================
    ttk.Label(
        unsupervised_block,
        text="Unsupervised Learning",
        font=("Arial", 20, "bold")
    ).pack(pady=(0, 20))

    unsupervised_algos = [
        ("12. K-Means", None),
        ("13. Dimensionality Reduction", None),
        ("14. PCA", None),
    ]

    # ===============================================================
    #                  BUTTON CREATOR FUNCTION
    # ===============================================================
    def create_buttons(frame, algo_list):
        for text, callback in algo_list:
            if callback is None:
                ttk.Button(
                    frame, text=text, width=28, state="disabled"
                ).pack(pady=4)
            else:
                ttk.Button(
                    frame, text=text, width=28,
                    command=lambda cb=callback: cb(main_frame, content_frame)
                ).pack(pady=4)

    # Add buttons to each section
    create_buttons(regression_frame, regression_algos)
    create_buttons(classification_frame, classification_algos)
    create_buttons(unsupervised_block, unsupervised_algos)

    root.mainloop()
