import tkinter as tk
from tkinter import ttk

from ui.algorithm_pages.linear_regression_page import load_linear_regression_page
from ui.algorithm_pages.logistic_regression_page import load_logistic_regression_page
from ui.algorithm_pages.perceptron_page import load_perceptron_page
from ui.algorithm_pages.centroid_classifier_page import load_centroid_classifier_page
from ui.algorithm_pages.naive_bayes_page import load_naive_bayes_page
from ui.algorithm_pages.SVM_page import load_svm_page
from ui.algorithm_pages.KNN_page import load_knn_page
from ui.algorithm_pages.decision_tree_page import load_decision_tree_page
from ui.algorithm_pages.kmeans_page import load_kmeans_page
def launch_main_ui():
    root = tk.Tk()
    root.title("ML Algorithm Demonstrator")
    root.geometry("1400x900")

    # Main menu frame
    main_frame = ttk.Frame(root, padding=20)
    main_frame.pack(expand=True, fill="both")

    # Content frame for algorithm pages
    content_frame = ttk.Frame(root, padding=20)

    # -----------------------------------------------------------
    # Title
    # -----------------------------------------------------------
    ttk.Label(
        main_frame,
        text="Machine Learning Algorithms",
        font=("Arial", 28, "bold")
    ).pack(pady=20)

    # -----------------------------------------------------------
    #   Main container (Supervised | separator | Unsupervised)
    # -----------------------------------------------------------
    container = ttk.Frame(main_frame)
    container.pack(pady=10, expand=True)

    # SUPERVISED
    supervised_block = ttk.Frame(container)
    supervised_block.grid(row=0, column=0, padx=40, sticky="n")

    # VERTICAL LINE
    vert_sep = ttk.Separator(container, orient="vertical")
    vert_sep.grid(row=0, column=1, sticky="ns", padx=30)

    # UNSUPERVISED
    unsupervised_block = ttk.Frame(container)
    unsupervised_block.grid(row=0, column=2, padx=40, sticky="n")

    # ============================================================
    #                     SUPERVISED
    # ============================================================
    ttk.Label(
        supervised_block,
        text="Supervised Learning",
        font=("Arial", 20, "bold")
    ).pack(pady=(0, 20))

    # Two subcolumns: Regression + Classification
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
        ("4. Centroid Classifier", load_centroid_classifier_page),   # ← ADDED
        ("5. Naive Bayes", load_naive_bayes_page),
        ("6. SVM", load_svm_page),
        ("7. KNN", load_knn_page),
        ("8. Decision Trees", load_decision_tree_page),
        ("9. Bagging", None),
        ("10. Random Forest", None),
        ("11. Boosting", None),
        ("12. Neural Networks", None),
    ]

    # ============================================================
    #                     UNSUPERVISED
    # ============================================================
    ttk.Label(
        unsupervised_block,
        text="Unsupervised Learning",
        font=("Arial", 20, "bold")
    ).pack(pady=(0, 20))

    unsupervised_algos = [
        ("1. K-Means", load_kmeans_page),
        ("14. Dimensionality Reduction", None),
        ("15. PCA", None),
    ]

    # ============================================================
    #                   BUTTON GENERATOR
    # ============================================================
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

    # Create buttons
    create_buttons(regression_frame, regression_algos)
    create_buttons(classification_frame, classification_algos)
    create_buttons(unsupervised_block, unsupervised_algos)

    root.mainloop()
