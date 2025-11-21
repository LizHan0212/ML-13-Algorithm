import tkinter as tk
from tkinter import ttk

# Import all algorithm pages
from ui.algorithm_pages.linear_regression_page import load_linear_regression_page
from ui.algorithm_pages.logistic_regression_page import load_logistic_regression_page
from ui.algorithm_pages.perceptron_page import load_perceptron_page
from ui.algorithm_pages.centroid_classifier_page import load_centroid_classifier_page
from ui.algorithm_pages.naive_bayes_page import load_naive_bayes_page
from ui.algorithm_pages.SVM_page import load_svm_page
from ui.algorithm_pages.KNN_page import load_knn_page
from ui.algorithm_pages.decision_tree_page import load_decision_tree_page
from ui.algorithm_pages.random_forest_page import load_random_forest_page
from ui.algorithm_pages.adaboost_page import load_adaboost_page
from ui.algorithm_pages.neural_network_page import load_neural_network_page
from ui.algorithm_pages.kmeans_page import load_kmeans_page


def launch_main_ui():
    root = tk.Tk()
    root.title("ML Algorithm Demonstrator")
    root.geometry("1200x850")

    # Main menu frame
    main_frame = ttk.Frame(root, padding=20)
    main_frame.pack(expand=True, fill="both")

    # Content frame (used when opening algorithm pages)
    content_frame = ttk.Frame(root, padding=20)

    # -----------------------------------------------------------
    # Title
    # -----------------------------------------------------------
    ttk.Label(
        main_frame,
        text="Common Machine Learning Models",
        font=("Arial", 30, "bold")
    ).pack(pady=40)

    # -----------------------------------------------------------
    #  Buttons 2 columns × 6 rows
    # -----------------------------------------------------------
    container = ttk.Frame(main_frame)
    container.pack(pady=20)

    left_col = ttk.Frame(container)
    right_col = ttk.Frame(container)

    left_col.grid(row=0, column=0, padx=50)
    right_col.grid(row=0, column=1, padx=50)

    # Left column: 1–6
    col1_models = [
        ("1. Linear Regression", load_linear_regression_page),
        ("2. Logistic Regression", load_logistic_regression_page),
        ("3. Perceptron", load_perceptron_page),
        ("4. Centroid Classifier", load_centroid_classifier_page),
        ("5. Naive Bayes", load_naive_bayes_page),
        ("6. SVM", load_svm_page),
    ]

    # Right column: 7–12
    col2_models = [
        ("7. KNN", load_knn_page),
        ("8. Decision Tree", load_decision_tree_page),
        ("9. Random Forest", load_random_forest_page),
        ("10. AdaBoost", load_adaboost_page),
        ("11. Neural Network", load_neural_network_page),
        ("12. K-Means", load_kmeans_page),
    ]

    # Button creator
    def create_buttons(frame, models):
        for text, callback in models:
            ttk.Button(
                frame,
                text=text,
                width=28,
                command=lambda cb=callback: cb(main_frame, content_frame)
            ).pack(pady=8)

    # Place buttons
    create_buttons(left_col, col1_models)
    create_buttons(right_col, col2_models)

    root.mainloop()
