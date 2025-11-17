import tkinter as tk
from tkinter import ttk

# Import algorithm pages directly
from ui.algorithm_pages.linear_regression_page import load_linear_regression_page
# (More pages will be added later)


def launch_main_ui():
    root = tk.Tk()
    root.title("ML Algorithm Demonstrator")
    root.geometry("1000x650")

    # ----- Main Frame (Menu) -----
    main_frame = ttk.Frame(root, padding=20)
    main_frame.pack(expand=True, fill="both")

    # ----- Content Frame (Algorithm Page) -----
    content_frame = ttk.Frame(root, padding=20)

    # Title
    ttk.Label(main_frame, text="Machine Learning Algorithms",
              font=("Arial", 18, "bold")).pack(pady=20)

    # Layout → Two columns
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

    # ---------------------------------------------------------
    # ALGORITHM ROUTER (dispatcher)
    # ---------------------------------------------------------
    def load_algorithm_page(algo_name):
        """Switch from main menu to the algorithm page."""

        # Hide main frame
        main_frame.pack_forget()

        # Create new left/right panels for the algorithm page
        content_frame.pack(expand=True, fill="both")

        left_panel = ttk.Frame(content_frame, width=300, padding=15)
        right_panel = ttk.Frame(content_frame, padding=15)

        left_panel.pack(side="left", fill="y")
        right_panel.pack(side="right", expand=True, fill="both")

        # BACK BUTTON
        def go_back():
            content_frame.pack_forget()
            main_frame.pack(expand=True, fill="both")

            # Clear content
            for w in content_frame.winfo_children():
                w.destroy()

        ttk.Button(left_panel, text="← Back to Main Menu",
                   command=go_back).pack(anchor="w", pady=(0, 10))

        # -------- ROUTING --------
        if algo_name.startswith("1."):
            load_linear_regression_page(left_panel, right_panel)

        else:
            ttk.Label(right_panel, text=f"{algo_name} page not implemented yet.",
                      font=("Arial", 14)).pack(pady=40)

    # ---------------------------------------------------------
    # BUTTON CREATOR
    # ---------------------------------------------------------
    def create_buttons(frame, algo_list):
        for algo in algo_list:
            ttk.Button(frame, text=algo, width=25,
                       command=lambda a=algo: load_algorithm_page(a)
                       ).pack(pady=5)

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

    create_buttons(supervised_frame, supervised_algos)
    create_buttons(unsupervised_frame, unsupervised_algos)

    root.mainloop()
