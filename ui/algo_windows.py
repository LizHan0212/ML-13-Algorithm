import tkinter as tk
from tkinter import ttk

# Import algorithm pages
from ui.algorithm_pages.linear_regression_page import load_linear_regression_page


def load_algorithm_page(algo_name, main_frame, content_frame):
    # Hide main menu
    main_frame.pack_forget()
    content_frame.pack(expand=True, fill="both")

    # Clear old content
    for widget in content_frame.winfo_children():
        widget.destroy()

    # Left & right panels
    left_panel = ttk.Frame(content_frame, width=250, padding=10)
    left_panel.pack(side="left", fill="y")

    right_panel = ttk.Frame(content_frame)
    right_panel.pack(side="right", expand=True, fill="both")

    # Title
    ttk.Label(left_panel, text=algo_name,
              font=("Arial", 16, "bold")).pack(pady=10)

    # ROUTING HERE
    if algo_name.startswith("1."):
        load_linear_regression_page(left_panel, right_panel)

    # Back button
    ttk.Button(
        left_panel,
        text="⬅ Back to Main Menu",
        command=lambda: back_to_main(main_frame, content_frame)
    ).pack(side="bottom", pady=20)


def back_to_main(main_frame, content_frame):
    content_frame.pack_forget()
    main_frame.pack(expand=True, fill="both")