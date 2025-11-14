from ui.algo_windows import open_algorithm_window


def test_open_algorithm_window():
    win = open_algorithm_window("Linear Regression")
    assert win.winfo_exists() == 1
    assert "Linear Regression" in win.title()
    win.destroy()