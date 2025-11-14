import pytest
from ui.main_window import create_algorithm_buttons


class DummyFrame:
    """Mock frame to test button creation without actual Tk window."""
    def __init__(self):
        self.children = []

    def pack(self, **kwargs):
        pass


def test_create_algorithm_buttons():
    frame = DummyFrame()
    algos = ["A", "B", "C"]

    buttons = create_algorithm_buttons(frame, algos)

    assert len(buttons) == 3
    assert all(hasattr(btn, "cget") for btn in buttons)