from utils.plots import scatter_plot
import numpy as np


def test_scatter_plot():
    x = np.random.rand(10, 2)
    fig = scatter_plot(x, title="Test Scatter")
    assert fig is not None