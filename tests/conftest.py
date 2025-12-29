import pytest

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


@pytest.fixture(autouse=True)
def close_plots_before_each_test():
    yield
    if plt is not None:
        plt.close("all")
