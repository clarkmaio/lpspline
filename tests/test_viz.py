"""Smoke tests for the plotting helpers."""
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from lpspline import bs, f, l  # noqa: E402
from lpspline.datasets import load_demo_dataset  # noqa: E402
from lpspline.viz import plot_diagnostic, plot_interactive  # noqa: E402


@pytest.fixture(scope="module")
def fitted():
    np.random.seed(0)
    X, y = load_demo_dataset(samples=200)
    model = l("xl", by="xfactor") + bs("xbs", knots=6) + f("xfactor")
    model.fit(X, y, summary=False)
    return model, X, y


@pytest.mark.parametrize("ncols", [1, 2, 4])
def test_plot_diagnostic(fitted, ncols):
    model, X, y = fitted
    fig, axes = plot_diagnostic(model=model, X=X, y=y, ncols=ncols)
    assert len(fig.axes) == len(model.splines)
    plt.close(fig)


def test_plot_diagnostic_without_y(fitted):
    model, X, _ = fitted
    fig, _ = plot_diagnostic(model=model, X=X)
    plt.close(fig)


def test_plot_interactive(fitted):
    model, X, y = fitted
    chart = plot_interactive(model, X, y, xcol="xbs", show_splines=["bspline", "factor"])
    spec = chart.to_dict()
    assert "vconcat" in spec
