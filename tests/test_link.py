"""Tests for the link wrappers: each is named after the function applied to the splines."""
import numpy as np
import polars as pl
import pytest

from lpspline import Exp, Link, Log, LpRegressor, Sigmoid, l


@pytest.fixture
def X():
    return pl.DataFrame({"x": np.linspace(1, 4, 100)})


def test_exp_means_y_equals_exp_of_splines(X):
    x = X["x"].to_numpy()
    y = pl.Series("y", np.exp(0.5 * x + 1))  # y = exp(1 + 0.5 x)
    model = Exp(+l("x"))
    model.fit(X, y, summary=False)
    np.testing.assert_allclose(model.splines[0].coefficients, [1.0, 0.5], atol=1e-5)
    np.testing.assert_allclose(model.predict(X), y.to_numpy(), rtol=1e-5)


def test_log_means_y_equals_log_of_splines(X):
    x = X["x"].to_numpy()
    y = pl.Series("y", np.log(2 * x + 3))  # y = log(3 + 2 x)
    model = Log(+l("x"))
    model.fit(X, y, summary=False)
    np.testing.assert_allclose(model.splines[0].coefficients, [3.0, 2.0], atol=1e-5)
    np.testing.assert_allclose(model.predict(X), y.to_numpy(), atol=1e-5)


def test_sigmoid_means_y_equals_sigmoid_of_splines(X):
    x = X["x"].to_numpy()
    y = pl.Series("y", 1 / (1 + np.exp(-(x - 2.5))))  # y = sigmoid(-2.5 + x)
    model = Sigmoid(+l("x"))
    model.fit(X, y, summary=False)
    np.testing.assert_allclose(model.splines[0].coefficients, [-2.5, 1.0], atol=1e-5)
    pred = model.predict(X)
    assert np.all((pred > 0) & (pred < 1))
    np.testing.assert_allclose(pred, y.to_numpy(), atol=1e-6)


def test_custom_link_and_identity_default(X):
    x = X["x"].to_numpy()
    y = pl.Series("y", (1 + x) ** 2)
    model = Link(+l("x"), link=np.sqrt, inv_link=np.square)
    model.fit(X, y, summary=False)
    np.testing.assert_allclose(model.predict(X), y.to_numpy(), rtol=1e-5)

    identity = Link(LpRegressor(l("x")))
    identity.fit(X, pl.Series("y", 2 * x), summary=False)
    np.testing.assert_allclose(identity.predict(X), 2 * x, atol=1e-5)


def test_components_are_on_the_spline_scale(X):
    x = X["x"].to_numpy()
    model = Exp(+l("x"))
    model.fit(X, pl.Series("y", np.exp(x)), summary=False)
    comps = model.predict(X, return_components=True)
    np.testing.assert_allclose(comps[:, 0], x, atol=1e-5)
