"""End-to-end checks that shape constraints hold on the fitted curves.

Each test fits data that violates the constraint (a sine wave) and checks the
fitted spline on a dense grid, so a wrong coefficient mapping shows up directly.
"""
from typing import Callable, Optional

import numpy as np
import polars as pl
import pytest

from lpspline import LpRegressor, bs, l, pwl
from lpspline.spline import Spline

TOL = 1e-6
GRID = np.linspace(0, 10, 2001)


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 10, 400)
    X = pl.DataFrame({"x": x, "g": rng.integers(0, 2, x.size)})
    y = pl.Series(3 * np.sin(x) + rng.normal(0, 0.1, x.size))
    return X, y


def _fit(spline: Spline, data) -> LpRegressor:
    X, y = data
    model = +spline
    model.fit(X, y, summary=False)
    assert model.problem.status == "optimal"
    return model


def _curves(spline: Spline):
    """Yield the fitted curve on GRID, once per `by` group if any."""
    if spline.by is None:
        yield spline.eval(GRID)
    else:
        for c in spline._by_classes:
            yield spline.eval(GRID, by=np.full(GRID.size, c))


def _assert_shape(spline: Spline, check: Callable[[np.ndarray], bool],
                  start: Optional[float] = None, end: Optional[float] = None) -> None:
    lo = -np.inf if start is None else start
    hi = np.inf if end is None else end
    mask = (GRID >= lo) & (GRID <= hi)
    for curve in _curves(spline):
        assert check(curve[mask])


def increasing(v): return np.all(np.diff(v) >= -TOL)
def decreasing(v): return np.all(np.diff(v) <= TOL)
def convex(v): return np.all(np.diff(v, 2) >= -TOL)
def concave(v): return np.all(np.diff(v, 2) <= TOL)


WINDOWS = [(None, None), (0, 10), (2, 6), (1.3, 7.7), (5, None), (None, 4)]
BSPLINE_SETUPS = [(deg, kn) for deg in (1, 2, 3) for kn in (6, 11)]


@pytest.mark.parametrize("degree,knots", BSPLINE_SETUPS)
@pytest.mark.parametrize("start,end", WINDOWS)
def test_bspline_monotonic(data, degree, knots, start, end):
    s = bs("x", knots=knots, degree=degree).increasing(start=start, end=end)
    _fit(s, data)
    _assert_shape(s, increasing, start, end)


@pytest.mark.parametrize("degree,knots", BSPLINE_SETUPS)
def test_bspline_monotonic_decreasing(data, degree, knots):
    s = bs("x", knots=knots, degree=degree).decreasing(start=3, end=8)
    _fit(s, data)
    _assert_shape(s, decreasing, 3, 8)


@pytest.mark.parametrize("degree,knots", BSPLINE_SETUPS)
@pytest.mark.parametrize("start,end", WINDOWS)
def test_bspline_convex(data, degree, knots, start, end):
    s = bs("x", knots=knots, degree=degree).convex(start=start, end=end)
    _fit(s, data)
    _assert_shape(s, convex, start, end)


@pytest.mark.parametrize("degree,knots", BSPLINE_SETUPS)
@pytest.mark.parametrize("start,end", WINDOWS)
def test_bspline_concave(data, degree, knots, start, end):
    s = bs("x", knots=knots, degree=degree).concave(start=start, end=end)
    _fit(s, data)
    _assert_shape(s, concave, start, end)


@pytest.mark.parametrize("method,check", [("convex", convex), ("increasing", increasing)])
def test_bspline_uneven_knots(data, method, check):
    s = getattr(bs("x", knots=np.array([0, 0.5, 1, 1.5, 4, 9, 10.0]), degree=3), method)()
    _fit(s, data)
    _assert_shape(s, check)


@pytest.mark.parametrize("method,check,start,end", [
    ("increasing", increasing, 1, 5),
    ("convex", convex, 2, 8),
])
def test_bspline_by_group_constraints(data, method, check, start, end):
    s = getattr(bs("x", knots=6, degree=3, by="g"), method)(start=start, end=end)
    _fit(s, data)
    _assert_shape(s, check, start, end)


def test_window_does_not_constrain_outside(data):
    """A local constraint must leave the rest of the curve free to follow the data."""
    s = bs("x", knots=11, degree=3).increasing(start=0, end=3)
    _fit(s, data)
    v = s.eval(GRID)
    assert not increasing(v[GRID >= 4])


@pytest.mark.parametrize("knots", [4, 8])
@pytest.mark.parametrize("start,end", WINDOWS)
def test_pwl_monotonic(data, knots, start, end):
    s = pwl("x", knots=knots).increasing(start=start, end=end)
    _fit(s, data)
    _assert_shape(s, increasing, start, end)


@pytest.mark.parametrize("knots", [4, 8])
@pytest.mark.parametrize("start,end", WINDOWS)
def test_pwl_convex_concave(data, knots, start, end):
    s = pwl("x", knots=knots).convex(start=start, end=end)
    _fit(s, data)
    _assert_shape(s, convex, start, end)
    s = pwl("x", knots=knots).concave(start=start, end=end)
    _fit(s, data)
    _assert_shape(s, concave, start, end)


def test_pwl_by_monotonic(data):
    s = pwl("x", knots=5, by="g").increasing(start=2, end=6)
    _fit(s, data)
    _assert_shape(s, increasing, 2, 6)


@pytest.mark.parametrize("by", [None, "g"])
def test_linear_monotonic(data, by):
    s = l("x", by=by).decreasing()
    _fit(s, data)
    _assert_shape(s, decreasing)


@pytest.mark.parametrize("make", [
    lambda: bs("x", knots=6, degree=1),
    lambda: pwl("x", knots=5),
])
def test_bound_is_exact_for_piecewise_linear(data, make):
    s = make().bounded(lower=-1, upper=1)
    _fit(s, data)
    v = s.eval(GRID)
    assert v.min() >= -1 - 1e-6 and v.max() <= 1 + 1e-6


@pytest.mark.parametrize("degree", [2, 3])
def test_bound_bspline_grid(data, degree):
    """Higher degrees are bounded on the grid; between grid points only a tiny overshoot is possible."""
    s = bs("x", knots=11, degree=degree).bounded(lower=-1, upper=1, n=200)
    _fit(s, data)
    v = s.eval(GRID)
    assert v.min() >= -1.01 and v.max() <= 1.01


@pytest.mark.parametrize("make", [
    lambda: bs("x", knots=8, degree=3),
    lambda: pwl("x", knots=5),
    lambda: l("x"),
])
def test_anchor(data, make):
    s = make().passes_through(points=[(5.0, 2.0), (1.0, -1.0)])
    _fit(s, data)
    np.testing.assert_allclose(s.eval(np.array([5.0, 1.0])), [2.0, -1.0], atol=1e-6)


def test_window_start_after_end_raises():
    with pytest.raises(ValueError):
        bs("x", knots=5).increasing(start=5, end=1)
    with pytest.raises(ValueError):
        bs("x", knots=5).convex(start=5, end=1)


def test_window_outside_domain_adds_no_constraints(data):
    s = bs("x", knots=6).increasing(start=20, end=30)
    _fit(s, data)
    assert s.constraints[0].build_constraint(s) == []
