"""Tests for the constraint API: spline methods, validation, groups, labels."""
import numpy as np
import polars as pl
import pytest

from lpspline import bs, c, cs, f, l, pwl
from lpspline.constraints import Anchor, Bound, ShapeConstraint

ALL = ["increasing", "decreasing", "convex", "concave", "bounded", "passes_through"]
SUPPORT = {
    "BSpline": (lambda: bs("x", knots=5), set(ALL)),
    "PiecewiseLinear": (lambda: pwl("x", knots=3), set(ALL)),
    "Linear": (lambda: l("x"), {"increasing", "decreasing", "bounded", "passes_through"}),
    "CyclicSpline": (lambda: cs("x", order=2), {"bounded", "passes_through"}),
    "Factor": (lambda: f("x"), {"bounded", "passes_through"}),
    "Constant": (lambda: c("x"), {"bounded", "passes_through"}),
}
ARGS = {"bounded": {"lower": 0}, "passes_through": {"points": [(1, 0)]}}


@pytest.mark.parametrize("name", SUPPORT)
@pytest.mark.parametrize("method", ALL)
def test_support_matrix_checked_when_adding(name, method):
    make, supported = SUPPORT[name]
    spline = make()
    call = getattr(spline, method)
    if method in supported:
        assert call(**ARGS.get(method, {})) is spline
        assert spline.constraints[-1].method == method
    else:
        with pytest.raises(TypeError, match=f"{name} does not support {method}\\(\\); supported:"):
            call(**ARGS.get(method, {}))


def test_methods_chain():
    s = bs("x", knots=8).increasing(start=1).convex(end=5).bounded(lower=0).passes_through(points=[(2, 1)])
    assert [c.method for c in s.constraints] == ["increasing", "convex", "bounded", "passes_through"]


@pytest.mark.parametrize("points", [None, "abc", [], [(1, 2, 3)], [1, 2], (1, 2)])
def test_passes_through_validates_points(points):
    with pytest.raises((TypeError, ValueError)):
        l("x").passes_through(points=points)


def test_bounded_validation():
    with pytest.raises(ValueError, match="at least one"):
        bs("x", knots=4).bounded()
    with pytest.raises(ValueError, match="lower"):
        bs("x", knots=4).bounded(lower=2, upper=1)


def test_groups_require_by():
    with pytest.raises(ValueError, match="by"):
        bs("x", knots=4).increasing(groups=["a"])


def test_factor_rejects_window():
    with pytest.raises(ValueError, match="start/end"):
        f("x").bounded(lower=0, start=1)


# ----------------------------------------------------------------- fitting

@pytest.fixture
def grouped():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 10, 400)
    g = np.where(np.arange(400) % 2 == 0, "up", "down")
    y = np.where(g == "up", 1, -1) * np.sin(x)
    return pl.DataFrame({"x": x, "g": g}), pl.Series(y + rng.normal(0, 0.05, 400))


def test_groups_constrain_only_selected(grouped):
    X, y = grouped
    s = bs("x", knots=8, by="g").increasing(groups=["up"])
    (+s).fit(X, y, summary=False)
    grid = np.linspace(0, 10, 300)
    up = s.eval(grid, by=np.full(300, "up"))
    down = s.eval(grid, by=np.full(300, "down"))
    assert np.all(np.diff(up) >= -1e-6)
    assert np.any(np.diff(down) < -1e-3)  # unconstrained group still follows the data


def test_unknown_group_raises_at_fit(grouped):
    X, y = grouped
    model = +bs("x", knots=5, by="g").increasing(groups=["nope"])
    with pytest.raises(ValueError, match="not found"):
        model.fit(X, y, summary=False)


def test_linear_bounded_uses_data_range():
    x = np.linspace(0, 10, 50)
    X = pl.DataFrame({"x": x})
    s = l("x").bounded(upper=5)
    (+s).fit(X, pl.Series(x), summary=False)
    assert s.eval(np.array([0.0, 10.0])).max() <= 5 + 1e-6


def test_factor_and_constant_bounded():
    X = pl.DataFrame({"k": ["a", "b", "c"] * 10})
    y = pl.Series(np.tile([-5.0, 0.0, 5.0], 10))
    model = f("k").bounded(lower=-1, upper=1) + c("k", tag="const").bounded(upper=0)
    model.fit(X, y, summary=False)
    assert np.all(np.abs(model.splines[0].coefficients) <= 1 + 1e-6)
    assert model.splines[1].coefficients[0] <= 1e-6


def test_factor_passes_through_category():
    X = pl.DataFrame({"k": ["a", "b"] * 10})
    model = +f("k").passes_through(points=[("a", 3.0)])
    model.fit(X, pl.Series(np.zeros(20)), summary=False)
    np.testing.assert_allclose(model.predict(pl.DataFrame({"k": ["a"]})), [3.0], atol=1e-6)


def test_conflict_detected_before_solving():
    X = pl.DataFrame({"x": np.linspace(0, 1, 20)})
    model = +bs("x", knots=4).bounded(lower=0, upper=1).passes_through(points=[(0.5, 2.0)])
    with pytest.raises(ValueError, match="conflicts with"):
        model.fit(X, pl.Series(np.zeros(20)), summary=False)


def test_anchor_outside_bound_window_is_not_a_conflict():
    X = pl.DataFrame({"x": np.linspace(0, 1, 20)})
    model = +bs("x", knots=4).bounded(upper=1, end=0.4).passes_through(points=[(0.8, 2.0)])
    model.fit(X, pl.Series(np.zeros(20)), summary=False)
    assert model.problem.status == "optimal"


def test_summary_shows_readable_constraints(capsys):
    X = pl.DataFrame({"hour": np.linspace(0, 23, 50)})
    model = +(
        bs("hour", knots=6, tag="h")
        .increasing(start=6)
        .bounded(lower=0, end=12)
        .passes_through(points=[(0, 0.0)])
    )
    model.fit(X, pl.Series(np.linspace(0, 1, 50)), summary=True)
    out = capsys.readouterr().out
    assert "increasing for hour ≥ 6" in out
    assert "bounded in [0, ∞] for hour ≤ 12" in out
    assert "passes through (0, 0)" in out


# --------------------------------------------------------- constraint objects

def test_constraint_objects_match_methods():
    s = (bs("x", knots=5).increasing().decreasing(end=1).convex().concave(start=2)
         .bounded(lower=0).passes_through(points=[(1, 2)]))
    kinds = [type(c) for c in s.constraints]
    assert kinds == [ShapeConstraint] * 4 + [Bound, Anchor]
    assert [(c.order, c.sign) for c in s.constraints[:4]] == [(1, 1), (1, -1), (2, 1), (2, -1)]
    assert s.constraints[-1].points == [(1, 2.0)]


def test_deprecated_api_is_removed():
    assert not hasattr(l("x"), "add_constraint")
    with pytest.raises(TypeError):
        Anchor((0.0, 1.0), (1.0, 2.0))
