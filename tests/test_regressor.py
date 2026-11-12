"""Feature tests for LpRegressor: fitting, prediction, composition, persistence."""
import numpy as np
import polars as pl
import pytest

from lpspline import LpRegressor, Exp, Link, Log, Sigmoid, bs, c, cs, f, l, pwl
from lpspline.constraints import Monotonic
from lpspline.datasets import load_by_dataset, load_demo_dataset
from lpspline.penalties import Lasso, Ridge


@pytest.fixture(scope="module")
def demo():
    np.random.seed(0)
    return load_demo_dataset(samples=600)


def _demo_model() -> LpRegressor:
    return (
        +l(term="xl", by="xfactor")
        + pwl(term="xpwl", knots=3)
        + bs(term="xbs", knots=10, degree=2)
        + cs(term="xcyc", order=3)
        + f(term="xfactor")
    )


def test_demo_model_fits_well(demo):
    X, y = demo
    model = _demo_model()
    model.fit(X, y, summary=False)
    assert model.problem.status == "optimal"
    rmse = np.sqrt(np.mean((model.predict(X) - y.to_numpy()) ** 2))
    assert rmse < 0.3  # noise std is 0.2


def test_components_sum_to_prediction(demo):
    X, y = demo
    model = _demo_model()
    model.fit(X, y, summary=False)
    comps = model.predict(X, return_components=True)
    assert comps.shape == (len(X), len(model.splines))
    np.testing.assert_allclose(comps.sum(axis=1), model.predict(X))


def test_summary_printed(demo, capsys):
    X, y = demo
    model = +bs("xbs", knots=5)
    model.fit(X, y)
    out = capsys.readouterr().out
    assert "Model Summary" in out and "optimal" in out
    assert "| 7 " in out  # 5 knots + degree 3 - 1 parameters


def test_summary_before_fit_raises():
    with pytest.raises(ValueError):
        (+l("x")).summary()


@pytest.mark.parametrize("kind", ["linear", "cubic", "cyclic"])
def test_by_dataset(kind):
    np.random.seed(1)
    X, y = load_by_dataset(samples=300, type=kind)
    spline = {"linear": l("x", by="by"), "cubic": bs("x", knots=6, by="by"), "cyclic": cs("x", order=2, period=2 * np.pi, by="by")}[kind]
    model = +spline
    model.fit(X, y, summary=False)
    assert model.problem.status == "optimal"
    assert np.corrcoef(model.predict(X), y.to_numpy())[0, 1] > 0.9


def test_load_by_dataset_invalid_type():
    with pytest.raises(ValueError):
        load_by_dataset(type="nope")


def test_string_factor_and_by():
    rng = np.random.default_rng(0)
    n = 300
    cat = rng.choice(["a", "b", "c"], n)
    x = rng.uniform(0, 1, n)
    effect = {"a": 1.0, "b": -2.0, "c": 0.5}
    slope = {"a": 1.0, "b": 3.0, "c": -1.0}
    y = np.array([effect[k] + slope[k] * xi for k, xi in zip(cat, x)])
    X = pl.DataFrame({"x": x, "cat": cat})

    model = l("x", by="cat", bias=False) + f("cat")
    model.fit(X, pl.Series(y), summary=False)
    np.testing.assert_allclose(model.predict(X), y, atol=1e-4)

    # Unseen category contributes 0 instead of crashing.
    pred = model.predict(pl.DataFrame({"x": [0.5], "cat": ["zzz"]}))
    np.testing.assert_allclose(pred, [0.0], atol=1e-8)


def test_constant_intercept():
    # A Constant still reads its `term` column, only to know the number of rows.
    X = pl.DataFrame({"intercept": np.zeros(10)})
    model = +c("intercept")
    model.fit(X, pl.Series(np.full(10, 4.2)), summary=False)
    np.testing.assert_allclose(model.predict(X), 4.2, atol=1e-6)


@pytest.mark.parametrize("penalty", [Ridge(alpha=1e4), Lasso(alpha=1e4)])
def test_penalties_shrink_coefficients(demo, penalty):
    X, y = demo
    free = +bs("xbs", knots=8, tag="free")
    free.fit(X, y, summary=False)
    pen = +bs("xbs", knots=8, tag="pen").add_penalty(penalty)
    pen.fit(X, y, summary=False)
    assert np.abs(pen.splines[0].coefficients).sum() < 0.1 * np.abs(free.splines[0].coefficients).sum()


def test_add_penalty_type_check():
    with pytest.raises(TypeError):
        l("x").add_penalty("ridge")


def test_refit_with_different_groups_and_range():
    rng = np.random.default_rng(0)
    X1 = pl.DataFrame({"x": np.linspace(0, 10, 100), "g": rng.integers(0, 2, 100)})
    X2 = pl.DataFrame({"x": np.linspace(0, 20, 100), "g": rng.integers(0, 3, 100)})
    y = pl.Series(np.sin(np.linspace(0, 10, 100)))

    model = bs("x", knots=5, by="g") + f("g")
    model.fit(X1, y, summary=False)
    model.fit(X2, y, summary=False)
    spline = model.get_spline("bspline")
    assert spline.coefficients.shape == (5 + 3 - 1, 3)
    np.testing.assert_allclose(spline.knots, np.linspace(0, 20, 5))
    assert model.get_spline("factor").n_classes == 3


def test_unique_tags_enforced_everywhere():
    with pytest.raises(ValueError):
        l("a") + l("b")
    with pytest.raises(ValueError):
        +l("a") + l("b")
    with pytest.raises(ValueError):
        l("a") + (+l("b"))
    model = l("a", tag="la") + l("b", tag="lb")
    assert model.get_spline("lb").term == "b"
    with pytest.raises(ValueError):
        model.get_spline("missing")


def test_add_order_and_type_errors():
    model = l("b", tag="b") + (+l("a", tag="a"))
    assert [s.tag for s in model.splines] == ["b", "a"]
    with pytest.raises(TypeError):
        l("a") + 3
    with pytest.raises(TypeError):
        (+l("a")) + 3


def test_missing_column_raises(demo):
    X, y = demo
    model = +l("nope")
    with pytest.raises(Exception):
        model.fit(X, y, summary=False)


def test_length_mismatch_raises(demo):
    X, y = demo
    with pytest.raises(ValueError):
        (+l("xl")).fit(X, y[:10], summary=False)


def test_unfitted_spline_eval_raises():
    with pytest.raises(ValueError, match="not been fitted"):
        bs("x", knots=4).eval(np.array([1.0]))


def test_eval_return_basis(demo):
    X, y = demo
    model = +pwl("xpwl", knots=3)
    model.fit(X, y, summary=False)
    basis = model.splines[0].eval(np.array([0.0, 5.0]), return_basis=True)
    assert basis.shape == (2, 5)


def test_save_load_roundtrip(demo, tmp_path):
    X, y = demo
    model = _demo_model()
    model.fit(X, y, summary=False)
    path = tmp_path / "model.pkl"
    model.save(path)
    assert model.problem is not None  # saving must not mutate the live model
    loaded = LpRegressor.load(path)
    assert loaded.problem is None
    np.testing.assert_allclose(loaded.predict(X), model.predict(X))


def test_solver_choice_and_kwargs(demo):
    X, y = demo
    model = +bs("xbs", knots=6)
    model.fit(X, y, summary=False, solver="SCS", eps=1e-6)
    assert model.problem.solver_stats.solver_name == "SCS"
    model.fit(X, y, summary=False)
    assert model.problem.solver_stats.solver_name == "CLARABEL"


def test_non_optimal_status_warns(demo):
    X, y = demo
    model = +bs("xbs", knots=6).add_constraint(Monotonic())
    with pytest.warns(RuntimeWarning, match="status"):
        model.fit(X, y, summary=False, solver="SCS", max_iters=1)


# ----------------------------------------------------------------- links

def _positive_data():
    x = np.linspace(0, 4, 200)
    return pl.DataFrame({"x": x}), x


def test_log_link_recovers_exponential():
    X, x = _positive_data()
    y = pl.Series("y", np.exp(0.5 * x + 1))
    model = Log(+l("x"))
    model.fit(X, y, summary=False)
    np.testing.assert_allclose(model.predict(X), y.to_numpy(), rtol=1e-4)
    np.testing.assert_allclose(model.splines[0].coefficients, [1, 0.5], atol=1e-4)


def test_sigmoid_link_bounded():
    X, x = _positive_data()
    p = 1 / (1 + np.exp(-(x - 2)))
    model = Sigmoid(+l("x"))
    model.fit(X, pl.Series("y", p), summary=False)
    pred = model.predict(X)
    assert np.all((pred > 0) & (pred < 1))
    np.testing.assert_allclose(pred, p, atol=1e-4)


def test_exp_link():
    X, x = _positive_data()
    y = pl.Series("y", np.log(x + 1))
    model = Exp(+l("x"))
    model.fit(X, y, summary=False)
    assert np.all(np.isfinite(model.predict(X)))


def test_link_components_not_transformed():
    X, x = _positive_data()
    model = Log(+l("x"))
    model.fit(X, pl.Series("y", np.exp(x)), summary=False)
    comps = model.predict(X, return_components=True)
    np.testing.assert_allclose(comps[:, 0], x, atol=1e-4)


@pytest.mark.parametrize("wrapper", [Log, Sigmoid, lambda r: Link(r)])
def test_link_save_load(tmp_path, wrapper):
    X, x = _positive_data()
    y = pl.Series("y", 1 / (1 + np.exp(-(x - 2))))
    model = wrapper(+bs("x", knots=5))
    model.fit(X, y, summary=False)
    model.save(tmp_path / "m.pkl")
    loaded = LpRegressor.load(tmp_path / "m.pkl")
    assert type(loaded) is type(model)
    np.testing.assert_allclose(loaded.predict(X), model.predict(X))

