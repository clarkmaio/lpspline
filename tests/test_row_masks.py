"""Tests for `Spline.train_on` and `Spline.active_where`."""
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import pytest  # noqa: E402

from lpspline import Exp, LpRegressor, bs, cs, f, l  # noqa: E402
from lpspline.viz import plot_diagnostic, plot_interactive  # noqa: E402

N = 2400


def profile(hour: np.ndarray) -> np.ndarray:
    return np.sin(2 * np.pi * hour / 24)


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    day = np.arange(N) // 2
    hour = np.arange(N) % 24
    f3 = rng.normal(size=N)
    X = pl.DataFrame({"day_index": day, "hour": hour.astype(float), "feature3": f3})
    return X, rng


def hourly(**kwargs):
    return cs("hour", order=1, period=24, **kwargs)


def _no_outside_warning(record) -> bool:
    return not any("outside train_on" in str(w.message) for w in record)


# ----------------------------------------------------------------- train_on

def test_train_on_ignores_rows_outside_subset(data):
    X, rng = data
    early = X["day_index"].to_numpy() < 1000
    # The late period is pure garbage: it must not affect the fitted profile.
    y = np.where(early, profile(X["hour"].to_numpy()), rng.normal(0, 50, N))

    model = +hourly().train_on(pl.col("day_index") < 1000)
    with pytest.warns(UserWarning, match="outside train_on"):
        model.fit(X, pl.Series(y), summary=False)

    # Prediction applies everywhere (train_on is fit-only), with the early profile.
    np.testing.assert_allclose(model.predict(X), profile(X["hour"].to_numpy()), atol=1e-6)


def test_train_on_sets_knots_from_subset():
    x = np.linspace(0, 10, 200)
    X = pl.DataFrame({"x": x})
    spline = bs("x", knots=5).train_on(pl.col("x") <= 4).active_where(pl.col("x") <= 4)
    (+spline).fit(X, pl.Series(x), summary=False)
    np.testing.assert_allclose(spline.knots, np.linspace(0, x[x <= 4].max(), 5))


def test_train_on_sets_by_groups_from_subset():
    rng = np.random.default_rng(1)
    g = rng.choice(["a", "b", "c"], 300)
    x = rng.uniform(0, 1, 300)
    X = pl.DataFrame({"x": x, "g": g})
    spline = l("x", by="g").train_on(pl.col("g") != "c").active_where(pl.col("g") != "c")
    (+spline).fit(X, pl.Series(2 * x), summary=False)
    assert list(spline._by_classes) == ["a", "b"]
    pred = (+spline).predict(pl.DataFrame({"x": [0.5], "g": ["c"]}))
    np.testing.assert_allclose(pred, [0.0], atol=1e-8)


def test_nan_features_outside_training_rows(data):
    X, _ = data
    cond = pl.col("day_index") < 1000
    early = X["day_index"] < 1000
    X = X.with_columns(pl.when(cond).then(pl.col("hour")).otherwise(None).alias("hour"))
    y = pl.Series(np.nan_to_num(profile(X["hour"].to_numpy().astype(float))))
    spline = hourly().train_on(cond).active_where(cond)
    model = +spline
    model.fit(X, y, summary=False)
    assert np.all(np.isfinite(spline.coefficients))
    pred = model.predict(X)
    assert np.all(np.isfinite(pred))
    np.testing.assert_allclose(pred[~early.to_numpy()], 0.0)


def test_train_on_with_constraint(data):
    X, rng = data
    x = X["hour"].to_numpy()
    y = np.where(X["day_index"].to_numpy() < 1000, -x, rng.normal(0, 10, N))
    spline = bs("hour", knots=6).train_on(pl.col("day_index") < 1000).increasing()
    with pytest.warns(UserWarning):
        (+spline).fit(X, pl.Series(y), summary=False)
    v = spline.eval(np.linspace(0, 23, 200))
    assert np.all(np.diff(v) >= -1e-6)


# ------------------------------------------------------------- active_where

def test_active_where_is_zero_in_fit_and_predict(data):
    X, _ = data
    hour, f3 = X["hour"].to_numpy(), X["feature3"].to_numpy()
    y = np.where(f3 <= 0, profile(hour), 3.0)
    X = X.with_columns(flag=pl.Series((f3 > 0).astype(int)))

    spline = hourly(tag="hourly").active_where(pl.col("feature3") <= 0)
    model = spline + l("flag", tag="flag")  # the flag explains the inactive rows
    model.fit(X, pl.Series(y), summary=False)

    np.testing.assert_allclose(model.predict(X), y, atol=1e-6)
    comps = model.predict(X, return_components=True)
    np.testing.assert_array_equal(comps[f3 > 0, 0], 0.0)
    np.testing.assert_allclose(comps[f3 <= 0, 0], profile(hour[f3 <= 0]) - comps[f3 <= 0, 1], atol=1e-6)


def test_active_where_inside_training_period_is_used_in_fit(data):
    """Inactive rows inside the training window must not inform the spline."""
    X, rng = data
    hour, f3 = X["hour"].to_numpy(), X["feature3"].to_numpy()
    early = X["day_index"].to_numpy() < 1000
    y = np.where(early & (f3 <= 0), profile(hour), rng.normal(0, 50, N))

    spline = (
        hourly()
        .train_on(pl.col("day_index") < 1000)
        .active_where(pl.col("feature3") <= 0)
    )
    model = +spline
    with pytest.warns(UserWarning, match="outside train_on"):
        model.fit(X, pl.Series(y), summary=False)

    pred = model.predict(X)
    np.testing.assert_allclose(pred[f3 <= 0], profile(hour[f3 <= 0]), atol=1e-6)
    np.testing.assert_array_equal(pred[f3 > 0], 0.0)


def test_no_warning_when_active_covers_untrained_rows(data):
    X, _ = data
    cond = pl.col("day_index") < 1000
    model = +hourly().train_on(cond).active_where(cond)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        model.fit(X, pl.Series(profile(X["hour"].to_numpy())), summary=False)
    assert _no_outside_warning(record)


def test_nulls_in_condition_count_as_false():
    X = pl.DataFrame({"x": [0.0, 1.0, 2.0, 3.0], "c": [1, None, 1, -1]})
    spline = l("x").active_where(pl.col("c") > 0)
    np.testing.assert_array_equal(spline.active_mask(X), [True, False, True, False])


# ------------------------------------------------------------------- errors

def test_empty_training_rows_raise(data):
    X, _ = data
    model = +hourly().train_on(pl.col("day_index") < 0)
    with pytest.raises(ValueError, match="no rows to train on"):
        model.fit(X, pl.Series(np.zeros(N)), summary=False)


@pytest.mark.parametrize("condition,error", [
    ("day_index < 1000", TypeError),
    (lambda X: X["day_index"] < 1000, TypeError),
])
def test_condition_must_be_polars_expression(condition, error):
    with pytest.raises(error, match="polars expression"):
        hourly().train_on(condition)
    with pytest.raises(error, match="polars expression"):
        hourly().active_where(condition)


def test_non_boolean_condition_raises(data):
    X, _ = data
    model = +hourly().active_where(pl.col("day_index"))
    with pytest.raises(TypeError, match="booleans"):
        model.fit(X, pl.Series(np.zeros(N)), summary=False)


def test_missing_condition_column_raises(data):
    X, _ = data
    model = +hourly().train_on(pl.col("nope") > 0)
    with pytest.raises(ValueError, match="column"):
        model.fit(X, pl.Series(np.zeros(N)), summary=False)


def test_aggregated_condition_raises(data):
    X, _ = data
    model = +hourly().train_on(pl.col("day_index").max() > 0)
    with pytest.raises(ValueError, match="one value per row"):
        model.fit(X, pl.Series(np.zeros(N)), summary=False)


def test_weakly_determined_warning(data):
    X, _ = data
    model = +bs("hour", knots=10).train_on(pl.col("day_index") < 2).active_where(pl.col("day_index") < 2)
    with pytest.warns(UserWarning, match="weakly determined"):
        model.fit(X, pl.Series(np.zeros(N)), summary=False)


# ----------------------------------------------------- integration / output

def test_save_load_keeps_conditions(data, tmp_path):
    X, _ = data
    hour, f3 = X["hour"].to_numpy(), X["feature3"].to_numpy()
    spline = hourly().train_on(pl.col("day_index") < 1000).active_where(pl.col("feature3") <= 0)
    model = +spline
    with pytest.warns(UserWarning):
        model.fit(X, pl.Series(np.where(f3 <= 0, profile(hour), 0.0)), summary=False)
    model.save(tmp_path / "m.pkl")
    loaded = LpRegressor.load(tmp_path / "m.pkl")
    assert str(loaded.splines[0].active_condition) == str(spline.active_condition)
    np.testing.assert_allclose(loaded.predict(X), model.predict(X))


def test_link_model_with_active_where(data):
    X, _ = data
    f3 = X["feature3"].to_numpy()
    y = np.where(f3 <= 0, np.exp(profile(X["hour"].to_numpy())), 1.0)
    model = Exp(+hourly().active_where(pl.col("feature3") <= 0))
    model.fit(X, pl.Series(y), summary=False)
    np.testing.assert_allclose(model.predict(X), y, atol=1e-5)


def test_summary_lists_row_subsets(data, capsys):
    X, _ = data
    model = +hourly().train_on(pl.col("day_index") < 1000).active_where(pl.col("day_index") < 1000)
    model.fit(X, pl.Series(profile(X["hour"].to_numpy())))
    out = capsys.readouterr().out
    assert "Details" in out and "train_on" in out and "active_where" in out
    assert "(2000 training rows)" in out


def test_plots_with_masks(data):
    X, _ = data
    f3 = X["feature3"].to_numpy()
    X = X.with_columns(g=pl.Series(np.arange(N) % 2))
    y = pl.Series(np.where(f3 <= 0, profile(X["hour"].to_numpy()), 0.0))
    model = (
        hourly(tag="hourly").active_where(pl.col("feature3") <= 0)
        + l("day_index", by="g", tag="trend").active_where(pl.col("day_index") < 500)
        + f("g", tag="g")
    )
    model.fit(X, y, summary=False)
    fig, _ = plot_diagnostic(model=model, X=X, y=y)
    plt.close(fig)
    chart = plot_interactive(model, X, y, xcol="hour")
    assert "vconcat" in chart.to_dict()
