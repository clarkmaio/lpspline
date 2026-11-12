import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl

    from lpspline import cs, l

    return cs, l, mo, np, pl, plt


@app.cell
def _(mo):
    mo.md(r"""
    # `train_on` and `active_where`

    Two chainable methods restrict a spline to a subset of rows, using a polars expression:

    | method | during `fit` | during `predict` |
    |---|---|---|
    | `train_on(cond)` | spline is 0 where `cond` is false | ignored: predicts everywhere |
    | `active_where(cond)` | spline is 0 where `cond` is false | spline is 0 where `cond` is false |

    ### API at a glance

    ```python
    import polars as pl
    from lpspline import cs, l

    spline = (
        cs("hour", order=3, period=24)
            .train_on(pl.col("day_index") < 1000)   # rows used to fit this spline
            .active_where(pl.col("feature3") <= 0)  # rows where this spline is non-zero
    )
    model = spline + l("flag")
    model.fit(X, y)                                 # prints a summary with the row subsets
    model.predict(X)                                # only active_where applies here
    model.predict(X, return_components=True)        # one column per spline, 0 where inactive

    # Inspect the conditions and the resulting boolean masks
    spline.train_condition, spline.active_condition  # the polars expressions
    spline.train_mask(X)                             # rows matching train_on
    spline.active_mask(X)                            # rows matching active_where
    spline.fit_mask(X)                               # train_on AND active_where: rows used in fit
    ```

    Conditions are any boolean polars expression (combine them with `&`, `|`, `~`);
    nulls count as `False`.

    We use a toy dataset: an hourly profile $\sin(2\pi\,hour/24)$ observed over 1500 days.
    """)
    return


@app.cell
def _(np, pl):
    rng = np.random.default_rng(0)
    n_days = 1500
    day_index = np.repeat(np.arange(n_days), 24)
    hour = np.tile(np.arange(24), n_days).astype(float)
    feature3 = rng.normal(size=day_index.size)
    time = day_index + hour / 24  # continuous time axis for plots

    true_profile = lambda h: np.sin(2 * np.pi * h / 24)

    X = pl.DataFrame({"day_index": day_index, "hour": hour, "feature3": feature3})
    hours_grid = np.linspace(0, 24, 200)
    return X, day_index, feature3, hour, hours_grid, rng, time, true_profile


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. `train_on`: learn a spline from a subset of the data

    After day 1000 a sensor breaks and the target becomes pure noise.
    We want to learn the hourly profile **only from the first 1000 days**.
    """)
    return


@app.cell
def _(day_index, hour, pl, rng, true_profile):
    y1 = true_profile(hour) + rng.normal(0, 0.1, hour.size)
    y1[day_index >= 1000] = rng.normal(0, 5, (day_index >= 1000).sum())  # broken sensor
    y1 = pl.Series("y", y1)
    return (y1,)


@app.cell
def _(day_index, plt, time, y1):
    _train = day_index < 1000
    fig_data1, ax_data1 = plt.subplots(figsize=(11, 3))
    ax_data1.scatter(time[_train], y1.to_numpy()[_train], s=1, alpha=0.3, color="C0",
                     label="day_index < 1000: used to fit (train_on)")
    ax_data1.scatter(time[~_train], y1.to_numpy()[~_train], s=1, alpha=0.3, color="C3",
                     label="day_index >= 1000: ignored by the fit")
    ax_data1.axvline(1000, color="k", lw=1)
    ax_data1.set_xlabel("day_index")
    ax_data1.set_ylabel("y")
    ax_data1.set_title("Whole training set")
    ax_data1.legend(loc="upper left", markerscale=10)
    fig_data1
    return


@app.cell
def _(mo):
    mo.md(r"""
    **The API**

    ```python
    model = +cs("hour", order=3, period=24).train_on(pl.col("day_index") < 1000)
    model.fit(X, y)
    ```
    """)
    return


@app.cell
def _(X, cs, pl, y1):
    # Without train_on: the broken period pollutes the fit
    model_all = +cs("hour", order=3, period=24)
    model_all.fit(X, y1, summary=False)

    # With train_on: only days < 1000 are used to fit the spline
    model_subset = +cs("hour", order=3, period=24).train_on(pl.col("day_index") < 1000)
    model_subset.fit(X, y1)  # warns: the spline will still predict on days >= 1000
    return model_all, model_subset


@app.cell
def _(hours_grid, model_all, model_subset, plt, true_profile):
    fig1, ax1 = plt.subplots(figsize=(7, 3.5))
    ax1.plot(hours_grid, true_profile(hours_grid), "k--", label="true profile")
    ax1.plot(hours_grid, model_all.splines[0].eval(hours_grid), label="fit on all rows")
    ax1.plot(hours_grid, model_subset.splines[0].eval(hours_grid), lw=3, alpha=0.7,
             label="train_on(day_index < 1000)")
    ax1.set_xlabel("hour")
    ax1.set_title("Learned hourly profile")
    ax1.legend()
    fig1
    return


@app.cell
def _(mo):
    mo.md(r"""
    The `train_on` spline recovers the true profile.

    The **warning** printed above is expected. `train_on` only affects fitting,
    so the spline still contributes on days ≥ 1000 at prediction time, even
    though it was 0 there during the fit. Add `active_where` if you want it off there too.

    ## 2. `active_where`: switch a spline off on some rows

    Now the hourly effect only exists when `feature3 <= 0`; when `feature3 > 0`
    the target is a flat level of 2. A `flag` column lets a linear term model that level.
    """)
    return


@app.cell
def _(X, feature3, hour, pl, rng, true_profile):
    y2 = pl.Series("y", (
        (feature3 <= 0) * true_profile(hour)
        + (feature3 > 0) * 2.0
        + rng.normal(0, 0.1, hour.size)
    ))
    X2 = X.with_columns(flag=(pl.col("feature3") > 0).cast(pl.Float64))
    return X2, y2


@app.cell
def _(feature3, hour, plt, rng, time, y2):
    _on = feature3 <= 0
    _jitter = rng.uniform(-0.3, 0.3, hour.size)
    fig_data2, (ax_t, ax_h) = plt.subplots(1, 2, figsize=(12, 3.5), width_ratios=[2, 1])
    for _mask, _color, _label in [(_on, "C0", "feature3 <= 0: spline active"),
                                  (~_on, "C1", "feature3 > 0: spline is 0")]:
        ax_t.scatter(time[_mask], y2.to_numpy()[_mask], s=1, alpha=0.2, color=_color, label=_label)
        ax_h.scatter(hour[_mask] + _jitter[_mask], y2.to_numpy()[_mask], s=1, alpha=0.1, color=_color)
    ax_t.set_xlabel("day_index")
    ax_t.set_ylabel("y")
    ax_t.set_title("Whole training set over time")
    ax_t.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2,
                markerscale=10, frameon=False)
    ax_h.set_xlabel("hour")
    ax_h.set_title("Same data by hour")
    fig_data2
    return


@app.cell
def _(mo):
    mo.md(r"""
    **The API**

    ```python
    model = (
        cs("hour", order=3, period=24, tag="hourly").active_where(pl.col("feature3") <= 0)
        + l("flag", bias=False, tag="flag")
    )
    model.fit(X, y)
    model.predict(X, return_components=True)[:, 0]  # hourly component: 0 when feature3 > 0
    ```
    """)
    return


@app.cell
def _(X2, cs, l, pl, y2):
    # Without active_where: the hourly spline tries to fit every row
    model_plain = cs("hour", order=3, period=24, tag="hourly") + l("flag", bias=False, tag="flag")
    model_plain.fit(X2, y2, summary=False)

    # With active_where: hourly spline is 0 when feature3 > 0, in fit AND predict
    model_active = (
        cs("hour", order=3, period=24, tag="hourly").active_where(pl.col("feature3") <= 0)
        + l("flag", bias=False, tag="flag")
    )
    model_active.fit(X2, y2)
    return model_active, model_plain


@app.cell
def _(X2, model_active, model_plain, np, plt, y2):
    def rmse(m):
        return np.sqrt(np.mean((m.predict(X2) - y2.to_numpy()) ** 2))

    comps = model_active.predict(X2, return_components=True)
    off = X2["feature3"].to_numpy() > 0

    fig2, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11, 3.5))
    first_days = slice(0, 24 * 3)
    ax_a.plot(y2.to_numpy()[first_days], "k.", label="target")
    ax_a.plot(model_plain.predict(X2)[first_days], label=f"plain (RMSE {rmse(model_plain):.2f})")
    ax_a.plot(model_active.predict(X2)[first_days], label=f"active_where (RMSE {rmse(model_active):.2f})")
    ax_a.set_title("Predictions, zoom on the first 3 days")
    ax_a.legend(fontsize=8)

    ax_b.hist(comps[off, 0], bins=20)
    ax_b.set_title("hourly component on rows with feature3 > 0")
    fig2
    return


@app.cell
def _(mo):
    mo.md(r"""
    With `active_where` the model fits the data, and the hourly component is
    **exactly 0** on every row with `feature3 > 0` (right panel).

    ## 3. Combining both

    Train on the first 1000 days **and** stay off when `feature3 > 0`.
    Each row of the training set falls in one of four groups:
    """)
    return


@app.cell
def _(day_index, feature3, np, plt, time, y2):
    _train, _active = day_index < 1000, feature3 <= 0
    _groups = [
        (_train & _active, "C0", "train & active: fits the spline"),
        (_train & ~_active, "C1", "train, inactive: spline is 0 (fit and predict)"),
        (~_train & _active, "C2", "outside train, active: 0 in fit, predicted"),
        (~_train & ~_active, "C7", "outside train, inactive: always 0"),
    ]
    fig_data3, ax_data3 = plt.subplots(figsize=(11, 3.5))
    for _mask, _color, _label in _groups:
        ax_data3.scatter(time[_mask], y2.to_numpy()[_mask], s=1, alpha=0.2, color=_color,
                         label=f"{_label} ({int(np.sum(_mask))} rows)")
    ax_data3.axvline(1000, color="k", lw=1)
    ax_data3.set_xlabel("day_index")
    ax_data3.set_ylabel("y")
    ax_data3.set_title("Whole training set, by row group")
    ax_data3.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2,
                    markerscale=10, fontsize=8, frameon=False)
    fig_data3
    return


@app.cell
def _(mo):
    mo.md(r"""
    **The API**

    ```python
    model = (
        cs("hour", order=3, period=24, tag="hourly")
            .train_on(pl.col("day_index") < 1000)
            .active_where(pl.col("feature3") <= 0)
        + l("flag", bias=False, tag="flag")
    )
    model.fit(X, y)
    ```

    The model summary below lists the conditions and the number of training rows
    (the blue group). The warning refers to the green group.
    """)
    return


@app.cell
def _(X2, cs, l, pl, y2):
    model_both = (
        cs("hour", order=3, period=24, tag="hourly")
            .train_on(pl.col("day_index") < 1000)
            .active_where(pl.col("feature3") <= 0)
        + l("flag", bias=False, tag="flag")
    )
    model_both.fit(X2, y2)
    hourly = model_both.get_spline("hourly")
    return (hourly,)


@app.cell
def _(X2, hourly, mo):
    mo.md(f"""
    **Inspecting the masks**

    ```python
    hourly = model_both.get_spline("hourly")
    hourly.train_mask(X).sum()   # {int(hourly.train_mask(X2).sum())} rows with day_index < 1000
    hourly.active_mask(X).sum()  # {int(hourly.active_mask(X2).sum())} rows with feature3 <= 0
    hourly.fit_mask(X).sum()     # {int(hourly.fit_mask(X2).sum())} rows actually used to fit the spline
    ```
    """)
    return


if __name__ == "__main__":
    app.run()
