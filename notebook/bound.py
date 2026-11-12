import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    from lpspline import bs, LpRegressor, l, pwl, cs
    import numpy as np
    import matplotlib.pyplot as plt
    import polars as pl
    from lpspline.datasets import load_by_dataset
    from lpspline.viz import plot_diagnostic

    return bs, np, pl, plot_diagnostic


@app.cell
def _(np, pl):
    X = pl.DataFrame({'x': np.linspace(0, 10, 1000)})
    y = pl.Series(np.sin(X['x']*2) * X['x'] + np.random.normal(0, 1, 1000))
    return X, y


@app.cell
def _(X, bs, plot_diagnostic, y):
    estimator = (
        +bs(term='x', knots=20)
            .bounded(lower=0, upper=3, n=1000, start=6, end=100)
    )
    estimator.fit(X, y)
    fig, _axes = plot_diagnostic(model=estimator, X=X, y=y)
    fig
    return


if __name__ == "__main__":
    app.run()
