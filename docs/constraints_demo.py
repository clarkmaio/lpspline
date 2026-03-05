import marimo

__generated_with = "0.20.4"
app = marimo.App()


@app.cell
def _():
    import numpy as np
    import polars as pl
    import matplotlib.pyplot as plt

    from lpspline import bs, l, pwl
    from lpspline.constraints import Anchor, Monotonic, Concave, Convex
    from lpspline.optimizer import LpRegressor
    from lpspline.penalties import Ridge
    from lpspline.viz import plot_diagnostic


    # Create a sample dataset
    np.random.seed(50)
    N = 1000
    x = np.linspace(-5, 5, N)
    by = np.random.randint(0, 3, N)
    y = np.sin(x) + np.random.normal(0, 0.2, N) + (0.9 + by)*x

    df = pl.DataFrame({"x": x, 'by': by,"y": y})
    return Convex, bs, df, np, plot_diagnostic


@app.cell
def _(Convex, bs, df, np, plot_diagnostic):
    anchor_points = [(0,1), (2, 0)]

    model = (
        +bs("x", knots=np.linspace(-10, 10, 20), degree=2, tag='bs', by='by')
        #+l('x', by='by')
        #.add_constraint(Monotonic(decreasing=True, start=0, end = 10))
        .add_constraint(Convex(start=0, end = 10))
    )

    model.fit(X=df, y=df['y'])



    plot_diagnostic(model=model, X=df, y=df["y"])
    return


if __name__ == "__main__":
    app.run()

