# LPSpline

LPSpline is a Python package for building and optimizing linear spline models using an intuitive additive API. It provides a flexible way to model non-linear relationships using various spline types like Piecewise Linear, B-Splines, Cyclic Splines, and Categorical Factors.

## Installation

Install `lpspline` via pip directly from the repository, or if published:

```bash
pip install lpspline
```


## Main Features

- Additive model definition
- CVXPY backend for optimization
- Multiple spline types: Linear, Piecewise Linear, B-Splines, Cyclic Splines, Categorical Factors, Constant
- Penalties on the splines: Ridge, Lasso
- Constraints on the splines: Monotonic, Anchor
- Polars DataFrame integration
- Nice plots using `matplotlib` and `pimpmyplot`

## Sandbox

Visit the [marimo playground](https://molab.marimo.io/notebooks/nb_MWvkutFSuuXinahopPY1NA) for a live demo

## Quick Start

Here a small code example:

```python
import numpy as np
import polars as pl
from lpspline import l, pwl, bs, cs
from lpspline.viz import plot_diagnostic


# ---------------------------------------- Data Generation

n = 1000

# Regressors
x_linear = np.linspace(0, 10, n)
x_pwl = np.linspace(0, 10, n)
x_bs = np.linspace(0, 10, n)
x_cyc = np.linspace(0, 2*np.pi, n)
x_factor = np.random.randint(0, 3, n)

# Target
y_linear = 0.5 * x_linear
y_pwl = np.where(x_pwl < 5, 0, x_pwl - 5)
y_bs = np.sin(x_bs) 
y_cyc = np.cos(x_cyc)
y_factor = np.array([0, 2, -1])[x_factor]

y = y_linear + y_pwl + y_bs + y_cyc + y_factor + np.random.normal(0, 0.2, n)

df = pl.DataFrame({
    "xl": x_linear,
    "xpwl": x_pwl,
    "xbs": x_bs,
    "xcyc": x_cyc,
    "xfactor": x_factor,
    "target": y
})

# ---------------------------------------- Model Definition
model = (
    +l(term='xl')
    + pwl(term='xpwl', knots=3)
    + bs(term="xbs", knots=10, degree=2)
    + cs(term="xcyc", order=3)
    + f(term="xfactor")
)
# ---------------------------------------- Model Fitting
model.fit(df, df["target"])

# ---------------------------------------- Model Prediction
predictions = model.predict(df)

# ---------------------------------------- Model Visualization
plot_diagnostic(model=model, X=df, y=df['target'], ncols=3)
```

## Expected output

Once the model is fitted, you will see a detailed summary to the console and a diagnostic plot showing the fitted splines.

```
========================================================================================================================
✨ Model Summary ✨
========================================================================================================================
Problem Status: ✅ optimal
------------------------------------------------------------------------------------------------------------------------
Spline Type          | Term         | Tag             | Constraints          | Penalties            | Params 
------------------------------------------------------------------------------------------------------------------------
🟢 Linear            | xl           | linear          | None                 | None                 | 2       
🟢 PiecewiseLinear   | xpwl         | pwl             | None                 | None                 | 5       
🟢 BSpline           | xbs          | bspline         | None                 | None                 | 11      
🟢 CyclicSpline      | xcyc         | cyclicspline    | None                 | None                 | 7       
🟢 Factor            | xfactor      | factor          | None                 | None                 | 5       
------------------------------------------------------------------------------------------------------------------------
📊 Total Parameters                                                                                 | 29
========================================================================================================================

Model fitted successfully.
```


![LPSpline Visualization](assets/demo_plot.png)
