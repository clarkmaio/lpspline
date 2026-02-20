# LPSpline

LPSpline is a Python package for building and optimizing linear spline models using an intuitive additive API. It provides a flexible way to model non-linear relationships using various spline types like Piecewise Linear, B-Splines, Cyclic Splines, and Categorical Factors.

## Installation

Install `lpspline` via pip directly from the repository, or if published:

```bash
pip install lpspline
```


## Quick Start

LPSpline allows you to easily compose additive models. Here's a quick example:

```python
import numpy as np
import polars as pl
from lpspline import l, pwl, bs



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
    "linear_col": x_linear,
    "pwl_col": x_pwl,
    "bs_col": x_bs,
    "cyc_col": x_cyc,
    "factor_col": x_factor,
    "target": y
})

# ---------------------------------------- Model Definition
model = (
    l(term='linear_col', bias=True)
    + pwl(term='pwl_col', knots=[5.])
    + bs(term="bs_col", knots=np.linspace(0, 10, 5), degree=3)
    + cs(term="cyc_col", period=2*np.pi, order=2)
    + f(term="factor_col", n_classes=3)
)

# ---------------------------------------- Model Fitting
model.fit(df, df["target"])

# ---------------------------------------- Model Prediction
predictions = model.predict(df)
```

## Expected output

Once the model is fitted, you will see a detailed summary to the console:

```
==================================================
✨ Model Summary ✨
==================================================
Problem Status: optimal
--------------------------------------------------
Spline Type       | Term            | Params    
--------------------------------------------------
🟢 Linear         | x1              | 2         
🟢 Piecewise      | x2              | 2         
🟢 BSpline        | x3              | 5         
--------------------------------------------------
📊 Total Parameters                 | 9         
==================================================
```

## Demo with multiple variables

Inside the `notebook/` folder you will find a `demo.ipynb` file which plots the learned spline components automatically:

![LPSpline Visualization](assets/demo_plot.png)
