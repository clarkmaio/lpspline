# LPSpline

LPSpline is a Python package for building and optimizing linear spline models using an intuitive additive API. It provides a flexible way to model non-linear relationships using various spline types like Piecewise Linear, B-Splines, Cyclic Splines, and Categorical Factors.

## Installation

Install `lpspline` via pip directly from the repository, or if published:

```bash
pip install lpspline
```


## Quick Start
Visit the [marimo playground](https://molab.marimo.io/notebooks/nb_MWvkutFSuuXinahopPY1NA) to try out the package.


Here a small code example:

```python
import numpy as np
import polars as pl
from lpspline import l, pwl, bs, cs



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
    l(term='xl', bias=True)
    + pwl(term='xpwl', knots=[5.])
    + bs(term="xbs", knots=np.linspace(0, 10, 5), degree=3)
    + cs(term="xcyc", period=2*np.pi, order=2)
    + f(term="xfactor", n_classes=3)
)

# ---------------------------------------- Model Fitting
model.fit(df, df["target"])

# ---------------------------------------- Model Prediction
predictions = model.predict(df)

# ---------------------------------------- Model Visualization
plot_components(model=model, df=df, ncols=3)
```

## Expected output

Once the model is fitted, you will see a detailed summary to the console:

```
==================================================
✨ Model Summary ✨
==================================================
Problem Status: ✅ optimal
--------------------------------------------------
Spline Type          | Term            | Params    
--------------------------------------------------
🟢 Linear            | linear_col      | 2         
🟢 PiecewiseLinear   | pwl_col         | 3         
🟢 BSpline           | bs_col          | 1         
🟢 CyclicSpline      | cyc_col         | 5         
🟢 Factor            | factor_col      | 3         
--------------------------------------------------
📊 Total Parameters                    | 14        
==================================================

Model fitted successfully.
```


![LPSpline Visualization](assets/demo_plot.png)
