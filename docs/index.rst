LPSpline Documentation
======================

LPSpline is a Python package for building and optimizing linear spline models using an intuitive additive API. It provides a flexible way to model non-linear relationships using various spline types like Piecewise Linear, B-Splines, Cyclic Splines, and Categorical Factors.

It uses a powerful `CVXPY` backend to formulate and solve the mathematical optimization problems behind the scenes.

Features
--------

- Intitutive Additive Model Definition
- Supports Linear, Piecewise Linear, B-Splines, Cyclic Splines, Categorical Factors, and Constants.
- Supports Shape Constraints: Monotonicity, Convexity, Concavity, Anchoring 
- Supports Penalties: Ridge, Lasso
- First-class support for `polars` DataFrames
- Built-in diagnostic visualizations

Installation
------------

You can install LPSpline via pip:

.. code-block:: bash

   pip install lpspline

Quick Start
-----------

Here is a small complete example demonstrating the core features:

.. code-block:: python

   import numpy as np
   import polars as pl
   from lpspline import l, pwl, bs, cs, f
   from lpspline.viz import plot_diagnostic

   # Data Generation
   n = 1000
   x_linear = np.linspace(0, 10, n)
   x_pwl = np.linspace(0, 10, n)
   x_bs = np.linspace(0, 10, n)
   x_cyc = np.linspace(0, 2*np.pi, n)
   x_factor = np.random.randint(0, 3, n)

   y = 0.5 * x_linear + np.where(x_pwl < 5, 0, x_pwl - 5) + np.sin(x_bs) + np.cos(x_cyc) + np.array([0, 2, -1])[x_factor] + np.random.normal(0, 0.2, n)

   df = pl.DataFrame({"xl": x_linear, "xpwl": x_pwl, "xbs": x_bs, "xcyc": x_cyc, "xfactor": x_factor, "target": y})

   # Model Definition
   model = (
       + l(term='xl')
       + pwl(term='xpwl', knots=3)
       + bs(term="xbs", knots=10, degree=2)
       + cs(term="xcyc", order=3)
       + f(term="xfactor")
   )

   # Fitting
   model.fit(df, df["target"])
   predictions = model.predict(df)

   # Visualization
   plot_diagnostic(model=model, X=df, y=df['target'], ncols=3)

Navigation
----------

.. toctree::
   :maxdepth: 2
   :caption: Resources:

   user_guide
   api

