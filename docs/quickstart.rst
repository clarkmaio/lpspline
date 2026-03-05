Quickstart
==========

Installation
------------

Install `lpspline` via pip directly from the command line:

.. code-block:: bash

   pip install lpspline

Basic Usage
-----------

LPSpline optimizes shape-constrained additive configurations utilizing convex methods (via CVXPY). 
It generates regression equations binding B-Splines, Linear blocks, or Categorical matrices smoothly.

.. code-block:: python

    import polars as pl
    from lpspline.optimizer import LpRegressor
    from lpspline.spline import BSpline, Linear
    from lpspline.constraints import Convex

    X = pl.DataFrame({"feature1": [1.0, 2.0, 3.0, 4.0, 5.0]})
    y = pl.Series([1.5, 3.2, 5.0, 6.7, 8.8])

    # Instantiate splines sequentially mapped against native DataFrame columns
    spline1 = BSpline(term="feature1", knots=5)

    # Attach desired mathematical shape-restraints seamlessly
    spline1.add_constraint(Convex())

    # Build the sequential model
    model = LpRegressor(splines=[spline1])

    # Fit against the data components
    model.fit(X, y)

    # Validate generated output distributions 
    predictions = model.predict(X)
