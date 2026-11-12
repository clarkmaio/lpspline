
:html_theme.sidebar_secondary.remove:


Quickstart
==========

Installation
------------

Install ``lpspline`` via pip directly from the command line:

.. code-block:: bash

   pip install lpspline

Basic Usage
-----------

LPSpline optimizes shape-constrained additive configurations utilizing convex methods (via CVXPY). 
It generates regression equations binding B-Splines, Linear blocks, or Categorical matrices smoothly.

For each spline type there is a dedicated class you can import

.. code-block:: python

   from lpspline import c, f, l, pwl, bs, cs


* ``c``: constant
* ``f``: factor
* ``l``: linear
* ``pwl``: piecewise linear
* ``bs``: B-spline
* ``cs``: cyclic spline


LpRegressor
-----------

The class ``LpRegressor`` is a sklearn-like interface to perform a regression with your data.

The easiest way to initialize an instance is to simply write down the formula.


.. code-block:: python

   from lpspline import c, f

   mymodel = (
      l(term='x')
      + f(term='c')
   )


`term` is the name of the column in the DataFrame X the spline is fitted on.


.. tip::

   If your model consist in just one spline you can initialize the instance like so:

   .. code-block:: python

      mymodel = +l(term='x')



You can still initialize the ``LpRegressor`` creating an instance in the classical way:

.. code-block:: python

   mymodel = LpRegressor(splines=[l(term='x'), f(term='c')])



Fit and predict   
----------------

``LpRegressor`` is compatible with sklearn syntax. It works with ``polars`` DataFrames and Series.


.. code-block:: python

   import polars as pl
   import numpy as np
   from lpspline import c, f

   X = pl.DataFrame({
      'x': np.linspace(0, 10, 100),
      'c': np.random.randint(0, 10, 100)
   })
   y = pl.Series(X['x'] + X['c'] + np.random.randn(100))

   mymodel = l(term='x') + f(term='c')
   mymodel.fit(X, y)
   y_pred = mymodel.predict(X)



Summary
-------

When you fit the model a summary is printed to the console with all main informations:

* spline type
* term
* applied constraints
* applied penalties
* number of parameters


.. code-block:: text

   ========================================================================================================================
   ✨ Model Summary ✨
   ========================================================================================================================
   Problem Status: ✅ optimal
   ------------------------------------------------------------------------------------------------------------------------
   Spline Type          | Term         | Tag             | Constraints          | Penalties            | Params  
   ------------------------------------------------------------------------------------------------------------------------
   🟢 Linear            | x            | linearspline    | None                 | None                 | 2       
   🟢 Factor            | c            | factorspline    | None                 | None                 | 10      
   ------------------------------------------------------------------------------------------------------------------------
   📊 Total Parameters                                                                                 | 12
   ========================================================================================================================





Training and activity subsets
-----------------------------

Each spline can be restricted to a subset of rows with two chainable methods
taking a boolean polars expression (nulls count as ``False``):

* ``train_on(condition)``: the spline is **fitted** only on the rows matching
  ``condition``. On the other rows it contributes 0 during the fit, so the rest
  of the model explains them. Automatic knots, cyclic period, ``by`` groups and
  factor classes are computed from these rows only. At prediction time the
  condition is ignored.
* ``active_where(condition)``: the spline returns 0 on the rows **not** matching
  ``condition``, both when fitting and when predicting.

.. code-block:: python

   import polars as pl
   from lpspline import cs, l

   model = (
       cs(term="hour", order=3, period=24)
           .train_on(pl.col("day_index") < 1000)    # learn the profile on early data
           .active_where(pl.col("feature3") <= 0)   # and switch it off when feature3 > 0
       + l(term="temperature")
   )
   model.fit(X, y)

During the fit, a spline contributes only where **both** conditions hold.
When predicting, only ``active_where`` is applied:

=================================  ==============  =================
Row                                During ``fit``  During ``predict``
=================================  ==============  =================
train and active                   fitted          contributes
inside train, not active           0               0
outside train, active              0               contributes
outside train, not active          0               0
=================================  ==============  =================

.. warning::

   A spline using ``train_on`` contributes at prediction time on rows where it
   was 0 during the fit ("outside train, active"). The other splines have
   already explained those rows, so this may double count. ``fit`` emits a
   ``UserWarning`` when this happens on the training data; add an
   ``active_where`` condition if the spline should be off there.

Saving and Loading
------------------

You can save your trained ``LpRegressor`` model to a file and load it back later. This is useful for persisting models or sharing them.

.. code-block:: python

   # Save the model
   model.save("my_model.pkl")

   # Load the model back
   from lpspline import LpRegressor
   loaded_model = LpRegressor.load("my_model.pkl")
   
   # Use the loaded model for predictions
   predictions = loaded_model.predict(X)


.. note::

   The model is saved using ``pickle``. When saving, the underlying optimization problem state is cleared to ensure the object is serializable. The loaded model is fully functional for predictions but cannot be further optimized without re-fitting.

Diagnostics
-----------

``plot_diagnostic`` is a function that shows the diagnostic of the model.

It shows each fitted spline separately.

If you pass target to argument ``y: pl.Series`` it will be shown residuals relative to each spline as well.

In practice for each spline :math:`s_i` it will be shown:

.. math::

   r_i = y - \sum_{j \neq i} s_j(X)




.. code-block:: python

   from lpspline.viz import plot_diagnostic

   plot_diagnostic(model=mymodel, X=X, y=y)



.. image:: ../assets/quickstart_diagnostic.png
   :align: center
   :scale: 100 %
   :alt: LPSpline Diagnostic Plot
