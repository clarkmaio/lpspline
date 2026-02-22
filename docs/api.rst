API Reference
=============

This page details the public API for the ``lpspline`` package, including the main regressor, the various spline types, penalties, and constraints.

Main Regressor
--------------

.. autoclass:: lpspline.optimizer.regressor.LpRegressor
   :members:
   :undoc-members:
   :show-inheritance:

Spline Fundamentals
-------------------

The building blocks of an additive spline model.

.. autoclass:: lpspline.spline.linear.Linear
   :members:
   :show-inheritance:

.. autoclass:: lpspline.spline.piecewise_linear.PiecewiseLinear
   :members:
   :show-inheritance:

.. autoclass:: lpspline.spline.bspline.BSpline
   :members:
   :show-inheritance:

.. autoclass:: lpspline.spline.cyclic_spline.CyclicSpline
   :members:
   :show-inheritance:

.. autoclass:: lpspline.spline.factor.Factor
   :members:
   :show-inheritance:

.. autoclass:: lpspline.spline.constant.Constant
   :members:
   :show-inheritance:

Constraints
-----------

.. autoclass:: lpspline.constraints.Monotonic
   :members:
   :show-inheritance:

.. autoclass:: lpspline.constraints.Convex
   :members:
   :show-inheritance:

.. autoclass:: lpspline.constraints.Concave
   :members:
   :show-inheritance:

.. autoclass:: lpspline.constraints.Anchor
   :members:
   :show-inheritance:

Penalties
---------

.. autoclass:: lpspline.penalties.Ridge
   :members:
   :show-inheritance:

.. autoclass:: lpspline.penalties.Lasso
   :members:
   :show-inheritance:
