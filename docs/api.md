# API Reference

This page details the public API for the `lpspline` package, including the main regressor, the various spline types, penalties, and constraints.

## Main Regressor

::: lpspline.optimizer.regressor.LpRegressor

## Spline Fundamentals

The building blocks of an additive spline model.

::: lpspline.spline.linear.Linear
::: lpspline.spline.piecewise_linear.PiecewiseLinear
::: lpspline.spline.bspline.BSpline
::: lpspline.spline.cyclic_spline.CyclicSpline
::: lpspline.spline.factor.Factor
::: lpspline.spline.constant.Constant

## Constraints

::: lpspline.constraints.Monotonic
::: lpspline.constraints.Convex
::: lpspline.constraints.Concave
::: lpspline.constraints.Anchor

## Penalties

::: lpspline.penalties.Ridge
::: lpspline.penalties.Lasso

## Datasets

::: lpspline.datasets
