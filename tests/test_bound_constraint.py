import pytest
import numpy as np
import polars as pl
from lpspline.spline import PiecewiseLinear, BSpline, CyclicSpline, Linear
from lpspline import LpRegressor
from lpspline.constraints import Bound

class TestBoundConstraint:
    def test_bspline_bound_lower(self):
        # Data: y follows x^2, but we bound it to be >= 0.5
        x = np.linspace(-1, 1, 20)
        y = x**2
        df = pl.DataFrame({"x": x})
        
        # Without bound, min(y) is 0
        # With bound >= 0.5, all preds should be >= 0.5
        s = BSpline("x", knots=5, degree=3).add_constraint(Bound(lower=0.5))
        model = LpRegressor(s)
        model.fit(df, pl.Series("y", y))
        
        preds = model.predict(df)
        assert np.all(preds >= 0.5 - 1e-5)

    def test_pwl_bound_upper(self):
        # Data: y follows x, but we bound it to be <= 0.2
        x = np.linspace(0, 1, 20)
        y = x
        df = pl.DataFrame({"x": x})
        
        s = PiecewiseLinear("x", knots=5).add_constraint(Bound(upper=0.2))
        model = LpRegressor(s)
        model.fit(df, pl.Series("y", y))
        
        preds = model.predict(df)
        assert np.all(preds <= 0.2 + 1e-5)

    def test_cyclic_bound_both(self):
        # Cyclic spline on [0, 2pi]
        x = np.linspace(0, 2*np.pi, 50)
        y = np.sin(x)
        df = pl.DataFrame({"x": x})
        
        # Bound sin(x) between -0.5 and 0.5
        s = CyclicSpline("x", period=2*np.pi, order=3).add_constraint(Bound(lower=-0.5, upper=0.5, n=200))
        model = LpRegressor(s)
        model.fit(df, pl.Series("y", y))
        
        preds = model.predict(df)
        assert np.all(preds >= -0.5 - 1e-5)
        assert np.all(preds <= 0.5 + 1e-5)

    def test_bound_interval(self):
        # Data: y = 0.2 everywhere, but we bound it to be >= 0.6 ONLY for x in [0, 0.5]
        x = np.linspace(0, 1, 50)
        y = np.full_like(x, 0.2)
        df = pl.DataFrame({"x": x})
        
        # Bound [0, 0.5] to be >= 0.6
        s = BSpline("x", knots=20, degree=1).add_constraint(Bound(lower=0.6, start=0.0, end=0.5))
        model = LpRegressor(s)
        model.fit(df, pl.Series("y", y))
        
        preds = model.predict(df)
        
        # Check x in [0, 0.5]
        mask_in = x <= 0.5
        assert np.all(preds[mask_in] >= 0.6 - 1e-5)
        
        # Check x > 0.7 - it should be able to follow y=0.2 and be less than 0.6
        mask_out = x > 0.7
        assert np.any(preds[mask_out] < 0.4)
