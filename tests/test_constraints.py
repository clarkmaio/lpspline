import pytest
import numpy as np
import polars as pl
from lpspline.spline import Linear, PiecewiseLinear, BSpline, CyclicSpline, Factor
from lpspline import LpRegressor
from lpspline.constraints import Anchor, Monotonic, Convex, Concave

class TestConstraints:
    def test_anchor(self):
        s = Linear("x", bias=True).add_constraint(Anchor([(0.0, 10.0)]))
        assert len(s.constraints) == 1

    def test_invalid_constraints(self):
        with pytest.raises(ValueError, match="cannot accept Convex constraint"):
            Linear("x").add_constraint(Convex())
            
        with pytest.raises(ValueError, match="cannot accept Concave constraint"):
            PiecewiseLinear("x", knots=[1.0]).add_constraint(Concave())

        with pytest.raises(ValueError, match="cannot accept Monotonic constraint"):
            CyclicSpline("x", period=24, order=4).add_constraint(Monotonic())

        with pytest.raises(ValueError, match="cannot accept Convex constraint"):
            Factor("x", n_classes=3).add_constraint(Convex())

    def test_valid_monotonic(self):
        # Should not raise
        Linear("x").add_constraint(Monotonic())
        PiecewiseLinear("x", knots=[1.0]).add_constraint(Monotonic())
        BSpline("x", knots=[0,1,2,3]).add_constraint(Monotonic())
        
    def test_optimizer_integration(self):
        df = pl.DataFrame({"x": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 0.0]})
        
        # Test monotonicity should force slope >= 0
        dec_spline = Linear("x", bias=True).add_constraint(Monotonic(decreasing=True))
        opt = LpRegressor(dec_spline)
        opt.fit(df, df["y"])
        
        # Original points go up then down: 0->1->0
        # Decreasing line fit on (0,0), (1,1), (2,0) will just be flat or similar due to sum_squares objective
        preds = opt.predict(df)
        assert preds[1] <= preds[0] + 1e-4  # Should be non-increasing
