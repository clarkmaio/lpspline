
import pytest
import numpy as np
import cvxpy as cp
from lpspline import Linear, PiecewiseLinear, BSpline, CyclicSpline, Factor, LpRegressor

class TestSplines:
    
    def test_linear_spline(self):
        spline = Linear(term="x", bias=True)
        assert repr(spline) == "Linear(term='x', bias=True)"
        
        x = np.array([0, 1, 2])
        basis = spline._build_basis(x)
        assert basis.shape == (3, 2)
        assert np.allclose(basis, [[0, 1], [1, 1], [2, 1]])
        
        vars = spline._build_variables()
        assert len(vars) == 1
        assert vars[0].shape == (2,)

    def test_linear_spline_no_bias(self):
        spline = Linear(term="x", bias=False)
        assert repr(spline) == "Linear(term='x', bias=False)"
        
        x = np.array([0, 1, 2])
        basis = spline._build_basis(x)
        assert basis.shape == (3, 1)
        assert np.allclose(basis, [[0], [1], [2]])
        
        vars = spline._build_variables()
        assert len(vars) == 1
        assert vars[0].shape == (1,)

    def test_piecewise_linear_spline(self):
        knots = [1.0]
        spline = PiecewiseLinear(term="x", knots=knots)
        assert repr(spline) == "PiecewiseLinear(term='x', knots=[1.0])"
        
        x = np.array([0, 1, 2])
        basis = spline._build_basis(x)
        # Basis: 1, x, (x-1)+
        # 0: 1, 0, 0
        # 1: 1, 1, 0
        # 2: 1, 2, 1
        assert basis.shape == (3, 3)
        expected = np.array([
            [1, 0, 0],
            [1, 1, 0],
            [1, 2, 1]
        ])
        assert np.allclose(basis, expected)
        
        vars = spline._build_variables()
        assert len(vars) == 1
        assert vars[0].shape == (3,)

    def test_bspline(self):
        # Knots: 0, 1, 2, 3, 4. Degree 1 (linear B-splines)
        knots = [0, 1, 2, 3, 4]
        degree = 1
        spline = BSpline(term="x", knots=knots, degree=degree)
        assert "BSpline(term='x', degree=1" in repr(spline)
        
        x = np.array([0.5, 1.5, 2.5, 3.5])
        basis = spline._build_basis(x)
        # 5 knots, degree 1 -> 5-1-1 = 3 basis functions
        assert basis.shape == (4, 3)
        
        vars = spline._build_variables()
        assert len(vars) == 1
        assert vars[0].shape == (3,)

    def test_cyclic_spline(self):
        period = 4.0
        order = 1
        spline = CyclicSpline(term="x", period=period, order=order)
        assert repr(spline) == f"CyclicSpline(term='x', period={period}, order={order})"
        
        x = np.array([0, 1, 2, 3])
        basis = spline._build_basis(x)
        # 1 + 2*order = 3
        assert basis.shape == (4, 3)
        
        # Check first column is all ones
        assert np.allclose(basis[:, 0], 1)
        
        vars = spline._build_variables()
        assert len(vars) == 1
        assert vars[0].shape == (3,)

    def test_factor_spline(self):
        n_classes = 3
        spline = Factor(term="x", n_classes=n_classes)
        assert repr(spline) == f"Factor(term='x', n_classes={n_classes})"
        
        x = np.array([0, 1, 2, 0])
        basis = spline._build_basis(x)
        assert basis.shape == (4, 3)
        
        expected = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0]
        ])
        assert np.allclose(basis, expected)
        
        vars = spline._build_variables()
        assert len(vars) == 1
        assert vars[0].shape == (3,)

    def test_optimizer_repr(self):
        s1 = Linear("a")
        s2 = Factor("b", n_classes=2)
        opt = LpRegressor([s1, s2])
        assert "LpRegressor(splines=[" in repr(opt)
        assert "Linear(term='a'" in repr(opt)
        assert "Factor(term='b'" in repr(opt)
