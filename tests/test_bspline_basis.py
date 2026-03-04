import numpy as np

def run_test():
    from lpspline.spline.bspline import BSpline
    # Mocking cp.Variable so we don't need cvxpy
    class MockVariable:
        def __init__(self, shape, name):
            self.shape = shape
            self.name = name
    import sys
    import types
    mock_cp = types.ModuleType("cvxpy")
    mock_cp.Variable = MockVariable
    sys.modules["cvxpy"] = mock_cp
    
    x = np.linspace(0, 10, 100)
    spline = BSpline(term='x', knots=4, degree=3)
    spline.init_spline(x)
    basis = spline._build_basis(x)
    
    var = spline._build_variables()
    print(f"Basis shape: {basis.shape}")
    print(f"Variables shape expected: {var.shape}")

run_test()
