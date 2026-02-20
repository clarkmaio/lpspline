
import pytest
from lpspline import Linear, LpRegressor

def test_single_spline_init():
    s = Linear("x")
    
    # Method 1: Flexible Constructor
    model1 = LpRegressor(s)
    assert len(model1.splines) == 1
    assert model1.splines[0] is s
    assert isinstance(model1, LpRegressor)
    
    # Method 2: Unary +
    model2 = +s
    assert len(model2.splines) == 1
    assert model2.splines[0] is s
    assert isinstance(model2, LpRegressor)
    
    print("SUCCESS: Single spline initialization works via constructor and unary +.")

if __name__ == "__main__":
    test_single_spline_init()
