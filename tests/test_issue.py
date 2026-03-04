import numpy as np
import polars as pl
from lpspline.spline.bspline import BSpline
from lpspline.optimizer.regressor import LpRegressor

def test():
    np.random.seed(42)
    x = np.linspace(0, 10, 100)
    y = np.sin(x) + np.random.normal(0, 0.1, 100)
    df = pl.DataFrame({"x": x, "y": y})

    # The user says "if I set 4 knots basis should consist in 4 element, instead is 6"
    spline = BSpline(term='x', knots=4, degree=3)
    
    reg = LpRegressor(splines=[spline])
    try:
        reg.fit(df, df["y"])
        print("Fit successful!")
    except Exception as e:
        print(f"Fit failed with error: {e}")

if __name__ == "__main__":
    test()
