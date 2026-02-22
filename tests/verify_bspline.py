
import polars as pl
import numpy as np
from lpspline.spline import BSpline
from lpspline import LpRegressor

def test_bspline_refactoring():
    # 1. Generate data
    n = 100
    x = np.linspace(0, 10, n)
    # Target is sin wave
    y = np.sin(x) + np.random.normal(0, 0.05, n)
    
    df = pl.DataFrame({"x": x})
    
    # 2. Define BSpline
    # Knots at 0, 2.5, 5, 7.5, 10
    knots = [0, 2.5, 5, 7.5, 10]
    bs = BSpline("x", knots=knots, degree=3)
    
    # 3. Fit
    opt = LpRegressor([bs])
    print("Fitting BSpline...")
    opt.fit(df, pl.Series(y))
    
    # 4. Predict
    print("Predicting...")
    preds = opt.predict(df)
    
    mse = np.mean((preds - y)**2)
    print(f"MSE: {mse}")
    
    if mse < 0.1: # Threshold for success
        print("SUCCESS: BSpline works after refactoring.")
    else:
        print("FAILURE: BSpline MSE is high.")

if __name__ == "__main__":
    test_bspline_refactoring()
