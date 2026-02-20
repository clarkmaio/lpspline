
import polars as pl
import numpy as np
from lpspline import LpRegressor, Linear, PiecewiseLinear, BSpline, CyclicSpline, Factor

def test_lpspline():
    # Generate synthetic data
    np.random.seed(42)
    n = 100
    x_linear = np.linspace(0, 10, n)
    x_pwl = np.linspace(0, 10, n)
    x_bs = np.linspace(0, 10, n)
    x_cyc = np.linspace(0, 2*np.pi, n)
    x_factor = np.random.randint(0, 3, n)
    
    # Ground truth
    y = (2 * x_linear + 1) + \
        (np.maximum(0, x_pwl - 5)) + \
        (np.sin(x_bs)) + \
        (np.cos(x_cyc)) + \
        (x_factor * 0.5) + \
        np.random.normal(0, 0.1, n)
        
    df = pl.DataFrame({
        "linear_term": x_linear,
        "pwl_term": x_pwl,
        "bs_term": x_bs,
        "cyc_term": x_cyc,
        "factor_term": x_factor,
        "y": y
    })
    
    # Define splines
    splines = [
        Linear("linear_term"),
        PiecewiseLinear("pwl_term", knots=[5.0]),
        BSpline("bs_term", knots=np.linspace(0, 10, 5), degree=3),
        CyclicSpline("cyc_term", period=2*np.pi, order=2),
        Factor("factor_term", n_classes=3)
    ]
    
    # LpRegressor
    opt = LpRegressor(splines)
    
    print("Fitting model...")
    opt.fit(df, df["y"])
    
    print("Predicting...")
    preds = opt.predict(df)
    
    mse = np.mean((preds - y)**2)
    print(f"MSE: {mse}")
    
    if mse < 0.5:
        print("SUCCESS: MSE is low.")
    else:
        print("FAILURE: MSE is high.")

if __name__ == "__main__":
    test_lpspline()
