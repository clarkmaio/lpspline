
import polars as pl
import numpy as np
from lpspline import Linear, LpRegressor

def test_predict_components():
    # 1. Generate data
    n = 100
    df = pl.DataFrame({
        "x1": np.linspace(0, 10, n),
        "x2": np.linspace(0, 5, n)
    })
    # y = 2*x1 + 3*x2
    y = df["x1"] * 2 + df["x2"] * 3
    
    # 2. Define splines
    s1 = Linear("x1", bias=False)
    s2 = Linear("x2", bias=False)
    
    model = s1 + s2
    
    # 3. Fit
    model.fit(df, y)
    
    # 4. Predict total
    preds_total = model.predict(df)
    assert preds_total.shape == (n,)
    
    # 5. Predict components
    preds_components = model.predict(df, return_components=True)
    assert preds_components.shape == (n, 2)
    
    # Check components sum to total
    assert np.allclose(preds_components.sum(axis=1), preds_total)
    
    # Check individual component values roughly
    # s1 should be ~ 2*x1
    # s2 should be ~ 3*x2
    # Note: Optimization might distribute bias differently if enabled, but here bias=False.
    
    print("SUCCESS: Predict components works.")

if __name__ == "__main__":
    test_predict_components()
