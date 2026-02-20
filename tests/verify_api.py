
import polars as pl
import numpy as np
from lpspline import LinearSpline, CyclicSpline

def test_api():
    # 1. Generate data
    n = 100
    df = pl.DataFrame({
        "temperature": np.random.rand(n) * 10,
        "hour": np.linspace(0, 24, n)
    })
    y = df["temperature"] * 2 + np.sin(df["hour"] * 2 * np.pi / 24) + np.random.normal(0, 0.1, n)
    
    # 2. Define API
    spline1 = CyclicSpline(period=24, order=4, term="hour")
    spline2 = LinearSpline(bias=True, term="temperature")
    
    # 3. Add them model = spline1 + spline2
    model = spline1 + spline2
    
    print(f"Model type: {type(model)}")
    print(f"Number of splines: {len(model.splines)}")
    
    # 4. Fit
    print("Fitting...")
    model.fit(df, y)
    
    # 5. Predict
    print("Predicting...")
    preds = model.predict(df)
    
    p_series = pl.Series(preds)
    y_series = y
    mse = (p_series - y_series).pow(2).mean()
    print(f"MSE: {mse}")
    
    if mse < 0.1:
        print("SUCCESS: API works and MSE is low.")
    else:
        print("FAILURE: MSE is high.")

if __name__ == "__main__":
    test_api()
