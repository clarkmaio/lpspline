import numpy as np
import polars as pl
from lpspline import LpRegressor, l, Log, Sigmoid, Exp

def test_log_link():
    print("Testing Log link...")
    X = pl.DataFrame({"x": np.linspace(0, 10, 100)})
    # y = exp(0.5 * x)
    y = np.exp(0.5 * X["x"].to_numpy()) + np.random.normal(0, 0.1, 100)
    y = np.maximum(y, 0.01) # Ensure positive
    y_series = pl.Series("y", y)
    
    reg = LpRegressor(l("x"))
    log_model = Log(reg)
    log_model.fit(X, y_series, summary=False)
    
    pred = log_model.predict(X)
    print(f"Log model max prediction: {pred.max():.2f}, Mean: {pred.mean():.2f}")
    assert np.all(pred > 0)

def test_sigmoid_link():
    print("\nTesting Sigmoid link...")
    X = pl.DataFrame({"x": np.linspace(-5, 5, 100)})
    # y = sigmoid(0.5 * x)
    y_true = 1 / (1 + np.exp(-0.5 * X["x"].to_numpy()))
    y = y_true + np.random.normal(0, 0.01, 100)
    y = np.clip(y, 0.01, 0.99) # Ensure between 0 and 1
    y_series = pl.Series("y", y)
    
    reg = LpRegressor(l("x"))
    sigmoid_model = Sigmoid(reg)
    sigmoid_model.fit(X, y_series, summary=False)
    
    pred = sigmoid_model.predict(X)
    print(f"Sigmoid model min/max prediction: {pred.min():.2f}/{pred.max():.2f}")
    assert np.all((pred >= 0) & (pred <= 1))

def test_exp_link():
    print("\nTesting Exp link...")
    X = pl.DataFrame({"x": np.linspace(1, 10, 100)})
    # y = log(x)
    y_true = np.log(X["x"].to_numpy())
    y = y_true + np.random.normal(0, 0.01, 100)
    y_series = pl.Series("y", y)
    
    reg = LpRegressor(l("x"))
    exp_model = Exp(reg)
    exp_model.fit(X, y_series, summary=False)
    
    pred = exp_model.predict(X)
    print(f"Exp model min/max prediction: {pred.min():.2f}/{pred.max():.2f}")

if __name__ == "__main__":
    test_log_link()
    test_sigmoid_link()
    test_exp_link()
    print("\nAll tests completed!")
