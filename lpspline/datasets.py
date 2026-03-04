import numpy as np
import polars as pl
from typing import Tuple



def load_by_dataset(samples: int = 1000) -> Tuple[pl.DataFrame, pl.Series]:
    """
    Generate a synthetic dataset for demonstrating LPSpline features.
    
    Args:
        samples (int): Number of samples to generate. Defaults to 1000.
        
    Returns:
        Tuple[pl.DataFrame, pl.Series]: A tuple containing the predictor 
        DataFrame (X) and the target Series (y).
    """
    x = np.linspace(-3, 8, samples)
    by = np.random.randint(0, 3, samples)
    
    y = np.where(by == 0, x**2, np.where(by == 1, 2 * x**2, 3 * x**2)) + np.random.normal(0, 0.2, samples)
    
    X = pl.DataFrame({
        "x": x,
        "by": by
    })
    
    y_series = pl.Series("target", y)
    
    return X, y_series

def load_demo_dataset(samples: int = 1000) -> Tuple[pl.DataFrame, pl.Series]:
    """
    Generate a synthetic dataset for demonstrating LPSpline features.
    
    Args:
        samples (int): Number of samples to generate. Defaults to 1000.
        
    Returns:
        Tuple[pl.DataFrame, pl.Series]: A tuple containing the predictor 
        DataFrame (X) and the target Series (y).
    """
    x_linear = np.linspace(0, 10, samples)
    x_pwl = np.linspace(0, 10, samples)
    x_bs = np.linspace(0, 10, samples)
    x_cyc = np.linspace(0, 2*np.pi, samples)
    x_factor = np.random.randint(0, 3, samples)
    
    # True functions
    y_linear = -1 * x_linear
    y_pwl = np.where(x_pwl < 5, 0, x_pwl - 5) # Hinge at 5
    y_bs = np.sin(x_bs) # Sine wave
    y_cyc = np.cos(x_cyc) # Cosine wave
    y_factor = np.array([0, 2, -1])[x_factor] # Categorical effects
    
    # Combined target with noise
    y = (
        y_linear + 
        y_pwl + 
        y_bs + 
        y_cyc + 
        y_factor + 
        np.random.normal(0, 0.2, samples)
    )
    
    X = pl.DataFrame({
        "xl": x_linear,
        "xpwl": x_pwl,
        "xbs": x_bs,
        "xcyc": x_cyc,
        "xfactor": x_factor
    })
    
    y_series = pl.Series("target", y)
    
    return X, y_series
