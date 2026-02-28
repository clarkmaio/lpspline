import cvxpy as cp
import numpy as np

x = cp.Variable(5)
idx = np.array([0, 2])
expr = x[idx+2] - 2*x[idx+1] + x[idx] >= 0
print(expr.shape)
