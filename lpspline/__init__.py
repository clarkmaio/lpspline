
from .optimizer import LpRegressor

from .spline import Linear as l
from .spline import PiecewiseLinear as pwl
from .spline import BSpline as bs
from .spline import CyclicSpline as cs
from .spline import Factor as f

from .constraints import Monotonic, Concave, Convex, Anchor

from .viz import splines_diagnostic
