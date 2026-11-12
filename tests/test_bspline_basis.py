"""Tests for the B-spline basis construction against scipy's reference implementation."""
import numpy as np
import pytest
from scipy.interpolate import BSpline as ScipyBSpline

from lpspline.spline import BSpline


def _reference_basis(x: np.ndarray, knots: np.ndarray, degree: int) -> np.ndarray:
    """Build the scipy basis on the same uniformly padded knot vector."""
    low, high = knots[1] - knots[0], knots[-1] - knots[-2]
    steps = np.arange(1, degree + 1)
    t = np.concatenate([knots[0] - low * steps[::-1], knots, knots[-1] + high * steps])
    return ScipyBSpline.design_matrix(x, t, degree).toarray()


@pytest.mark.parametrize("degree", [0, 1, 2, 3])
@pytest.mark.parametrize("knots", [5, 11])
def test_basis_matches_scipy(degree: int, knots: int) -> None:
    x = np.linspace(0, 10, 301)[:-1]  # scipy's design matrix is half-open at the end
    spline = BSpline(term="x", knots=knots, degree=degree)
    spline.init_spline(np.linspace(0, 10, 50))

    basis = spline._build_basis(x)
    expected = _reference_basis(x, spline.knots, degree)

    assert basis.shape == (len(x), knots + degree - 1)
    np.testing.assert_allclose(basis, expected, atol=1e-12)


def test_basis_uneven_knots_matches_scipy() -> None:
    knots = np.array([0.0, 1.0, 1.5, 4.0, 9.0, 10.0])
    spline = BSpline(term="x", knots=knots, degree=3)
    spline.init_spline(np.linspace(0, 10, 50))
    x = np.linspace(0, 9.99, 200)
    np.testing.assert_allclose(spline._build_basis(x), _reference_basis(x, knots, 3), atol=1e-12)


def test_padded_knots_are_increasing() -> None:
    spline = BSpline(term="x", knots=np.array([0.0, 1, 2, 3, 4]), degree=3)
    padded = spline._pad_knots(spline.knots, 3)
    np.testing.assert_allclose(padded, np.arange(-3.0, 8.0))


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_partition_of_unity_including_right_edge(degree: int) -> None:
    spline = BSpline(term="x", knots=6, degree=degree)
    x = np.linspace(0, 10, 101)
    spline.init_spline(x)
    np.testing.assert_allclose(spline._build_basis(x).sum(axis=1), 1.0, atol=1e-12)


def test_numpy_integer_knots() -> None:
    spline = BSpline(term="x", knots=np.int64(4))
    spline.init_spline(np.linspace(0, 9, 10))
    np.testing.assert_allclose(spline.knots, [0, 3, 6, 9])


def test_too_few_knots_raises() -> None:
    spline = BSpline(term="x", knots=np.array([1.0]))
    spline.init_spline(np.linspace(0, 1, 10))
    with pytest.raises(ValueError):
        spline._build_basis(np.array([0.5]))
