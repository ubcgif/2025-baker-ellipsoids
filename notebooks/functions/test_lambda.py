import numpy as np
from .utils import calculate_lambda


def test_lambda():
    """Test that lambda fits characteristic equation."""
    x, y, z = 6, 5, 4
    a, b, c = 3, 2, 1
    lmbda = calculate_lambda(x, y, z, a, b, c)

    # test for lambda is within parameters for an ellipsoid, as suppposed to sh$
    assert lmbda > -(c**2)

    # check lambda fits the characteristic equation
    np.testing.assert_allclose(
        x**2 / (a**2 + lmbda) + y**2 / (b**2 + lmbda) + z**2 / (c**2 + lmbda), 1.0
    )
