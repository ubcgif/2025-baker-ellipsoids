from .get_gravity_ellipsoids import (
    calculate_delta_gs_triaxial,
    calculate_delta_gs_oblate,
    calculate_delta_gs_prolate,
)
import numpy as np
from choclo.point import gravity_u as pointgrav
import verde as vd
from .projection import gz_rotated_ellipsoid
from .get_gravity_ellipsoids import get_gz_array


def test_ellipsoid_at_distance():
    """**Idea** to test that the triaxial ellipsoid function produces the same result as the scipy point mass
    for spherical bodies at distance."""

    dg1, dg2, dg3 = calculate_delta_gs_triaxial(0, 0, 100, 3, 2, 1, density=1000)
    mass = 1000 * 4 / 3 * np.pi * 3 * 2 * 1
    point_grav = pointgrav(0, 0, 100, 0, 0, 0, mass)

    assert np.allclose(dg3, point_grav)


def test_symmetry_ellipsoids():
    """Test that the gravity anomaly produced shows symmetry across the axes."""

    _, _, dg3_tri_up = calculate_delta_gs_triaxial(10, 0, 0, 3, 2, 1, density=1000)
    _, _, dg3_tri_down = calculate_delta_gs_triaxial(-10, 0, 0, 3, 2, 1, density=1000)

    _, _, dg3_obl_up = calculate_delta_gs_oblate(10, 0, 0, 1, 3, 3, density=1000)
    _, _, dg3_obl_down = calculate_delta_gs_oblate(-10, 0, 0, 1, 3, 3, density=1000)

    _, _, dg3_pro_up = calculate_delta_gs_prolate(10, 0, 0, 3, 2, 2, density=1000)
    _, _, dg3_pro_down = calculate_delta_gs_prolate(-10, 0, 0, 3, 2, 2, density=1000)

    np.testing.assert_allclose(np.abs(dg3_tri_down), np.abs(dg3_tri_up))
    np.testing.assert_allclose(np.abs(dg3_pro_down), np.abs(dg3_pro_up))
    np.testing.assert_allclose(np.abs(dg3_obl_down), np.abs(dg3_obl_up))


def test_symmetry_prolate_oblate():

    a, b, c = (3, 2, 2)
    d, f, g = (2, 3, 3)
    R = 5
    e = 0

    theta = np.linspace(0, 2 * np.pi, 20)
    n = R * np.cos(theta)
    u = R * np.sin(theta)

    _, ogn, ogu = calculate_delta_gs_oblate(e, n, u, d, f, g, density=1000)

    _, pgn, pgu = calculate_delta_gs_prolate(e, n, u, a, b, c, density=1000)

    for i in range(19):
        np.testing.assert_allclose(
            np.sqrt(ogn[i] ** 2 + ogu[i] ** 2),
            np.sqrt(ogn[i + 1] ** 2 + ogu[i + 1] ** 2),
        )
        np.testing.assert_allclose(
            np.sqrt(pgn[i] ** 2 + pgu[i] ** 2),
            np.sqrt(pgn[i + 1] ** 2 + pgu[i + 1] ** 2),
        )


def test_opposite_planes():

    e, n, u1 = vd.grid_coordinates(region=(-10, 10, -10, 10), spacing=1, extra_coords=5)
    e, n, u2 = vd.grid_coordinates(
        region=(-10, 10, -10, 10), spacing=1, extra_coords=-5
    )
    _, _, gu1 = gz_rotated_ellipsoid(5, 4, 3, 0, 30, 0, e, n, u1, density=1000)
    _, _, gu2 = gz_rotated_ellipsoid(5, 4, 3, 0, 30, 0, e, n, u2, density=1000)
    np.testing.assert_allclose(gu1, -np.flip(gu2))


def test_int_ext_boundary():

    # compare a set value apart

    a, b, c = (5, 4, 3)
    x = np.linspace(0, 10, 100)
    y = np.zeros(x.shape)
    z = np.zeros(x.shape)
    internal_mask = (x**2) / (a**2) + (y**2) / (b**2) + (z**2) / (c**2) < 1

    ge, _, _ = get_gz_array(internal_mask, 5, 4, 3, x, y, z, density=1000)
    first_false = np.argmax(~internal_mask)
    np.testing.assert_allclose(ge[first_false], ge[first_false - 1], atol=1e-06)
