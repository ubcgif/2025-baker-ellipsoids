from .create_ellipsoid import TriaxialEllipsoid, OblateEllipsoid, ProlateEllipsoid
from .magnetic_calcs import ellipsoid_magnetics
import verde as vd
import numpy as np
from scipy.constants import mu_0

def test_magnetic_symmetry():
    """
    
    Check the symmetry of magentic calculations at surfaces above and below the body.
    
    """
    a, b, c = (4, 3, 2) # triaxial ellipsoid
    yaw = 0
    pitch = 0
    roll = 0
    n = [1, 2, 3]
    H0 = np.array([5, 5, 5])
    triaxial_example = TriaxialEllipsoid(a, b, c, yaw, pitch, roll, (0, 0, 0))
    triaxial_example2 = TriaxialEllipsoid(a, b, c, yaw, pitch, roll, (0, 0, 0))
    
    # define observation points (2D grid) at surface height (z axis, 'Upward') = 5
    coordinates = vd.grid_coordinates(region = (-20, 20, -20, 20), spacing = 0.5, extra_coords = 5)
    coordinates2 = vd.grid_coordinates(region = (-20, 20, -20, 20), spacing = 0.5, extra_coords = -5)

    be1, bn1, bu1 = ellipsoid_magnetics(coordinates, triaxial_example, 0.1, H0, field="b")
    be2, bn2, bu2 = ellipsoid_magnetics(coordinates2, triaxial_example2, 0.1, H0, field="b")
    
    np.testing.assert_allclose(np.abs(be1), np.flip(np.abs(be2)))
    np.testing.assert_allclose(np.abs(bn1), np.flip(np.abs(bn2)))
    np.testing.assert_allclose(np.abs(bu1), np.flip(np.abs(bu2)))

    
def test_flipped_h0():
    
    """
    Check that reversing the magentising field produces the same (reversed) field.
    
    """
    
    a, b = (2, 4) # triaxial ellipsoid
    yaw = 0
    pitch = 0
    n = [1, 2, 3]
    H01 = np.array([5, 5, 5])
    H02 = np.array([-5, -5, -5])
    oblate_example = OblateEllipsoid(a, b, yaw, pitch, (0, 0, 0))

    # define observation points (2D grid) at surface height (z axis, 'Upward') = 5
    coordinates = vd.grid_coordinates(region = (-20, 20, -20, 20), spacing = 0.5, extra_coords = 5)
   
    be1, bn1, bu1 = ellipsoid_magnetics(coordinates, oblate_example, 0.1, H01, field="b")
    be2, bn2, bu2 = ellipsoid_magnetics(coordinates, oblate_example, 0.1, H02, field="b")
    
    np.testing.assert_allclose(np.abs(be1), np.abs(be2))
    np.testing.assert_allclose(np.abs(bn1), np.abs(bn2))
    np.testing.assert_allclose(np.abs(bu1), np.abs(bu2))
    
def test_zero_susceptability():
    """
    Test for the case of 0 susceptabililty == inducing field.
    """
    
    a, b = 1, 2
    H0 = np.array([10, 0, 0])  # Arbitrary field
    k = 0  # No contrast = no magnetisation

    ellipsoid = OblateEllipsoid(a, b, yaw=0, pitch=0, centre=(0, 0, 0))
    coordinates = vd.grid_coordinates(region=(-10, 10, -10, 10), spacing=1.0, extra_coords=5)

    be, bn, bu = ellipsoid_magnetics(coordinates, ellipsoid, k, H0, field="b")

    np.testing.assert_allclose(be[0], 1e9 * mu_0 * H0[0])
    np.testing.assert_allclose(bn[0], 1e9 * mu_0 * H0[1])
    np.testing.assert_allclose(bu[0], 1e9 * mu_0 * H0[2])
    
def test_zero_field():
    
    """
    Test that zero field produces zero anomalies.
    
    """
    
    a, b = 1, 2
    H0 = np.array([0, 0, 0])  # Arbitrary field
    k = 0  # No contrast = no magnetisation

    ellipsoid = OblateEllipsoid(a, b, yaw=0, pitch=0, centre=(0, 0, 0))
    coordinates = vd.grid_coordinates(region=(-10, 10, -10, 10), spacing=1.0, extra_coords=5)

    be, bn, bu = ellipsoid_magnetics(coordinates, ellipsoid, k, H0, field="b")

    np.testing.assert_allclose(be[0], 0)
    np.testing.assert_allclose(bn[0], 0)
    np.testing.assert_allclose(bu[0], 0)

def test_mag_ext_int_boundary():
    """
    Check the boundary between internal and external field calculations is consistent.
    """
    # aribtrary parameters
    a, b = 1, 2
    H0 = np.array([10.0, 0.0, 0.0])
    k = 0.1

    ellipsoid = OblateEllipsoid(a=2, b=3, yaw=0, pitch=0, centre=(0, 0, 0))
    
    e = np.array([[1.999, 2.001]])
    n = np.array([[0.0, 0.0]])
    u = np.array([[0.0, 0.0]])
    coordinates = (e, n, u)

    be, bn, bu = ellipsoid_magnetics(coordinates, ellipsoid, k, H0, field="b")

    np.testing.assert_allclose(be[0, 0], be[0, 1], rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(bn[0, 0], bn[0, 1], rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(bu[0, 0], bu[0, 1], rtol=1e-3, atol=1e-3)

def test_mag_flipped_ellipsoid():
    """
    Check that rotating the ellipsoid in various ways maintains expected results.
    """
    ...
    
def test_mag_symmetry_through_axis():
    """
    With no rotation of the ellipsoid and an external field aligned with the axis,
    check that the symmetry of the returned magnetic field is as expected.
    """
    ...

def test_obl_pro_symmetry():
    ...
    
    