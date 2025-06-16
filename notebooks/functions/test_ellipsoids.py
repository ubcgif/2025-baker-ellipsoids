from .create_ellipsoid import TriaxialEllipsoid, OblateEllipsoid, ProlateEllipsoid
from .get_gravity_ellipsoids import ellipsoid_gravity
import numpy as np
import verde as vd

def test_degenerate_ellipsoid_cases():
    
    """
    
    Test cases where the axes lengths are close to the boundary of accepted values.
    
    """
    tri = TriaxialEllipsoid(5, 4.99999999, 4.99999998, 0, 0, 0, (0, 0, 0))
    pro = ProlateEllipsoid(5, 4.99999999, 0, 0, (0, 0, 0))
    obl = OblateEllipsoid(4.99999999, 5, 0, 0, (0, 0, 0))
    
    coordinates = vd.grid_coordinates(region = (-20, 20, -20, 20), spacing = 0.5, extra_coords = 5)

    _, _, gu1 = ellipsoid_gravity(coordinates, tri, 2000, field="g")
    _, _, gu2 = ellipsoid_gravity(coordinates, pro, 2000, field="g")
    _, _, gu3 = ellipsoid_gravity(coordinates, obl, 2000, field="g")
    
    ...
    
def test_euler_rotation_symmetry():
    """
    Check that euler rotations (e.g. 180 or 360 rotations) produce the expected result.
    
    """
    ...
    