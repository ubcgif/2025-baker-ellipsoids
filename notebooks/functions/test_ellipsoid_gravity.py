from .get_gravity_ellipsoids import calculate_delta_gs_triaxial, calculate_delta_gs_oblate, calculate_delta_gs_prolate
import numpy as np
from choclo.point import gravity_u as pointgrav

def test_triaxial():
    
    """ **Idea** to test that the triaxial ellipsoid function produces the same result as the scipy point mass
    for spherical bodies at distance. """

    dg1, dg2, dg3 = calculate_delta_gs_triaxial(0, 0, 100, 3, 2, 1, density=1000)
    mass = 1000 * 4/3 * np.pi * 3  * 2 * 1
    point_grav = pointgrav(0, 0, 100, 0, 0, 0, mass)
    print(dg3, point_grav)
    
    assert np.allclose(dg3, point_grav)
        
def test_symmetry_ellipsoids():
    
    """Test that the gravity anomaly produced shows symmetry across the axes."""
    
    _, _, dg3_tri_up = calculate_delta_gs_triaxial(10, 0, 0, 3, 2, 1, density=1000)
    _, _, dg3_tri_down = calculate_delta_gs_triaxial(-10, 0, 0, 3, 2, 1, density=1000)
    
    _, _, dg3_obl_up = calculate_delta_gs_oblate(10, 0, 0, 1, 3, 3, density=1000)
    _, _, dg3_obl_down = calculate_delta_gs_oblate(-10, 0, 0, 1, 3, 3, density=1000)
    
    _, _, dg3_pro_up = calculate_delta_gs_prolate(10, 0, 0, 3, 2, 2, density=1000)
    _, _, dg3_pro_down = calculate_delta_gs_prolate(-10, 0, 0, 3, 2, 2, density=1000)
    
    assert (np.allclose(np.abs(dg3_tri_down), np.abs(dg3_tri_up)), 
            np.allclose(np.abs(dg3_pro_down), np.abs(dg3_pro_up)),
            np.allclose(np.abs(dg3_obl_down), np.abs(dg3_obl_up))
            )
    
    