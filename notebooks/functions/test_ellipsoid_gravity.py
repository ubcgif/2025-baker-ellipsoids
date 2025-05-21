def test_triaxial():
    
    """ **Idea** to test that the triaxial ellipsoid function produces the same result as the scipy point mass
    for spherical bodies at distance. """

    dg1, dg2, dg3 = calculate_delta_gs_triaxial(0, 0, 100, 3, 2, 1, density=1000)
    mass = 1000 * 4/3 * np.pi * 3  * 2 * 1
    point_grav = pointgrav(0, 0, 100, 0, 0, 0, mass)
    print(dg3, point_grav)
    assert np.allclose(dg3, point_grav)
        
