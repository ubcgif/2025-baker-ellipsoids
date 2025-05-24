import numpy as np 
import verde as vd

def calculate_lambda(x, y, z, a, b, c): # takes semiaxes and observation coordinates
    
    """
    Calculate the value of lambda (parameter defining surfaces in confocal family,
    colloquially, the inflation or deflation parameter), for given ellipsoid semiaxes
    and given points of observation.
    x, y, z are positions in the local co-ordinate system.
    
    Parameters
    ----------
    Semiaxes (length) (integer): a, b, c
    Observation coordinates (integer): x, y, z


    Returns
    -------
    lambda (float): the value of lambda.
    
    """
    # Calculate lambda using x, y, z 

    # Check that the input values of x, y, z are greater than a, b, c semi axis lengths
    if not (np.abs(np.any(x)) >= a or np.abs(np.any(y)) >= b 
            or np.abs(np.any(z)) >= c):
        raise ValueError(
            "Arrays x, y, z should contain points which lie outside"
            " of the surface defined by a, b, c"
            )
    p_0 = a**2 * b**2 * c**2 - b**2 * c**2 * x**2 - c**2 * a**2 * y**2 - a**2 * b**2 * z**2
    p_1 = a**2 * b**2 + b**2 * c**2 + c**2 * a**2 - (b**2 + c**2) * x**2 - (c**2 + a**2) * y**2 - (a**2 + b**2) * z**2
    p_2 = a**2 + b**2 + c**2 - x**2 - y**2 - z**2

    p = p_1 - (p_2**2)/3 

    q = p_0 - ((p_1*p_2)/3) + 2*(p_2/3)**3

    theta = np.arccos(-q / (2 * np.sqrt((-p/3)**3))) 
    
    lmbda = 2 * np.sqrt((-p/3)) * np.cos(theta/3) - p_2/3 
    

    return lmbda

   
def get_ellipsoid_mass(a, b, c, density):
    """
    Get mass of ellipsoid from volume,
    In order to compare to point mass (spherical) source.
    
    Parameters
    ----------
    a, b, c (m) = ellipsoid semiaxes
    density (kg/m^3) = uniform density of the ellipsoid
    
    Returns
    -------
    mass of the ellpsoid (kg)
    
    """
    volume = 4/3 * np.pi * a * b * c

    return density * volume

def get_coords_and_mask(region, spacing, extra_coords, a, b, c):
    """
    Return the  coordinates and mask which separates points 
    within the given ellipsoid and on or outside
    of the given ellipsoid.
    
    Parameters
    ----------
    region (list)[N, S, E, W]: end points of the coordinate grid
    spacing (float): separation between the points (default = 1)
    extra_coords (float or list): surfaces of constant height to test (default = 0)
    a, b, c (float): semiaxes of the ellipsoid
    
    Returns 
    -------
    x, y, z (arrays): 2D coordinate arrays for grid 
    internal (array): mask for the internal points of the ellipsoid
    
    NOTES:
    Consider making it possible to pass a varying array as a set of z coords.
    """
    
    coords = vd.grid_coordinates(region, spacing=1, extra_coords=0)
    x, y, z = coords
    
    internal = (x**2)/a + y**2/b + z**2/c < 1

    return x, y, z, internal

