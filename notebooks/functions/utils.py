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
    if not (np.abs(x) >= a or np.abs(y) >= b or np.abs(z) >= c):
        raise ValueError(
            f"Location (x, y, z) should lie on or outside of semiaxes (a, b, c), "
            f"but got x = {x}, y = {y}, z = {z}, and a = {a}, b = {b}, c = {c}")

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
