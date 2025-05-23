from functions.utils import calculate_lambda
import numpy as np 
from scipy.special import ellipkinc, ellipeinc 
import verde as vd 


def calculate_delta_gs_oblate(x, y, z, a, b, c, density=1000): # takes semiaxes, lambda value, density
    
    """
    Calculate the components of delta_g_i for i=1,2,3, for the oblate ellipsoid case (a < b = c 
    Delta_g_i represent the local axes system (for now, the only axes system). 
    x, y, z are positions of observation in the local co-ordinate system.
    
    
    Parameters
    ----------
    lambda (float): the parameter defining surfaces in confocal family, for an ellipsoid.
    Observation coordinates (integer): x, y, z


    Returns
    -------
    # is this the best way to do this? will individual delta_g components be used later? maybe just return delta_g_z??
    
    dg1 (float): change in gravity for the x axis. (is this the right explanation?)
    dg2 (float): change in gravity for the y axis.
    dg3 (float): change in gravity for the z axis.

    """
    # constants
    G = 6.6743e-11

    # call and use lambda function 
    lmbda = calculate_lambda(x, y, z, a, b, c)
    
    # check the function is used for the correct type of ellipsoid
    if not (a < b and b == c):
        raise ValueError(f"Invalid ellipsoid axis lengths for oblate ellipsoid:" 
            f"expected a < b = c but got a = {a}, b = {b}, c = {c}")
    
    # compute the coefficient of the three delta_g equations 
    numerator = np.pi * a * b**2 * G * density
    denominator = (b**2 - a**2)**1.5 
    co_eff1 = numerator / denominator

    # compute repeated arctan term
    arc_tan_term = np.arctan(((b**2 - a**2) / (a**2 + lmbda))**0.5)
    
    # compute the terms within the brackets for delta_g 1,2,3
    bracket_term_g1 = arc_tan_term - ((b**2 - a**2) / (a**2 + lmbda) )**0.5
    
    bracket_term_g2g3 = ((((b**2 - a**2) * (a**2 + lmbda))**0.5) / (b**2 + lmbda)) - arc_tan_term

    # compile constants, coefficients, bracket terms to calculate final value of the delta_g terms
    dg1 = 4 * co_eff1 * x * bracket_term_g1
    dg2 = 2 * co_eff1 * y * bracket_term_g2g3
    dg3 = 2 * co_eff1 * z * bracket_term_g2g3

    return dg1, dg2, dg3

def calculate_delta_gs_prolate(x, y, z, a, b, c, density=1000): # takes semiaxes, lambda value, density
    
    """
    Calculate the components of delta_g_i for i=1,2,3, for the prolate ellipsoid case.
    Delta_g_i represent the local axes system (for now, the only axes system). 
    x, y, z are positions of observation in the local co-ordinate system.
    
    Parameters
    ----------
    lambda (float): the parameter defining surfaces in confocal family, for an ellipsoid.
    Observation coordinates (integer): x, y, z


    Returns
    -------
    # is this the best way to do this? will individual delta_g components be used later? maybe just return delta_g_z??
    
    dg1 (float): change in gravity for the x axis. (is this the right explanation?)
    dg2 (float): change in gravity for the y axis.
    dg3 (float): change in gravity for the z axis.

    """
    # constants
    G = 6.6743e-11

    # call and use lambda function 
    lmbda = calculate_lambda(x, y, z, a, b, c)
    
    # check the function is used for the correct type of ellipsoid
    if not (a > b and b == c):
        raise ValueError(f"Invalid ellipsoid axis lengths for prolate ellipsoid: expected a > b = c but got a = {a}, b = {b}, c = {c}")
    
    # compute the coefficient of the three delta_g equations 
    numerator = np.pi * a * b**2 * G * density
    denominator = (a**2 - b**2)**1.5 
    co_eff1 = numerator / denominator

    # compute repeated log_e term
    log_term = np.log(((a**2 - b**2)**0.5 + (a**2 + lmbda)**0.5)/ ((b**2 + lmbda)**0.5))

    # compute repeated f_2 second term 
    f_2_term_2 = (((a**2 - b**2) * (a**2 + lmbda))**0.5)/(b**2 + lmbda)

    # compile terms 
    dg1 = 4 * co_eff1 * x * (((a**2 - b**2)/(a**2 + lmbda))**0.5 - log_term)
    dg2 = 2 * co_eff1 * y * (log_term - f_2_term_2) 
    dg3 = 2 * co_eff1 * z * (log_term - f_2_term_2) 
    
    return dg1, dg2, dg3

def calculate_delta_gs_triaxial(x, y, z, a, b, c, density=1000): # takes semiaxes, lambda value, density
    
    """
    Calculate the components of delta_g_i for i=1,2,3, for the triaxial ellipsoid case.
    Delta_g_i represent the local axes system (for now, the only axes system). 
    x, y, z are positions of observation in the local co-ordinate system.
    
    Parameters
    ----------
    lambda (float): the parameter defining surfaces in confocal family, for an ellipsoid.
    Observation coordinates (integer): x, y, z


    Returns
    -------
    # is this the best way to do this? will individual delta_g components be used later? maybe just return delta_g_z??
    
    dg1 (float): change in gravity for the x axis. (is this the right explanation?)
    dg2 (float): change in gravity for the y axis.
    dg3 (float): change in gravity for the z axis.

    """
    # constants
    G = 6.6743e-11

    # call and use lambda function 
    lmbda = calculate_lambda(x, y, z, a, b, c)
    
    # check the function is used for the correct type of ellipsoid
    if not (a > b > c):
        raise ValueError(f"Invalid ellipsoid axis lengths for triaxial ellipsoid:"
            f"expected a > b > c but got a = {a}, b = {b}, c = {c}")
    
    # compute the coefficient of the three delta_g equations 
    co_eff = -2 * np.pi * a * b * c * G * density
    k = np.sqrt((a**2 - b**2)/(a**2 - c**2))
    theta_prime = np.arcsin(np.sqrt((a**2 - c**2)/(a**2 + lmbda))) # for theta in range (0 =< theta =< np.pi/2) - check how to code this??

    # compute terms associated with A(lambda) 
    A_coeff = 2/((a**2 - b**2)*np.sqrt(a**2 - c**2))
    A_elliptic_integral = ellipkinc(theta_prime, k)  - ellipeinc(theta_prime, k)
    A_lmbda = A_coeff * A_elliptic_integral


    # compute terms associated with B(lambda) 
    B_coeff = (2 * np.sqrt(a**2 - c**2)) / ((a**2 - b**2) * (b**2 - c**2))
    B_fk_coeff = ((b**2 - c**2)/(a**2 - c**2))
    B_fk_subtracted_term = (k**2 * np.sin(theta_prime) * np.cos(theta_prime)) / np.sqrt(1 - k**2 * np.sin(theta_prime)**2)
    B_lmbda = B_coeff * (ellipeinc(theta_prime, k) - B_fk_coeff * ellipkinc(theta_prime, k) - B_fk_subtracted_term) # check this is right

    # compute terms associated with C(lambda) 
    C_coeff = 2/((b**2 - c**2) * np.sqrt(a**2 - c**2))
    C_ek_subtracted_term = (np.sin(theta_prime) * np.sqrt(1 - k**2 * np.sin(theta_prime)**2)) / np.cos(theta_prime) # check the brackets here ??
    C_lmbda = C_coeff * (C_ek_subtracted_term - ellipeinc(theta_prime, k))

    # compile all terms 
    dg1 = co_eff * x * A_lmbda
    dg2 = co_eff * y * B_lmbda
    dg3 = co_eff * z * C_lmbda
    
    return dg1, dg2, dg3

def calc_gz_array(func, spacing, region, height, a, b, c, density):
    
    """Function to call one of the gz functions,
    and use it for an array of coordinates.
    
    Parameters
    ----------
    func (function): the function for the desired ellipsoid (triaxial, prolate or oblate)
    spacing (float, tuple)): grid spacing. 1 value means equal in all directions. 
                            tuple = (spacing_north, spacing_east)
                            
    region (list): [W, E, S, N] for the boundaries in each direction.
    height (float): the z-plane to produce the 2D slice. 
                    NOTE: if plane disects the ellipsoid, the internal values will 
                    be retruned as NaNs.
    a, b, c (floats): semiaxes lengths of the ellipsoid. 
                    NOTE: these must comply with chosen ellipsoid type.
    density (float): uniform density of the ellipsoid.
                    
                    
    Returns 
    -------
    easting (array): the easting value of each point on the coordinate grid.
    northing (array): the northing value of each point on the cooridnate grid.
    gz (array) : a value of gz for each point on the coordinate grid.
    
    
    TO DO:
    
    return all g components?
    
    """
  

    # set the coord points
    easting, northing, elv = vd.grid_coordinates(region=region, spacing=spacing, extra_coords=5)
    gz_2d = np.empty_like(easting)

    # check the function runs and that the correct a, b, c ratio has been given for chosen function
    func(easting[0, 0], northing[0, 0], elv[0, 0], a, b, c, density)

    # loop over coordinate points 
    # bit concerned about how clunky this is?
    for i in range(easting.shape[0]):
        for j in range(easting.shape[1]):
            try:
                _, _, gz = func(easting[i, j], northing[i, j], elv[i, j], a, b, c, density)
                gz_2d[i, j] = gz
            except ValueError: # give nan values when ValueError thrown in the ellips. function
                gz_2d[i, j] = np.nan 
                
    return easting, northing, gz_2d

