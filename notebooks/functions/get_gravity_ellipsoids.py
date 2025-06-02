from .utils import calculate_lambda, get_coords_and_mask
import numpy as np 
from scipy.special import ellipkinc, ellipeinc 
import verde as vd 

def get_ABC(x, y, z, a, b, c, lmbda):
    """
    Calculate the A(lmbda), B(lmbda), C(lmbda) functions using elliptic 
    integrals as given in Clark et al (1986) and Takenhasi et al (2018).
    
    Parameters
    ----------
    x, y, z (array, integer): Observation coordinates (2D or 1D)
    a, b, c (floats): semiaxes lengths of the ellipsoid. 
                    NOTE: these must comply with chosen ellipsoid type.
    lmbda (integer, array): the value/s of lambda associated with the confocal
                            surfaces of the defined ellipsoid.
    Returns
    -------
    A_lmbda (array): the A(lambda) values
    B_lmbda (array): the B(lambda) values
    C_lmbda (array): the C(lambda) values
                            
    """
    
    # compute the k and theta terms for the elliptic integrals
    k = np.sqrt((a**2 - b**2)/(a**2 - c**2))
    theta_prime = np.arcsin(np.sqrt((a**2 - c**2)/(a**2 + lmbda)))

    
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
    
    return A_lmbda, B_lmbda, C_lmbda


def calculate_internal_g(x, y, z, a, b, c, density):
    """
    Calculate the field inside the ellipsoid due to the ellipsoid body.
    
    Parameters
    ----------
    x, y, z (array, integer): Observation coordinates (2D or 1D)
    a, b, c (floats): semiaxes lengths of the ellipsoid. 
                    NOTE: these must comply with chosen ellipsoid type.
    density (float): uniform density of the ellipsoid.
    
    Returns
    -------
    g_int_x (array): values of the gx component within the ellispoid.
    g_int_y(array): values of the gy component within the ellispoid.
    g_int_z (array): values of the gz component within the ellispoid.
    
    """
    # calculate functions with lambda = 0
    # in the triaxial case 
    if (b!=c):
        g_int_x, g_int_y, g_int_z = calculate_delta_gs_triaxial(x, y, z, a, b, c, density)
        
    # in the prolate case
    elif (a>b):
        g_int_x, g_int_y, g_int_z = calculate_delta_gs_prolate(x, y, z, a, b, c, density, lmbda=0)
        
    # in the oblate case
    else:
        g_int_x, g_int_y, g_int_z = calculate_delta_gs_oblate(x, y, z, a, b, c, density, lmbda=0)
        
        
    return g_int_x, g_int_y, g_int_z
    

def calculate_delta_gs_oblate(x, y, z, a, b, c, density, lmbda=None): # takes semiaxes, lambda value, density
    
    """
    Calculate the components of delta_g_i for i=1,2,3, for the oblate ellipsoid case (a < b = c 
    Delta_g_i represent the local axes system (for now, the only axes system). 
    x, y, z are positions of observation in the local co-ordinate system.
    
    
    Parameters
    ----------
    x, y, z (array, integer): Observation coordinates (2D or 1D)
    a, b, c (floats): semiaxes lengths of the ellipsoid. 
                    NOTE: these must comply with chosen ellipsoid type.
    density (float): density of the body of interest in kg/m^3
    lmbda (float, array): a set of lambda values to use instead of those 
    calculated in calculate_lambda, which is computed in the function. E.g., if
    you wish to use lambda=0 for interal fields. When lmbda=None, the correct
    lambda values are calculated within this function.

    Returns
    -------
    dg1 (float): change in gravity for the x axis. (is this the right explanation?)
    dg2 (float): change in gravity for the y axis.
    dg3 (float): change in gravity for the z axis.

    """
    # constants
    G = 6.6743e-11

    # call and use lambda function 
    if lmbda==None:
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

def calculate_delta_gs_prolate(x, y, z, a, b, c, density, lmbda=None): # takes semiaxes, lambda value, density
    
    """
    Calculate the components of delta_g_i for i=1,2,3, for the prolate ellipsoid case.
    Delta_g_i represent the local axes system (for now, the only axes system). 
    x, y, z are positions of observation in the local co-ordinate system.
    
    Parameters
    ----------
    x, y, z (array, integer): Observation coordinates (2D or 1D)
    a, b, c (floats): semiaxes lengths of the ellipsoid. 
                    NOTE: these must comply with chosen ellipsoid type.
    density (float): density of the body of interest in kg/m^3
    lmbda (float, array): a set of lambda values to use instead of those 
    calculated in calculate_lambda, which is computed in the function. E.g., if
    you wish to use lambda=0 for interal fields. When lmbda=None, the correct
    lambda values are calculated within this function.
    

    Returns
    -------
    dg1 (float): change in gravity for the x axis. (is this the right explanation?)
    dg2 (float): change in gravity for the y axis.
    dg3 (float): change in gravity for the z axis.

    """
    # constants
    G = 6.6743e-11

    # call and use lambda function 
    if lmbda==None:
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

def calculate_delta_gs_triaxial(x, y, z, a, b, c, density): # takes semiaxes, lambda value, density
    
    """
    Calculate the components of delta_g_i for i=1,2,3, for the triaxial ellipsoid case.
    Delta_g_i represent the local axes system (for now, the only axes system). 
    x, y, z are positions of observation in the local co-ordinate system.
    
    Parameters
    ----------
    x, y, z (array, integer): Observation coordinates (2D or 1D)
    a, b, c (floats): semiaxes lengths of the ellipsoid. 
                    NOTE: these must comply with chosen ellipsoid type.
    density (float): density of the body of interest in kg/m^3

    Returns
    -------
    dg1 (float): change in gravity for the x axis. (is this the right explanation?)
    dg2 (float): change in gravity for the y axis.
    dg3 (float): change in gravity for the z axis.

    """
    # constants
    G = 6.6743e-11

    # call and use calc_lambda abd get_ABC functions 
    lmbda = calculate_lambda(x, y, z, a, b, c)
    A_lmbda, B_lmbda, C_lmbda = get_ABC(x, y, z, a, b, c, lmbda)
    
    # check the function is used for the correct type of ellipsoid
    if not (a > b > c):
        raise ValueError(f"Invalid ellipsoid axis lengths for triaxial ellipsoid:"
            f"expected a > b > c but got a = {a}, b = {b}, c = {c}")
    
    # compute the coefficient of the three delta_g equations 
    co_eff = -2 * np.pi * a * b * c * G * density

    # compile all terms 
    dg1 = co_eff * x * A_lmbda
    dg2 = co_eff * y * B_lmbda
    dg3 = co_eff * z * C_lmbda
    
    return dg1, dg2, dg3

def get_gz_array(local_coords, internal_mask, a, b, c, density, topo_h=None):
    """
    
    Takes the chosen ellipsoid function, the internal potential function,
    runs these functions with necessary parameters,
    and combines into a single array to return a total ellipsoid function for
    any given coordinate.
    
    Parameters
    ----------
    local_coords (x, y, z): array/coordinate system of the observation points 
    rotated by some matrix R to be in the local coordinate system of the ellipsoid.
    a, b, c (floats): semiaxes lengths of the ellipsoid. 
                    NOTE: these must comply with chosen ellipsoid type.
    density (float): density of the body of interest in kg/m^3
    internal_mask (boolean array): denoting where the surface of the ellipsoid lies
    where True is inside the ellipsoid.
    
    
    Returns
    -------
    xresults (array):
    yresults (array):
    zresults (array):
    
    NOTES:
    Get it to produce one output array?
        
    """
    # select function to use based on ellipsoid parameters
    if (a > b > c):
        func = calculate_delta_gs_triaxial
    elif (a > b and b == c):
        func = calculate_delta_gs_prolate
    elif (a < b and b == c):
        func = calculate_delta_gs_oblate
    
    # unpack input 
    x, y, z = local_coords
    
    # create array to hold values
    xresults = np.zeros(x.shape)
    yresults = np.zeros(y.shape)
    zresults = np.zeros(z.shape)
    
    # call functions to produce g values, external and internal
    g_ext_x, g_ext_y, g_ext_z = func(x[~internal_mask], y[~internal_mask], z[~internal_mask], a, b, c, density)
    g_int_x, g_int_y, g_int_z = calculate_internal_g(x[internal_mask], y[internal_mask], z[internal_mask], a, b, c, density)
    
    # assign external and internal values to the arrays created
    xresults[internal_mask] = g_int_x
    xresults[~internal_mask]= g_ext_x
    
    yresults[internal_mask] = g_int_y
    yresults[~internal_mask] = g_ext_y
    
    zresults[internal_mask]= g_int_z
    zresults[~internal_mask]= g_ext_z
    
    return xresults, yresults, zresults

    