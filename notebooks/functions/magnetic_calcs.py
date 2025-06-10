# calculations to approximate magnetic field 

from .get_gravity_ellipsoids import _get_ABC
from .coord_rotations import _get_V_as_Euler
from .utils import _calculate_lambda
import numpy as np
from scipy.special import ellipkinc, ellipeinc

# internal field N matrix functions

def _depol_triaxial_int(a, b, c):
    """
    Calculate the internal depolarisation tensor (N(r)) for the triaxial case.
    """
    
    phi = np.arccos(c/a)
    k = np.sqrt(((a**2 - b**2) / (a**2 - c**2)))
    coeff = (a * b * c) / (np.sqrt(a**2 - c**2) * (a**2 - b**2))
    
    nxx = coeff * (ellipkinc(phi, k) - ellipeinc(phi, k))
    nyy = -nxx + coeff * ellipeinc(phi, k) - c**2 / (b**2 - c**2)
    nzz = -coeff * ellipeinc(phi, k) + b**2 / (b**2 - c**2)
    
    return nxx, nyy, nzz
    
def _depol_prolate_int(a, b, c):
    """ 
    Calcualte internal depolarisation factors for prolate case.
    """
    m = a/b
    
    nxx = 1/(m**2 -1) * ((m/np.sqrt(m**2 -1)) * np.log(m + np.sqrt(m**2 -1)) - 1)
    nyy = nzz = 0.5 * (1 - nxx)
    
    return nxx, nyy, nzz
     
def _depol_oblate_int(a, b, c):
    """ 
    Calcualte internal depolarisation factors for prolate case.
    """
    
    m = a/b
    
    nxx = 1/(1 - m**2) * (1 - (m/np.sqrt(1-m**2)) * np.arccos(m))
    nyy = nzz = 0.5 * (1 - nxx)
    
    return nxx, nyy, nzz

def _construct_N_matrix_internal(x, y, z, a, b, c, lmbda):
    
    """ Construct the N matrix for the internal field"""
    
    # only diagonal elements 
    # Nii corresponds to the above functions
    
    if (a > b > c):
        func = _depol_triaxial_int(a, b, c)
    if (a > b and b == c):
        func = _depol_prolate_int(a, b, c)
    if (a < b and b == c):
        func = _depol_oblate_int(a, b, c)
    
    # construct identity matrix
    N = np.eye(3)
    
    for i in range(3):
        N[i][i] *= func[i]
    
    return N
    


# construct components of the external matrix




def _get_h_values(a, b, c, lmbda):
    
    """ Get the h values for the N matrix """
    
    axes = a, b, c 
    h_i = np.zeros(3)
    R = np.sqrt(((a**2 + lmbda) * (b**2 + lmbda) * (c**2 + lmbda)))
    for index, value in enumerate(axes):
        h = - 1 / ((value**2 + lmbda) * R)
        h_i[index] = h
        
    return h_i

def _spatial_deriv_lambda(x, y, z, a, b, c, lmbda):
    
    """ Get the spatial derivative of lambda with respect to the x,y,z """
    
    r = (x, y, z)
    e = (a, b, c)
    vals = np.zeros(len(e))
    
    for i in range(len(r)):
        numerator = 2 * r[i] / (e[i]**2 + lmbda)
        denominator = (x/a**2 + lmbda)**2 + (y/b**2 + lmbda)**2 + (z/c**2 + lmbda)**2
        vals[i] = numerator/denominator
    return vals

def _get_g_values_magnetics(x, y, z, a, b, c, lmbda):
    
    """Get the gravity values for the three ellipsoid types."""
    
    gvals = np.zeros(3)
    if (a > b > c):
        func = _get_ABC(x, y, z, a, b, c, lmbda)
        gvals[0], gvals[1], gvals[2] = func[0], func[1], func[2]
        
    if (a > b and b == c):
        g1 = 2/((a**2 - b **2)**3/2) \
            * (np.log(((a**2 - b**2)**0.5 + (a**2 + lmbda)**0.5)/(b**2 + lmbda)**0.5) \
               - ((a**2 - b**2)/(a**2 + lmbda))**0.5)
        g2 = 1/((a**2 - b**2)**3/2) \
            * (((a**2 - b**2)*(a**2 + lmbda)**0.5)/(b**2 + lmbda)) \
                - (np.log(((a**2 - b**2)**0.5 + (a**2 + lmbda)**0.5)/(b**2 + lmbda)**0.5))
        gvals[0], gvals[1], gvals[2] = g1, g2, g2
        
    if (a < b and b == c):
        g1 = 2/((b**2 - a**2)**3/2) * ((((b**2 - a**2)/(a**2 + lmbda))**0.5) \
            - np.arctan(((b**2 - a**2) / (a**2 + lmbda)) ** 0.5))
        g2 = 1/((b**2 - a**2)**3/2) * (np.arctan(((b**2 - a**2) / (a**2 + lmbda)) ** 0.5) \
                                       - (((b**2 - a**2) * (a**2 + lmbda)) ** 0.5) / (b**2 + lmbda))
                
        gvals[0], gvals[1], gvals[2] = g1, g2, g2
    
    return gvals
    


def _construct_N_matrix_external(x, y, z, a, b, c, lmbda):
    
    """ Construct the N matrix for the external field"""
    
    # g values here are equivalent to the A(lambda) etc values previously.
    # h values as above 
    # lambda derivatives as above
    N = np.eye(3)
    r = [x, y, z]
    gvals = _get_g_values_magnetics(x, y, z, a, b, c, lmbda)
    derivs_lmbda = _spatial_deriv_lambda(x, y, z, a, b, c, lmbda)
    h_vals = _get_h_values(a, b, c, lmbda)
    
    for i in range(len(N)):
        for j in range(len(N[0])):
            if i == j:
                N[i][j] = (- a*b*c/2) * (derivs_lmbda[i] * h_vals[i] * r[i] + gvals[i])
            else:
                N[i][j] = (- a*b*c/2) * (derivs_lmbda[i] * h_vals[j] * r[j])
                
    return N
                       

# construct total magnetic components
# just for 1 obs point rn 


def get_magnetic_components(x, y, z, a, b, c, yaw, pitch, roll, k, H0, mu0):
    
    """get internal and external components of magnetism """
    
    # construct feeder components
    internal_mask = (x**2) / (a**2) + (y**2) / (b**2) + (z**2) / (c**2) < 1
    lmbda = _calculate_lambda(x, y, z, a, b, c)
    R = _get_V_as_Euler(yaw, pitch, roll)
    K = k * np.eye(3)
    N_cross = _construct_N_matrix_internal(lmbda, x, y, z, a, b, c)
    # construct depol matrix
    
    if internal_mask is True:
        N = N_cross
    
    else:
        N = _construct_N_matrix_external(x, y, z, a, b, c, lmbda)
        
    Nr = R.T @ N @ R
    H_cross = np.linalg.inv(np.eye(3) + N_cross @ K) @ H0
    Hr = H0 + (Nr @ K) @ H_cross
    Br = 1e9 * mu0 * Hr
    
    return H_cross, Br
lmbda = _calculate_lambda(5, 5, 5, 3, 2, 1)
N = _construct_N_matrix_external(5, 5, 5, 3, 2, 1, lmbda)
N_int = _construct_N_matrix_internal(5, 5, 5, 3, 2, 1, lmbda)
print("N:", N, N.shape)
print("N_int:", N_int, N_int.shape)
#print("Nr:", Nr.shape)
#print("K:", K.shape)
#print("H_cross:", np.shape(H_cross))
#print("H0:", H0.shape)
# how to construct the nii and nij given there is a derivative??
# otherwise just extract the gi components and add them to the h components using 
# the equations on pafe 3596 takenhashi 
