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
    
    if np.all((a > b) & (b > c)):
        func = _depol_triaxial_int(a, b, c)
    if np.all((a > b) & (b == c)):
        func = _depol_prolate_int(a, b, c)
    if np.all((a < b) & (b == c)):
        func = _depol_oblate_int(a, b, c)
    
    # construct identity matrix
    N = np.eye(3)
    
    for i in range(3):
        N[i][i] *= func[i]
    
    return N
    


# construct components of the external matrix




def _get_h_values(a, b, c, lmbda):
    
    """ Get the h values for the N matrix """
    
    axes = np.array([a, b, c])
    h= np.zeros(3)
    R = np.sqrt(np.prod(axes**2 + lmbda))
    return -1 / ((axes**2 + lmbda) * R)

def _spatial_deriv_lambda(x, y, z, a, b, c, lmbda):
    
    """ Get the spatial derivative of lambda with respect to the x,y,z """
    
    # Numerators (shape: same as x, y, z)
    num_x = 2 * x / (a**2 + lmbda)
    num_y = 2 * y / (b**2 + lmbda)
    num_z = 2 * z / (c**2 + lmbda)
    
    # Denominator (broadcasts naturally)
    denom = ((x / (a**2 + lmbda))**2 +
             (y / (b**2 + lmbda))**2 +
             (z / (c**2 + lmbda))**2)
    
    # Avoid divide-by-zero
    denom = np.where(denom == 0, 1e-12, denom)
    
    
    vals = np.stack([num_x / denom, num_y / denom, num_z / denom], axis=-1)
    
    return vals

def _get_g_values_magnetics(a, b, c, lmbda):
    
    """Get the gravity values for the three ellipsoid types."""
        
    if (a > b > c):
        func = _get_ABC(a, b, c, lmbda)
        gvals_x, gvals_y, gvals_z = func[0], func[1], func[2]
        
    if (a > b and b == c):
        g1 = 2/((a**2 - b **2)**3/2) \
            * (np.log(((a**2 - b**2)**0.5 + (a**2 + lmbda)**0.5)/(b**2 + lmbda)**0.5) \
               - ((a**2 - b**2)/(a**2 + lmbda))**0.5)
        g2 = 1/((a**2 - b**2)**3/2) \
            * (((a**2 - b**2)*(a**2 + lmbda)**0.5)/(b**2 + lmbda)) \
                - (np.log(((a**2 - b**2)**0.5 + (a**2 + lmbda)**0.5)/(b**2 + lmbda)**0.5))
        gvals_x, gvals_y, gvals_z = g1, g2, g2
        
    if (a < b and b == c):
        g1 = 2/((b**2 - a**2)**3/2) * ((((b**2 - a**2)/(a**2 + lmbda))**0.5) \
            - np.arctan(((b**2 - a**2) / (a**2 + lmbda)) ** 0.5))
        g2 = 1/((b**2 - a**2)**3/2) * (np.arctan(((b**2 - a**2) / (a**2 + lmbda)) ** 0.5) \
                                       - (((b**2 - a**2) * (a**2 + lmbda)) ** 0.5) / (b**2 + lmbda))
                
        gvals_x, gvals_y, gvals_z = g1, g2, g2
    
    return gvals_x, gvals_y, gvals_z
    


def _construct_N_matrix_external(x, y, z, a, b, c, lmbda):
    
    """ Construct the N matrix for the external field"""
    
    # g values here are equivalent to the A(lambda) etc values previously.
    # h values as above 
    # lambda derivatives as above
    N = np.eye(3)
    r = [x, y, z]
    gvals = _get_g_values_magnetics(a, b, c, lmbda)
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


def get_magnetic_components(coordinates, ellipsoids, k, H0, mu0):
    
    """get internal and external components of magnetism """
    
    e, n, u = coordinates[0], coordinates[1], coordinates[2]
    cast = np.broadcast(e, n, u)
    be, bn, bu = np.zeros(e.shape), np.zeros(e.shape), np.zeros(e.shape)
    

    if type(ellipsoids) is not list:
        ellipsoids = [ellipsoids]
    if type(H0) is not np.ndarray:
        raise ValueError("H0 values of the regional field  must be an array.")
    
    for index, ellipsoid in enumerate(ellipsoids):
        a, b, c = ellipsoid.a, ellipsoid.b, ellipsoid.c
        yaw, pitch, roll = ellipsoid.yaw, ellipsoid.pitch, ellipsoid.roll
        ox, oy, oz = ellipsoid.centre
        lmbda = _calculate_lambda(e, n, u, a, b, c)
        
        # preserve ellipsoid shape, translate origin of ellipsoid
        cast = np.broadcast(e, n, u)
        obs_points = np.vstack(((e - ox).ravel(), (n - oy).ravel(), (u - oz).ravel()))
        
        # get observation points, rotate them
        R = _get_V_as_Euler(yaw, pitch, roll)
        rotated_points = R.T @ obs_points
        x, y, z = tuple(c.reshape(cast.shape) for c in rotated_points)

        # create boolean for internal vs external field points
        internal_mask = (x**2) / (a**2) + (y**2) / (b**2) + (z**2) / (c**2) < 1

        # create K and N_cross matrix
        K = k * np.eye(3)
        
        for i, j in np.ndindex(lmbda.shape):
            lam = lmbda[i, j]
            xi, yi, zi = x[i, j], y[i, j], z[i, j]
            is_internal = internal_mask[i, j]
            
            N_cross = _construct_N_matrix_internal(xi, yi, zi, a, b, c, lam)
            
            if is_internal:
                N = N_cross
            else:
                N = _construct_N_matrix_external(xi, yi, zi, a, b, c, lam)
        
            Nr = R.T @ N @ R
            H_cross = np.linalg.inv(np.eye(3) + N_cross @ K) @ H0
            Hr = H0 + (Nr @ K) @ H_cross
        
            be[i, j], bn[i, j], bu[i, j] = 1e9 * mu0 * Hr
    
    return be, bn, bu
lmbda = _calculate_lambda(5, 5, 5, 3, 2, 1)


#todo 
# insert as an ellipsoid / coords etc into function 
# gte ir to work for multiple values.
