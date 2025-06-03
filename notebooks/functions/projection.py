# to project gravity surface in local coords onto global coordinate system

import numpy as np
from .coord_rotations import get_V_as_Euler
from .get_gravity_ellipsoids import get_gz_array


def rotate_obs_points(obs_points, r):
    """
    Transforms and rotates observation points from a global system into a local
    system which is defined by a translation and matrix rotation.
    
    Parameters
    ----------
    obs_points: observation points (global): Eastsing, northing, upward coord components
    r: rotation matrix: rotation angles which define the ellipsoid
    
    Returns
    -------
    local_coords [x, y, z]: 
    
    """
    
    local_coords = [x, y, z] = r.T @ obs_points
    
    return local_coords

def project_gravity_global(gs, r):
    
    """
    Project the components of g onto the global vertical system. 
    
    Parameters
    ----------
    gs [gx, gy, gz]: the three components of gravity in the local system 
    R (rotation matrix): rotation matrix from the local system into the global
    system.
    
    Returns 
    -------
    g_global [ge, gn, gu]: the three global components of gravity.
    
    """
    
    g_global = [ge, gn, gu] = r @ gs
    
    return g_global

# pass the grid points e, n , u into function instead
# get it to calculate the internal mask within this function

def gz_rotated_ellipsoid(ellipsoid, yaw, pitch, roll, e, n, u, density):
    
    """
    Function which sews everything together:
    - Creates global coordinate system as user defines
    - Creates rotation matrix based on input angles
    - Rotates observation points     
    - Calculates the gravity components for the rotated points 
    - Projects the gravity components onto the U axis (upward gravity component)
    - Returns array of the upward gravity component
    
    Parameters
    ----------
    
    Returns 
    -------
    """
    a, b, c = ellipsoid.a, ellipsoid.b, ellipsoid.c
    
    # create boolean for internal vs external field points
    internal_mask = (e**2)/(a**2) + (n**2)/(b**2) + (u**2)/(c**2) < 1
    cast = np.broadcast(e, n, u)
    obs_points = np.vstack((e.ravel(), n.ravel(), u.ravel()))
    
    # create rotation matrix 
    R = get_V_as_Euler(yaw, pitch, roll)
    
    # rotate observation points
    rotated_points = R.T @ obs_points
    x, y, z = tuple(c.reshape(cast.shape) for c in rotated_points)
    
    # calculate gravity component for the rotated points
    gx, gy, gz = get_gz_array(internal_mask, a, b, c, x, y, z, density)
    G = np.vstack((gx.ravel(), gy.ravel(), gz.ravel()))
    
    # project onto upward unit vector, axis U
    g_projected = R @ G
    ge, gn, gu = tuple(c.reshape(cast.shape) for c in g_projected)
    
    
    
    return ge, gn, gu 