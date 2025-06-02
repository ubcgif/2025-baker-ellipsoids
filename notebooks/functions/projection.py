# to project gravity surface in local coords onto global coordinate system

import numpy as np
from .utils import get_coords_and_mask
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


def gz_rotated_ellipsoid(a, b, c, yaw, pitch, roll, obs_points, internal_mask, density):
    
    """
    Function which sews everything together:
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
    
    # create rotation matrix 
    R = get_V_as_Euler(yaw, pitch, roll)
    
    # rotate observation points
    local_coords = [x, y, z] = R.T @ obs_points 
    
    # calculate gravity component for the rotated points
    G = [gx, gy, gz] = get_gz_array(local_coords, internal_mask, a, b, c, density)
    
    # project onto upward unit vector, axis U
    G_global = [ge, gn, gu] = R @ G
    
    return G_global