# file for doing rotations between global and local coordinate systems.

#definitions for triaxial

# alpha = azimuth of plunge of major axis (a) (clockwise from +x)
# beta = plunge of major axis (angle between major axis and horizonal)
# gamma = angle between upwards dorected intermediate axis and vertical plane containing major axis
# posiitve clockwise (?)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.spatial.transform import Rotation as R


def get_V_as_Euler(yaw, pitch, roll):
    """
    Produce rotation matrix (V) from Tait-Bryan yaw pitch roll axis rotations.
    
    parameters
    ----------
    Yaw, pitch, roll (float 0<=i<360): angles of rotation in degrees for each 
    axis, respectively. Rotations are applied in order: 1st = yaw - rotates the 
    vertical or z axis, 2nd = pitch - rotates the northing or y axis, and 3rd
    = roll - rotates the easting or x axis. 
    
    Returns
    -------
    V (matrix): rotation matrix in degrees.
    
    """
    
    
    # using scipy rotation package
    # this produces the local to global rotation matri (or what would be defined 
    # as R.T from global to local)
    r = R.from_euler('zyx', [yaw, -pitch, roll], degrees=True)
    V = r.as_matrix()
    
    return V

def structural_angles_to_abg(strike, dip, rake): # we can decide if this is needed
    """
    Takes structural (geological) angles of strike, dip, rake and converts them
    into alpha, beta, gamma angles.
    
    parameters 
    ----------
    strike (float):
    dip (0<= dip <= 90)(float):
    rake (float):
    
    
    returns
    -------
    alpha (float, 0<=alpha<=360) = azimuth of plunge of major axis (a) (clockwise from +x)
    beta (float, 0<=beta<=90) = plunge of major axis (angle between major axis and horizonal)
    gamma (float, -90<=gamma<=90) =  angle between upwards directed intermediate 
                                    axis and vertical plane containing major axis
    """

    alpha = strike - np.arcos(np.cos(rake)/np.sqrt(1 - np.sin(dip)**2 * np.sin(rake)**2))
    
    beta = np.arcsin(np.sin(dip) * np.sin(rake))
    
    gamma = -np.arctan(np.cot(dip) * np.sec(rake))
    
    return alpha, beta, gamma

def get_body_rotation_sdr(strike, dip, rake):
    """
    Creates the unit vectors of the body (local) coordinate axis from the 
    global coordinate axis, using STRIKE, DIP and RAKE
    
    parameters
    ----------
    strike (float):
    dip (0<= dip <= 90)(float):
    rake (float):
        
    
    Returns
    -------    
    V:  v1 [l1, m1, n1] (vector) 
        v2 [l2, m2, n2] (vector) 
        v3 [l3, m3, n3] (vector)
        
    """
    
    if not (0<=strike<=360):
        raise ValueError("Invalid value for strike."
                         f"Expected 0<=strike<=360, got {strike}.")
        
    if not (0<=dip<=90):
        raise ValueError("Invalid value for dip."
                         f"Expected 0<=dip<=90, got {dip}.")
        
    if not (0<=rake<=180):
        raise ValueError("Invalid value for rake."
                         f"Expected 0<=rake<=180, got {rake}.")
        
    if 0 <= rake <= 90:
        sign = 1
    else: 
        sign = -1
        
    strike = np.radians(strike)
    dip = np.radians(dip)
    rake = np.radians(rake)
        
    v1 = [l1, m1, n1] = (-np.cos(strike) * np.cos(rake) - np.sin(strike) \
                         * np.cos(dip) * np.sin(rake), -np.sin(strike) * np.cos(rake) \
                             + np.cos(strike) * np.cos(dip) * np.sin(rake), \
                                 -np.sin(dip) * np.sin(rake)
                                 )
    v2 = [l2, m2, n2] = sign * np.array((np.cos(strike) * np.sin(rake) - np.sin(strike) * np.cos(dip) \
                         * np.cos(rake), np.sin(strike) * np.sin(rake) + np.cos(strike)\
                             * np.cos(dip) * np.cos(rake), -np.sin(dip) * np.cos(rake)
                         ))
    v3 = [l3, m3, n3] = sign * np.array((np.sin(strike) * np.sin(dip), -np.cos(strike) * np.sin(dip), \
                         -np.sin(dip)
                         ))
    
    V = [v1, v2, v3]
    
    return V

#V1 = get_body_rotation_sdr(0, 90, 0)
#V2 = get_body_rotation_sdr(0, 0, 0)
#print(V1)
#print(V2)
    
def get_body_rotation_abg(alpha, beta, gamma): # in degrees
    """
    Creates the unit vectors of the body (local) coordinate axis from the 
    global coordinate axis using ALPHA, BETA and GAMMA.
    
    parameters
    ----------
    alpha (float, 0<=alpha<=360) = azimuth of plunge of major axis (a) (clockwise from +x)
    beta (float, 0<=beta<=90) = plunge of major axis (angle between major axis and horizonal)
    gamma (float, -90<=gamma<=90) =  angle between upwards directed intermediate 
                                    axis and vertical plane containing major axis
                                    
    returns
    ------- 
    # not sure how is best to lay this out 
    V: v1 [l1, m1, n1] (vector) 
       v2 [l2, m2, n2] (vector) 
       v3 [l3, m3, n3] (vector) :
           
    
    """
    
    # check inputs are valid 
    if not (0<=alpha<=360):
        raise ValueError("Invalid value for alpha."
                         f"Expected 0<=alpha<=360, got {alpha}.")
        
    if not (0<=beta<=90):
        raise ValueError("Invalid value for beta."
                         f"Expected 0<=beta<=90, got {beta}.")
        
    if not (-90<=gamma<=90):
        raise ValueError("Invalid value for gamma."
                         f"Expected -90<=gamma<=90, got {gamma}.")
    
    # convert to radians
    alpha = np.radians(alpha)
    beta = np.radians(beta)
    gamma = np.radians(gamma)
    
    v1 = [l1, m1, n1] = (-np.cos(alpha) * np.cos(beta),
                         -np.sin(alpha) * np.cos(beta),
                         -np.sin(beta))
    
    v2 = [l2, m2, n2] = (np.cos(alpha) * np.cos(gamma) * np.sin(beta) + np.sin(alpha) * np.sin(gamma),
                         np.sin(alpha) * np.cos(gamma) * np.sin(beta) - np.cos(alpha) * np.sin(gamma),
                         - np.cos(gamma) * np.cos(beta))
    
    v3 = [l3, m3, n3] = (np.sin(alpha) * np.cos(gamma) - np.cos(alpha) * np.sin(gamma) * np.sin(beta),
                         -np.cos(alpha) * np.cos(gamma) - np.sin(alpha) * np.sin(beta) * np.sin(gamma),
                         np.sin(gamma) * np.cos(beta))
    
    V = [v1, v2, v3]
    #L = (l1, l2, l3)
    #M = (m1, m2, m3)
    #N = (n1, n2, n3)
    
    return V


def global_to_local(northing, easting, extra_coords, depth, V):
    
    """
    Conversion of a point from global coordinates, which we refer to as 
    Northing, easting, height, to local coordinates which we refer to as x, y, z.
    
    Parameters
    ----------
    abg (list)[alpha, beta, gamma]:
    sdr (list) [strike, dip, rake]:
        NOTE: only one of these parameters should be included, otherwise an error
        will be raised
        
    northing, easting, extra_coords (arrays): observation plane to be converted into 
    local coordinates. NOTE: 'extra_coords' as given in vd.grid_coordinates refers
    to the height of the plane above the surface. 
    depth (float): the depth of the body below the surface.
    
    Returns
    -------
    x, y, z (arrays, floats): Observatin points to convert from global to local
    
    NOTES:
        
    Currently only translates body below surface, will need to translate body 
    from an origin point which also varies in northing/easting.
    
    """
    # get unit rotations (V)
    #if abg!=None:
    #    V = get_body_rotation_abg(abg[0], abg[1], abg[2])
    #    
    #elif sdr!=None:
    #    V = get_body_rotation_sdr(sdr[0], sdr[1], sdr[2])
    #else:
    #    raise ValueError("Input is only required for either alpha-beta-gamma"
    #                    "OR strike-dip-rake. Please choose one.")

    
    # create arrays to hold local coords
    x = np.ones(northing.shape)
    y = np.ones(northing.shape)
    z = np.ones(northing.shape)
    local_coords = [x, y, z]
    
    # calculate local_coords for each x, y, z
    for i in range(len(local_coords)):
        local_coords[i] = northing * V[i][0] + easting * V[i][1] \
            - (depth - extra_coords) * V[i][2]
            
    return local_coords

# plot the rotation vector as a rotation of the surface.
def plot_axis_rotation(northing, easting, extra_coords, depth, V):
    """
    Plots the plane of rotation of the ellipsoid, as a rotation of the 'surface'.
    
    Parameters
    ----------
    alpha (float, 0<=alpha<=360) = azimuth of plunge of major axis (a) (clockwise from +x)
    beta (float, 0<=beta<=90) = plunge of major axis (angle between major axis and horizonal)
    gamma (float, -90<=gamma<=90) =  angle between upwards directed intermediate 
                                    axis and vertical plane containing major axis
        
    northing, easting, extra_coords (arrays): observation plane to be converted into 
    local coordinates. NOTE: 'extra_coords' as given in vd.grid_coordinates refers
    to the height of the plane above the surface. 
    depth (float): the depth of the body below the surface.
    
    Returns
    -------
    None. Produces 3D plot of the surfaces.
    
    
    """
    
    # get local coordinates via rotation
    local_coords = global_to_local(northing, easting, extra_coords, depth, V)
     
    # plot both original surface and rotated plane
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    ax.plot_surface(northing, easting, extra_coords, cmap=cm.jet)
    ax.plot_surface(local_coords[0], local_coords[1], local_coords[2], cmap=cm.jet)
    
    return 


def generate_basic_ellipsoid(a, b, c):
    
    """
    Generates the basic ellipsoid with spherical angles to be plotted in 3D.
    
    parameters
    ----------
    a, b, c (floats): semiaxes lengths of the ellipsoid. 
    
    returns
    -------
    x1, y1, z1: components of the equation of the ellipsoid in spherical coords
                NOTE: not to be confused with x, y, z (local coord system observation points).
    """
    
    # Set of all spherical angles:
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    
    # Cartesian coordinates that correspond to the spherical angles:
        # np.outer is the outer product of the two arrays (ellipsoid surfce)
    x1 = a * np.outer(np.cos(u), np.sin(v))
    y1 = b * np.outer(np.sin(u), np.sin(v))
    z1 = c * np.outer(np.ones_like(u), np.cos(v))
    
    return x1, y1, z1

#x1, y1, z1 = generate_basic_ellipsoid()

def plot_rotated_ellispoid(a, b, c, depth, V):
    
    """
    Plots original ellipsoid corresponding with coordinate axis, and plots
    rotated ellipsoid corresponding to a 'local' system.
    
    Parameters
    ----------
    a, b, c (floats): semiaxes lengths of the ellipsoid.
    alpha (float, 0<=alpha<=360) = azimuth of plunge of major axis (a) (clockwise from +x)
    beta (float, 0<=beta<=90) = plunge of major axis (angle between major axis and horizonal)
    gamma (float, -90<=gamma<=90) =  angle between upwards directed intermediate 
                                    axis and vertical plane containing major axis
        
    Returns 
    -------
    None, plots the two ellispoids (one axis)
    """
    # generate ellipsoid as spherical coords to plot
    x1, y1, z1 = generate_basic_ellipsoid(a, b, c)
    
    # generate rotated ellispoid fromt the original
    local_coords = global_to_local(x1, y1, z1, V)
    
    # create plot
    fig = plt.figure(figsize=plt.figaspect(1))  # Square figure
    ax = fig.add_subplot(111, projection='3d')
    
    # plot both ellipsoid cases
    ax.plot_surface(x1, y1, z1,  rstride=4, cstride=4, color='r')
    ax.plot_surface(local_coords[0], local_coords[1], local_coords[2],  rstride=4, cstride=4, color='b')
    
    max_radius = max(a, b, c)
    for axis in 'xyz':
        getattr(ax, 'set_{}lim'.format(axis))((-max_radius, max_radius))
    
    plt.show()
    return

#plot_rotated_ellispoid(a=5, b=3, c=1, depth=0, sdr=[0, 0, 70])