from .utils import calculate_lambda, get_ellipsoid_mass
from  .get_gravity_ellipsoids import calculate_delta_gs_oblate, calculate_delta_gs_prolate, calculate_delta_gs_triaxial, calc_gz_array
import  matplotlib.pyplot as plt
from choclo.point import gravity_u as pointgrav
import numpy as np
import verde as vd

# plot visualising how delta_g_z changes with distance from the ellipsoid 

def plot_colourmap_gz(func, spacing, region, z_height, a, b, c, density):
    """
    Creates northing and easting (x, y) coordinates (fixed area for now),
    Eliminates those within the ellipsoid body,
    Calculates gz for those outside the body,
    Plots 2D colourmap of the gz value with location.
    
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
    None. Plots colourmap of easting, northing at chosen z surface height.    
    """

    e, n, g = calc_gz_array(func, spacing, region, z_height, a, b, c, density)
    plt.pcolormesh(e, n, g)
    plt.gca().set_aspect("equal")
    plt.colorbar(label="gz")
    plt.title("Vertical gravity on a plane of constant height")
    plt.xlabel("Easting")
    plt.tight_layout()
    plt.ylabel("Northing")
    plt.show()
    
    return 


def plot_gz_decay_comparison(func, num_points, a, b, c, density):
    """
    Plots the curves for z gravity component as it decays with distance,
    for a user defined ellipsoid,
    and the point mass equivalent. 
    
    Parameters
    ----------
    func (function): the function for the desired ellipsoid (triaxial, prolate or oblate)
    
    Returns 
    -------
    
    """
    
    smallest_zval = max([a, b, c])
        
    z = np.linspace(smallest_zval, 2*smallest_zval**2, num_points)
    x = y = np.zeros_like(z)

    # lists to hold variables
    mag = []
    gz = []
    point_gz = []
    
    #calculate ellipsoid mass
    ellipsoid_mass = get_ellipsoid_mass(a, b, c, density)
    

    # iterate over z component and append the calculated values
    for i in range(len(x)):
        _ , _ , gz_val = func(x[i], y[i], z[i], a, b, c, density)
        point_grav = pointgrav(x[i], y[i], z[i], 0, 0, 0, ellipsoid_mass)
        point_gz.append(point_grav)
        gz.append(gz_val)

    # plot findings
    plt.figure(figsize = (10, 8))
    plt.title("Comparing ellipsoid gravity to point mass gravity at distance")
    plt.plot(z, np.abs(gz), color = 'blue', label='Ellipsoid decay')
    plt.plot(z, np.abs(point_gz), color = 'red', label='Point mass decay')
    plt.legend()
    plt.xlabel('Distance from source surface (m)')
    plt.ylabel('change in gravity (m/s^2)')
    plt.grid(alpha=0.3)
    
    return 
