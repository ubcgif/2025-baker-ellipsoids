import matplotlib.pyplot as plt
from choclo.point import gravity_u as pointgrav
import numpy as np
from .get_gravity_ellipsoids import get_gz_array
from .utils import  get_ellipsoid_mass, get_coords_and_mask
import verde as vd
from matplotlib import cm

# plot visualising how delta_g_z changes with distance from the ellipsoid 

def plot_colourmap_gz(region, spacing, extra_coords, a, b, c, density, func, topo_h=None):
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
    a, b, c (floats): semiaxes lengths of the ellipsoid. 
                    NOTE: these must comply with chosen ellipsoid type.
    density (float): uniform density of the ellipsoid.
                    
    
    Returns
    -------
    None. Plots colourmap of easting, northing at chosen z surface height.    
    """

    x, y, z, _ = get_coords_and_mask(region, spacing, extra_coords, a, b, c, topo_h)

    xresults, yresults, zresults = get_gz_array(region, spacing, extra_coords, a, b, c, density, func, topo_h)
   
    
    plt.pcolormesh(x, y, zresults)
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
    None. Produces two plots: decay with distance and difference between the two
    models.
    
    """
    
    # plot decay when moving along the z-axis
    z = np.linspace(c, 10*c**2, num_points)
    x = y = np.zeros_like(z)

    # lists to hold variables
    gz = []
    point_gz = []
    
    # calculate ellipsoid mass
    ellipsoid_mass = get_ellipsoid_mass(a, b, c, density)
    

    # iterate over z component and append the calculated values
    for i in range(len(x)):
        _ , _ , gz_val = func(x[i], y[i], z[i], a, b, c, density)
        point_grav = pointgrav(x[i], y[i], z[i], 0, 0, 0, ellipsoid_mass)
        point_gz.append(point_grav)
        gz.append(gz_val)

    
    # plotting
    fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharex=True)

    # plot 1: gravity decay of the two models
    axs[0].set_title("Comparing ellipsoid gravity to point mass gravity at distance")
    axs[0].plot(z, np.abs(gz), color='blue', label='Ellipsoid decay')
    axs[0].plot(z, np.abs(point_gz), color='red', label='Point mass decay')
    axs[0].set_ylabel('Gravity (m/s²)')
    axs[0].set_xlabel('Distance from source (surface) (m)')
    axs[0].legend()
    axs[0].grid(alpha=0.3)
    
    # plot 2: difference between the gravity models
    axs[1].set_title("Difference between ellipsoid and point mass gravity")
    axs[1].plot(z, np.abs(gz) - np.abs(point_gz), color='purple', label='Difference')
    axs[1].set_xlabel('Distance from source (surface) (m)')
    axs[1].set_ylabel('Change in gravity (m/s²)')
    axs[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()

    return 
