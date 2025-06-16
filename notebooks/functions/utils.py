import numpy as np
import verde as vd
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.spatial.transform import Rotation as R

def _calculate_lambda(x, y, z, a, b, c):  # takes semiaxes and observation coordinates
    """
    Calculate the value of lambda, the parameter defining surfaces in a confocal family
    of ellipsoids (i.e., the inflation/deflation parameter), for a given ellipsoid and
    observation point.

    Parameters
    ----------
    x : float or array
        X-coordinate(s) of the observation point(s) in the local coordinate system.
    y : float or array
        Y-coordinate(s) of the observation point(s).
    z : float or array
        Z-coordinate(s) of the observation point(s).
    a : float
        Semi-major axis of the ellipsoid along the x-direction.
    b : float
        Semi-major axis of the ellipsoid along the y-direction.
    c : float
        Semi-major axis of the ellipsoid along the z-direction.

    Returns
    -------
    lmbda : float or array-like
        The computed value(s) of the lambda parameter.

    """
    # Calculate lambda using x, y, z

    # Check that the input values of x, y, z are greater than a, b, c semi axis lengths
    if not (np.any(np.abs(x) >= a) or np.any(np.abs(y) >= b) or np.any(np.abs(z) >= c)):
        raise ValueError(
            "Arrays x, y, z should contain points which lie outside"
            " of the surface defined by a, b, c"
        )

    # compute lambda
    p_0 = (
        a**2 * b**2 * c**2
        - b**2 * c**2 * x**2
        - c**2 * a**2 * y**2
        - a**2 * b**2 * z**2
    )
    p_1 = (
        a**2 * b**2
        + b**2 * c**2
        + c**2 * a**2
        - (b**2 + c**2) * x**2
        - (c**2 + a**2) * y**2
        - (a**2 + b**2) * z**2
    )
    p_2 = a**2 + b**2 + c**2 - x**2 - y**2 - z**2

    p = p_1 - (p_2**2) / 3

    q = p_0 - ((p_1 * p_2) / 3) + 2 * (p_2 / 3) ** 3

    theta_internal = -q / (2 * np.sqrt((-p / 3) ** 3))

    # clip to remove floating point precision errors
    theta_internal_1 = np.clip(theta_internal, -1.0, 1.0)

    theta = np.arccos(theta_internal_1)

    lmbda = 2 * np.sqrt((-p / 3)) * np.cos(theta / 3) - p_2 / 3

    return lmbda

def _get_V_as_Euler(yaw, pitch, roll):
    """
    Generate a rotation matrix (V) from Tait-Bryan angles: yaw, pitch, and roll.

    Parameters
    ----------
    yaw : float
        Rotation about the vertical (z) axis, in degrees.
    pitch : float
        Rotation about the northing (y) axis, in degrees.
    roll : float
        Rotation about the easting (x) axis, in degrees.

    These rotations are applied in the following order order as above, (zyx).

    Returns
    -------
    V : ndarray of shape (3, 3)
        Rotation matrix that transforms coordinates from the local ellipsoid-aligned
        frame to the global coordinate system.

    Notes
    -----
    All angles must be given in degrees.

    """

    # using scipy rotation package
    # this produces the local to global rotation matrix (or what would be defined
    # as R.T from global to local)
    r = R.from_euler("zyx", [yaw, -pitch, roll], degrees=True)
    V = r.as_matrix()

    return V


def _global_to_local(northing, easting, extra_coords, depth, V):
    """
    Convert observation points from global coordinates (Northing, Easting, Height)
    to local ellipsoid-aligned coordinates (x, y, z).

    Parameters
    ----------
    northing : array_like
        Northing (Y) coordinates in the global system.

    easting : array_like
        Easting (X) coordinates in the global system.

    extra_coords : array_like
        Height or vertical offset above the surface (commonly from `vd.grid_coordinates`).

    depth : float
        Depth of the ellipsoid’s center below the surface (positive downward).

    V : ndarray of shape (3, 3)
        Rotation matrix used to transform from global to local coordinates.

    Returns
    -------
    x, y, z : ndarray
        Coordinates of the observation points in the local ellipsoid-aligned frame.

    Notes
    -----
    Needs to handle translation component.

    """

    # create arrays to hold local coords
    x = np.ones(northing.shape)
    y = np.ones(northing.shape)
    z = np.ones(northing.shape)
    local_coords = [x, y, z]

    # calculate local_coords for each x, y, z
    for i in range(len(local_coords)):
        local_coords[i] = (
            northing * V[i][0] + easting * V[i][1] - (depth - extra_coords) * V[i][2]
        )

    return local_coords



##############################################################################

# likely redundant functions


def _get_ellipsoid_mass(a, b, c, density):
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
    volume = 4 / 3 * np.pi * a * b * c

    return density * volume


def _get_coords_and_mask(region, spacing, extra_coords, a, b, c, topo_h=None):
    """
    Return the  coordinates and mask which separates points
    within the given ellipsoid and on or outside
    of the given ellipsoid.

    Parameters
    ----------
    region (list)(W, E, S, N): end points of the coordinate grid
    spacing (float): separation between the points (default = 1)
    extra_coords (float or list): surfaces of constant height to test (default = 0)
    a, b, c (float): semiaxes of the ellipsoid

    Returns
    -------
    x, y, z (arrays): 2D coordinate arrays for grid
    internal (array): mask for the internal points of the ellipsoid

    NOTES:
    Consider making it possible to pass a varying array as a set of z coords.
    """
    if topo_h == None:
        e, n, u = vd.grid_coordinates(
            region=region, spacing=spacing, extra_coords=extra_coords
        )

    else:
        e, n = vd.grid_coordinates(region=region, spacing=spacing)
        u = topo_h * np.exp(-(e**2) / (np.max(e) ** 2) - n**2 / (np.max(n) ** 2))

    internal = (e**2) / (a**2) + (n**2) / (b**2) + (u**2) / (c**2) < 1

    return e, n, u, internal
