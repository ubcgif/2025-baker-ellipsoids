# to project gravity surface in local coords onto global coordinate system

import numpy as np
from .coord_rotations import _get_V_as_Euler
from .get_gravity_ellipsoids import _get_gravity_array


def ellipsoid_gravity(coordinates, ellipsoids, density, field="g"):
    """

    Compute the three gravity components for an ellipsoidal body at specified o
    bservation locations.

    - Unpacks ellipsoid instance parameters (a, b, c, yaw, pitch, roll, origin)
    - Constructs Euler rotation matrix
    - Rotates observation points (e, n, u) into local ellipsoid system (x, y, z)
    - Computes gravity components in local coordinate system
    - Projects these gravity components back into the original coordinate
      system (e, n, u)
    - Returns the gravity field components as arrays

    Parameters
    ----------

    ellipsoid* : value, or list of values
        instance(s) of TriaxialEllipsoid, ProlateEllipsoid,
                 OblateEllipsoid
        Geometric description of the ellipsoid:
            - Semiaxes : a, b, c**
            - Orientation : yaw, pitch, roll**
            - Origin : centre point (x, y, z)

    e : ndarray
        Easting coordinates, in the form:
            - A scalar value (float or int)
            - A 1D array of shape (N,)
            - A 2D array (meshgrid) of shape (M, N)

    n : ndarray
        Northing coordinates, same shape and rules as 'e'.

    u : ndarray
        Upward coordinates, e.g. the surface height desired to compute the
        gravity value. Same shape and rules as 'e'.

    density***: float or list
        The uniform density of the ellipsoid in kg/m^3, or an array of densities
        for multiple ellipsoid cases, with the same size as 'ellipsoids'.

    Returns
    -------

    ge: ndarray
        Easting component of the gravity field.

    gn ndarray
        Northing component of the gravity field.

    gu: ndarray
        Upward component of the gravity field.

    NOTES

    -----
    * : ellipsoid may be defined using one of the three provided classes:
        TriaxialEllipsoid where a > b > c, OblateEllipsoid where a < b = c,
        and ProlateEllipsoid where a > b = c.
    ** : the c value and roll angle are only explicitly defined for trixial
        ellipsoids by definition of the ellipsoid. Otherwise, ProlateEllipsoid
        and OblateEllipsoid take b = c and roll = 0.
    ***: Must be an array even if multiple ellipsoids are passes, EVEN IF
        the ellipsoids have the same desnity.
    Input arrays must match in shape.

    """
    # unpack coordinates and create array to hold final g values
    e, n, u = coordinates[0], coordinates[1], coordinates[2]
    cast = np.broadcast(e, n, u)
    ge, gn, gu = np.zeros(e.shape), np.zeros(e.shape), np.zeros(e.shape)

    # deal with the case of a single ellipsoid being passed
    if type(ellipsoids) is not list:
        ellipsoids = [ellipsoids]
        density = [density]

    # case of multiple ellipsoids but only one density value
    if type(ellipsoids) is list and type(density) is not list:
        raise ValueError(
            "Ellipsoids is a list, but density is not."
            "Perhaps multiple arguments were given for ellipsoids"
            " but only one value was given for density?"
        )

    # case of passing density as a list of differing size to ellipsoids
    if len(ellipsoids) != len(density):
        raise ValueError(
            f"{len(ellipsoids)} arguments were given for ellipsoids"
            f" but {len(density)} values were given for density."
        )

    for index, ellipsoid in enumerate(ellipsoids):
        # unpack instances
        a, b, c = ellipsoid.a, ellipsoid.b, ellipsoid.c
        yaw, pitch, roll = ellipsoid.yaw, ellipsoid.pitch, ellipsoid.roll
        ox, oy, oz = ellipsoid.centre

        # preserve ellipsoid shape, translate origin of ellipsoid
        cast = np.broadcast(e, n, u)
        obs_points = np.vstack(((e - ox).ravel(), (n - oy).ravel(), (u - oz).ravel()))

        # create rotation matrix
        R = _get_V_as_Euler(yaw, pitch, roll)

        # rotate observation points
        rotated_points = R.T @ obs_points
        x, y, z = tuple(c.reshape(cast.shape) for c in rotated_points)

        # create boolean for internal vs external field points
        internal_mask = (x**2) / (a**2) + (y**2) / (b**2) + (z**2) / (c**2) < 1

        # calculate gravity component for the rotated points
        gx, gy, gz = _get_gravity_array(internal_mask, a, b, c, x, y, z, density[index])
        gravity = np.vstack((gx.ravel(), gy.ravel(), gz.ravel()))

        # project onto upward unit vector, axis U
        g_projected = R @ gravity
        ge_i, gn_i, gu_i = tuple(c.reshape(cast.shape) for c in g_projected)

        # sum contributions from each ellipsoid
        ge += ge_i
        gn += gn_i
        gu += gu_i

    if field == "e":
        return ge
    elif field == "n":
        return gn
    elif field == "u":
        return gu

    return ge, gn, gu
