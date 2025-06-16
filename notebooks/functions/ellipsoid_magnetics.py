# calculations to approximate magnetic field

from .ellipsoid_gravity import _get_ABC
from .utils import (_get_V_as_Euler, _calculate_lambda)

import numpy as np
from scipy.special import ellipkinc, ellipeinc
from scipy.constants import mu_0

# internal field N matrix functions

def ellipsoid_magnetics(coordinates, ellipsoids, k, H0, field="b"):
    """
    Produces the components for the magnetic field components (be, bn, bu):

        - Unpacks ellipsoid instance parameters (a, b, c, yaw, pitch, roll, origin)
        - Constructs Euler rotation matrix
        - Rotates observation points (e, n, u) into local ellipsoid system (x, y, z)
        - constructs susceptability matrix and depolarisation matrix for internal
          and external points of observation
        - Calculates the magentic field due to the magnetised ellipsoid (H())
        - Converts this to magentic induction (B()) in nT.

    Parameters
    ----------

    coordinates: tuple of easting (e), northing (n), upward (u) points
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


    ellipsoid* : value, or list of values
        instance(s) of TriaxialEllipsoid, ProlateEllipsoid,
                 OblateEllipsoid
        Geometric description of the ellipsoid:
            - Semiaxes : a, b, c**
            - Orientation : yaw, pitch, roll**
            - Origin : centre point (x, y, z)

    k : float or list of floats
        Susceptibilty value. Assumes isotropy within the body. Should be of the
        same length as the number of ellipsoids given.

    H0 : ndarray
        Three components of the uniform inducing field.

    field : (optional) str, one of either "e", "n", "u".
        if no input is given, the function will return all three components of
        magentic induction.

    Returns
    -------

    be: ndarray
        Easting component of the magnetic field.

    bn ndarray
        Northing component of the magnetic field.

    bu: ndarray
        Upward component of the magnetic field.

    NOTES
    -----

    """
    # unpack coordinates, set up arrays to hold results
    e, n, u = coordinates[0], coordinates[1], coordinates[2]
    cast = np.broadcast(e, n, u)
    be, bn, bu = np.zeros(e.shape), np.zeros(e.shape), np.zeros(e.shape)

    # check inputs are of the correct type
    if type(ellipsoids) is not list:
        ellipsoids = [ellipsoids]

    if type(k) is not list:
        k = [k]

    if len(ellipsoids) != len(k):
        raise ValueError(
            "Magnetic susceptibilty must be a list containing the value"
            " of k for each ellipsoid. Instead, number of ellipsoids"
            f" given is {len(ellipsoids)} and number of k values is"
            f" {len(k)}."
        )

    if type(H0) is not np.ndarray:
        raise ValueError("H0 values of the regional field  must be an array.")

    # loop over each given ellipsoid
    for index, ellipsoid in enumerate(ellipsoids):

        # unpack instance
        a, b, c = ellipsoid.a, ellipsoid.b, ellipsoid.c
        yaw, pitch, roll = ellipsoid.yaw, ellipsoid.pitch, ellipsoid.roll
        ox, oy, oz = ellipsoid.centre

        # preserve ellipsoid shape, translate origin of ellipsoid
        cast = np.broadcast(e, n, u)
        obs_points = np.vstack(((e - ox).ravel(), (n - oy).ravel(), (u - oz).ravel()))

        # get observation points, rotate them
        R = _get_V_as_Euler(yaw, pitch, roll)
        rotated_points = R.T @ obs_points
        x, y, z = tuple(c.reshape(cast.shape) for c in rotated_points)

        # create boolean for internal vs external field points
        # and compute lambda for each coordinate point
        lmbda = _calculate_lambda(x, y, z, a, b, c)
        internal_mask = (x**2) / (a**2) + (y**2) / (b**2) + (z**2) / (c**2) < 1

        # create K matrix
        K = k[index] * np.eye(3)

        # create N matricies for each given point
        for i, j in np.ndindex(lmbda.shape):
            lam = lmbda[i, j]
            xi, yi, zi = x[i, j], y[i, j], z[i, j]
            is_internal = internal_mask[i, j]

            N_cross = _construct_N_matrix_internal(a, b, c)

            if is_internal:
                N = N_cross
            else:
                N = _construct_N_matrix_external(xi, yi, zi, a, b, c, lam)

            # compute rotation and final H() values
            Nr = R.T @ N @ R
            H_cross = np.linalg.inv(np.eye(3) + N_cross @ K) @ H0
            Hr = H0 + (Nr @ K) @ H_cross

            # sum across all components and ellipsoids
            be[i, j] += 1e9 * mu_0 * Hr[0]
            bn[i, j] += 1e9 * mu_0 * Hr[1]
            bu[i, j] += 1e9 * mu_0 * Hr[2]

    # return according to user
    if field == "e":
        return be
    elif field == "n":
        return bn
    elif field == "u":
        return bu

    return be, bn, bu


def _depol_triaxial_int(a, b, c):
    """
    Calculate the internal depolarisation tensor (N(r)) for the triaxial case.

    parameters
    ----------
    a, b, c : floats
        Semiaxis lengths of the triaxial ellipsoid (a ≥ b ≥ c).

    returns
    -------
    nxx, nyy, nzz : floats
        individual diagonal components of the x, y, z matrix.


    """

    phi = np.arccos(c / a)
    k = np.sqrt(((a**2 - b**2) / (a**2 - c**2)))
    coeff = (a * b * c) / (np.sqrt(a**2 - c**2) * (a**2 - b**2))

    nxx = coeff * (ellipkinc(phi, k) - ellipeinc(phi, k))
    nyy = -nxx + coeff * ellipeinc(phi, k) - c**2 / (b**2 - c**2)
    nzz = -coeff * ellipeinc(phi, k) + b**2 / (b**2 - c**2)

    return nxx, nyy, nzz


def _depol_prolate_int(a, b, c):
    """
    Calcualte internal depolarisation factors for prolate case.


    parameters
    ----------
    a, b, c : floats
        Semiaxis lengths of the prolate ellipsoid (a > b = c).

    returns
    -------
    nxx, nyy, nzz : floats
        individual diagonal components of the x, y, z matrix.

    """
    m = a / b

    nxx = 1 / (m**2 - 1) * ((m / np.sqrt(m**2 - 1)) * np.log(m + np.sqrt(m**2 - 1)) - 1)
    nyy = nzz = 0.5 * (1 - nxx)

    return nxx, nyy, nzz


def _depol_oblate_int(a, b, c):
    """
    Calcualte internal depolarisation factors for oblate case.

    parameters
    ----------
    a, b, c : floats
        Semiaxis lengths of the oblate ellipsoid (a < b = c).

    returns
    -------
    nxx, nyy, nzz : floats
        individual diagonal components of the x, y, z matrix.

    """

    m = a / b

    nxx = 1 / (1 - m**2) * (1 - (m / np.sqrt(1 - m**2)) * np.arccos(m))
    nyy = nzz = 0.5 * (1 - nxx)

    return nxx, nyy, nzz


def _construct_N_matrix_internal(a, b, c):
    """
    Construct the N matrix for the internal field using the above functions.

    parameters
    ----------
    a, b, c : floats
        Semiaxis lengths of the given ellipsoid.

    returns
    -------
    N : matrix
        depolarisation matrix (diagonal-only values) for the given ellipsoid.

    """

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
    """
    Get the h values for the N matrix. Each point has its own h value and hence
    external N matrix.

    parameters
    ----------

    a, b, c : floats
        Semiaxis lengths of the given ellipsoid.

    lmbda : float
        the given lmbda value for the point we are considering with this matrix.

    returns
    -------

    h : float
        the h value for the given point.

    """

    axes = np.array([a, b, c])
    h = np.zeros(3)
    R = np.sqrt(np.prod(axes**2 + lmbda))
    return -1 / ((axes**2 + lmbda) * R)


def _spatial_deriv_lambda(x, y, z, a, b, c, lmbda):
    """
    Get the spatial derivative of lambda with respect to the x,y,z.

    parameters
    ----------
    x, y, z : floats
        A singular observation point in the local coordinate system.

    a, b, c : floats
        Semiaxis lengths of the given ellipsoid.

    lmbda : float
        the given lmbda value for the point we are considering with this matrix.


    returns
    -------

    vals : array of x, y, z components
        The spatial derivative for the given point.


    """

    # Numerators (shape: same as x, y, z)
    num_x = 2 * x / (a**2 + lmbda)
    num_y = 2 * y / (b**2 + lmbda)
    num_z = 2 * z / (c**2 + lmbda)

    # Denominator (broadcasts naturally)
    denom = (
        (x / (a**2 + lmbda)) ** 2
        + (y / (b**2 + lmbda)) ** 2
        + (z / (c**2 + lmbda)) ** 2
    )

    # Avoid divide-by-zero
    denom = np.where(denom == 0, 1e-12, denom)

    vals = np.stack([num_x / denom, num_y / denom, num_z / denom], axis=-1)

    return vals


def _get_g_values_magnetics(a, b, c, lmbda):
    """
    Get the gravity values for the three ellipsoid types.

    parameters
    ----------

    a, b, c : floats
        Semiaxis lengths of the given ellipsoid.

    lmbda : float
        the given lmbda value for the point we are considering.

    returns
    -------

    gvals (x, y, z) : floats
        the g values for the given ellipsoid type, and given observation point.


    """

    if a > b > c:
        func = _get_ABC(a, b, c, lmbda)
        gvals_x, gvals_y, gvals_z = func[0], func[1], func[2]

    if a > b and b == c:
        g1 = (
            2
            / ((a**2 - b**2) ** 3 / 2)
            * (
                np.log(
                    ((a**2 - b**2) ** 0.5 + (a**2 + lmbda) ** 0.5)
                    / (b**2 + lmbda) ** 0.5
                )
                - ((a**2 - b**2) / (a**2 + lmbda)) ** 0.5
            )
        )
        g2 = 1 / ((a**2 - b**2) ** 3 / 2) * (
            ((a**2 - b**2) * (a**2 + lmbda) ** 0.5) / (b**2 + lmbda)
        ) - (
            np.log(
                ((a**2 - b**2) ** 0.5 + (a**2 + lmbda) ** 0.5) / (b**2 + lmbda) ** 0.5
            )
        )
        gvals_x, gvals_y, gvals_z = g1, g2, g2

    if a < b and b == c:
        g1 = (
            2
            / ((b**2 - a**2) ** 3 / 2)
            * (
                (((b**2 - a**2) / (a**2 + lmbda)) ** 0.5)
                - np.arctan(((b**2 - a**2) / (a**2 + lmbda)) ** 0.5)
            )
        )
        g2 = (
            1
            / ((b**2 - a**2) ** 3 / 2)
            * (
                np.arctan(((b**2 - a**2) / (a**2 + lmbda)) ** 0.5)
                - (((b**2 - a**2) * (a**2 + lmbda)) ** 0.5) / (b**2 + lmbda)
            )
        )

        gvals_x, gvals_y, gvals_z = g1, g2, g2

    return gvals_x, gvals_y, gvals_z


def _construct_N_matrix_external(x, y, z, a, b, c, lmbda):
    """
    Construct the N matrix for the external field.

    parameters
    ----------
    x, y, z : floats
        A singular observation point in the local coordinate system.

    a, b, c : floats
        Semiaxis lengths of the given ellipsoid.

    lmbda : float
        the given lmbda value for the point we are considering with this matrix.


    returns
    -------

    N : matrix
        External points' depolarisation matrix for the given point.

    """

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
                N[i][j] = (-a * b * c / 2) * (
                    derivs_lmbda[i] * h_vals[i] * r[i] + gvals[i]
                )
            else:
                N[i][j] = (-a * b * c / 2) * (derivs_lmbda[i] * h_vals[j] * r[j])

    return N


# construct total magnetic components
# just for 1 obs point rn
