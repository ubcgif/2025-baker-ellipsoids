# definitions of ellipsoid classes to pass into the functions
import numpy as np

class TriaxialEllipsoid:
    """
    Class creates case of a trixial ellipsoid, storing geometric properties.
    Trixial ellipsoids defined as a > b > c, for semiaxes lengths.

    Parameters
    ----------
    a, b, c : float
        Semiaxis lengths of the ellipsoid. Must satisfy the condition a > b > c
        for the triaxial ellipsoid case.

    yaw : float
        Rotation about the vertical (z) axis, in degrees.
    pitch : float
        Rotation about the northing (y) axis, in degrees.
    roll : float
        Rotation about the easting (x) axis, in degrees.

    origin : array
        (ox, oy, oz) - origin position as an offset from some global
        coordinate system.

    Properties
    ----------
    None.

    """

    def __init__(self, a, b, c, yaw, pitch, roll, centre):

        if not (a > b > c):
            raise ValueError(
                "Invalid ellipsoid axis lengths for triaxial ellipsoid:"
                f"expected a > b > c but got a = {a}, b = {b}, c = {c}"
            )

        # semiaxes
        self.a = a  # major_axis
        self.b = b  # intermediate_axis
        self.c = c  # minor_axis

        # euler angles
        self.yaw = yaw
        self.pitch = pitch
        self.roll = roll

        # centre of ellipsoid
        self.centre = centre


class ProlateEllipsoid:
    """
    Class creates case of a prolate ellipsoid, storing geometric properties.
    Prolate ellipsoids defined as a > b = c, for semiaxes lengths. Hence, values
    'c' and 'roll' are not required as input as, by definition, c = b, and roll
    has no effect due to symmetry, and this is set equal to zero.

    Parameters
    ----------
    a, b: floats
        Semiaxis lengths of the ellipsoid. Must satisfy the condition a > b = c
        for the prolate ellipsoid case.

    yaw : float
        Rotation about the vertical (z) axis, in degrees.
    pitch : float
        Rotation about the northing (y) axis, in degrees.

    origin : array
        (ox, oy, oz) - origin position as an offset from some global
        coordinate system.

    Properties
    ----------

    c : set equal to b
        Due to the nature of prolate ellipsoids.
    roll : set equal to 0
        Due to the nature of prolate ellipsoids.
    """

    def __init__(self, a, b, yaw, pitch, centre):

        if not (a > b):
            raise ValueError(
                "Invalid ellipsoid axis lengths for prolate ellipsoid:"
                f"expected a > b (= c ) but got a = {a}, b = {b}"
            )

        # semiaxes
        self.a = a  # major_axis
        self.b = b  # minor axis

        # euler angles
        self.yaw = yaw
        self.pitch = pitch

        # centre of ellipsoid
        self.centre = centre

    @property
    def c(self):
        return self.b

    @property
    def roll(self):
        return 0.0


class OblateEllipsoid:
    """
    Class creates case of a oblate ellipsoid, storing geometric properties.
    Oblate ellipsoids defined as a < b = c, for semiaxes lengths. Hence, values
    'c' and 'roll' are not required as input as, by definition, c = b, and roll
    has no effect due to symmetry, and this is set equal to zero.

    Parameters
    ----------
    a, b: floats
        Semiaxis lengths of the ellipsoid. Must satisfy the condition a > b > c
        for the triaxial ellipsoid case.

    yaw : float
        Rotation about the vertical (z) axis, in degrees.
    pitch : float
        Rotation about the northing (y) axis, in degrees.

    origin : array
        (ox, oy, oz) - origin position as an offset from some global
        coordinate system.

    Properties
    ----------

    c : set equal to b
        Due to the nature of oblate ellipsoids.
    roll : set equal to 0
        Due to the nature of oblate ellipsoids.
    """

    def __init__(self, a, b, yaw, pitch, centre):

        if not (a < b):
            raise ValueError(
                "Invalid ellipsoid axis lengths for oblate ellipsoid:"
                f"expected a < b (= c ) but got a = {a}, b = {b}"
            )

        # semiaxes
        self.a = a  # minor ais
        self.b = b  # major axis

        # euler angles
        self.yaw = yaw
        self.pitch = pitch

        # centre of ellipsoid
        self.centre = centre

    @property
    def c(self):
        return self.b

    @property
    def roll(self):
        return 0.0
    
    
def _generate_basic_ellipsoid(a, b, c):
    """
    Generate the surface of an ellipsoid using spherical angles for 3D plotting.
    This function is seperate from gravity calculations and is purely for
    visualisation of 3D ellipsoids.
    
    Parameters
    ----------
    a, b, c : float
        Semiaxis lengths of the ellipsoid along the x, y, and z axes, respectively.
    
    Returns
    -------
    x1, y1, z1 : ndarray
        Arrays representing the ellipsoid surface coordinates in 3D space, computed
        from spherical angles. T
    
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
