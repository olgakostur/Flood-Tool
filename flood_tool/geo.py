from numpy import array, asarray, mod, sin, cos, tan, sqrt, arctan2, \
    floor, rad2deg, deg2rad, stack # noqa
from scipy.linalg import inv

__all__ = ['get_easting_northing_from_gps_lat_long',
           'get_gps_lat_long_from_easting_northing']


class Ellipsoid(object):
    """Class to hold Ellipsoid information."""
    def __init__(self, a, b, F_0):
        self.a = a
        self.b = b
        self.n = (a-b)/(a+b)
        self.e2 = (a**2-b**2)/a**2
        self.F_0 = F_0
        self.H = 0


class Datum(Ellipsoid):
    """Class to hold datum information."""

    def __init__(self, a, b, F_0, phi_0, lam_0, E_0, N_0, H):
        super().__init__(a, b, F_0)
        self.phi_0 = phi_0
        self.lam_0 = lam_0
        self.E_0 = E_0
        self.N_0 = N_0
        self.H = H


def rad(deg, min=0, sec=0):
    """Convert degrees/minutes/seconds into radians.

    Parameters
    ----------

    deg: float/arraylike
       Value(s) in degrees
    min: float/arraylike
       Value(s) in minutes
    sec: float/arraylike
       Value(s) in (angular) seconds

    Returns
    -------
    numpy.ndarray
         Equivalent values in radians
    """
    deg = asarray(deg)
    min = asarray(min)
    sec = asarray(sec)
    return deg2rad(deg+min/60.+sec/3600.)


def deg(rad, dms=False):
    """Convert degrees into radians.

    Parameters
    ----------

    deg: float/arraylike
        Value(s) in degrees

    Returns
    -------
    np.ndarray
        Equivalent values in radians.
    """
    rad = asarray(rad)
    deg = rad2deg(rad)
    if dms:
        min = 60.0*mod(deg, 1.0)
        sec = 60.0*mod(min, 1.0)
        return stack((deg.round(4),  min.round(4), sec.round(4)))
    else:
        return deg


# data for OSGB36 lat/long datum.
osgb36 = Datum(a=6377563.396,
               b=6356256.910,
               F_0=0.9996012717,
               phi_0=rad(49.0),
               lam_0=rad(-2.),
               E_0=400000,
               N_0=-100000,
               H=24.7)

# data for WGS84 GPS datum.
wgs84 = Ellipsoid(a=6378137,
                  b=6356752.3142,
                  F_0=0.9996)


def lat_long_to_xyz(phi, lam, rads=False, datum=osgb36):
    """Convert latitude/longitude in a given datum into
    Cartesian (x, y, z) coordinates.
    """
    if not rads:
        phi = rad(phi)
        lam = rad(lam)

    nu = datum.a*datum.F_0/sqrt(1-datum.e2*sin(phi)**2)

    return array(((nu+datum.H)*cos(phi)*cos(lam),
                  (nu+datum.H)*cos(phi)*sin(lam),
                  ((1-datum.e2)*nu+datum.H)*sin(phi)))


def xyz_to_lat_long(x, y, z, rads=False, datum=osgb36):
    """Convert Cartesian (x,y,z) coordinates into
    latitude and longitude in a given datum.
    """

    p = sqrt(x**2+y**2)

    lam = arctan2(y, x)
    phi = arctan2(z, p*(1-datum.e2))

    for _ in range(10):

        nu = datum.a*datum.F_0/sqrt(1-datum.e2*sin(phi)**2)
        dnu = (-datum.a*datum.F_0*cos(phi)*sin(phi)
               / (1-datum.e2*sin(phi)**2)**1.5)

        f0 = (z + datum.e2*nu*sin(phi))/p - tan(phi)
        f1 = datum.e2*(nu**cos(phi)+dnu*sin(phi))/p - 1.0/cos(phi)**2
        phi -= f0/f1

    if not rads:
        phi = deg(phi)
        lam = deg(lam)

    return phi, lam


def get_easting_northing_from_gps_lat_long(phi, lam, rads=False):
    """ Get OSGB36 easting/northing from GPS latitude and
    longitude pairs.

    Parameters
    ----------

    phi: float/arraylike
        GPS (i.e. WGS84 datum) latitude value(s)
    lam: float/arraylike
        GPS (i.e. WGS84 datum) longitude value(s).
    rads: bool (optional)
        If true, specifies input is is radians.

    Returns
    -------
    numpy.ndarray
        Easting values (in m)
    numpy.ndarray
        Northing values (in m)

    Examples
    --------

    >>> get_easting_northing_from_gps_lat_long([55.5], [-1.54])
    (array([429157.0]), array([623009]))

    References
    ----------

    Based on the formulas in "A guide to coordinate systems in Great Britain".

    See also https://webapps.bgs.ac.uk/data/webservices/convertForm.cfm
    """

    if not rads:
        phi = rad(array(phi))
        lam = rad(array(lam))

    # convert WGS lat long to OS lat long
    phi, lam = WGS84toOSGB36(phi, lam, rads=True)
    phi = array(phi)
    lam = array(lam)

    # compute the coefficient rho, nu and eta
    rho = osgb36.a * osgb36.F_0 * (1 - osgb36.e2) / (
         (1 - osgb36.e2 * (sin(phi)) ** 2) ** 1.5)
    nu = osgb36.a * osgb36.F_0 / sqrt(1 - osgb36.e2 * (sin(phi)) ** 2)
    eta = sqrt(nu / rho - 1)

    # set the variable names for conveniently refer
    delta_latitude = phi - osgb36.phi_0
    plus_latitude = phi + osgb36.phi_0
    M1 = 1 + osgb36.n + 5 / 4 * osgb36.n ** 2 + 5 / 4 * osgb36.n ** 3
    M2 = 3 * osgb36.n + 3 * osgb36.n ** 2 + 21 / 8 * osgb36.n ** 3
    M3 = 15 / 8 * osgb36.n ** 2 + 15 / 8 * osgb36.n ** 3
    M4 = 35 / 24 * osgb36.n ** 3
    M = osgb36.b * osgb36.F_0 * (
                        M1 * delta_latitude
                        - M2 * sin(delta_latitude) * cos(plus_latitude)
                        + M3 * sin(2 * delta_latitude) * cos(2 * plus_latitude)
                        - M4 * sin(3 * delta_latitude) * cos(3 * plus_latitude)
                        )

    delta_longitude = (lam - osgb36.lam_0)
    eta2 = eta ** 2
    sin_lat = sin(phi)
    cos_lat = cos(phi)
    tan_lat = sin_lat/cos_lat
    tan_lat2 = tan_lat ** 2
    tan_lat4 = tan_lat2 ** 2
    cos_lat3 = cos_lat ** 3
    cos_lat5 = cos_lat ** 5

    Initial = M + osgb36.N_0
    II = nu / 2 * sin_lat * cos_lat
    III = nu / 24 * sin_lat * (cos_lat3 * ((5 - tan_lat2) + 9 * eta2))
    IIIA = nu / 720 * sin_lat * cos_lat5 * (61 - 58 * tan_lat2) + tan_lat4
    IV = nu * cos_lat
    V = nu / 6 * cos_lat3 * (nu / rho - tan_lat2)
    VI = nu / 120 * cos_lat5 * ((5 - 18 * tan_lat2)
                                + tan_lat4 + 14 * eta2 - 58 * tan_lat2 * eta2)
    e_os = osgb36.E_0 + (IV * delta_longitude + V
                         * delta_longitude ** 3 + VI * delta_longitude ** 5)
    n_os = Initial + (II * delta_longitude ** 2 +
                      III * delta_longitude ** 4 + IIIA * delta_longitude ** 6)
    e_os = array(e_os)
    n_os = array(n_os)
    return e_os, n_os


def get_gps_lat_long_from_easting_northing(east, north,
                                           rads=False, dms=False):
    """ Get GPS latitude and longitude pairs from
    OSGB36 easting/northing.

    Parameters
    ----------

    east: float/arraylike
        OSGB36 easting value(s) (in m).
    north: float/arraylike
        OSGB36 easting value(s) (in m).
    rads: bool (optional)
        If true, specifies ouput is is radians.
    dms: bool (optional)
        If true, output is in degrees/minutes/seconds. Incompatible
        with rads option.

    Returns
    -------
    numpy.ndarray
        GPS (i.e. WGS84 datum) latitude value(s).
    numpy.ndarray
        GPS (i.e. WGS84 datum) longitude value(s).

    Examples
    --------

    >>> get_gps_lat_long_from_easting_northing([429157], [623009])
    (array([55.5]), array([-1.540008]))
    >>> get_gps_lat_long_from_easting_northing([429157], [623009],
                                                rads=False, dms=True)
    (array([55.50001947, 30.00208694, 59.92798956]),
            array([-1.54001445, 27.69329395, 41.77411121]))
    References
    ----------

    Based on the formulas in "A guide to coordinate systems in Great Britain".

    See also https://webapps.bgs.ac.uk/data/webservices/convertForm.cfm
    """

    east = array(east)
    north = array(north)

    M, phip = 0, osgb36.phi_0
    # If abs(𝑁−𝑁0−𝑀)>0.01𝑚𝑚 go to 2, otherwise halt
    while abs(north - osgb36.N_0 - M) >= 1.e-5:
        phip = (north - osgb36.N_0 - M)/(osgb36.a * osgb36.F_0) + phip
        # set the variable name for conveniently refer
        delta_latitude = phip - osgb36.phi_0
        plus_latitude = phip + osgb36.phi_0
        M1 = 1 + osgb36.n + 5 / 4 * osgb36.n ** 2 + 5 / 4 * osgb36.n ** 3
        M2 = 3 * osgb36.n + 3 * osgb36.n ** 2 + 21 / 8 * osgb36.n ** 3
        M3 = 15 / 8 * osgb36.n ** 2 + 15 / 8 * osgb36.n ** 3
        M4 = 35 / 24 * osgb36.n ** 3
        M = osgb36.b * osgb36.F_0 * (M1 * delta_latitude
                                     - M2 * sin(delta_latitude)
                                     * cos(plus_latitude)
                                     + M3 * sin(2 * delta_latitude)
                                     * cos(2 * plus_latitude)
                                     - M4 * sin(3 * delta_latitude)
                                     * cos(3 * plus_latitude))

    rho = osgb36.a * osgb36.F_0 * (1 - osgb36.e2) / (
        (1 - osgb36.e2 * (sin(phip)) ** 2) ** 1.5)
    nu = osgb36.a * osgb36.F_0 / sqrt(1 - osgb36.e2 * (sin(phip)) ** 2)
    eta = sqrt(nu / rho - 1)
    eta2 = eta * eta

    tan_phip = tan(phip)
    tan_phip2 = tan_phip**2
    nu3, nu5 = nu**3, nu**5
    sec_phip = 1./cos(phip)

    VII = tan_phip/2/rho/nu
    VIII = tan_phip/24/rho/nu3 * (5 + 3*tan_phip2 + eta2 * (1 - 9*tan_phip2))
    IX = tan_phip / 720/rho/nu5 * (61 + tan_phip2*(90 + 45 * tan_phip2))
    X = sec_phip / nu
    XI = sec_phip / 6 / nu3 * (nu/rho + 2*tan_phip2)
    XII = sec_phip / 120 / nu5 * (5 + tan_phip2*(28 + 24*tan_phip2))
    XIIA = sec_phip / 5040 / nu**7 * (61 + tan_phip2*(662
                                      + tan_phip2 * (1320 + tan_phip2*720)))
    E_E0 = east - osgb36.E_0
    E_E02 = E_E0 ** 2

    phi = phip + E_E0 ** 2 * (-VII + E_E02 * (VIII - IX * E_E02))
    lam = osgb36.lam_0 + E_E0 *\
        (X + E_E02*(-XI + E_E02*(XII - XIIA * E_E02)))
    phi1 = deg(phi)
    lam1 = deg(lam)
    latitude, longtitude = OSGB36toWGS84(phi1, lam1, rads=False)
    # set different conditions for rads and dms
    if rads is True and dms is False:
        phi2 = rad(latitude)
        lam2 = rad(longtitude)
        return phi2, lam2
    if dms is True and rads is False:
        phi3 = deg(phi, dms=True)
        lam3 = deg(lam, dms=True)
        latitude1, longtitude1 = OSGB36toWGS84(phi3, lam3, rads=False)
        return latitude1, longtitude1
    if dms is True and rads is True:
        return print("Invalid input!" + '\n'
                     + "dms is incompatible with rads option")

    return latitude, longtitude


class HelmertTransform(object):
    """Class to perform a Helmert Transform."""
    def __init__(self, s, rx, ry, rz, T):

        self.T = T.reshape((3, 1))

        self.M = array([[1+s, -rz, ry],
                        [rz, 1+s, -rx],
                        [-ry, rx, 1+s]])

    def __call__(self, X):
        X = X.reshape((3, -1))
        return self.T + self.M@X


class HelmertInverseTransform(object):
    """Class to perform the inverse of a Helmert Transform."""
    def __init__(self, s, rx, ry, rz, T):

        self.T = T.reshape((3, 1))

        self.M = inv(array([[1+s, -rz, ry],
                            [rz, 1+s, -rx],
                            [-ry, rx, 1+s]]))

    def __call__(self, X):
        X = X.reshape((3, -1))
        return self.M@(X-self.T)


OSGB36transform = HelmertTransform(20.4894e-6,
                                   -rad(0, 0, 0.1502),
                                   -rad(0, 0, 0.2470),
                                   -rad(0, 0, 0.8421),
                                   array([-446.448, 125.157, -542.060]))

WGS84transform = HelmertInverseTransform(20.4894e-6,
                                         -rad(0, 0, 0.1502),
                                         -rad(0, 0, 0.2470),
                                         -rad(0, 0, 0.8421),
                                         array([-446.448, 125.157, -542.060]))


def WGS84toOSGB36(lat, long, rads=False):
    """Convert WGS84 lat/long to OSGB36 lat/long."""
    X = OSGB36transform(lat_long_to_xyz(asarray(lat), asarray(long),
                                        rads=rads, datum=wgs84))
    return xyz_to_lat_long(*X, rads=rads, datum=osgb36)


def OSGB36toWGS84(lat, long, rads=False):
    """Convert OSGB36 lat/long to WGS84 lat/long."""
    X = WGS84transform(lat_long_to_xyz(asarray(lat), asarray(long),
                                       rads=rads, datum=osgb36))
    return xyz_to_lat_long(*X, rads=rads, datum=wgs84)


# print(get_gps_lat_long_from_easting_northing([429157], [623009]))
# print(get_gps_lat_long_from_easting_northing([429157], [623009],
#                                              rads=False, dms=True))
