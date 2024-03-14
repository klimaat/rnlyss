# -*- coding: utf-8 -*-
#
# Copyright 2019 Klimaat

import numpy as np

_EARTH_RADIUS = 6370000.0


def center(lon, rad=False):
    """
    Ensure longitude is within -180 to +180
    """
    if rad:
        return ((lon + np.pi) % (2 * np.pi)) - np.pi
    else:
        return ((lon + 180.0) % 360) - 180.0


class Grid(object):
    def __init__(
        self,
        shape=(181, 360),
        origin=(-90, -180),
        delta=(1, 1),
        periodic=True,
        pole_to_pole=True,
        r=_EARTH_RADIUS,
        **kwargs
    ):

        self.shape = shape
        self.origin = origin
        self.delta = delta
        self.periodic = periodic
        self.pole_to_pole = pole_to_pole

        # Default coordinates in map'd space
        self.y0, self.x0 = origin
        self.dy, self.dx = delta

        # Radius of Earth (m)
        self.r = r

    def __getitem__(self, args):
        """
        Given (i, j), what is corresponding (lat, lon)?
        """

        # Convert to x, y
        x, y = self.ij2xy(*map(np.asarray, args))

        # Convert to lat, lon
        lat, lon = self.xy2ll(x, y)

        return (
            np.asscalar(lat) if np.isscalar(args[0]) else lat,
            np.asscalar(lon) if np.isscalar(args[1]) else lon,
        )

    def __call__(self, lat, lon, snap=False, limit=False):
        """
        Return location in grid given latitude, longitude.
        """

        # Convert to x, y
        x, y = self.ll2xy(np.asarray(lat), np.asarray(lon))

        # Find i, j
        i, j = self.xy2ij(x, y)

        if limit:
            eps = np.finfo(i.dtype).eps
            i = np.clip(i, -0.5 + eps, self.shape[0] - 0.5 - eps)
            j = np.clip(j, -0.5 + eps, self.shape[1] - 0.5 - eps)

        if snap:
            i, j = np.rint(i).astype(int), np.rint(j).astype(int)

        if self.periodic:
            j %= self.shape[1]

        return (
            np.asscalar(i) if np.isscalar(lat) else i,
            np.asscalar(j) if np.isscalar(lon) else j,
        )

    def lats(self):
        # These should be consistent with Gaussian... i.e. method vs. value
        return self[np.arange(self.shape[0]), 0][0]

    def lons(self):
        return self[0, np.arange(self.shape[1])][1]

    def indices(self):
        """
        Return a full set of i, j indices
        """
        return np.meshgrid(
            np.arange(self.shape[0]), np.arange(self.shape[1]), indexing="ij"
        )

    def bbox(self, lat, lon):
        """
        Return indices and weights of (i, j) vertices of grid given lat, lon.

        Handles a few tricky edge cases:
        - extrapolation if outside data (e.g. Gaussian latitudes)
        - interpolation across origin if periodic; extrapolation otherwise
        """
        i, j = self(lat, lon)

        # Allow extrapolation in i
        i1 = int(np.floor(i))
        i1 = np.where(i1 < 0, 0, i1)
        i1 = np.where(i1 >= self.shape[0] - 1, self.shape[0] - 2, i1)
        i2 = i1 + 1

        j1 = int(np.floor(j))
        if not self.periodic:
            # Allow Extrapolation in j
            j1 = np.where(j1 < 0, 0, j1)
            j1 = np.where(j1 >= self.shape[1] - 1, self.shape[1] - 2, j1)
        j2 = j1 + 1

        # Weights
        fi = (i2 - i) / (i2 - i1)
        fj = (j2 - j) / (j2 - j1)

        if self.periodic:
            # Correct j
            j2 %= self.shape[1]

        # Return indices & weights
        return (
            ((i1, j1), (i1, j2), (i2, j1), (i2, j2)),
            np.array([fi * fj, fi * (1 - fj), (1 - fi) * fj, (1 - fi) * (1 - fj)]),
        )

    def ij2xy(self, i, j):
        """
        Convert grid indices to (x,y)-coordinates
        """
        return self.x0 + self.dx * j, self.y0 + self.dy * i

    def xy2ij(self, x, y):
        """
        Convert (x,y)-coordinates to grid indices.
        """
        i, j = (y - self.y0) / self.dy, (x - self.x0) / self.dx

        if self.periodic:
            j %= self.shape[1]

        return i, j

    def ll2xy(self, lat, lon):
        """
        Default projection x=lon, y=lat.
        """
        return center(lon), lat

    def xy2ll(self, x, y):
        """
        Default projection lat=y, lon=x
        """
        return np.clip(y, -90, 90), center(x)

    def cell_corners(self, lat, lon):
        """
        Return latitudes of cell centered at (lat, lon)

        Nominally this is (lat ± dlat/2, lon ± dlon/2) but have to check
        limits

        Don't apply periodic boundary modulus but *do* clip latitudes
        """
        return [
            (np.clip(lat + dlat, -90, 90), lon + dlon)
            for dlat in [-self.dy / 2, self.dy / 2]
            for dlon in [-self.dx / 2, self.dx / 2]
        ]

    def areas(self, r=None):
        """
        Return *all* areas in matrix ni x nj matrix
        """

        if r is None:
            r = self.r

        # Longitude constant
        dlon = np.abs(self.dx)

        # Latitudes could change based on limits or Gaussian
        lats = self[np.arange(self.shape[0] + 1) - 0.5, 0][0]

        # If grid spans pole-to-pole, force latitude limits
        if self.pole_to_pole:
            lats[lats.argmax()] = 90.0
            lats[lats.argmin()] = -90

        # Center of cell (not necessarily where cell stored)
        latc = (lats[:-1] + lats[1:]) / 2
        dlat = np.abs(np.diff(lats))

        # Calculate areas
        areas = (
            np.radians(dlon)
            * 2
            * np.cos(np.radians(latc))
            * np.sin(np.radians(dlat) / 2)
            * r ** 2
        )

        # Copy in longitude dir'n
        return np.repeat(areas[:, None], self.shape[1], 1)

    def extents(self):
        """
        Return latitude and longitudes of corner cells.
        """
        i = [0, 0, self.shape[0] - 1, self.shape[0] - 1]
        j = [0, self.shape[1] - 1, self.shape[1] - 1, 0]
        return self[i, j]


def gaussian_latitudes(n):
    """
    Calculate latitudes of Gaussian grid of n values.
    """
    from scipy.special.orthogonal import p_roots

    return np.degrees(np.arcsin(np.real(p_roots(n)[0])))


class GaussianGrid(Grid):
    def __init__(self, shape, origin, delta, **kwargs):

        super(GaussianGrid, self).__init__(
            shape, origin, delta, periodic=True, **kwargs
        )

        # Gaussian latitudes, ordered
        self.lats = gaussian_latitudes(self.shape[0])

        # Overwrite y0 and dy
        if self.origin[0] > 0:
            self.y0 = self.shape[0] - 1
            self.dy = -1
        else:
            self.y0 = 0
            self.dy = 1

        # Set up re-usable piece-wise interpolators
        from scipy.interpolate import InterpolatedUnivariateSpline

        self.lat2y = InterpolatedUnivariateSpline(
            self.lats, np.arange(len(self.lats)), k=1, ext="extrapolate", bbox=[-90, 90]
        )
        self.y2lat = InterpolatedUnivariateSpline(
            np.arange(len(self.lats)),
            self.lats,
            k=1,
            ext="extrapolate",
            bbox=self.lat2y((-90, 90)),
        )

    def ll2xy(self, lat, lon):
        """
        Convert (lat, lon) to (x, y).
        """

        # Piece-wise interpolate lat into lats returning y
        y = self.lat2y(lat)

        return center(lon), y

    def xy2ll(self, x, y):
        """
        Convert (x, y) to (lat, lon)
        """

        # Piece-wise interpolate y into yi returning y
        lat = self.y2lat(y)

        return lat, center(x)


def test():

    # Regular boring grid
    grid = Grid(shape=(181, 360), origin=(-90, -180), delta=(1, 1))
    A = grid.areas(r=1.0)
    print(A.shape, np.sum(A) / (4 * np.pi))

    # MERRA-2
    grid = Grid(shape=(361, 576), origin=(-90, -180), delta=(1 / 2, 5 / 8))
    A = grid.areas(r=1.0)
    print(A.shape, np.sum(A) / (4 * np.pi))

    # ERA5
    # NB. Data downloaded from Copernicus in netCDF4 format is 0.25°×0.25°
    grid = Grid(shape=(721, 1440), origin=(90, 0), delta=(-1 / 4, 1 / 4))
    A = grid.areas(r=1.0)
    print(A.shape, np.sum(A) / (4 * np.pi))

    # CFSv2
    grid = GaussianGrid(shape=(880, 1760), origin=(90, 0), delta=(-1, 360 / 1760))
    A = grid.areas(r=1.0)
    print(A.shape, np.sum(A) / (4 * np.pi))


if __name__ == "__main__":
    test()
