# -*- coding: utf-8 -*-
#
# Copyright 2017 Klimaat

import os
import urllib.request as request
from urllib.error import URLError
from contextlib import closing
import numpy as np
import netCDF4
from tqdm import tqdm
from rnlyss.dataset import Dataset
from rnlyss.grid import Grid, center


class LambertGrid(Grid):
    def __init__(
        self, shape, origin, delta, lon0, lat0, lat1, lat2, r=6370000.0, **kwargs
    ):
        super(LambertGrid, self).__init__(
            shape=shape,
            origin=origin,
            delta=delta,
            periodic=False,
            pole_to_pole=False,
            **kwargs,
        )

        self.lon0 = lon0
        self.lat0 = np.radians(lat0)
        self.lat1 = np.radians(lat1)
        self.lat2 = np.radians(lat2)
        self.r = r

        if self.lat1 == self.lat2:
            self.n = np.sin(np.abs(self.lat1))
        else:
            self.n = np.log(np.cos(self.lat1) / np.cos(self.lat2)) / np.log(
                np.tan(np.pi / 4 + self.lat2 / 2) / np.tan(np.pi / 4 + self.lat1 / 2)
            )

        self.F = (
            np.cos(self.lat1) / self.n * np.tan(np.pi / 4 + self.lat1 / 2) ** self.n
        )

        self.rho0 = self.r * self.F / np.tan(np.pi / 4 + self.lat0 / 2) ** self.n

        # Easting & northing of origin of grid
        self.x0, self.y0 = self.ll2xy(*self.origin)

    def ll2xy(self, lat, lon):
        """
        Convert (lat, lon) to (x, y)
        """
        rho = self.r * self.F / np.tan(np.pi / 4 + np.radians(lat) / 2) ** self.n
        theta = np.radians(self.n * center(lon - self.lon0))
        return rho * np.sin(theta), self.rho0 - rho * np.cos(theta)

    def xy2ll(self, x, y):
        """
        Convert (x, y) to (lat, lon)
        """
        rho = np.sign(self.n) * np.sqrt(x**2 + (self.rho0 - y) ** 2)
        theta = np.degrees(np.arctan(x / (self.rho0 - y)))
        lat = np.degrees(
            2 * np.arctan((self.r * self.F / rho) ** (1.0 / self.n)) - np.pi / 2
        )
        lon = center(theta / self.n + self.lon0)
        return lat, lon

    def crs(self):
        """
        Equivalent Cartopy projection
        """
        try:
            import cartopy.crs as ccrs
        except ImportError:
            return None

        return ccrs.LambertConformal(
            central_longitude=self.lon0,
            central_latitude=np.degrees(self.lat0),
            false_easting=0,
            false_northing=0,
            standard_parallels=(np.degrees(self.lat1), np.degrees(self.lat2)),
        )

    def alpha(self, lat, lon):
        """
        Angle that positive geographical (eastward) x-axis is away from
        positive Lambert x-axis.
        """
        return np.sign(lat) * center(lon - self.lon0) * self.n

    def rotate(self, u, v, lat, lon):
        """
        Rotate Lambert vector onto geographic coordinates
        (u=east/west, v=north/south).
        """
        a = np.radians(self.alpha(lat, lon))
        ca, sa = np.cos(a), np.sin(a)
        return v * sa + u * ca, v * ca - u * sa

    def map_factor(self, lat, lon):
        """
        Calculate the map_factor
        c.f. Snyder, "Map Projections" eqn (15-4)
        """
        return (
            np.cos(self.lat1)
            / np.cos(np.radians(lat))
            * (
                np.tan(np.pi / 4 + self.lat1 / 2)
                / np.tan(np.pi / 4 + np.radians(lat) / 2)
            )
            ** self.n
        )

    def areas(self, r=None):
        """
        Return area

        area = dx * dy/ h**2 -> m²

        """
        if r is None:
            # dx & dy already in m
            f = 1.0
        else:
            # scaling factor
            f = (r / self.r) ** 2

        i, j = self.indices()
        lat, lon = self[i, j]
        h = self.map_factor(lat, lon)
        return f * self.dx * self.dy / h

    def plot_extents(self):
        """
        Quick plot of extents.
        """
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs

        crs = self.crs()
        lat, lon = self.extents()
        x, y = self.ll2xy(lat, lon)
        ax = plt.subplot(111, projection=ccrs.PlateCarree(central_longitude=self.lon0))
        ax.coastlines(color="grey")
        ax.set_global()
        ax.fill(x, y, color="orange", transform=crs, alpha=0.4)
        ax.plot(0, 0, "x", color="black", transform=crs)
        text_props = {
            "bbox": {"fc": "lightgrey", "alpha": 0.7, "boxstyle": "round"},
            "color": "black",
            "transform": crs,
            "multialignment": "right",
        }
        for c in range(4):
            txt = f"{lat[c]:.3f}°\n{lon[c]:.3f}°"
            ha = "right" if c in [0, 3] else "left"
            va = "top" if c in [0, 1] else "bottom"
            ax.text(x[c], y[c], txt, ha=ha, va=va, **text_props)
        ax.gridlines()
        plt.tight_layout()
        plt.show()


class NARR(Dataset):
    dvars = {
        "air.2m": {
            "scale": 1e-2,
            "offset": 330,
            "units": "K",
            "role": "tas",
            "label": "air",
        },
        "dpt.2m": {
            "scale": 1e-2,
            "offset": 330,
            "units": "K",
            "role": "tdps",
            "label": "dpt",
        },
        # "shum.2m": {
        #     "scale": 1e-6,
        #     "offset": 0.03,
        #     "unit": "kg/kg",
        #     "role": "huss",
        #     "label": "shum",
        # },
        "uwnd.10m": {"scale": 1e-2, "units": "m/s", "role": "uas", "label": "uwnd"},
        "vwnd.10m": {"scale": 1e-2, "units": "m/s", "role": "vas", "label": "vwnd"},
        "pres.sfc": {
            "scale": 1,
            "offset": 75000,
            "units": "Pa",
            "role": "ps",
            "label": "pres",
        },
        "hgt.sfc": {"role": "hgt", "constant": True, "label": "hgt"},
        "land": {"role": "land", "constant": True, "label": "land"},
        "dswrf": {"scale": 0.1, "units": "W/mn2", "role": "rsds", "label": "dswrf"},
        # "dlwrf": {"scale": 0.1, "units": "W/m2", "role": "rlds", "label": "dlwrf"},
        # "apcp": {
        #     "scale": 1 / 36000,
        #     "units": "kg/m2/s",
        #     "role": "pr",
        #     "collection": "acpcp",
        # },
    }

    # Time starts 1979; 3-hourly; 00:00
    years = [1979, None]
    freq = 3
    h0 = 0

    grid = LambertGrid(
        shape=(277, 349),
        origin=(1, -145.5),
        # delta=(32463, 32463),
        delta=(32463.41, 32463.41),
        lon0=-107.0,
        lat0=50.0,
        lat1=50.0,
        lat2=50,
        r=6371200,
    )

    def download(self, dvars=None, years=None, months=None, ignore=False, **kwargs):
        """
        Download NARR NC files.
        """

        if dvars is None:
            dvars = list(self.dvars.keys())

        if not isinstance(dvars, list):
            dvars = [dvars]

        def get_file(url, dst):
            # Ensure directory exists
            os.makedirs(os.path.dirname(dst), exist_ok=True)

            try:
                with closing(request.urlopen(url)) as req:
                    content_type = req.headers["Content-type"]
                    if content_type != "application/x-netcdf":
                        print(f"{dst} unavailable... skipping")

                    content_length = int(req.headers["Content-length"])
                    if os.path.isfile(dst):
                        file_size = os.stat(dst).st_size
                        if file_size == content_length:
                            print(f"{dst} unchanged... skipping")
                            return

                    # print(f"{dst} available... downloading {content_length} bytes")

                    with open(dst, "wb") as file, tqdm(
                        desc=dst,
                        total=content_length,
                        unit="iB",
                        unit_scale=True,
                        unit_divisor=1024,
                    ) as bar:
                        while True:
                            buffer = req.read(1024)
                            if not buffer:
                                break
                            size = file.write(buffer)
                            bar.update(size)
                    # try:
                    #     shutil.copyfileobj(req, open(dst, "wb"))
                    # except BaseException:
                    #     # Problem; delete file
                    #     if os.path.isfile(dst):
                    #         print(f"{dst} interrupted... deleting")
                    #         os.remove(dst)
                    #     raise
            except URLError:
                print(f"{dst} unavailable ...skipping")

            return True

        # Release the sloths...
        root_url = "ftp://ftp.cdc.noaa.gov/Datasets/NARR"

        for dvar in sorted(dvars):
            if dvar not in self:
                print("%s not in dataset... skipping" % dvar)
                continue

            print(dvar)

            if self.isconstant(dvar):
                url = f"{root_url}/time_invariant/{dvar}.nc"
                dst = self.get_data_path("nc", f"{dvar}.nc")
                get_file(url, dst)

            else:
                # Hourly file
                for year in self.iter_year(years):
                    url = f"{root_url}/monolevel/{dvar}.{year}.nc"
                    dst = self.get_data_path("nc", str(year), f"{dvar}.{year}.nc")
                    get_file(url, dst)

    def stack(self, dvars=None, years=None, months=None, force=False, **kwargs):
        """
        Fill element HDF with available CERES data
        """
        if dvars is None:
            dvars = list(self.dvars.keys())

        for dvar in sorted(dvars):
            # Check dvar
            if dvar not in self:
                print("%s not in dataset... skipping" % dvar)
                print("choices: %r" % list(self.dvars.keys()))
                continue

            # Get full name
            print(dvar)

            # Get converter
            converter = self.dvars[dvar].get("converter", None)
            label = self.dvars[dvar].get("label", None)
            if label is None:
                raise ValueError(f"Specify label for {dvar}")

            # Special case: constant
            if self.isconstant(dvar):
                with self[dvar] as slab:
                    if not slab:
                        slab.create(
                            shape=self.grid.shape,
                            year=self.years[0],
                            freq=0,
                            **self.dvars[dvar],
                        )

                    if slab.isfull(0) and not force:
                        print("already stacked... skipping")
                        continue

                    path = self.get_data_path("nc", f"{dvar}.nc")
                    if not os.path.isfile(path):
                        print("missing... skipping")
                        continue

                    # Retrieve slice from netcdf4 file
                    print("reading...", end=" ")
                    with netCDF4.Dataset(path) as nc:
                        try:
                            missing_value = nc[label].missing_value
                            nc.set_auto_false(False)
                        except AttributeError:
                            missing_value = None

                        x = np.expand_dims(nc.variables[label][0, :], axis=-1)

                        if missing_value:
                            x[x == missing_value] = np.nan

                    # Insert into HDF
                    print("writing...")
                    slab.fill(0, slab.to_int(x, converter))

                # Done
                continue

            # Loop over request years
            for year in self.iter_year(years):
                # if self.isscalar(dvar):
                shape = self.grid.shape

                with self[dvar, year] as slab:
                    if slab:
                        print(year, "exists... updating...", end=" ", flush=True)
                    else:
                        print(
                            year, "does not exist... creating...", end=" ", flush=True
                        )
                        slab.create(
                            shape=shape, year=year, freq=self.freq, **self.dvars[dvar]
                        )

                    # for month in self.iter_month(year, months):

                    # Insert point
                    # i = slab.date2ind(months=month, hours=hour0)

                    # Number of hours in year
                    nh = len(slab)
                    # nh = slab.month_len(month)

                    # Check fill status
                    if slab.isfull() and not force:
                        print("already stacked... skipping")
                        continue

                    # Check that data exists
                    path = self.get_data_path("nc", str(year), f"{dvar}.{year}.nc")
                    if not os.path.isfile(path):
                        print("missing... skipping")
                        continue

                    # Shape of this month; i.e. nlat x nlon x nh values
                    shape = self.grid.shape + (nh,)

                    with netCDF4.Dataset(path) as nc:
                        # Check that month isn't truncated
                        if nc.variables[label].shape[0] != nh:
                            print(year, "incomplete... skipping", flush=True)
                            continue

                        print("complete... reading", end=" ", flush=True)
                        x = np.transpose(
                            slab.to_int(nc.variables[label][:], converter), (1, 2, 0)
                        )

                        # Write whole year
                        print("... writing", flush=True)
                        slab.fill(0, x)
                        x = None


def main():
    N = NARR()

    # c.f. https://www.nco.ncep.noaa.gov/pmb/docs/on388/tableb.html#GRID221
    # NB. i, j are zero-indexed vs. one-indexed values in table
    print("Extents", N.grid.extents())
    print("Pole point", N.grid(90, -107))
    print("40N,70W", N.grid(40, -107))

    # Map factor along parallel should be unity
    h = N.grid.map_factor(50, -107)
    print("Map factor along 50N", h)

    # Design map factor
    h = N.grid.map_factor(40, -107)
    # This should be ~32km
    print("Distance along 40N", N.grid.dx / h)

    # Test areas
    A = N.grid.areas(r=1)
    print("Area min, mean, max", A.min(), A.mean(), A.max())
    print("Area fraction of Earth", A.sum() / (4 * np.pi))

    N.grid.plot_extents()


if __name__ == "__main__":
    main()
