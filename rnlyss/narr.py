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
from rnlyss.grid import LambertGrid


class NARR(Dataset):
    # NARR dataset variables;
    # NB. "label" refers to the name of variable stored within the netcdf
    #     files. e.g. files named "air.2m." have "air" as the stored variable
    dvars = {
        # Dry bulb temperature @ 2m (K)
        "air.2m": {
            "scale": 1e-2,
            "offset": 330,
            "units": "K",
            "role": "tas",
            "label": "air",
        },
        # Dew point temperature @ 2m (K)
        "dpt.2m": {
            "scale": 1e-2,
            "offset": 330,
            "units": "K",
            "role": "tdps",
            "label": "dpt",
        },
        # Zonal wind (east-west) @ 10m (m/s)
        "uwnd.10m": {"scale": 1e-2, "units": "m/s", "role": "uas", "label": "uwnd"},
        # Meridional wind (north-south) @ 2m (m/s)
        "vwnd.10m": {"scale": 1e-2, "units": "m/s", "role": "vas", "label": "vwnd"},
        # Surface pressure (Pa)
        "pres.sfc": {
            "scale": 1,
            "offset": 75000,
            "units": "Pa",
            "role": "ps",
            "label": "pres",
        },
        # Surface elevation/height (m)
        "hgt.sfc": {"role": "hgt", "constant": True, "label": "hgt"},
        "land": {"role": "land", "constant": True, "label": "land"},
        # Cloud fraction (0 to 1); convert from %
        "tcdc": {
            "scale": 1e-4,
            "units": "1",
            "role": "clt",
            "label": "tcdc",
            "converter": lambda x: x / 100.0,
        },
        # Shortwave downwelling @ surface (W/m²)
        "dswrf": {"scale": 0.1, "units": "W/m2", "role": "rsds", "label": "dswrf"},
        # Longwave downwelling @ surface (W/m²)
        "dlwrf": {"scale": 0.1, "units": "W/m2", "role": "rlds", "label": "dlwrf"},
    }

    # Time starts 1979; 3-hourly; 00:00
    years = [1979, None]
    freq = 3
    h0 = 0

    grid = LambertGrid(
        shape=(277, 349),
        origin=(0, 0),
        delta=(32463, 32463),
        lon0=-107.0,
        lat0=50.0,
        lat1=50.0,
        lat2=50,
        false_easting=5632642.225474948,
        false_northing=4612545.651374279,
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
        Fill element HDF with available NARR data
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

                    # Number of hours in year
                    nh = len(slab)

                    # Check fill status
                    if slab.isfull() and not force:
                        print("already stacked... skipping")
                        continue

                    # Check that data exists
                    path = self.get_data_path("nc", str(year), f"{dvar}.{year}.nc")
                    if not os.path.isfile(path):
                        print("missing... skipping")
                        continue

                    # Shape of this year; i.e. nlat x nlon x nh values
                    shape = self.grid.shape + (nh,)

                    with netCDF4.Dataset(path) as nc:
                        # Check that year isn't truncated
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
    from rnlyss.grid import plot_extents
    import matplotlib.pyplot as plt

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

    plot_extents(N.grid)
    plt.show()


if __name__ == "__main__":
    main()
