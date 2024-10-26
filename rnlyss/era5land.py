#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
from datetime import datetime
import numpy as np
import calendar
import requests
import shutil

from rnlyss.era5 import ERA5
from rnlyss.grid import Grid
from rnlyss.util import syslog_elapsed_time

try:
    import netCDF4
except ImportError:
    raise NotImplementedError("netcdf4 req'd to read some ERA5Land datasets")

try:
    import cdsapi
except ImportError:
    raise NotImplementedError("cdsapi req'd to download ERA5Land datasets")

try:
    import pygrib
except ImportError:
    raise NotImplementedError("pygrib req'd to read hourly ERA5Land datasets")


class ERA5Land(ERA5):

    # fmt: off
    dvars = {
        # Dry bulb temperature @ 2m (K)
        "t2m": {
            "role": "tas",
            "scale": 1e-2,
            "offset": 330,
            "full": "2m_temperature",
            "type": "hourly",
        },
        # Dew point temperature @ 2m (K)
        "d2m": {
            "role": "tdps",
            "scale": 1e-2,
            "offset": 330,
            "full": "2m_dewpoint_temperature",
            "type": "hourly",
        },
        # Zonal wind (east-west) @ 10m (m/s)
        "u10": {
            "role": "uas",
            "scale": 1e-2,
            "offset": 0,
            "full": "10m_u_component_of_wind",
            "type": "hourly",
        },
        # Meridional wind (north-south) @ 10m (m/s)
        "v10": {
            "role": "vas",
            "scale": 1e-2,
            "offset": 0,
            "full": "10m_v_component_of_wind",
            "type": "hourly",
        },
        # Surface pressure (Pa)
        "sp": {
            "role": "ps",
            "scale": 1,
            "offset": 75000,
            "full": "surface_pressure",
            "type": "hourly",
        },
        # Surface geopotential (m); convert from m2/s2 to m
        "z": {
            "role": "hgt",
            "scale": 1,
            "constant": True,
            "converter": lambda x: x / 9.80665,
            "full": "orography",
            "url": "https://confluence.ecmwf.int/download/attachments/140385202/geo_1279l4_0.1x0.1.grib2_v4_unpack.nc?version=1&modificationDate=1591979822003&api=v2"
        },
        # Land area fraction (0 to 1)
        "lsm": {
            "role": "land",
            "scale": 1e-2,
            "constant": True,
            "full": "land_sea_mask",
            "url": "https://confluence.ecmwf.int/download/attachments/140385202/lsm_1279l4_0.1x0.1.grb_v4_unpack.nc?version=1&modificationDate=1591979822208&api=v2"
        },
        # Total precipitation (m); convert to (mm/s)
        "tp": {
            "role": "pr",
            "scale": 1 / 36000,
            "converter": lambda x: x / 3.6,
            "full": "total_precipitation",
            "type": "monthly",
        },
    }
    # fmt: on

    # Time
    years = [1950, None]
    freq = 1

    # Grid
    # NB. Data downloaded from Copernicus in netCDF4 format is 0.1°×0.1°
    grid = Grid(shape=(1801, 3600), origin=(90, 0), delta=(-0.1, 0.1))

    def download(self, dvars=None, years=None, months=None, ignore=False, **kwargs):
        """
        Download ERA5 netcdf4 files.
        """

        if dvars is None:
            dvars = list(self.dvars.keys())

        if not isinstance(dvars, list):
            dvars = [dvars]

        # Establish CDS API client
        CDS = cdsapi.Client()

        def n_months(year, month):
            """
            Months between now and (year, month)
            """
            now = datetime.utcnow()
            then = datetime(year, month, 1)
            return (now.year - then.year) * 12 + (now.month - then.month)

        def get_era5_args(dvar, year, month, monthly=False):
            """
            Return variable name, selection dict, and target path
            """

            # Full name of variable
            full_name = self.dvars[dvar]["full"]

            target_path = self.get_filename(full_name, year, month, monthly=monthly)

            # Check that we're not within 3 months
            if n_months(year, month) <= 3:
                print("%s likely contains ERA5T... skipping" % target_path)
                return None

            os.makedirs(os.path.dirname(target_path), exist_ok=True)

            if os.path.isfile(target_path):
                print("%s exists... skipping" % target_path)
                return None

            if monthly:
                # Request monthly average

                return (
                    "reanalysis-era5-land-monthly-means",
                    {
                        "format": "netcdf",
                        "variable": full_name,
                        "year": "%04d" % year,
                        "month": "%02d" % month,
                        "product_type": "monthly_averaged_reanalysis",
                        "time": "00:00",
                    },
                    target_path,
                )
            else:
                # Request hourly for entire month
                days = [
                    "%02d" % (day + 1,)
                    for day in range(calendar.monthrange(year, month)[1])
                ]

                # Request 24 hours
                hours = ["%02d:00" % h for h in range(24)]

                return (
                    "reanalysis-era5-land",
                    {
                        "format": "grib",
                        "variable": full_name,
                        "year": "%04d" % year,
                        "month": "%02d" % month,
                        "day": days,
                        "time": hours,
                    },
                    target_path,
                )

        def get_era5_file(name, request, target):
            """
            Wrap CDS API download routine forcing delete if ANY problem

            NB. Avoids truncated netcdf3 files, which will still return valid
                numpy arrays and so are hard to identify
            """
            try:
                # Stream to file
                CDS.retrieve(name, request, target)

            except BaseException:
                # Problem; delete file
                if os.path.isfile(target):
                    print("%s interrupted... deleting" % target)
                    os.remove(target)
                raise

        def get_file(url, dst):

            # Ensure directory exists
            os.makedirs(os.path.dirname(dst), exist_ok=True)

            # Start request
            for i in range(3):
                request = requests.get(url, stream=True)
                if request.status_code == 200:
                    break
                print(
                    "%s unreachable (%d)... " % (dst, request.status_code), "retrying"
                )

            else:
                print(
                    "%s unavailable (%d)... " % (dst, request.status_code), "skipping"
                )

                return False

            content_length = int(request.headers["Content-Length"])

            print(
                "%s available... " % dst,
                "downloading %d bytes" % content_length,
                flush=True,
            )

            try:
                # Stream to file
                shutil.copyfileobj(request.raw, open(dst, "wb"))

            except BaseException:
                # Problem; delete file
                if os.path.isfile(dst):
                    print("%s interrupted... deleting" % dst)
                    os.remove(dst)
                raise

            return True

        def get_constant_file(dvar):
            url = self.dvars[dvar]["url"]
            target = self.get_data_path(
                "constant", "nc4", self.dvars[dvar]["full"] + ".nc4"
            )
            if os.path.isfile(target):
                print("%s exists... skipping" % target)
                return None
            return get_file(url, target)

        # Release the sloths...
        for dvar in sorted(dvars):
            if dvar not in self:
                print("%s not in dataset... skipping" % dvar)
                continue

            print(dvar, self.dvars[dvar]["full"])

            if self.isconstant(dvar):
                # Constant
                get_constant_file(dvar)

            else:
                type_ = self.dvars[dvar].get("type", "hourly")
                for year, month in self.iter_year_month(years, months):
                    monthly = True if self.dvars[dvar]["type"] == "monthly" else False
                    args = get_era5_args(dvar, year, month, monthly=monthly)
                    if args is not None:
                        get_era5_file(*args)

        return

    def get_filename(self, full_name, year, month, monthly=False):
        sdir = "monthly" if monthly else "hourly"
        ext = "nc4" if monthly else "grb"
        return os.path.join(
            self.get_data_path(sdir, ext),
            "%04d" % year,
            "%s_%04d_%02d.%s" % (full_name, year, month, ext),
        )

    def stack(self, dvars=None, years=None, months=None, force=False, **kwargs):
        """
        Fill element HDF with available GRB data
        """

        if dvars is None:
            dvars = list(self.dvars.keys())

        for dvar in sorted(dvars):

            # Check dvar
            if dvar not in self:
                print("%s not in dataset... skipping" % dvar)
                continue

            # Get full name
            full_name = self.dvars[dvar]["full"]
            print(dvar, full_name)

            # Get converter
            converter = self.dvars[dvar].get("converter", None)

            # Get hour0
            hour0 = self.dvars[dvar].get("hour0", 0)

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
                        print(dvar, "already stacked... skipping")
                        continue

                    path = self.get_data_path("constant", "nc4", f"{full_name}.nc4")

                    if not os.path.isfile(path):
                        print(dvar, "missing... skipping")
                        continue

                    # Retrieve slice from netcdf4 file
                    with netCDF4.Dataset(path) as nc:
                        x = np.expand_dims(nc.variables[dvar][0, :], axis=-1)

                    # Insert into HDF
                    slab.fill(0, slab.to_int(x, converter))

                # Done
                continue

            # Loop over request years
            for year in self.iter_year(years):

                if self.dvars[dvar].get("type", "hourly") != "hourly":
                    print(dvar, "is not hourly... skipping")
                    continue

                if self.isscalar(dvar):
                    shape = self.grid.shape

                if self.isvector(dvar):
                    shape = self.grid.shape + (2,)

                with self[dvar, year] as slab:

                    if slab:
                        print(dvar, "exists... updating")
                    else:
                        print(dvar, "does not exist... creating")
                        slab.create(
                            shape=shape, year=year, freq=self.freq, **self.dvars[dvar]
                        )

                    for month in self.iter_month(year, months):

                        start_time = time.time()

                        # Insert point
                        i = slab.date2ind(months=month, hours=hour0)

                        # Number of hours in month
                        nh = slab.month_len(month)

                        # Check fill status
                        if slab.isfull(np.s_[i : (i + nh)]) and not force:
                            print(year, month, "already stacked... skipping")
                            continue

                        # Check that data exists
                        path = self.get_filename(full_name, year, month, monthly=False)
                        if not os.path.isfile(path):
                            print(year, month, "missing... skipping")
                            continue

                        # Shape of this month; i.e. nlat x nlon x nh values
                        shape = self.grid.shape + (nh,)

                        # Store entire month; big!
                        X = np.zeros(shape, dtype=np.int16)

                        with pygrib.open(path) as grb:
                            h = 0
                            for msg in grb:
                                if msg.stepType != "instant":
                                    raise NotImplementedError(
                                        "%s not implemented" % msg.type
                                    )
                                print(msg)
                                # Extract, unmasking array, and convert to int
                                X[..., h] = slab.to_int(
                                    msg.values.filled(np.nan), converter
                                )
                                h += 1

                        # Write whole month
                        print("... writing", flush=True)
                        slab.fill(i, X)
                        X = None

                        # Write to syslog
                        syslog_elapsed_time(
                            time.time() - start_time,
                            "%s %s %04d-%02d written." % (str(self), dvar, year, month),
                        )


def main():
    # Create instance
    E = ERA5Land()

    # Download some basics
    E.download(dvars=["geo", "lsm"])

    # Grab all the temperature available
    x = E("tas", 33.640, -84.430)
    print(x.head())


if __name__ == "__main__":
    main()
