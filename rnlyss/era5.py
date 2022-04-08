# -*- coding: utf-8 -*-

import os
import time
from datetime import datetime
import numpy as np
import calendar

from rnlyss.dataset import Dataset
from rnlyss.grid import Grid
from rnlyss.util import syslog_elapsed_time

try:
    import netCDF4
except ImportError:
    raise NotImplementedError("cdsapi req'd to read ERA5 datasets")

try:
    import cdsapi
except ImportError:
    raise NotImplementedError("cdsapi req'd to download ERA5 datasets")


class ERA5(Dataset):

    # fmt: off
    dvars = {
        # Dry bulb temperature @ 2m (K)
        "t2m": {
            "role": "tas",
            "scale": 1e-2,
            "offset": 330, "full": "2m_temperature",
        },
        # Dew point temperature @ 2m (K)
        "d2m": {
            "role": "tdps",
            "scale": 1e-2,
            "offset": 330,
            "full": "2m_dewpoint_temperature",
        },
        # Zonal wind (east-west) @ 10m (m/s)
        "u10": {
            "role": "uas",
            "scale": 1e-2,
            "offset": 0,
            "full": "10m_u_component_of_wind",
        },
        # Meridional wind (north-south) @ 10m (m/s)
        "v10": {
            "role": "vas",
            "scale": 1e-2,
            "offset": 0,
            "full": "10m_v_component_of_wind",
        },
        # Surface pressure (Pa)
        "sp": {
            "role": "ps",
            "scale": 1,
            "offset": 75000,
            "full": "surface_pressure",
        },
        # Surface geopotential (m); convert from m2/s2 to m
        "z": {
            "role": "hgt",
            "scale": 1,
            "constant": True,
            "converter": lambda x: x / 9.80665,
            "full": "orography",
        },
        # Land area fraction (0 to 1)
        "lsm": {
            "role": "land",
            "scale": 1e-2,
            "constant": True,
            "full": "land_sea_mask",
        },
        # Shortwave downwelling @ surface (J/m²); convert to (W/m²)
        "ssrd": {
            "role": "rsds",
            "scale": 0.1,
            "offset": 0,
            "converter": lambda x: x / 3600,
            "full": "surface_solar_radiation_downwards",
        },
        # Shortwave (clear sky) downwelling at surface (J/m²); convert to (W/m²)
        "ssrdc": {
            "role": "rsdsc",
            "scale": 0.1,
            "offset": 0,
            "converter": lambda x: x / 3600,
            "full": "surface_solar_radiation_downward_clear_sky",
        },
        # Shortwave (direct) downwelling at surface (J/m²); convert to (W/m²)
        # NB. This is horizontal direct i.e. fdir = direct_normal × cos(zenith)
        "fdir": {
            "role": "rsdsd",
            "scale": 0.1,
            "offset": 0,
            "converter": lambda x: x / 3600,
            "full": "total_sky_direct_solar_radiation_at_surface",
        },
        # Shortwave downwelling @ top of atmosphere (J/m²); convert to (W/m²)
        "tisr": {
            "role": "rsdt",
            "scale": 0.1,
            "offset": 0,
            "converter": lambda x: x / 3600,
            "full": "toa_incident_solar_radiation",
        },
        # Longwave downwelling at surface (J/m²); convert to (W/m²)
        "strd": {
            "role": "rlds",
            "scale": 0.1,
            "offset": 0,
            "converter": lambda x: x / 3600,
            "full": "surface_thermal_radiation_downwards",
        },
        # Total precipitation (m); convert to (mm/s)
        "tp": {
            "role": "pr",
            "scale": 1 / 36000,
            "converter": lambda x: x / 3.6,
            "full": "total_precipitation",
        },
        # Cloud fraction (0 to 1)
        "tcc": {
            "role": "clt",
            "scale": 1e-4,
            "full": "total_cloud_cover",
        },
        # Precipitable water (kg/m²)
        "tcwv": {
            "role": "pwat",
            "scale": 1e-2,
            "full": "total_column_water_vapour",
        },
        # # Albedo (0 to 1)
        "fal": {
            "role": "albedo",
            "scale": 1e-4,
            "full": "forecast_albedo",
        },
    }
    # fmt: on

    # Time
    years = [1979, None]
    freq = 1

    # Grid
    # NB. Data downloaded from Copernicus in netCDF4 format is 0.25°×0.25°
    grid = Grid(shape=(721, 1440), origin=(90, 0), delta=(-1 / 4, 1 / 4))

    def stack(self, dvars=None, years=None, months=None, force=False, **kwargs):
        """
        Fill element HDF with available NC3 data
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
                            **self.dvars[dvar]
                        )

                    if slab.isfull(0) and not force:
                        print(dvar, "already stacked... skipping")
                        continue

                    path = self.get_nc3_filename(full_name, 1979, 1)
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
                        path = self.get_nc3_filename(full_name, year, month)
                        if not os.path.isfile(path):
                            print(year, month, "missing... skipping")
                            continue

                        # Shape of this month; i.e. nlat x nlon x nh values
                        shape = self.grid.shape + (nh,)

                        with netCDF4.Dataset(path) as nc:
                            # Check for ERA5T
                            if "expver" in nc.variables:
                                print(year, month, "expver... skipping")
                                continue

                            # Check that month isn't truncated
                            if nc.variables[dvar].shape[0] != nh:
                                print(year, month, "incomplete... skipping")
                                continue

                            # Store entire month; big!
                            X = np.zeros(shape, dtype=np.int16)

                            print(
                                year, month, "complete... reading", end="", flush=True
                            )

                            h = 0
                            for d in range(nh // 24):
                                # Read in 24 hours, convert, cast to int16, & transpose
                                x = np.transpose(
                                    slab.to_int(
                                        nc.variables[dvar][h : (h + 24), ...], converter
                                    ),
                                    (1, 2, 0),
                                )
                                # Store day
                                X[..., h : (h + 24)] = x
                                h += 24

                        # Write whole month
                        print("... writing", flush=True)
                        slab.fill(i, X)
                        X = None

                        # Write to syslog
                        syslog_elapsed_time(
                            time.time() - start_time,
                            "%s %s %04d-%02d written." % (str(self), dvar, year, month),
                        )

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

        def get_era5_args(dvar, year, month, constant=False):
            """
            Return variable name, selection dict, and target path
            """

            # Full name of variable
            full_name = self.dvars[dvar]["full"]

            target_path = self.get_nc3_filename(full_name, year, month)

            # Check that we're not within 3 months
            if n_months(year, month) <= 3:
                print("%s likely contains ERA5T... skipping" % target_path)
                return None

            os.makedirs(os.path.dirname(target_path), exist_ok=True)

            if os.path.isfile(target_path):
                print("%s exists... skipping" % target_path)
                return None

            if constant:
                # Request single hour
                days = ["01"]
                hours = ["00:00"]

            else:
                # Request entire month
                days = [
                    "%02d" % (day + 1,)
                    for day in range(calendar.monthrange(year, month)[1])
                ]

                # Request 24 hours
                hours = ["%02d:00" % h for h in range(24)]

            return (
                "reanalysis-era5-single-levels",
                {
                    "product_type": "reanalysis",
                    "format": "netcdf",
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

        # Release the sloths...
        for dvar in sorted(dvars):
            if dvar not in self:
                print("%s not in dataset... skipping" % dvar)
                continue

            print(dvar, self.dvars[dvar]["full"])

            if self.isconstant(dvar):
                # Constant
                args = get_era5_args(dvar, 1979, 1, constant=True)
                if args is not None:
                    get_era5_file(*args)

            else:
                # Hourly file
                for year, month in self.iter_year_month(years, months):
                    args = get_era5_args(dvar, year, month, constant=False)
                    if args is not None:
                        get_era5_file(*args)

        return

    def get_nc3_filename(self, full_name, year, month):
        return os.path.join(
            self.get_data_path("nc3"),
            "%04d" % year,
            "%s_%04d_%02d.nc3" % (full_name, year, month),
        )


def main():

    # Create MERRA-2 instance
    E = ERA5()

    # Atlanta
    lat, lon = 33.640, -84.430

    # Height
    h = E.hgt(lat, lon)
    print("elevation", h)

    # Land mask
    m = E.land(lat, lon)
    print("land mask", m)

    # Extract air temperature at 2m at a given location into a Pandas Series
    # (return the nearest location)
    x = E("tdps", lat, lon)
    print(x.head())


if __name__ == "__main__":
    main()
