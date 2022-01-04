# -*- coding: utf-8 -*-
#
# Copyright 2019 Klimaat

import os
import time
import numpy as np
import calendar
import shutil
import requests
from email.utils import parsedate_tz, mktime_tz
from urllib.parse import urlencode

from rnlyss.dataset import Dataset
from rnlyss.grid import Grid
from rnlyss.util import syslog_elapsed_time

try:
    import netCDF4
except ImportError:
    raise NotImplementedError("netCDF4 req'd to read MERRA2 datasets")


class MERRA2(Dataset):

    # fmt: off
    dvars = {
        # Dry bulb temperature @ 2m, 10m (K)
        'T2M': {'role': 'tas', 'scale': 1e-2, 'offset': 330,
                'collection': 'inst1_2d_asm_Nx', 'hour0': 0},
        # 'T10M': {'role': 'tas10', 'scale': 1e-2, 'offset': 330,
        #         'collection': 'inst1_2d_asm_Nx', 'hour0': 0},
        # Surface temperature (K)
        'TS': {'role': 'ts', 'scale': 1e-2, 'offset': 330,
               'collection': 'inst1_2d_asm_Nx', 'hour0': 0},
        # Specific humidity @ 2m, 10m (kg/kg)
        'QV2M': {'role': 'huss', 'scale': 1e-6, 'offset': 0.03,
                 'collection': 'inst1_2d_asm_Nx', 'hour0': 0},
        # 'QV10M': {'role': 'huss10', 'scale': 1e-6, 'offset': 0.03,
        #          'collection': 'inst1_2d_asm_Nx', 'hour0': 0},
        # Zonal wind (east-west) @ 2m, 10m (m/s)
        # 'U2M': {'role': 'uas2', 'scale': 1e-2, 'offset': 0,
        #          'collection': 'inst1_2d_asm_Nx', 'hour0': 0},
        'U10M': {'role': 'uas', 'scale': 1e-2, 'offset': 0,
                 'collection': 'inst1_2d_asm_Nx', 'hour0': 0},
        # Meridional wind (north-south) @ 10m (m/s)
        # 'V2M': {'role': 'vas2', 'scale': 1e-2, 'offset': 0,
        #          'collection': 'inst1_2d_asm_Nx', 'hour0': 0},
        'V10M': {'role': 'vas', 'scale': 1e-2, 'offset': 0,
                 'collection': 'inst1_2d_asm_Nx', 'hour0': 0},
        # Surface pressure (Pa)
        'PS': {'role': 'ps', 'scale': 1, 'offset': 75000,
               'collection': 'inst1_2d_asm_Nx', 'hour0': 0},
        # Surface geopotential (m)
        # Convert from m2/s2 to m
        'PHIS': {'role': 'hgt', 'scale': 1,
                 'constant': True, 'hour0': 0,
                 'converter': lambda x: x/9.80665},
        # Land area fraction (0 to 1)
        'FRLAND': {'role': 'frland', 'scale': 1e-2,
                   'constant': True, 'hour0': 0},
        # Land-ice area fraction (0 to 1)
        'FRLANDICE': {'role': 'frlandice', 'scale': 1e-2,
                      'constant': True, 'hour0': 0},
        # Ocean area fraction (0 to 1)
        'FROCEAN': {'role': 'frocean', 'scale': 1e-2,
                    'constant': True, 'hour0': 0},
        # Lake area fraction (0 to 1)
        'FRLAKE': {'role': 'frlake', 'scale': 1e-2,
                   'constant': True, 'hour0': 0},
        # Longwave downwelling @ surface (W/m²)
        'LWGAB': {'role': 'rlds', 'scale': 0.1, 'offset': 0,
                  'collection': 'tavg1_2d_rad_Nx', 'hour0': 1},
        # Shortwave downwelling @ surface (W/m²)
        'SWGDN': {'role': 'rsds', 'scale': 0.1, 'offset': 0,
                  'collection': 'tavg1_2d_rad_Nx', 'hour0': 1},
        # Shortwave (clear sky) downwelling @ surface (W/m²)
        'SWGDNCLR': {'role': 'rsdsc', 'scale': 0.1,
                     'collection': 'tavg1_2d_rad_Nx', 'hour0': 1},
        # Shortwave downwelling @ top of atmosphere (W/m²)
        'SWTDN': {'role': 'rsdt', 'scale': 0.1,
                  'collection': 'tavg1_2d_rad_Nx', 'hour0': 1},
        # Total precipitation (kg/m²/s) <-> (mm/s)
        'PRECTOT': {'role': 'pr', 'scale': 1/36000,
                    'collection': 'tavg1_2d_flx_Nx', 'hour0': 1},
        # Cloud fraction (0 to 1)
        'CLDTOT': {'role': 'clt', 'scale': 1e-4,
                   'collection': 'tavg1_2d_rad_Nx', 'hour0': 1},
        # Precipitable water (kg/m²)
        'TQV': {'role': 'pwat', 'scale': 1e-2,
                'collection': 'inst1_2d_asm_Nx', 'hour0': 0},
        # Ozone (Dobsons)
        'TO3': {'role': 'ozone', 'scale': 1e-2, 'offset': 320,
                'collection': 'inst1_2d_asm_Nx', 'hour0': 0},
        # Albedo (0 to 1)
        'ALBEDO': {'role': 'albedo', 'scale': 1e-4,
                   'collection': 'tavg1_2d_rad_Nx', 'hour0': 1},
        # Aerosol Angstrom exponent (0 to 1)
        'TOTANGSTR': {'role': 'alpha', 'scale': 1e-4,
                      'collection': 'tavg1_2d_aer_Nx', 'hour0': 1},
        # Aerosol optical depth @ 550nm (0 to 1)
        'TOTEXTTAU': {'role': 'aod550', 'scale': 1e-4,
                      'collection': 'tavg1_2d_aer_Nx', 'hour0': 1},
        # Aerosol scattering (0 to 1)
        'TOTSCATAU': {'role': 'scatter', 'scale': 1e-4,
                      'collection': 'tavg1_2d_aer_Nx', 'hour0': 1},
        #  Surface roughness (m); store log(z0)
        # 'Z0M': {'role': 'z0', 'scale': 1e-3, 'converter': lambda x: np.log(x),
        #        'collection': 'tavg1_2d_flx_Nx', 'hour0': 1},
        # Boundary layer height (m)
        # 'PBLH': {'role': 'pblh', 'scale': 1,
        #        'collection': 'tavg1_2d_flx_Nx', 'hour0': 1},
        # Surface flux Richardson number (-)
        # 'RISFC': {'role': 'rif', 'scale': 1e-2,
        #        'collection': 'tavg1_2d_flx_Nx', 'hour0': 1},
    }
    # fmt: on

    # Time
    years = [1980, None]
    freq = 1

    # Grid
    grid = Grid(shape=(361, 576), origin=(-90, -180), delta=(1 / 2, 5 / 8))

    def land(self, lat, lon, order=0):
        """
        Return land mask given lat/lon.  0=100% lake/ocean, 1=100% land/ice.

        order=0: snap to nearest grid point horizontally.
        order=1: perform bi-linear interpolation

        MERRA-2 is FRLAND+FRLANDICE
        """
        return self("frland", lat, lon, order=order) + self(
            "frlandice", lat, lon, order=order
        )

    def stack(self, dvars=None, years=None, months=None, force=False, **kwargs):
        """
        Fill element HDF with available GRB data.
        """

        if dvars is None:
            dvars = list(self.dvars.keys())

        nc_path = self.get_data_path("nc4")

        for dvar in sorted(dvars):

            # Check dvar
            if dvar not in self:
                print("%s not in dataset... skipping" % dvar)
                continue

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

                    path = os.path.join(nc_path, dvar + ".nc4")

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

                        if slab.isfull(np.s_[i : (i + nh)]) and not force:
                            print(year, month, "already stacked... skipping")
                            continue

                        # Check that all days are available in this month

                        complete = True
                        for day in range(1, nh // 24 + 1):
                            fn = self.get_nc4_filename(dvar, year, month, day)
                            if not os.path.isfile(fn):
                                complete = False
                                break

                        # Don't build unless complete
                        if not complete:
                            print(year, month, "incomplete... skipping")
                            continue

                        # Store entire month
                        shape = self.grid.shape + (nh,)
                        X = np.zeros(shape, dtype=np.int16)

                        # Have full month; build
                        h = 0
                        for day in range(1, nh // 24 + 1):
                            fn = self.get_nc4_filename(dvar, year, month, day)
                            print(fn)
                            with netCDF4.Dataset(fn) as nc:
                                x = np.transpose(
                                    slab.to_int(nc.variables[dvar][:], converter),
                                    (1, 2, 0),
                                )

                            X[..., h : (h + 24)] = x
                            h += 24

                        # Store month
                        print(year, month, "complete... writing", flush=True)
                        slab.fill(i, X)
                        X = None

                        # Write to syslog
                        syslog_elapsed_time(
                            time.time() - start_time,
                            "%s %s %04d-%02d written." % (str(self), dvar, year, month),
                        )

    def download(self, dvars=None, years=None, months=None, ignore=False, **kwargs):
        """
        Download MERRA files.
        """

        if dvars is None:
            dvars = list(self.dvars.keys())

        if not isinstance(dvars, list):
            dvars = [dvars]

        # Establish session to store cookies

        session = requests.Session()

        def get_file(url, dst):

            # Ensure directory exists
            os.makedirs(os.path.dirname(dst), exist_ok=True)

            # Start request
            for i in range(3):
                request = session.get(url, stream=True)
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

        def get_constants():

            # Need to get a single constant file collection

            constants = self.constants()

            for dvar in constants:
                fn = os.path.join(self.get_data_path("nc4"), "%s.nc4" % dvar)
                if not os.path.isfile(fn):
                    break
            else:
                print("All constants downloaded and extracted")
                return

            fn = "MERRA2_101.const_2d_asm_Nx.00000000.nc4"

            url = r"https://goldsmr4.gesdisc.eosdis.nasa.gov/"
            url += r"data/MERRA2_MONTHLY/M2C0NXASM.5.12.4/1980/"
            url += fn

            dst = os.path.join(self.get_data_path("nc4"), fn)

            # Ensure netcdf directory exists
            os.makedirs(self.get_data_path("nc4"), exist_ok=True)

            # Grab constant file
            if not get_file(url, dst):
                raise IOError("Error getting constants file")

            # Extract required constants

            def copy_var(v, ds_in, ds_out):
                x = ds_in.variables[v]
                y = ds_out.createVariable(v, x.datatype, x.dimensions)
                y.setncatts({k: x.getncattr(k) for k in x.ncattrs()})
                y[:] = x[:]

            with netCDF4.Dataset(dst) as nc_in:
                for dvar in constants:
                    if dvar in nc_in.variables:
                        print("Extracting %s" % dvar)
                        fn = os.path.join(self.get_data_path("nc4", "%s.nc4" % dvar))
                        with netCDF4.Dataset(fn, "w") as nc_out:
                            nc_out.setncatts(
                                {k: nc_in.getncattr(k) for k in nc_in.ncattrs()}
                            )
                            for k, v in nc_in.dimensions.items():
                                nc_out.createDimension(
                                    k, len(v) if not v.isunlimited() else None
                                )
                                copy_var(k, nc_in, nc_out)
                            copy_var(dvar, nc_in, nc_out)

            # Clean up
            if os.path.isfile(dst):
                os.remove(dst)

        shortnames = {
            "inst1_2d_asm_Nx": "M2I1NXASM",
            "tavg1_2d_slv_Nx": "M2T1NXSLV",
            "tavg1_2d_rad_Nx": "M2T1NXRAD",
            "tavg1_2d_flx_Nx": "M2T1NXFLX",
            "tavg1_2d_aer_Nx": "M2T1NXAER",
        }

        def get_hourly_file(dvar, year, month, day):

            dst = self.get_nc4_filename(dvar, year, month, day)

            # Quick exit
            if os.path.isfile(dst) and ignore:
                return True

            # Determine stream SV;
            if year <= 1991:
                stream = "10"
            elif 1991 < year <= 2000:
                stream = "20"
            elif 2000 < year <= 2010:
                stream = "30"
            else:
                stream = "40"

            collection = self.dvars[dvar].get("collection", "inst1_2d_asm_Nx")

            shortname = shortnames[collection]

            def get_fn(ver):
                """
                There may be different versions that we need to ping
                """
                fn = "/data/MERRA2/%s.5.12.4/" % shortname
                fn += "%04d/%02d/" % (year, month)
                fn += "MERRA2_%s%s.%s." % (stream, ver, collection)
                fn += "%04d%02d%02d.nc4" % (year, month, day)
                return fn

            # Check time of server file from which the subsetter works
            # i.e. subsetter always returns a brand new file

            # First determine if any versions exist
            for ver in ["0", "1"]:
                fn = get_fn(ver)
                url = r"https://goldsmr4.gesdisc.eosdis.nasa.gov" + fn
                request = session.head(url)
                if request.status_code == 200:
                    break
            else:
                print("%s unavailable... skipping" % fn)
                return False

            print("%s available... checking" % fn)

            # Assume that if we have a file of the same date we can skip
            last_modified = mktime_tz(parsedate_tz(request.headers["Last-Modified"]))
            if os.path.isfile(dst):
                if int(os.path.getmtime(dst)) == last_modified:
                    print("%s unchanged... skipping" % dst)
                    return True

            # Build subsetter url
            params = {
                "FILENAME": fn,
                "FORMAT": "bmM0Lw",
                "BBOX": "-90,-180,90,180",
                "SHORTNAME": shortname,
                "SERVICE": "SUBSET_MERRA2",
                "VERSION": "1.02",
                "LAYERS": "",
                "VARIABLES": "%s" % dvar.upper(),
            }

            url = r"http://goldsmr4.gesdisc.eosdis.nasa.gov/"
            url += r"daac-bin/OTF/HTTP_services.cgi?"
            url += urlencode(params)

            # Grab it
            result = get_file(url, dst)

            # Set modification date
            if result and os.path.isfile(dst):
                os.utime(dst, (last_modified, last_modified))
                return True

            return False

        # Release the sloths...
        for dvar in sorted(dvars):
            if dvar not in self:
                print("%s not in dataset... skipping" % dvar)
                continue

            print(dvar)

            if self.isconstant(dvar):
                # Constant
                get_constants()

            else:
                # Hourly file
                for year, month in self.iter_year_month(years, months):
                    days = range(1, calendar.monthrange(year, month)[1] + 1)
                    for day in days:
                        result = get_hourly_file(dvar, year, month, day)
                        if not result:
                            break
                    else:
                        continue
                    break

        return

    def get_nc4_filename(self, dvar, year, month, day):
        return os.path.join(
            self.get_data_path("nc4"),
            "%04d" % year,
            "%02d" % month,
            "%s.%04d%02d%02d.nc4" % (dvar, year, month, day),
        )


def main():

    # Create MERRA-2 instance
    M = MERRA2()

    # Extract air temperature at 2m at a given location into a Pandas Series
    # (return the nearest location)
    x = M("tas", 33.640, -84.430)
    print(x.head())

    # The same call but applying bi-linear interpolation of the surrounding
    # 4 grid locations and restricting data to the year 2018.
    y = M("tas", 33.640, -84.430, hgt=313, order=1, years=[2018])
    print(y.head())

    # Calculate the ASHRAE tau coefficients and optionally the fluxes at noon
    tau = M.to_clearsky(33.640, -84.430, years=[2018], noon_flux=True)
    print(tau)

    # Produces the average monthly (and annual) daily-average all sky radiation
    # for every requested year
    rad = M.to_allsky(lat=33.640, lon=-84.430, years=[2018])

    # Which again can be massaged into the required statistics (mean, std)
    print(rad.describe().round(decimals=1))

    # Extract the solar components
    solar = M.solar_split(33.640, -84.430, years=[2018])
    print(solar[12:24])


if __name__ == "__main__":
    main()
