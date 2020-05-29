# -*- coding: utf-8 -*-
#
# Copyright 2019 Klimaat

import os
import time
import calendar
import numpy as np
import netrc
import shutil
import requests
import tarfile
from email.utils import parsedate_tz, mktime_tz

from rnlyss.dataset import Dataset
from rnlyss.grid import GaussianGrid
from rnlyss.util import syslog_elapsed_time

try:
    import pygrib
except ImportError:
    raise NotImplementedError("pygrib req'd to read CFS datasets")


class CFSV2(Dataset):

    # Dataset variables
    # NB: Due to storage of source files, some positive-only variables may
    #     have negative values.
    # fmt: off
    dvars = {
        # Surface geopotential (m)
        'orog': {'role': 'hgt', 'scale': 1, 'constant': True, 'hour0': 0},
        # Land surface mask (0 or 1)
        'lsm': {'role': 'land', 'scale': 1e-2, 'constant': True, 'hour0': 0},
        # Dry bulb temperature @ 2m (K)
        'tmp2m': {'role': 'tas', 'scale': 1e-2, 'offset': 330, 'hour0': 1},
        # Specific humidity @ 2m (kg/kg)
        'q2m': {'role': 'huss', 'scale': 1e-6, 'offset': 0.03, 'hour0': 1},
        # Wind velocity vector (u, v) @ 10m (m/s)
        'wnd10m': {'role': ('uas', 'vas'), 'scale': 1e-2, 'hour0': 1,
                   'vector': True},
        # Surface pressure (Pa)
        'pressfc': {'role': 'ps', 'scale': 1, 'offset': 75000, 'hour0': 0},
        # Downwelling longwave surface radiation (W/m²)
        'dlwsfc': {'role': 'rlds', 'scale': 0.1, 'offset': 0, 'hour0': 1},
        # Downwelling shortwave surface radiation (W/m²)
        'dswsfc': {'role': 'rsds', 'scale': 0.1, 'offset': 0, 'hour0': 1},
        # Downwelling clear sky shortwave surface radiation (W/m²)
        'dcsswsfc': {'role': 'rsdsc', 'scale': 0.1, 'offset': 0, 'hour0': 1,
                     'uglpr': (2, 0, 5, 5, 0)},
        # Downwelling shortwave radiation at top of atmosphere (W/m²)
        'dswtoa': {'role': 'rsdt', 'scale': 0.1, 'offset': 0, 'hour0': 1,
                   'uglpr': (2, 0, 21, 13, 0)},
        # Precipitation rate (kg/m²/s = 1 mm/s = 3600 mm/hr)
        'prate': {'role': 'pr', 'scale': 1/36000, 'hour0': 1},
        # Cloud cover (convert from % to fraction)
        'cldcovtot': {'role': 'clt', 'scale': 1e-4, 'hour0': 1,
                      'converter': lambda x: x/100},
        # Precipitable water (kg/m²)
        'pwat': {'role': 'pwat', 'scale': 1e-2, 'hour0': 0},
        # Surface albedo (convert from % to fraction) (download & assemble)
        'albedo': {'role': 'albedo', 'scale': 1e-4, 'offset': 0, 'hour0': 1,
                   'uglpr': (2, 0, 5, 1, 0), 'converter': lambda x: x/100},
    }
    # fmt:on

    # Time
    years = [2011, None]
    freq = 1

    # Grid
    grid = GaussianGrid(shape=(880, 1760), origin=(90, 0), delta=(-1, 360 / 1760))

    # CFSv2 RDA dataset
    dataset = "ds094"

    def stack(self, dvars=None, years=None, months=None, force=False, **kwargs):
        """
        Fill element HDF with available GRB data.
        """

        if dvars is None:
            dvars = list(self.dvars.keys())

        grb_path = self.get_data_path("grb2")

        for dvar in sorted(dvars):

            # Check dvar
            if dvar not in self:
                print("%s not in dataset... skipping" % dvar)
                continue

            # Get converter
            converter = self.dvars[dvar].get("converter", None)

            # Get hour offset
            hour0 = self.dvars[dvar].get("hour0", 0)

            # Special case:  constants
            if self.isconstant(dvar):

                with self[dvar] as slab:

                    # Create slab is it doesn't exist
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

                    path = os.path.join(grb_path, dvar + ".grb2")

                    if not os.path.isfile(path):
                        print(dvar, "missing... skipping")
                        continue

                    # Open GRB
                    grb = pygrib.open(path)
                    msg = grb.readline()
                    print(msg)

                    slab.fill(
                        0, slab.to_int(np.expand_dims(msg.values, axis=-1), converter)
                    )

                continue

            # Loop over request years
            for year in self.iter_year(years):

                # Scalars
                if self.isscalar(dvar):
                    shape = self.grid.shape

                # Vectors
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

                        i = slab.date2ind(months=month, hours=hour0)

                        nh = slab.month_len(month)

                        if slab.isfull(np.s_[i : (i + nh)]) and not force:
                            print(year, month, "already stacked... skipping")
                            continue

                        fn = self.get_grb2_filename(dvar, year, month)

                        # Open GRB; should'nt be missing as we glob'd it above
                        try:
                            grb = pygrib.open(fn)
                        except IOError:
                            print(year, month, "missing... skipping")
                            continue

                        if self.isvector(dvar):

                            # Shape of this month
                            # i.e. {nlat} x {nlon} x {x, y components} x {nh}
                            shape = self.grid.shape + (2, nh)

                            # Store entire month; big!
                            X = np.zeros(shape, dtype=np.int16)

                            for hexa in range(grb.messages // 12):
                                for step in range(6):
                                    # Loop over x- and y- component
                                    for comp in range(2):
                                        msg = grb.readline()
                                        print(repr(msg))
                                        h = (msg.day - 1) * 24 + msg.hour + step
                                        if msg.stepType == "instant":
                                            X[:, :, comp, h] = slab.to_int(
                                                msg.values, converter
                                            )
                                        else:
                                            raise NotImplementedError(msg.stepType)
                        else:

                            # Shape of this month; i.e. nlat x nlon x nh values
                            shape = self.grid.shape + (nh,)

                            # Store entire month; big!
                            X = np.zeros(shape, dtype=np.int16)

                            for hexa in range(grb.messages // 6):
                                values = np.zeros(self.grid.shape)
                                prev_values = np.zeros(self.grid.shape)
                                for step in range(6):
                                    msg = grb.readline()
                                    print(repr(msg))
                                    np.copyto(values, msg.values)
                                    h = (msg.day - 1) * 24 + msg.hour + step
                                    if msg.stepType == "instant":
                                        X[..., h] = slab.to_int(values, converter)
                                    elif msg.stepType == "avg":
                                        X[..., h] = slab.to_int(
                                            (step + 1) * values - step * prev_values,
                                            converter,
                                        )
                                        np.copyto(prev_values, values)
                                    elif msg.stepType == "accum":
                                        X[..., h] = slab.to_int(
                                            values - prev_values, converter
                                        )
                                        np.copyto(prev_values, values)
                                    else:
                                        raise NotImplementedError(msg.stepType)

                        # Close the GRB
                        grb.close()

                        # Store it
                        print(year, month, "complete... writing", flush=True)
                        slab.fill(i, X)
                        X = None

                        # Write to syslog
                        syslog_elapsed_time(
                            time.time() - start_time,
                            "%s %s %04d-%02d written." % (str(self), dvar, year, month),
                        )

    def download(self, dvars=None, years=None, months=None, **kwargs):
        """
        Download CFSv2 GRB files.
        """

        if dvars is None:
            # Default is all of them
            dvars = list(self.dvars.keys())

        if not isinstance(dvars, list):
            dvars = [dvars]

        # Establish connection

        session = requests.Session()
        machine = "rda.ucar.edu"
        auth = netrc.netrc().authenticators(machine)

        if auth is None:
            raise SystemExit("Add rda.ucar.edu credentials to .netrc")

        email, _, passwd = auth

        request = session.post(
            r"https://rda.ucar.edu/cgi-bin/login",
            data={"email": email, "password": passwd, "action": "login"},
        )

        if request.status_code != 200:
            raise SystemExit(request.headers)

        def get_file(url, dst):

            # Ensure directory exists
            os.makedirs(os.path.dirname(dst), exist_ok=True)

            # Start request
            request = session.get(url, stream=True)

            if request.status_code != 200:
                print("%s unavailable... skipping" % url)
                return False

            content_length = int(request.headers["Content-Length"])
            last_modified = mktime_tz(parsedate_tz(request.headers["Last-Modified"]))

            if os.path.isfile(dst):
                if os.path.getsize(dst) == content_length:
                    if os.path.getmtime(dst) == last_modified:
                        print("%s unchanged... skipping" % url)
                        return False

            print("%s available..." % url, "downloading %d bytes" % content_length)

            try:
                # Stream to file
                shutil.copyfileobj(request.raw, open(dst, "wb"))
                # Set time on disk to server time
                os.utime(dst, (last_modified, last_modified))
            except:
                # Problem; delete file
                if os.path.isfile(dst):
                    print("%s deleted... skipping" % dst)
                    os.remove(dst)
                raise

            return True

        def get_partial_file(url, dst, byte_range):
            """
            Download file and append to existing dst
            """

            # Ensure directory exists
            os.makedirs(os.path.dirname(dst), exist_ok=True)

            # Start request, downloading only within byte range
            for i in range(3):
                try:
                    headers = {"Range": "bytes={0}-{1}".format(*byte_range)}
                    request = session.get(url, headers=headers, stream=True)
                except:
                    # Retry
                    print("%s unavailable... retrying" % url)
                    continue
                else:
                    # Success
                    break

            else:
                # Failure
                if os.path.isfile(dst):
                    os.remove(dst)
                    raise SystemExit("%s problem... deleting & exiting" % dst)
                else:
                    raise SystemExit("%s problem... exiting" % dst)

            if request.status_code != 206:
                print("%s unavailable... skipping" % url)
                return False

            content_length = int(request.headers["Content-Length"])
            print(
                "%s available..." % url,
                "partially downloading %d bytes" % content_length,
            )

            try:
                # Stream to file
                shutil.copyfileobj(request.raw, open(dst, "ab"))
            except:
                # Delete file at the slighest whiff of trouble
                if os.path.isfile(dst):
                    os.remove(dst)
                    raise SystemExit("%s problem... deleting & exiting" % dst)
                else:
                    raise SystemExit("%s problem... exiting" % dst)

            return True

        def get_constants():

            # Need to get a single, large TAR from the 6-hourly products
            # ds93.0 (CFSR) and ds94.0 (CFSv2)

            constants = self.constants()

            for dvar in constants:
                fn = os.path.join(self.get_data_path("grb2"), "%s.grb2" % dvar)
                if not os.path.isfile(fn):
                    break
            else:
                print("All constants downloaded and extracted")
                return

            if self.dataset == "ds093":
                fn = r"flxf01.gdas.19790101-19790105.tar"
                url = r"https://rda.ucar.edu/data/ds093.0/1979/" + fn
            elif self.dataset == "ds094":
                fn = r"flxf01.gdas.20110101-20110105.tar"
                url = r"https://rda.ucar.edu/data/ds094.0/2011/" + fn
            else:
                raise NotImplemented(self.dataset)

            dst = os.path.join(self.get_data_path("grb2"), fn)

            # Ensure grb directory exists
            os.makedirs(self.get_data_path("grb2"), exist_ok=True)

            # Grab tar
            get_file(url, dst)

            # Open it up and pull out first GRB
            print("Inspecting %s" % fn)
            with tarfile.open(dst, "r") as tar:
                for member in tar:
                    fn = os.path.join(self.get_data_path("grb2"), member.name)
                    if not os.path.isfile(fn):
                        print("Extracting %s" % fn)
                        with open(fn, "wb") as f:
                            shutil.copyfileobj(tar.fileobj, f, member.size)
                    break

            # Extract constants
            with pygrib.open(fn) as grb:
                for msg in grb:
                    for dvar in constants[:]:
                        if msg.shortName == dvar:
                            fn = os.path.join(
                                self.get_data_path("grb2", "%s.grb2" % dvar)
                            )
                            with open(fn, "wb") as f:
                                f.write(msg.tostring())
                            constants.remove(dvar)

        def get_hourly_file(dvar, year, month):

            # Build paths
            dst = self.get_grb2_filename(dvar, year, month)
            url = "https://rda.ucar.edu/data/%s.1/%04d/%s" % (
                self.dataset,
                year,
                os.path.basename(dst),
            )

            # Grab grb2
            get_file(url, dst)

        def get_byte_ranges(invUrl, uglpr, nForecasts=1):
            """
            Download inventory URL and scan for UGLPR

            Inventory ranges are provided for each specific variable coded as
            U: Product e.g. 0: 1-hour Average, 1: 1-hour Forecast, etc.
            G: Grid: e.g. 0: Gaussian
            L: Level: e.g. 0: Surface, 8: Top-of-atmosphere
            P: Variable: e.g. 59=clear sky (look for 0.4.192)
            R: Process: e.g. 0
            """

            request = session.get(invUrl)

            if request.status_code != 200:
                print("%s unavailable... skipping" % invUrl)
                return []

            # Content is gzip'd; extract into str
            content = request.content.decode("utf-8")

            # Build search string(s)
            # If number of forecasts > 1, the U repeats every 4
            uglprStr = []
            U, G, L, P, R = uglpr
            for f in range(nForecasts):
                uglprStr.append("|%d|%d|%d|%d|%d" % (U + 4 * f, G, L, P, R))

            # Find ranges associated with each message in tar
            byte_ranges = []
            for line in content.splitlines():
                for f in range(nForecasts):
                    if uglprStr[f] in line:
                        fields = line.split("|")
                        offset, length = int(fields[0]), int(fields[1])
                        byte_ranges.append((offset, offset + length - 1))

            return byte_ranges

        def assemble_hourly_file(dvar, year, month, uglpr):

            # Build path
            dst = self.get_grb2_filename(dvar, year, month)

            # Check if it exists
            if os.path.isfile(dst):
                print("%s exists... skipping" % dst)
                return

            # Get inventories and generate a download work list
            print("%s..." % dst, "getting server inventory and building download list")
            numDays = calendar.monthrange(year, month)[1]

            work_list = [None] * (24 * numDays)

            stream = self.get_stream(year, month)

            if stream == "gdas":

                # CFSR and CFSv2, the early days

                for (d1, d2) in [
                    (1, 5),
                    (6, 10),
                    (11, 15),
                    (16, 20),
                    (21, 25),
                    (26, numDays),
                ]:
                    for fHour in range(6):
                        # Build URLs
                        dayRange = "-".join(
                            (
                                "%04d%02d%02d" % (year, month, d1),
                                "%04d%02d%02d" % (year, month, d2),
                            )
                        )
                        fn = "flxf%02d.gdas.%s.tar" % (fHour + 1, dayRange)

                        tarUrl = "https://rda.ucar.edu/data/"
                        tarUrl += "%s.0/%d/%s" % (self.dataset, year, fn)

                        invUrl = "https://rda.ucar.edu/"
                        invUrl += "cgi-bin/datasets/inventory?"
                        invUrl += "df=%04d/%s&" % (year, fn)
                        invUrl += "ds=%s.0&" % self.dataset[2:]
                        invUrl += "type=GrML"

                        # Get ranges for this particular inventory
                        ranges = get_byte_ranges(invUrl, uglpr)

                        if len(ranges):
                            for ir, r in enumerate(ranges):
                                # Absolute hour in month
                                hour = (d1 - 1) * 24 + ir * 6 + fHour
                                # Add to work list
                                work_list[hour] = (tarUrl, r)

            elif stream == "cdas1":

                # CFSv2, the later days

                for day in range(1, numDays + 1):
                    # Build URLs
                    fn = "cdas1.%04d%02d%02d.sfluxgrbf.tar" % (year, month, day)

                    tarUrl = "https://rda.ucar.edu/data/"
                    tarUrl += "%s.0/%d/%s" % (self.dataset, year, fn)

                    invUrl = "https://rda.ucar.edu/cgi-bin/datasets/inventory?"
                    invUrl += "df=%04d/%s&" % (year, fn)
                    invUrl += "ds=%s.0&" % self.dataset[2:]
                    invUrl += "type=GrML"

                    # Get ranges for this particular inventory
                    ranges = get_byte_ranges(invUrl, uglpr, nForecasts=6)

                    if len(ranges):
                        for ir, r in enumerate(ranges):
                            # Absolute hour in month
                            hour = (day - 1) * 24 + ir
                            # Add to work list
                            work_list[hour] = (tarUrl, r)

            else:
                raise NotImplementedError(stream)

            # Check if complete inventory
            for val in work_list:
                if val is None:
                    print("%s incomplete... skipping" % dst)
                    if os.path.isfile(dst):
                        os.remove(dst)
                    return False

            # Now loop over in hourly order and concatenate to dst
            for tarUrl, tarRange in work_list:
                get_partial_file(tarUrl, dst, tarRange)

            return True

        # Release the sloths...
        for dvar in sorted(dvars):
            if dvar not in self:
                print("%s not in dataset... skipping" % dvar)
                print("available: %r" % self.dvars.keys())
                continue
            print(dvar)
            if self.isconstant(dvar):
                # Constant file
                get_constants()
            else:
                # Hourly file

                uglpr = self.dvars[dvar].get("uglpr", None)

                for year, month in self.iter_year_month(years, months):
                    if uglpr is None:
                        get_hourly_file(dvar, year, month)
                    else:
                        assemble_hourly_file(dvar, year, month, uglpr)

        return

    def get_stream(self, year, month):
        """
        Return stream based on year and month
        """
        # Determine dataset and stream
        if self.dataset == "ds093":
            return "gdas"
        else:
            if year == 2011 and month < 4:
                return "gdas"
            else:
                return "cdas1"

    def get_grb2_filename(self, dvar, year, month):
        stream = self.get_stream(year, month)
        return os.path.join(
            self.get_data_path(
                "grb2",
                "%04d" % year,
                "%s.%s.%04d%02d.grb2" % (dvar.lower(), stream, year, month),
            )
        )

    def calc_tsi(self, year):
        """
        Calculate CFSR total solar irradiance based on year

        c.f. radiation_astronomy.f
        """

        # van den Dool data (1979-2006)
        # fmt: off
        dTSI = np.array([
            6.70, 6.70, 6.80, 6.60, 6.20, 6.00, 5.70, 5.70, 5.80, 6.20, 6.50,
            6.50, 6.50, 6.40, 6.00, 5.80, 5.70, 5.70, 5.90, 6.40, 6.70, 6.70,
            6.80, 6.70, 6.30, 6.10, 5.90, 5.70
        ])
        # fmt: on

        # Index into dTSI
        i = np.asarray(year) - 1979

        # Extend backward and/or forward assuming 11-year sunspot cycle
        while np.any(i < 0):
            i[i < 0] += 11
        while np.any(i > 27):
            i[i > 27] -= 11

        # Add base
        return 1360.0 + dTSI[i]


def main():
    # Create CFSv2 instance
    C = CFSV2()

    # Extract air temperature at 2m at a given location into a Pandas Series
    # (return the nearest location)
    x = C("tas", 33.640, -84.430)
    print(x.head())

    # The same call but applying bi-linear interpolation of the surrounding
    # 4 grid locations and restricting data to the year 2018.
    y = C("tas", 33.640, -84.430, hgt=313, order=1, years=[2018])
    print(y.head())

    # Calculate the ASHRAE tau coefficients and optionally the fluxes at noon
    tau = C.to_clearsky(33.640, -84.430, years=[2011, 2015], noon_flux=True)
    print(tau)

    # Produces the average monthly (and annual) daily-average all sky radiation
    # for every requested year
    rad = C.to_allsky(lat=33.640, lon=-84.430, years=range(2011, 2015))

    # Which again can be massaged into the required statistics (mean, std)
    print(rad.describe().round(decimals=1))

    # Extract the solar components
    solar = C.solar_split(33.640, -84.430, years=[2018])
    print(solar[12:24])


if __name__ == "__main__":
    main()
