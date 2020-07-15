# -*- coding: utf-8 -*-
#
# Copyright 2019 Klimaat

"""
Input/Output routines
"""

import calendar
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rnlyss.psychro import calc_relative_humidity
from rnlyss.solar import perez
from rnlyss.ground import calc_ground_temperatures


def is_leap(year):
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def epw_years(epw):
    """
    Given an EPW file, determine the twelve years used for each month of the
    typical year.
    """

    # Load year and month
    df = pd.read_csv(
        epw, header=None, usecols=[0, 1], dtype=int, skiprows=8, names=["Y", "M"]
    )

    # Drop all duplicate months leaving one row per month
    df = df.drop_duplicates(subset=["M"], keep="first")

    # Pull out the remaining years
    return df["Y"].tolist()


def epw_header(epw):
    """
    Given an EPW, get the header
    """

    with open(epw, "r") as f:
        header = f.readline().strip().split(",")

    return {
        "city": header[1],
        "state": header[2],
        "country": header[3],
        "source": header[4],
        "WMO": header[5],
        "lat": float(header[6]),
        "lon": float(header[7]),
        "tz": float(header[8]),
        "hgt": float(header[9]),
    }


def epw_col(epw, col=6):
    """
    Load a column of an EPW into a DataFrame, default is dry bulb.

    0 Year,
    1 Month,
    2 Day,
    3 Hour,
    4 Minute,
    5 Data Source and Uncertainty Flags,
    6 Dry Bulb,
    7 Dew Point,
    8 RH,
    9 Atmospheric Station Pressure,
    10 Extraterrestrial Horizontal Radiation,
    11 Extraterrestrial Direct Normal Radiation,
    12 Horizontal Infrared Radiation from Sky,
    13 Global Horizontal Radiation,
    14 Direct Normal Radiation,
    15 Diffuse Horizontal Radiation,
    16 Global Horizontal Illuminance,
    17 Direct Normal Illuminance,
    18 Diffuse Horizontal Illuminance,
    19 Zenith Luminance,
    20 Wind Direction,
    21 Wind Speed,
    22 Total Sky Cover,
    23 Opaque Sky Cover,
    24 Visibility,
    25 Ceiling Height,
    26 Present Weather Observation,
    27 Present Weather Codes,
    28 Precipitable Water,
    29 Aerosol Optical Depth,
    30 Snow Depth,
    31 Days Since Last Snowfall.
    """

    # Load year, month, day, hour and col
    df = pd.read_csv(
        epw,
        header=None,
        usecols=[0, 1, 2, 3, col],
        skiprows=8,
        names=["Y", "M", "D", "H", "X"],
    )

    def join_date(y=1970, m=1, d=1, hh=0, mm=0, ss=0):
        """
        Join date/time components into datetime64 object
        """
        y = (np.asarray(y) - 1970).astype("<M8[Y]")
        m = (np.asarray(m) - 1).astype("<m8[M]")
        d = (np.asarray(d) - 1).astype("<m8[D]")
        hh = np.asarray(hh).astype("<m8[h]")
        mm = np.asarray(mm).astype("<m8[m]")
        ss = np.asarray(ss).astype("<m8[s]")
        return y + m + d + hh + mm + ss

    df["Date"] = join_date(y=df["Y"], m=df["M"], d=df["D"], hh=df["H"] - 1, mm=30)

    return df.set_index("Date").drop(["Y", "M", "D", "H"], axis=1)


def blend_months(df):
    def unwrap(x1, x2):
        return x1 + (x2 - x1 + 180) % 360 - 180

    def blend(i, j, wi, wj):
        # Unwrap wind
        df.iat[j, colWD] = unwrap(df.iat[i, colWD], df.iat[j, colWD])
        df.iloc[i] = wi * df.iloc[i] + wj * df.iloc[j]
        df.iat[i, colWD] %= 360
        df.iloc[j] = np.nan

    # Find where the year changes
    rows = np.where(df.index.year[:-1] != df.index.year[1:])[0] + 1

    colWD = df.columns.get_loc("WD")

    for row0 in rows[1:]:

        # Index of this row
        t0 = df.index[row0]

        if df.index.hour[row0] == 0:
            # Midnight:  Year-end linear between Dec & Jan
            for i in range(12):
                w = (i + 0.5) / 12
                if i >= 6:
                    blend(i, row0 + i - 5, w, 1 - w)
                else:
                    blend(row0 + i - 5, i, 1 - w, w)

        elif df.index.hour[row0] == 19:
            # Midyear linear blending
            for i in range(12):
                w = (i + 0.5) / 12
                if i >= 6:
                    blend(row0 + i, row0 + i - 12, w, 1 - w)
                else:
                    blend(row0 + i - 12, row0 + i, 1 - w, w)

        else:
            raise ValueError("Year change at unplanned hour %r" % t0)

    # Get rid of empty rows
    return df.dropna()


def write_solar_csv(path, dsets, years, lat=0, lon=0, hgt=0, tz=0, **kwargs):

    # Instance the requested datasets (e.g. CFSR, CFSv2 and/or MERRA-2)
    dset_names = [dset.upper() for dset in dsets]
    dsets = [load_dataset(dset) for dset in dset_names]

    # Get unique years to retrieve

    # Need to prepend first year's December and last year's January
    data_years = [years[0] - 1] + list(years) + [years[-1] + 1]

    # Get unique years req'd
    data_years = sorted(list(set(data_years)))

    # Check that these years are possible
    for year in data_years:
        for dset in dsets:
            if dset.isvalid(year):
                break
        else:
            raise ValueError("Required year %d not in %r" % (year, dset_names))

    # Get shortwave fluxes
    df = pd.concat([dset.solar_split(lat, lon, years=data_years) for dset in dsets])

    # Get longwave
    lw = pd.concat(
        [dset("rlds", lat, lon, hgt=hgt, years=data_years) for dset in dsets]
    )
    if lw is not None:
        df["LW"] = lw

    # Shift to local time
    df = df.shift(int(np.rint(tz * 60)), "min")

    # Only save years requested
    i = np.in1d(df.index.year, years, assume_unique=True)
    df[i].to_csv(path, float_format="%.0f")


def write_epw(path, dsets, years, lat=0, lon=0, hgt=0, tz=0, **kwargs):
    """
    Write an EPW file, given: a path; dsets, a list of potential datasets to
    use; years, a list of the 12 years to use for Jan, Feb, Mar, ..., Dec;
    geographical location (lat, lon, hgt); time zone offset; and additional
    kwargs metadata like city, state, country, and WMO.
    """

    # Allow passing a single instance
    if not isinstance(dsets, (list, tuple)):
        dsets = [dsets]

    dset_names = [str(dset) for dset in dsets]

    # Need an iterable of length 12
    if len(years) != 12:
        raise ValueError("Must specify list of 12 years")

    # Get unique years to retrieve
    if tz > 0:
        # Need to append each year's next year
        data_years = [year + i for year in years for i in [0, 1]]
    elif tz < 0:
        # Need to prepend each year's previous year
        data_years = [year + i for year in years for i in [-1, 0]]
    else:
        data_years = list(years)

    # Get unique years req'd
    data_years = sorted(list(set(data_years)))

    # Check that these years are possible
    for year in data_years:
        for dset in dsets:
            if dset.isvalid(year):
                break
        else:
            raise ValueError("Required year %d not in %r" % (year, dset_names))

    # Get HOF dataframe for pressure, dry bulb, dew point, wind speed
    # and wind direction
    df = pd.concat([dset.to_hof(lat, lon, hgt=hgt, years=data_years) for dset in dsets])

    # Get shortwave fluxes
    sw = pd.concat([dset.solar_split(lat, lon, years=data_years) for dset in dsets])
    df = pd.concat([df, sw], axis=1)

    # Get longwave
    lw = pd.concat(
        [dset("rlds", lat, lon, hgt=hgt, years=data_years) for dset in dsets]
    )
    if lw is not None:
        df["LW"] = lw

    # Get preciptable water
    pwat = pd.concat(
        [dset("pwat", lat, lon, hgt=hgt, years=data_years) for dset in dsets]
    )
    if pwat is not None:
        df["PWat"] = pwat

    # Get cloud fraction, limit to 0, 1 (thank you again CFSR)
    clt = pd.concat(
        [dset("clt", lat, lon, hgt=hgt, years=data_years) for dset in dsets]
    )

    if clt is not None:
        clt = np.minimum(np.maximum(clt, 0), 1)
        # Convert to cloud cover in tenths, limiting to [0,10]
        df["TCC"] = 10 * clt

    # Get precipitation rate
    pr = pd.concat([dset("pr", lat, lon, hgt=hgt, years=data_years) for dset in dsets])
    if pr is not None:
        # Convert to mm
        df["Pr"] = 3600 * np.maximum(pr, 0)

    # Get aerosol optical depth
    aod = pd.concat(
        [dset("aod550", lat, lon, hgt=hgt, years=data_years) for dset in dsets]
    )
    if aod is not None:
        df["AOD"] = aod

    # Get albedo
    albedo = pd.concat(
        [dset("albedo", lat, lon, hgt=hgt, years=data_years) for dset in dsets]
    )
    if aod is not None:
        df["Albedo"] = albedo

    # Shift to local time
    df = df.shift(int(np.rint(tz * 60)), "min")

    # Loop over the months and create a time index for each month adding
    # 6 hours to the beginning and end of each month for blending
    ts = []
    for i in range(12):

        month = i + 1
        year = years[i]

        # Number of days in this month
        n_days = calendar.monthrange(year, month)[1]

        # Requested month
        t0 = np.datetime64("%04d-%02d-01T01" % (year, month))

        # Expanded window
        w = 6
        dt = np.timedelta64(w, "h")

        ts.append(
            pd.Series(
                index=pd.date_range(
                    start=t0 - dt, periods=24 * n_days + 2 * w, freq="h"
                ),
                data=month,
                name="Month",
            )
        )

    # Concatenate months
    ts = pd.concat(ts)

    # Get rid of the duplicates to eliminate any unnecessary 6 hours
    # when the years are identical
    ts = ts[~ts.index.duplicated(keep="last")]

    # In preparation for interpolation, need to unwrap the wind direction
    i = df["WD"].notnull()
    df.loc[i, "WD"] = np.degrees(np.unwrap(np.radians(df.loc[i, "WD"])))

    # Insert this time index into df, sort the time index, interpolate, and
    # select only the required months
    df = df.join(ts, how="outer").sort_index().interpolate(method="values")
    df = df[~df.index.duplicated(keep="last")].reindex(ts.index)

    # Map wind back into 0-360
    df["WD"] %= 360

    # Blend months
    df = blend_months(df)

    # Add RH (%) based on values
    df["RH"] = 100 * calc_relative_humidity(df["DB"], df["DP"])

    # Add illuminance values
    df["It"], df["Ib"], df["Id"], df["Lz"] = perez(
        df["Eb"], df["Ed"], df["E0"], df["E0h"], df["DP"]
    )

    # Add weather observations
    if "Pr" in df.columns:
        # Have a weather observation
        df["WObs"] = 0
        # Assume clear (Pr = 0)
        df["WCode"] = 999999999
        # Light rain  0 < Pr <= 2.5mm/hr
        df.loc[(df["Pr"] > 0.05) & (df["Pr"] <= 2.5), "WCode"] = 909999999
        # Moderate rain 2.5 < Pr <= 7.6/hr
        df.loc[(df["Pr"] > 2.5) & (df["Pr"] <= 7.6), "WCode"] = 919999999
        # Heavy rain Pr > 7.6 mm/hr
        df.loc[df["Pr"] > 7.6, "WCode"] = 929999999

    # Round wind direction to nearest 10deg
    df["WD"] = np.around(df["WD"], -1)

    # Degree zero should be calm; throw non-calm into 360
    i = (df["WD"] == 0) & (df["WS"] >= 0.3)
    df.loc[i, "WD"] = 360

    # And calm wind should be in zero deg
    i = df["WS"] < 0.3
    df.loc[i, "WD"] = 0

    # Build up source, comments, and snapped lat, lon, hgt for writing
    # Determine which dsets were used and corresponding (lat, lon, hgt)
    snaps = {}

    # Get location
    for name, dset in zip(dset_names, dsets):
        dlat, dlon = dset.snap(lat, lon)
        dhgt = dset.hgt(lat, lon)
        dland = dset.land(lat, lon)
        snaps[name] = {
            "lat": dlat,
            "lon": dlon,
            "hgt": dhgt,
            "land": dland,
            "years": [],
        }

    # Tack on years
    for year in years:
        for name, dset in zip(dset_names, dsets):
            if dset.isvalid(year):
                snaps[name]["years"].append(year)
                break

    # Find most common dataset
    name_max = None
    years_max = 0
    comments = []
    sources = []
    for name in snaps:
        n_years = len(snaps[name]["years"])

        if n_years == 0:
            continue

        sources.append(name)
        comments.append("%s %r" % (name, sorted(list(set(snaps[name]["years"])))))

        if n_years > years_max:
            name_max = name
            years_max = n_years

    # Overwrite lat, lon, & hgt with most common dataset
    lat = snaps[name_max]["lat"]
    lon = snaps[name_max]["lon"]
    hgt = snaps[name_max]["hgt"]
    land = snaps[name_max]["land"]

    # EPWs have hour=1..24, we have hour=1..23,00
    # -> subtract 30min from index
    df.index -= pd.Timedelta(minutes=30)

    df["Year"] = df.index.year
    df["Month"] = df.index.month
    df["Day"] = df.index.day
    df["Hour"] = df.index.hour + 1

    # Annual average temperature
    T_mean = df["DB"].mean()

    # Monthly average temperatures
    T_mth_avg = df["DB"].resample("M").mean()

    # Low and high monthly temperatures and coldest month
    T_min = T_mth_avg.min()
    T_max = T_mth_avg.max()
    coldest_month = T_mth_avg.idxmin().month

    # Calc ground temperatures
    depths = [0.5, 2, 4]
    diffusivity = 2.3225760e-3
    T_gnd = calc_ground_temperatures(
        T_mean,
        T_min,
        T_max,
        coldest_month,
        depth=depths,
        leap_year=is_leap(years[1]),
        diffusivity=diffusivity,
    )

    # Build ground temperature string
    ground = ["%d" % len(depths)]

    for i, depth in enumerate(depths):
        # Ground depth (m)
        ground.append("%.1f" % depth)
        # Soil conductivity, density, and specific heat left blank
        ground.extend(["", "", ""])
        # Add temperatures (C)
        ground.extend(["%.2f" % tmp for tmp in T_gnd[i]])

    ground = ",".join(ground)

    fmts = [
        ("Year", None, None),
        ("Month", None, None),
        ("Day", None, None),
        ("Hour", None, None),
        ("Minute", None, 60),
        ("Flags", None, "*"),
        ("DB", 1, None),
        ("DP", 1, None),
        ("RH", 0, None),
        ("SP", 0, None),
        ("E0h", 0, None),
        ("E0", 0, None),
        ("LW", 0, None),
        ("Et", 0, None),
        ("Eb", 0, None),
        ("Ed", 0, None),
        ("It", 0, 999999),
        ("Ib", 0, 999999),
        ("Id", 0, 999999),
        ("Lz", 0, 9999),
        ("WD", 0, None),
        ("WS", 1, None),
        ("TCC", 0, 99),
        ("OCC", 0, 99),
        ("Viz", 0, 9999),
        ("Ceiling", 0, 99999),
        ("WObs", 0, 9),
        ("WCode", 0, 999999999),
        ("PWat", 1, 999),
        ("AOD", 3, 999),
        ("Snow", 0, 999),
        ("Days", 0, 99),
        ("Albedo", 3, 999),
        ("Pr", 1, 0),
        ("PrHr", 1, 1),
    ]

    # Add columns that don't exist

    for col, prec, default in fmts:
        if col in df.columns:
            # Clean-up
            if prec is not None:
                df[col] = df[col].map(
                    lambda x: "{val:.{prec}f}".format(val=x, prec=prec)
                )
        else:
            if default is None:
                raise ValueError("Need a value for {0}".format(col))
            if prec is None:
                df[col] = default
            else:
                df[col] = "{val:.{prec}f}".format(val=default, prec=prec)

    cols = [fmt[0] for fmt in fmts]

    with open(path, "w") as epw:

        # Location
        location = [
            "LOCATION",
            kwargs.get("city", "N/A"),
            kwargs.get("state", "N/A"),
            kwargs.get("country", "N/A"),
            "/".join(sources),
            str(kwargs.get("WMO", "999999")),
            "%.3f" % lat,
            "%.3f" % lon,
            "%.2f" % tz,
            "%.1f" % hgt,
        ]
        epw.write(",".join(location) + "\n")

        # Design conditions
        epw.write("DESIGN CONDITIONS,0\n")

        # Typical/extreme periods
        epw.write("TYPICAL/EXTREME PERIODS,0\n")

        # Ground temperatures
        epw.write("GROUND TEMPERATURES,%s\n" % ground)

        # Leap year (Feb)
        if is_leap(int(years[1])):
            epw.write("HOLIDAYS/DAYLIGHT SAVINGS,Yes,0,0,0\n")
        else:
            epw.write("HOLIDAYS/DAYLIGHT SAVINGS,No,0,0,0\n")

        # Comments
        epw.write(
            "COMMENTS 1, Compiled from %s reanalysis data; %d%% land\n"
            % (" and ".join(comments), int(np.rint(100 * land)))
        )
        epw.write(
            "COMMENTS 2, Ground temps produced with a standard soil"
            "diffusivity of 2.3225760E-03 {m**2/day}\n"
        )
        epw.write("DATA PERIODS,1,1,Data,Sunday, 1/ 1,12/31\n")

        # Write data
        df.to_csv(epw, header=False, index=False, columns=cols)


def main():

    from rnlyss.dataset import load_dataset

    # Instance the requested datasets (e.g. CFSR, CFSv2 and/or MERRA-2)
    dsets = [load_dataset("MERRA2")]

    # Create an Atlanta EPW from MERRA2 for months chosen
    # from selected years
    years = [2011, 2008, 2005, 2005, 2012, 1990, 2003, 1985, 2017, 1998, 1998, 1981]
    meta = {"city": "Atlanta", "state": "GA", "country": "USA", "tz": -5}
    loc = {"lat": 33.640, "lon": -84.430, "hgt": 313}

    write_epw("Atlanta.epw", dsets=dsets, years=years, **meta, **loc)


if __name__ == "__main__":
    main()
