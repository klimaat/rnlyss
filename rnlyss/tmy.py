#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Routines to determine a Typical Meteorological Year (TMY)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rnlyss.dataset import load_dataset
from rnlyss.psychro import calc_relative_humidity

_METHODS = {
    "ISO_15927": {"DB_Avg": 1, "RH_Avg": 1, "Et_Avg": 1},
    "Sandia": {
        "DB_Max": 1,
        "DB_Min": 1,
        "DB_Avg": 2,
        "DP_Max": 1,
        "DP_Min": 1,
        "DP_Avg": 2,
        "WS_Max": 2,
        "WS_Avg": 2,
        "Et_Avg": 12,
    },
    "TMY3": {
        "DB_Max": 1,
        "DB_Min": 1,
        "DB_Avg": 2,
        "DP_Max": 1,
        "DP_Min": 1,
        "DP_Avg": 2,
        "WS_Max": 1,
        "WS_Avg": 1,
        "Et_Avg": 5,
        "Eb_Avg": 5,
    },
    "IWEC": {
        "DB_Max": 5,
        "DB_Min": 5,
        "DB_Avg": 30,
        "DP_Max": 2.5,
        "DP_Min": 2.5,
        "DP_Avg": 5,
        "WS_Max": 5,
        "WS_Avg": 5,
        "Et_Avg": 40,
    },
}


_OPS = {
    "Avg": np.mean,
    "Min": np.min,
    "Max": np.max,
    "Rng": np.ptp,
    "Sum": np.mean,
}


def cdf_distance(u, v, p=1, norm=True):
    """
    Calculate pth Wasserstein Distance representing the normed area between the cdfs of u & v.

    If norm is True, normalize by the range of data to force an area between 0 and 1.
    """

    u = np.asarray(u, dtype=float)
    u_sorter = np.argsort(u)

    v = np.asarray(v, dtype=float)
    v_sorter = np.argsort(v)

    uv = np.concatenate((u, v))
    uv.sort(kind="mergesort")

    scale = uv.max() - uv.min() if norm else 1

    if scale == 0:
        return 1.0

    duv = np.diff(uv) / scale

    u_cdf = np.searchsorted(u, uv[:-1], side="right", sorter=u_sorter) / len(u)
    v_cdf = np.searchsorted(v, uv[:-1], side="right", sorter=v_sorter) / len(v)

    if p == 1:
        # First Wasserstein Distance
        return np.sum(np.multiply(np.abs(u_cdf - v_cdf), duv))

    if p == 2:
        # Energy Distance
        return np.sqrt(np.sum(np.multiply(np.square(u_cdf - v_cdf), duv)))

    # p-Wasserstein Distance
    return np.power(np.sum(np.multiply(np.power(np.abs(u_cdf - v_cdf), p), duv)), 1 / p)


def tmy(
    dsets=["MERRA2"],
    years=None,
    lat=0,
    lon=0,
    hgt=0,
    weights=None,
    method=None,
    full_output=False,
):
    """
    Given dsets, a list of potential datasets to use, and years to choose from,
    generate a list of 12 years, representing a typical Jan, Feb, ..., Dec.
    """

    # Default weights
    if weights is None:
        if method is None:
            method = "ISO_15927"
        weights = _METHODS[method]

    # Normalize weights
    sum_weights = sum(weights.values())
    weights = {k: v / sum_weights for k, v in weights.items()}

    # Instance the requested datasets (e.g. CFSR, CFSv2 and/or MERRA-2)
    dset_names = [dset.upper() for dset in dsets]
    dsets = [load_dataset(dset) for dset in dset_names]

    # Figure out required variables
    quants = list(set([x.split("_")[0] for x in weights]))

    # Figure out required operations
    ops = list(set([x.split("_")[1] for x in weights]))
    if any(x not in _OPS for x in ops):
        raise ValueError("Only %r operations allowed" % list(_OPS.keys()))

    # Figure out clean years
    years = sorted(sum([list(dset.iter_year(years=years)) for dset in dsets], []))
    years = list(set(years))

    # Need at least *some* years
    if len(years) == 0:
        raise ValueError("None of the requested years is available in the dataset")

    # Get SP, DB, DP, WS, SD as a minimum
    df = pd.concat([dset.to_hof(lat, lon, hgt=hgt, years=years) for dset in dsets])

    # Get shortwave fluxes if necessary
    if any(x in quants for x in ["Et", "Eb", "Ed"]):
        sw = pd.concat([dset.solar_split(lat, lon, years=years) for dset in dsets])
        df = pd.concat([df, sw], axis=1)

    # Add RH (%) if necessary
    if "RH" in quants:
        df["RH"] = 100 * calc_relative_humidity(df["DB"], df["DP"])

    # Loop over each quantity
    dist_ym = {}

    months = range(1, 13)

    for key, weight in weights.items():

        quant, op = key.split("_")

        # Default distance is 1.0 (i.e. maximum)
        dist_ym[key] = pd.DataFrame(1.0, index=years, columns=months)

        # Resample hourly data to daily applying operation
        daily = df[quant].resample("D").agg(_OPS[op]).dropna()

        # Groupby month
        for m, monthly in daily.groupby(daily.index.month):
            # Groupby year within month
            for y, yearly in monthly.groupby(monthly.index.year):
                if y in years:
                    dist_ym[key].at[y, m] = cdf_distance(monthly, yearly)

    # Sum weighted distances
    dist_ym["Total"] = pd.DataFrame(0.0, index=years, columns=months)
    for key, weight in weights.items():
        for m in months:
            for y in years:
                dist_ym["Total"].at[y, m] += dist_ym[key].at[y, m] * weight

    # Minimum distance score
    years = list(dist_ym["Total"].idxmin())

    # Check for scores == 1
    for m, y in zip(months, years):
        if dist_ym["Total"].at[y, m] == 1:
            years[m - 1] = None

    if full_output:
        # Return distance matrices
        return years, dist_ym

    return years


def main():

    # Atlanta
    loc = {"lat": 33.640, "lon": -84.430, "hgt": 313}
    years, dist_ym = tmy(
        dsets=["MERRA2"],
        method="ISO_15927",
        full_output=True,
        years=range(2000, 2021),
        **loc
    )
    pd.set_option("display.width", 10000)
    for k, df in dist_ym.items():
        print(k)
        print(df)

    print("Typical month/years:")
    print(years)


if __name__ == "__main__":
    main()
