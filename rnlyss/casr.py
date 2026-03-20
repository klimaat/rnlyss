#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Canadian Surface Reanalysis (CaSR) v3.2
https://hpfx.collab.science.gc.ca/~scar700/rcas-casr/index.html
~10km resolution from 1980--2024
"""

from pathlib import Path
import numpy as np
import pandas as pd
import netCDF4

from rnlyss.dataset import Dataset, TiledDataset
from rnlyss.grid import RotatedGrid
from rnlyss.util import create_session, download_file


class CaSR_v32_Forecast(TiledDataset):
    """
    Forecast variables (1-hourly)
    """

    freq = 1

    dvars = {
        # Dry-bulb temperature; convert from °C to K
        "P_TT_1.5m": {
            "role": "tas",
            "scale": 1e-2,
            "offset": 330,
            "converter": lambda x: x + 273.15,
        },
        # Dew-point temperature; convert from °C to K
        "P_TD_1.5m": {
            "role": "tdps",
            "scale": 1e-2,
            "offset": 330,
            "converter": lambda x: x + 273.15,
        },
        # Surface pressure; convert from mb to Pa
        "P_P0_SFC": {
            "role": "ps",
            "scale": 1,
            "offset": 75000,
            "converter": lambda x: lambda x: x * 100,
        },
        # Geopotential; convert from dam to m
        "P_GZ_SFC": {
            "role": "hgt",
            "scale": 1,
            "constant": True,
            "converter": lambda x: x * 10,
        },
        # Zonal wind (east-west) @ 10m; convert from kt to m/s
        "P_UUC_10m": {
            "role": "uas",
            "scale": 1e-2,
            "offset": 0,
            "converter": lambda x: x * 1852 / 3600,
        },
        # Meridional wind (north-south) @ 10m; convert from kt to m/s
        "P_VVC_10m": {
            "role": "vas",
            "scale": 1e-2,
            "offset": 0,
            "converter": lambda x: x * 1852 / 3600,
        },
        # Shortwave downwelling flux @ surface; W/m²
        "P_FB_SFC": {
            "role": "rsds",
            "scale": 0.1,
            "offset": 0,
        },
        # Longwave downwelling flux @ surface; W/m²
        "P_FI_SFC": {
            "role": "rlds",
            "scale": 0.1,
            "offset": 0,
        },
    }

    # Time (last day of 1979 and first day of 2025)
    years = [1979, 2024]

    # Overall grid
    grid = RotatedGrid(
        shape=(778, 706),
        origin=(-44.1, -35.397217),
        delta=(0.09, 0.09),
        latp=31.758312454493154,
        lonp=87.59703130293302,
        r=6370997,
    )

    def __getitem__(self, args):
        i, j = args
        return CaSR_v32_Tile(self, i - i % 35, j - j % 35)

    def __call__(self, lat, lon):
        i, j = self.grid(lat, lon, snap=True)
        return self[i, j]


class CaSR_v32_Analysis(CaSR_v32_Forecast):
    """
    Analysis variables (3-hourly)
    """

    freq = 3

    dvars = {
        # Dry-bulb temperature; convert from °C to K
        "A_TT_1.5m": {
            "role": "tas",
            "scale": 1e-2,
            "offset": 330,
            "converter": lambda x: x + 273.15,
        },
        # Dew-point temperature; convert from °C to K
        "A_TD_1.5m": {
            "role": "tdps",
            "scale": 1e-2,
            "offset": 330,
            "converter": lambda x: x + 273.15,
        },
        # Geopotential; convert from dam to m
        "P_GZ_SFC": {
            "role": "hgt",
            "scale": 1,
            "constant": True,
            "converter": lambda x: x * 10,
        },
    }


class CaSR_v32_Tile(Dataset):
    """
    Tiled dataset, nominally shape=35×35
    """
    periods = [f"{_:04d}-{min(_+3, 2024):04d}" for _ in range(1980, 2025, 4)]

    def __init__(self, parent, tile_i, tile_j):
        self.dset = "CaSR_v3.2"

        # Final tiles are slightly larger
        if tile_i == 735:
            if tile_j == 665:
                shape = 43, 41
            else:
                shape = 43, 35
        else:
            if tile_j == 665:
                shape = 35, 41
            else:
                shape = 35, 35

        self.name = "_".join(
            [
                f"rlon{tile_j+1:03d}-{tile_j+shape[1]:03d}",
                f"rlat{tile_i+1:03d}-{tile_i+shape[0]:03d}",
            ]
        )

        self.tile = tile_i, tile_j
        x0, y0 = parent.grid.ij2xy(tile_i, tile_j)
        self.grid = RotatedGrid(
            shape=shape,
            origin=(y0, x0),
            delta=parent.grid.delta,
            latp=parent.grid.latp,
            lonp=parent.grid.lonp,
            r=parent.grid.r,
        )
        self.dvars = parent.dvars
        self.years = parent.years
        self.freq = parent.freq

        super().__init__(data_dir=Path(parent.data_dir) / Path(self.name))

    def __str__(self):
        return self.name

    def download(self, dvars=None, years=None, force=False, **kwargs):
        """
        Download CaSR files
        """
        if dvars is None:
            dvars = list(self.dvars.keys())

        if not isinstance(dvars, list):
            dvars = [dvars]

        session = create_session()

        root_url = (
            r"https://hpfx.collab.science.gc.ca/"
            "~scar700/rcas-casr/data/CaSRv3.2/netcdf_tile/"
        )
        for dvar in sorted(dvars):
            if dvar not in self:
                print("%s not in dataset... skipping" % dvar)
                continue

            print(dvar)

            if self.isconstant(dvar):
                # Download last period (2024; smallest download)
                url = (
                    f"{root_url}/{self.name}/"
                    f"{self.dset}_{dvar}_{self.name}_{self.periods[-1]}.nc"
                )
                dst = self.get_data_path("nc", f"{dvar}.nc")
                download_file(url, dst, session=session)
            else:
                # Download all periods
                for period in self.periods:
                    url = (
                        f"{root_url}/{self.name}/"
                        f"CaSR_v3.2_{dvar}_{self.name}_{period}.nc"
                    )
                    dst = self.get_data_path("nc", f"{dvar}_{period}.nc")
                    download_file(url, dst, session=session)

    def stack(self, dvars=None, years=None, force=False, **kwargs):
        """
        Fill element HDF with available GRB data
        """
        if dvars is None:
            dvars = list(self.dvars.keys())

        if not isinstance(dvars, list):
            dvars = [dvars]

        for dvar in sorted(dvars):
            if dvar not in self:
                print("%s not in dataset... skipping" % dvar)
                continue

            print(dvar)

            # Get converter
            converter = self.dvars[dvar].get("converter", None)

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

                    nc_path = Path(self.get_data_path("nc", f"{dvar}.nc"))

                    if not nc_path.is_file():
                        print(dvar, "missing... skipping")
                        continue

                    print(dvar, "writing...", end="")
                    with netCDF4.Dataset(nc_path) as nc:
                        dset_dvar = f"{self.dset}_{dvar}"
                        x = np.expand_dims(nc[dset_dvar][0, :], axis=-1)
                    slab.fill(0, slab.to_int(x, converter))
                    print("done")
                continue
            else:
                # Grab all the files available
                nc_paths = Path(self.get_data_path("nc")).glob(f"*{dvar}*.nc")
                # print(list(nc_paths))

                dset_dvar = f"{self.dset}_{dvar}"
                for nc_path in sorted(nc_paths):
                    with netCDF4.Dataset(nc_path) as nc:
                        X = nc.variables[dset_dvar][:]
                        if isinstance(X, np.ma.MaskedArray):
                            X = np.array(X.filled(fill_value=np.nan))
                        t = pd.DatetimeIndex(
                            netCDF4.num2date(
                                nc["time"][:],
                                units=nc["time"].units,
                                calendar=nc["time"].calendar,
                                only_use_cftime_datetimes=False,
                                only_use_python_datetimes=True,
                            )
                        )

                    # Analysis is 3-hourly; cull
                    if self.freq == 3:
                        t = t[2::3]
                        X = X[2::3, ...]

                    # Each dataset has 12 hours from a previous year
                    years = np.unique(t.year)

                    for year in years:
                        with self[dvar, year] as slab:
                            # Create as necessary
                            if slab:
                                print(dvar, year, "exists... updating... ", end="")
                            else:
                                print(
                                    dvar, year, "does not exist... creating...", end=""
                                )
                                slab.create(
                                    shape=self.grid.shape,
                                    year=year,
                                    freq=self.freq,
                                    **self.dvars[dvar],
                                )

                            # Insertion point for this slab
                            i = slab.time2ind(t[t.year == year][0])

                            # Indices of array x
                            j = np.nonzero(t.year == year)[0]
                            nh = len(j)

                            # Check if already filled with this dataset
                            if slab.isfull(np.s_[i : (i + nh)]) and not force:
                                print("already stacked... skipping")
                                continue

                            print(f"writing {nh} hours... ", end="")
                            slab.fill(
                                i,
                                np.transpose(
                                    slab.to_int(X[j, ...], converter=converter),
                                    (1, 2, 0),
                                ),
                            )
                            print("done")


def main():
    import calendar
    import matplotlib.pyplot as plt
    from rnlyss.dataset import load_dataset

    _, ax = plt.subplots(dpi=200)

    name, lat, lon = "St. John's", 47.62, -52.7494
    # name, lat, lon = "Guelph", 43.5448, -80.2482

    month = 6

    F = CaSR_v32_Forecast()
    tile = F(lat, lon)
    # tile.download(["P_TT_1.5m", "P_GZ_SFC"])
    # tile.stack(["P_TT_1.5m", "P_GZ_SFC"])
    h = tile.hgt(lat, lon)
    x = tile("tas", lat, lon).dropna()
    y = x.loc[(x.index.month == month)]
    z = y.groupby(y.index.hour).mean()
    z[24] = z[0]
    print(z)
    ax.plot(
        z.index, z - 273.15, "o-", markerfacecolor="none", label=f"casr forecast {h=}m"
    )

    A = CaSR_v32_Analysis()
    tile = A(lat, lon)
    # tile.download(["A_TT_1.5m", "P_GZ_SFC"])
    # tile.stack(["A_TT_1.5m", "P_GZ_SFC"])
    h = tile.hgt(lat, lon)
    x = tile("tas", lat, lon).dropna()
    y = x.loc[(x.index.month == month)]
    z = y.groupby(y.index.hour).mean()
    z[24] = z[0]
    print(z)
    ax.plot(
        z.index, z - 273.15, "o-", markerfacecolor="none", label=f"casr analysis {h=}m"
    )

    N = load_dataset("narr")
    h = N.hgt(lat, lon)
    x = N("tas", lat, lon).dropna()
    y = x.loc[(x.index.month == month)]
    z = y.groupby(y.index.hour).mean()
    z[24] = z[0]
    ax.plot(z.index, z - 273.15, "o-", markerfacecolor="none", label=f"narr {h=}")

    M = load_dataset("merra2")
    h = M.hgt(lat, lon)
    x = M("tas", lat, lon).dropna()
    y = x.loc[(x.index.month == month)]
    z = y.groupby(y.index.hour).mean()
    z[24] = z[0]
    ax.plot(z.index, z - 273.15, "o-", markerfacecolor="none", label=f"merra2 {h=}")

    C = load_dataset("era5land")
    h = C.hgt(lat, lon)
    x = C("tas", lat, lon).dropna()
    y = x.loc[(x.index.month == month)]
    z = y.groupby(y.index.hour).mean()
    z[24] = z[0]
    ax.plot(z.index, z - 273.15, "o-", markerfacecolor="none", label=f"era5land {h=}")
    ax.legend()

    E5 = load_dataset("era5")
    h = E5.hgt(lat, lon)
    x = E5("tas", lat, lon).dropna()
    y = x.loc[(x.index.month == month)]
    z = y.groupby(y.index.hour).mean()
    z[24] = z[0]
    ax.plot(z.index, z - 273.15, "o-", markerfacecolor="none", label=f"era5 {h=}")
    ax.legend()

    ax.set_xlabel("UTC Hour")
    ax.set_ylabel("Temperature °C")
    ax.set_title(f"{name} {calendar.month_abbr[month]} {lat:.2f}N {abs(lon):.2f}W ")
    plt.savefig(f"{name}_{calendar.month_abbr[month]}.png", bbox_inches="tight")


if __name__ == "__main__":
    main()
