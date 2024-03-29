# -*- coding: utf-8 -*-
#
# Copyright 2019 Klimaat

import os
import datetime
import numpy as np
import pandas as pd
import importlib
import configparser
from scipy.spatial import KDTree
from scipy.ndimage import map_coordinates

from rnlyss.hyperslab import HyperSlab
from rnlyss.grid import Grid
from rnlyss.sphere import to_cartesian
from rnlyss.humidity import calc_dp_from_q_and_p

# Register possible dataset sources
DATASETS = ["CFSv2", "CFSR", "MERRA2", "ERA5", "ERA5Land", "NARR"]


def import_module(dset, root="rnlyss"):
    """
    Import library for dset
    """
    try:
        return importlib.import_module("%s.%s" % (root.lower(), dset.lower()))
    except ImportError:
        raise ValueError("Specify one of %r" % (DATASETS))


def match_class(dset):
    """
    Search for matching class (case-insensitive) given dataset
    """
    for cls_ in DATASETS:
        if cls_.upper() == dset.upper():
            return cls_
    raise ValueError("Specify one of %r" % DATASETS)


def import_class(dset):
    """
    Import corresponding class for dset
    """
    mod = import_module(dset)
    cls_ = match_class(dset)
    return getattr(mod, cls_)


def load_dataset(dset, **kwargs):
    """
    Load a reanalsysis source by name. e.g. "CFSv2" via magic.
    """
    cls_ = import_class(dset)
    return cls_(**kwargs)


def get_config_dir(config_file, dset):
    config = configparser.ConfigParser()
    config.read(config_file)
    data_dir = config.get("Data", dset, fallback=None)
    if data_dir is None:
        return os.path.join(os.path.expanduser("~"), dset)
    else:
        return os.path.normpath(os.path.join(data_dir, dset))


class Dataset(object):
    """
    Base class for gridded datasets e.g. CFSR.
    Point to a set of 3D HDF5 hyperslabs.
    """

    # Dataset variables
    dvars = {}

    # Time
    years = [1970, None]

    # Hour frequency
    freq = 1

    # Grid
    grid = Grid()

    def __init__(self, data_dir=None, **kwargs):
        # Set data_dir
        self.set_data_dir(data_dir)

        # Ensure we have a valid end year
        self.now = datetime.datetime.now()
        if self.years[1] is None:
            self.years[1] = self.now.year

        # For tree lookups
        self.land_indices = None
        self.land_tree = None

    def stack(self, **kwargs):
        raise NotImplementedError

    def download(self, **kwargs):
        raise NotImplementedError

    def __str__(self):
        return self.__class__.__name__

    def set_data_dir(self, data_dir=None):
        """
        Set data directory for this dataset
        """
        dset = str(self).lower()
        if data_dir is None:
            if dset.upper() in os.environ:
                # Use environment variable directly
                self.data_dir = os.path.normpath(
                    os.path.join(os.environ[dset.upper()], dset)
                )

            else:
                # Check for existence of a config file
                home_dir = os.path.expanduser("~")

                # Look in XDG_CONFIG_HOME (or $HOME/.config by default)
                xdg_config_home = os.environ.get(
                    "XDG_CONFIG_HOME", os.path.join(home_dir, ".config")
                )
                config_path = os.path.join(xdg_config_home, "rnlyss.conf")

                if os.path.isfile(config_path):
                    # Read from rnlyss.conf
                    self.data_dir = get_config_dir(config_path, dset)

                else:
                    # Default is $HOME/dset
                    self.data_dir = os.path.join(home_dir, dset)

        else:
            # Specified
            self.data_dir = os.path.normpath(data_dir)

        # Ensure directory exists
        os.makedirs(self.get_data_path("h5"), exist_ok=True)

    def get_data_path(self, *sub_dirs):
        """
        Return path to dset data + all sub_dirs.
        """
        return os.path.join(self.data_dir, *sub_dirs)

    def __contains__(self, dvar):
        return dvar in self.dvars

    def __getitem__(self, key):
        """
        Return Hyperslab based on key
        """
        if isinstance(key, tuple):
            if len(key) == 2:
                dvar, year = key
                path = self.get_data_path("h5", "%04d" % year, "%s.h5" % dvar)
                return HyperSlab(path)
            else:
                raise NotImplementedError(key)
        elif isinstance(key, str):
            if self.isconstant(key):
                path = self.get_data_path("h5", "%s.h5" % key)
                return HyperSlab(path)
            else:
                raise NotImplementedError(key)
        else:
            raise NotImplementedError(key)

    def get_dvar(self, role):
        """
        Return dvar related to role (if it exists)
        """

        if isinstance(role, tuple):
            dvars = [None] * len(role)
            for dvar in self.dvars:
                stored = self.dvars[dvar].get("role", None)
                if isinstance(stored, tuple):
                    if stored == role:
                        return dvar
                else:
                    for i, comp in enumerate(role):
                        if stored == comp:
                            dvars[i] = dvar
            return tuple(dvars)

        else:
            for dvar in self.dvars:
                stored = self.dvars[dvar].get("role", None)
                if isinstance(stored, tuple):
                    for comp in stored:
                        if comp == role:
                            return dvar
                else:
                    if stored == role:
                        return dvar

        return None

    def min(self, role, years=None):
        """
        Return minimum of dvar if available
        """
        return self.maxmin(role=role, years=years, maxmin="min")

    def max(self, role, years=None):
        """
        Return maximum of dvar if available
        """
        return self.maxmin(role=role, years=years, maxmin="max")

    def maxmin(self, role, years=None, maxmin="max"):
        s = []

        dvar = self.get_dvar(role)
        if dvar is None:
            return None

        for year in self.iter_year(years):
            if self.isconstant(dvar):
                with self[dvar] as slab:
                    if not slab:
                        return None

                    val = getattr(slab, maxmin)()

                    if val is None:
                        return None

                    # Quick exit
                    return np.squeeze(val)

            else:
                with self[dvar, year] as slab:
                    if slab:
                        t = slab.time()
                        val = getattr(slab, maxmin)()
                    else:
                        continue

                    if val is None:
                        s.append(pd.Series(index=t, data=np.nan, name=role))
                    else:
                        s.append(pd.Series(index=t, data=val, name=role))

        # Convert to Pandas series
        if len(s):
            return pd.concat(s)

        return None

    def __call__(
        self,
        role,
        lat,
        lon,
        hgt=None,
        years=None,
        order=0,
        scale_height=None,
        lapse_rate=None,
        offset=None,
    ):
        """
        Extract time series of role at (lat, lon, hgt).

        order=0: snap to nearest grid point horizontally.
        order=1: perform bi-linear interpolation

        scale_height: if scale_height is provided, do exponential interpolation
        e.g. for pressure, perhaps scale_height = RT/g = (287*260)/9.8 ~ 7600m

        lapse_rate:  if lapse_rate is provided, apply linear interpolation
        e.g. for temperature, perhaps lapse_rate = -6.5/1000 deg C/m

        offset:  if offset is provided, the time index will be shifted
        by the amount in minutes
        """

        if hasattr(lat, "__len__") or hasattr(lon, "__len__"):
            raise NotImplemented("Specify only single location.")

        # Get related dvar for role
        dvar = self.get_dvar(role)
        if dvar is None:
            return None

        t, x = [], []

        if order == 0:
            # Nearest point gets full weight
            i, j = self.grid(lat, lon, snap=True)
            inds = [(i, j)]
            wgts = [1.0]

        elif order == 1:
            # Bi-linear interpolation
            inds, wgts = self.grid.bbox(lat, lon)

        else:
            #  Other orders...
            raise ValueError("Order must be 0 (nearest) or 1 (bi-linear)")

        if hgt is None:
            hgts = [None] * len(inds)
        else:
            # Get heights at all grid points
            with self[self.get_dvar("hgt")] as slab:
                hgts = [slab[i, j, 0] for i, j in inds]

        def extract(slab):
            if not slab:
                return None

            val = None

            for (i, j), hgt_ij, wgt_ij in zip(inds, hgts, wgts):
                val_ij = slab[i, j, ...]

                if val_ij is None:
                    return None

                if hgt is not None:
                    # Scale to hgt
                    if scale_height is not None:
                        val_ij *= np.exp((hgt_ij - hgt) / scale_height)

                    if lapse_rate is not None:
                        val_ij += lapse_rate * (hgt - hgt_ij)

                if val is None:
                    val = wgt_ij * val_ij
                else:
                    val += wgt_ij * val_ij

            return val

        for year in self.iter_year(years):
            if self.isconstant(dvar):
                with self[dvar] as slab:
                    val = extract(slab)

                    if val is None:
                        return None

                    # Quick exit
                    return np.asscalar(val)

            else:
                with self[dvar, year] as slab:
                    val = extract(slab)

                    if val is None:
                        continue

                    # Stack data
                    t.append(slab.time())
                    x.append(np.transpose(val))

        # Convert to Pandas series
        if len(t):
            # Create time index
            t = np.concatenate(t)

            if offset is not None:
                t += np.timedelta64(int(np.rint(offset)), "m")

            if self.isvector(dvar):
                return pd.DataFrame(index=t, data=np.concatenate(x), columns=list(role))
            else:
                return pd.Series(index=t, data=np.concatenate(x), name=role)

        return None

    def snap(self, lat, lon, land=False, max_dist=np.inf):
        """
        Given a requested (lat, lon) return the nearest grid (lat, lon)

        If land=True, search for nearest land-mask >= 0.5
        """

        # Snap to (i, j)
        i, j = self.grid(lat, lon, snap=True)

        # Don't care if land or not
        if not land:
            return self.grid[i, j]

        if self.land_tree is None:
            # Establish land mask
            land_bool = self.land_grid().flatten() >= 0.5

            # Full lat, lon matrices
            lats, lons = np.meshgrid(
                self.grid.lats(), self.grid.lons(), indexing="ij"
            )

            # Build tree of land locations
            self.land_tree = KDTree(
                to_cartesian(lats.flatten()[land_bool], lons.flatten()[land_bool])
            )

            # Store indices of these locations
            self.land_indices = np.arange(np.prod(self.grid.shape))[land_bool]

        # Query our location
        dist, ind = self.land_tree.query(to_cartesian(lat, lon), k=1, p=2)

        # Check we're within our query distance; else return NaNs
        if dist > max_dist:
            return np.nan, np.nan

        # Magic unravel of our 1D indices
        i, j = np.unravel_index(self.land_indices[ind], self.grid.shape)
        return self.grid[np.squeeze(i), np.squeeze(j)]

    def hgt(self, lat, lon, order=0):
        """
        Return surface elevation/height given lat/lon.

        order=0: snap to nearest grid point horizontally.
        order=1: perform bi-linear interpolation
        """
        return self("hgt", lat, lon, order=order)

    def land(self, lat, lon, order=0):
        """
        Return land mask given lat/lon.  0=100% lake/ocean, 1=100% land/ice.

        order=0: snap to nearest grid point horizontally.
        order=1: perform bi-linear interpolation

        May need to be over-ridden by subclass e.g. MERRA-2
        """
        return self("land", lat, lon, order=order)

    def hgts(self, lats, lons, order=0):
        """
        Return surface elevation/height given lat(s) and lon(s).

        order=0: snap to nearest grid point horizontally.
        order=1: perform bi-linear interpolation

        Vectorized.
        """

        with self[self.get_dvar("hgt")] as slab:
            # Grab entire array
            hgt = slab[:, :, 0]

            if order == 0:
                # Grab nearest point
                i, j = self.grid(lats, lons, snap=True)
                hgts = hgt[i, j]

            elif order == 1:
                # Bi-linear interpolation
                i, j = self.grid(lats, lons, snap=False)

                ij = np.array([np.atleast_1d(i), np.atleast_1d(j)])
                # Map float indices into hgt
                hgts = map_coordinates(hgt, ij, mode="nearest", order=1)

            else:
                #  Other orders...
                raise ValueError("Order must be 0=nearest or 1=bi-linear")

        return np.squeeze(hgts)

    def hgt_grid(self):
        """
        Return entire height grid
        """
        with self[self.get_dvar("hgt")] as slab:
            # Grab entire array
            return slab[:, :, 0]

    def land_grid(self):
        """
        Return entire land mask grid

        May need to be over-ridden by subclass e.g. MERRA-2
        """
        with self[self.get_dvar("land")] as slab:
            # Grab entire array
            return slab[:, :, 0]

    def wind(self, lat, lon, hgt=None, order=0, years=None):
        """
        Create a Pandas DataFrame with wind speed & direction.

        order = 0:  Nearest grid-point magnitude and direction
        order = 1:  Bi-linear interpolation of magnitude and
                    circular interpolation of direction

        No vertical interpolation.
        """

        if hasattr(lat, "__len__") or hasattr(lon, "__len__"):
            raise ValueError("Specify only single location.")

        if order == 0:
            # Nearest point gets full weight
            i, j = self.grid(lat, lon, snap=True)
            inds = [(i, j)]
            wgts = [1.0]

        elif order == 1:
            # Bi-linear interpolation
            inds, wgts = self.grid.bbox(lat, lon)
        else:
            #  Other orders...
            raise ValueError("Order must be 0=nearest or 1=bi-linear")

        # Get dvar(s) corresponding to wind
        # e.g. "wnd10m" for CFSR or ("U10M", "V10M") for MERRA
        dvars = self.get_dvar(("uas", "vas"))

        t, x, y = [], [], []

        for year in self.iter_year(years):
            wd, ws, u, v = None, None, None, None

            if isinstance(dvars, tuple):
                # e.g. MERRA, separate slabs for each component
                with self[dvars[0], year] as slabx:
                    if not slabx:
                        continue
                    with self[dvars[1], year] as slaby:
                        if not slaby:
                            continue
                        for (i, j), wgt_ij in zip(inds, wgts):
                            uas_ij = slabx[i, j, ...]
                            vas_ij = slaby[i, j, ...]
                            # Wind speed
                            ws_ij = np.hypot(uas_ij, vas_ij)
                            # Normalize
                            with np.errstate(invalid="ignore"):
                                # NaNs and zeros will be False
                                k = ws_ij > 0
                            uas_ij[k] /= ws_ij[k]
                            vas_ij[k] /= ws_ij[k]
                            # Weighted-average of speed and normalized (u, v)
                            if ws is None:
                                ws = wgt_ij * ws_ij
                                u = wgt_ij * uas_ij
                                v = wgt_ij * vas_ij
                            else:
                                ws += wgt_ij * ws_ij
                                u += wgt_ij * uas_ij
                                v += wgt_ij * vas_ij

                    t.append(slabx.time())

            else:
                # e.g. CFSR, one vector slab
                with self[dvars, year] as slab:
                    if not slab:
                        continue
                    for (i, j), wgt_ij in zip(inds, wgts):
                        uas_ij = slab[i, j, 0, ...]
                        vas_ij = slab[i, j, 1, ...]
                        # Wind speed
                        ws_ij = np.hypot(uas_ij, vas_ij)
                        # Normalize
                        with np.errstate(invalid="ignore"):
                            # NaNs and zeros will be False
                            k = ws_ij > 0
                        uas_ij[k] /= ws_ij[k]
                        vas_ij[k] /= ws_ij[k]
                        # Weighted-average of speed and normalized (u, v)
                        if ws is None:
                            ws = wgt_ij * ws_ij
                            u = wgt_ij * uas_ij
                            v = wgt_ij * vas_ij
                        else:
                            ws += wgt_ij * ws_ij
                            u += wgt_ij * uas_ij
                            v += wgt_ij * vas_ij

                    t.append(slab.time())

            # Convert normalized components average to wind direction
            with np.errstate(invalid="ignore"):
                wd = np.degrees(np.arctan2(-u, -v)) % 360

            x.append(np.transpose(ws))
            y.append(np.transpose(wd))

        # Convert to Pandas series
        if len(t):
            return (
                pd.Series(index=np.concatenate(t), data=np.concatenate(x), name="WS"),
                pd.Series(index=np.concatenate(t), data=np.concatenate(y), name="WD"),
            )

        return None, None

    def solar_position(self, lat, lon, utc, method=None):
        """
        Calculate solar vector (x,y,z) given (lat, lon) and UTC
        """

        from rnlyss.solar import position

        if method is None:
            method = self.__class__.__name__.lower()

        return position(lat, lon, utc, method=method)

    def solar_elevation(self, lat, lon, utc, method=None, interval=None):
        """
        Calculate the elevation z and extraterrestrial radiation E0 at
        (lat, lon) and UTC time.  Result is either "instantaneous" or
        the average over an "hourly" or "daily" interval.
        """

        from rnlyss.solar import elevation

        if method is None:
            method = self.__class__.__name__.lower()

        return elevation(lat, lon, utc, interval=interval, method=method)

    def solar_orbit(self, utc, method=None):
        """
        Calculate the solar orbital parameters sin & cos of declination,
        equation of time, and distance ratio squared (solFactor)
        """
        from rnlyss.solar import orbit

        if method is None:
            method = self.__class__.__name__.lower()

        return orbit(utc, method=method)

    def solar_split(self, lat, lon, years=None, plot=False):
        """
        Split the global horizontal shortwave solar radiation into its
        beam normal and diffuse components
        """
        from rnlyss.solar import erbs, engerer, hour_angle, sunset_hour_angle

        # Extract downwelling SW at surface
        Et = self("rsds", lat, lon, years=years)
        if Et is None:
            return None
        # Can have negative (looking at you CFSR)
        Et = Et.clip(lower=0)

        # Extract downwelling shortwave at top-of-atmosphere
        E0h = self("rsdt", lat, lon, years=years)
        if E0h is None:
            return None
        E0h = E0h.clip(lower=0)

        # Extract maximum E0h, representing E0
        E0 = self.max("rsdt", years=years)
        if E0 is None:
            return None

        # Extract downwelling clear-sky at surface
        # (if not available, Erbs?)
        Etc = self("rsdsc", lat, lon, years=years)
        if Etc is None:
            return None
        Etc = Etc.clip(lower=0)

        # UTC
        utc = Et.index.values - np.timedelta64(30, "m")

        # Calculate solar orbit; equation of time
        sinDec, cosDec, eot, solFactor = self.solar_orbit(utc)

        # Hour angle (radians)
        h = hour_angle(lon, utc, eot)

        # Sunset hour angle (radians)
        lat = np.radians(lat)
        h0 = sunset_hour_angle(np.sin(lat), np.cos(lat), sinDec, cosDec)

        # Add 30 minutes
        h0 += 0.5 * np.pi / 12

        # Avoid division by zero, sensible fluxes, and ensure daytime
        # NB. The last check is because CFS returns some non-zero fluxes
        #     at nighttime.
        i = (E0h > 0) & (Et > 0) & (Etc > 0) & (E0h > Et) & (np.abs(h) < h0)

        # Calculate altitude
        z = E0h[i] / E0[i]

        # Calculate clear sky clearness index
        Ktc = np.clip(Etc[i] / E0h[i], 0, 1)

        # Calculate all sky clearness index
        Kt = np.clip(Et[i] / E0h[i], 0, 1)

        # Engerer diffuse split
        Kd = engerer(Kt, Ktc, z, h[i.values])

        # Beam normal coefficient
        Kn = Kt * (1 - Kd)

        # Diffuse
        Ed = pd.Series(index=Et.index, data=np.nan, name="Ed")
        Ed[i] = Kd * Et[i]

        # Beam normal
        Eb = pd.Series(index=Et.index, data=np.nan, name="Eb")
        Eb[i] = Kn * E0[i]

        # Correct surface and TOA horizontal for export
        # NB. leave NaNs as NaNs
        E0h[~i & np.isfinite(E0h)] = 0
        Et[~i & np.isfinite(Et)] = 0
        Eb[~i & np.isfinite(Et)] = 0
        Ed[~i & np.isfinite(Et)] = 0
        Etc[~i & np.isfinite(Etc)] = 0

        if plot:
            import matplotlib.pyplot as plt

            plt.rc("text", usetex=True)
            plt.rc("text.latex", unicode=True)
            plt.rc("text.latex", preamble=r"\usepackage{cmbright}")
            f, ax = plt.subplots(figsize=(5, 5), dpi=200)
            ax.plot(Kt, Kd, ".", color="orange", markersize=2, label="Engerer2")
            x = np.linspace(0, 1, 100)
            y = erbs(x)
            ax.plot(x, y, "-", label="Erbs")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            leg = ax.legend(
                loc="upper right", markerscale=4, numpoints=1, fontsize="smaller"
            )
            for line in leg.get_lines():
                line.set_linewidth(2.0)
            ax.set_xlabel(r"Clearness Index $K_t$", fontsize="smaller")
            ax.set_ylabel(r"Diffuse Fraction $K_d$", fontsize="smaller")
            ax.text(
                0.02,
                0.02,
                "%d points" % (len(Kt),),
                ha="left",
                va="bottom",
                transform=ax.transAxes,
            )
            plt.tight_layout()
            f.savefig("solar_split_%s.png" % str(self), dpi=f.dpi, bbox_inches="tight")

        return pd.DataFrame(
            {"E0": E0, "E0h": E0h, "Et": Et, "Etc": Etc, "Eb": Eb, "Ed": Ed}
        )

    def to_prec(self, lat, lon, years=None, order=1):
        """
        Return a DataFrame of monthly precipitation

        No correction for timezone.
        """

        # Grab hourly precipitation (kg/m2/s)
        pr = self("pr", lat, lon, years=years, order=order, offset=-30)

        # Convert to mm/hr
        pr *= 3600

        # Aggregate to monthly sum
        pr = pr.resample("M", how="sum")

        # Pivot
        pt = pd.pivot(pr.index.year, pr.index.month, pr)

        # Add annual
        pt.insert(0, 0, pt.sum(axis=1))

        return pt

    def to_allsky(self, lat, lon, hgt=None, years=None, order=1):
        """
        Return a DataFrame of monthly average daily insolation (kW·hr/m²/day)

        No correction for timezone.
        """

        # Grab hourly global horizontal radiation (W/m2)
        E = self("rsds", lat, lon, years=years, order=order, offset=-30)

        # Aggregate to daily sum
        E = E.resample("D", how="sum")

        # Aggregate to monthly average
        E = E.resample("M", how="mean")

        # Pivot
        pt = pd.pivot(E.index.year, E.index.month, E)

        # Convert to kW·hr/m²/day
        return pt / 1000.0

    def clear_sky_index(self, lat, lon, years=None, order=0, alt_lim=5):
        # Extract downwelling clear-sky at surface
        Ec = self("rsdsc", lat, lon, years=years, order=order)
        if Ec is None:
            return None

        # Extract downwelling shortwave at top-of-atmosphere
        E0h = self("rsdt", lat, lon, years=years, order=order)
        if E0h is None:
            return None

        # Extract maximum E0h, representing E0
        E0 = self.max("rsdt", years=years)
        if E0 is None:
            return None

        # Calculate altitude
        z = E0h / E0
        z.name = "z"

        # Limit where Ec > 0, E0h > 0, and z > zLimit (5deg)
        zLim = np.sin(np.radians(alt_lim))
        i = (Ec > 0) & (E0h > 0) & (z > zLim)
        z = z[i]

        # Calculate clearness index
        Kt = Ec[i] / E0h[i]
        Kt.name = "Kt"

        # Create aligned dataframe
        return pd.concat([z, Kt], axis=1)

    def to_clearsky(
        self, lat, lon, years=None, order=0, noon_flux=False, alt_lim=5, plot=False
    ):
        """
        Return a DataFrame with the ASHRAE clear sky pseudo-optical
        coefficients given location (lat, lon), selected list of
        years (default: all), and whether to include the noon beam and
        diffuse fluxes (default: False)
        """

        from rnlyss.solar import fit_monthly_taus

        # Calculate elevation and clear sky clearness index
        df = self.clear_sky_index(lat, lon, years=years, order=order, alt_lim=alt_lim)

        # Return fitted monthly taus
        return pd.DataFrame(
            index=range(1, 13),
            data=fit_monthly_taus(
                df.z, df.Kt, noon_flux=noon_flux, lat=lat, lon=lon, plot=plot
            ),
        )

    def to_hof(
        self,
        lat,
        lon,
        hgt=None,
        years=None,
        interp_xy=False,
        interp_z=False,
        exact=True,
        lapse_rate=-6.5 / 1000,
        dew_lapse_rate=-2 / 1000,
    ):
        """
        Return a dataframe containing drybulb, dewpoint, station pressure,
        wind speed, wind direction as required for the CSV format
        """

        if hgt is None:
            interp_z = False

        if interp_xy:
            order = 1
        else:
            order = 0

        if not interp_z:
            lapse_rate = None
            dew_lapse_rate = None

        # Bi-linearly interpolate dry-bulb temperature at site (K)
        db = self(
            "tas", lat, lon, hgt=hgt, years=years, order=order, lapse_rate=lapse_rate
        )
        if db is None:
            return None
        db.name = "DB"

        if interp_z:
            # Calculate pressure scale height based on avg temperature at site
            Tm = db.mean()
            Hp = 287.042 * Tm / 9.80665
        else:
            Hp = None
        # Bi-linearly interpolate station pressure (Pa)
        p = self("ps", lat, lon, hgt=hgt, years=years, order=order, scale_height=Hp)
        p.name = "SP"

        # Convert dry bulb K to C
        db -= 273.15

        # Check if we have dew point directly
        dp = self(
            "tdps",
            lat,
            lon,
            hgt=hgt,
            years=years,
            order=order,
            lapse_rate=dew_lapse_rate,
        )

        if dp is None:
            # Calculate dew point from specific humidity

            # Bi-linearly interpolate specific humidity (kg water / kg moist air)
            if interp_z:
                # Scale height for vapor pressure
                Hw = 2500.0
                # Scale height for Y is geometric average of Hw & Hp
                Hy = (Hw * Hp) / (Hp - Hw)
            else:
                # No scaling
                Hy = None

            q = self(
                "huss", lat, lon, hgt=hgt, years=years, order=order, scale_height=Hy
            )

            # Calculate dew point temperature (C) from specific humidity and presure
            dp = pd.Series(data=calc_dp_from_q_and_p(q, p), index=q.index, name="DP")

        else:
            dp.name = "DP"
            dp -= 273.15

        # Bi-linearly interpolate wind
        ws, wd = self.wind(lat, lon, order=order, years=years)

        # Create time-aligned dataframe
        df = pd.concat([p, db, dp, ws, wd], axis=1)

        # Correct dew point > dry bulb
        i = df.DB < df.DP
        df.loc[i, "DP"] = df.loc[i, "DB"]

        # Return dataframe
        return df

    def isvector(self, dvar):
        if dvar in self:
            return self.dvars[dvar].get("vector", False)

    def isscalar(self, dvar):
        if dvar in self:
            return not self.isvector(dvar)

    def isconstant(self, dvar):
        if dvar in self:
            return self.dvars[dvar].get("constant", False)

    def constants(self):
        return [k for k, v in self.dvars.items() if v.get("constant", False)]

    def take_inventory(self):
        """
        Return inventory of dataset
        """
        inv = {}

        for dvar in self.dvars:
            slab = self[dvar]
            if slab:
                inv[dvar] = slab.count()
            else:
                inv[dvar] = 0

        return inv

    def iter_year(self, years=None):
        available_years = list(range(self.years[0], self.years[1] + 1))

        if years is None:
            years = available_years
        else:
            if isinstance(years, int):
                years = [years]
            years = [year for year in years if year in available_years]

        for year in years:
            yield year

    def iter_month(self, year, months=None):
        available_months = range(1, 12 + 1)

        if months is None:
            months = available_months
        else:
            if isinstance(months, int):
                months = [months]
            months = [month for month in months if month in available_months]

        for month in months:
            if year == self.now.year and month > self.now.month:
                break
            yield month

    def iter_year_month(self, years=None, months=None):
        """
        Create an interator over all available years and months
        (or selected years and months)
        """

        for year in self.iter_year(years):
            for month in self.iter_month(year, months):
                yield year, month

    def isvalid(self, year):
        if (year >= self.years[0]) and (year <= self.years[1]):
            return True
        return False

    def get_stacked_dates(self, dvars=None, years=None):
        """
        See what years are fully available.
        """

        if dvars is None:
            dvars = self.dvars.keys()

        elif isinstance(dvars, str):
            dvars = [dvars]

        stacked = {}

        for dvar in dvars:
            if self.isconstant(dvar):
                continue

            index = []
            data = []

            for year in self.iter_year(years):
                with self[dvar, year] as slab:
                    if slab:
                        index.append(year)
                        data.append(slab.months_full())

            if len(index):
                stacked[dvar] = pd.DataFrame(
                    data=np.vstack(data), index=index, columns=range(1, 13)
                )

        return stacked


def test():
    # Attempt to load the banana reanalyses
    b = load_dataset("banana")


if __name__ == "__main__":
    test()
