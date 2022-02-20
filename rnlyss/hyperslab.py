# -*- coding: utf-8 -*-
#
# Copyright 2019 Klimaat

import os
import numpy as np

# fmt: off
import mpi4py
mpi4py.rc.initialize = False
import h5py
# fmt: on

from rnlyss.util import DelayedKeyboardInterrupt


class HyperSlab(object):
    def __init__(self, path, **kwargs):
        self.path = os.path.abspath(path)
        self.hdf = None
        if os.path.isfile(self.path):
            self.connect(mode="r")

    def connect(self, mode="r", **kwargs):
        """
        Connect to an existing hdf5 file
        """
        self.close()
        try:
            self.hdf = h5py.File(self.path, mode=mode, **kwargs)
        except OSError as ex:
            return None
        except Exception:
            raise
        self.setup()

    def setup(self):

        # Insert HDF attributes into class attributes
        for attr in ["year", "freq", "scale", "offset", "missing", "hour0"]:
            setattr(self, attr, self.hdf.attrs[attr])

        # Hyperslab time origin; New Year's + hour0 to closest minute
        # NB. set hour0 to 0.5 for 00:30
        self.t0 = np.datetime64("%04d-01-01" % self.year) + np.timedelta64(
            int(np.rint(self.hour0 * 60)), "m"
        )

        # Span between slabs
        self.dt = np.timedelta64(int(self.freq), "h")

    def close(self):
        if self:
            self.hdf.close()

    def flush(self):
        if self:
            self.hdf.flush()

    def __enter__(self):
        """
        Allow use of with-statement with class.
        """
        return self

    def __exit__(self, *ignored):
        """
        Safely close hdf file.
        """
        self.close()

    def __str__(self):
        return self.path

    def __repr__(self):
        return "%s(path=%r)" % (self.__class__.__name__, self.path)

    def __bool__(self):
        """
        Check if initialized.
        """
        return False if self.hdf is None else True

    def to_int(self, x, converter=None):
        """
        Convert to integer; storage representation.

        Converter is a function that is applied to the variable *before*
        storage.  e.g. to convert units.  Default is to store whatever
        is provided.
        """
        # Handle masked arrays, converted to regular if necessary
        x = np.ma.filled(x, fill_value=np.nan)
        if converter is None:
            return np.where(
                np.isnan(x),
                self.missing,
                np.rint((x - self.offset) / self.scale).astype(np.int16),
            )
        else:
            return np.where(
                np.isnan(x),
                self.missing,
                np.rint((converter(x) - self.offset) / self.scale).astype(np.int16),
            )

    def to_float(self, i, dtype=np.float):
        """
        Convert to float; real representation
        """
        return np.where(
            i == self.missing,
            np.nan,
            np.array(i * self.scale + self.offset, dtype=dtype),
        )

    def __getitem__(self, i):
        """
        Numpy indexing into full data array.
        """
        if self:
            return self.to_float(self.hdf["data"][i])

    def __setitem__(self, i, x):
        """
        Numpy setting into full data array.
        Will bork on out of bounds.
        """
        if self:
            self.connect(mode="r+")
            self.hdf["data"][i] = self.to_int(x)
            self.flush()
            self.connect(mode="r")

    def create(
        self,
        shape,
        year=1970,
        freq=1,
        scale=1,
        offset=0,
        hour0=0,
        missing=np.iinfo(np.int16).max,
        **kwargs
    ):
        """
        Create a data hyperslab of any shape with final dimension
        expandable.

        e.g. chunks=(4,5,6)
        """

        # Initialize HDF file
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.hdf = h5py.File(self.path, "w", libver="latest")

        # Set attrs
        self.hdf.attrs["year"] = year
        self.hdf.attrs["freq"] = freq
        self.hdf.attrs["scale"] = scale
        self.hdf.attrs["offset"] = offset
        self.hdf.attrs["missing"] = missing
        self.hdf.attrs["hour0"] = hour0

        # Setup attributes etc.
        self.setup()

        # Length of time series
        nt = len(self)

        # Total shape
        shape = shape + (nt,)

        # Create a dataset of integers
        self.hdf.create_dataset("data", shape=shape, dtype=np.int16, fillvalue=missing)

        # Create a boolean dataset initialized to False to store
        # whether given time slice has been filled
        self.hdf.create_dataset("full", shape=(nt,), dtype=np.bool, fillvalue=False)

        # Create a int16 dataset initialized to missing to store
        # the minimum value of a given time slice
        self.hdf.create_dataset("min", shape=(nt,), dtype=np.int16, fillvalue=missing)

        # Create a int16 dataset initialized to missing to store
        # the maximum value of a given time slice
        self.hdf.create_dataset("max", shape=(nt,), dtype=np.int16, fillvalue=missing)

        # Flush buffers
        self.hdf.flush()

        # Re-open in read mode
        self.connect(mode="r")

    def fill(self, t, x):
        """
        Fill hyperslab starting at integer index t with slab x.
        NB. x is already scaled and integer.
        """

        if not isinstance(t, (int, np.integer)):
            raise NotImplementedError("Need integer t")

        if np.isscalar(x):
            nt = 1

        else:
            if self.ndim() != x.ndim:
                raise ValueError(
                    "Fill requires that inserted slab has same "
                    "dimensionality as existing"
                )

            if self.shape()[:-1] != x.shape[:-1]:
                raise ValueError(
                    "Fill requires that inserted slab non-time " "dimensions are equal"
                )
            nt = x.shape[-1]

        with DelayedKeyboardInterrupt():

            self.connect(mode="r+")
            self.hdf["data"][..., t : (t + nt)] = x
            self.hdf["full"][t : (t + nt)] = True
            if "min" in self.hdf:
                self.hdf["min"][t : (t + nt)] = np.nanmin(
                    x, axis=tuple(range(x.ndim - 1))
                )
            if "max" in self.hdf:
                self.hdf["max"][t : (t + nt)] = np.nanmax(
                    x, axis=tuple(range(x.ndim - 1))
                )
            self.flush()
            self.connect(mode="r")

    def empty(self, t):
        """
        Set time slice to missing.
        """
        self.connect(mode="r+")
        self.hdf["data"][..., t] = self.missing
        self.hdf["full"][t] = False
        self.connect(mode="r")

    def shape(self):
        return self.hdf["data"].shape if self else None

    def ndim(self):
        return self.hdf["data"].ndim if self else None

    def ind2date(self, i):
        """
        Given integer index i, return corresponding datetime64
        """
        return self.t0 + np.array(i, dtype="timedelta64[h]") * self.freq

    def time2ind(self, t):
        """
        Given datetime64 datetime t, return corresponding index into slab
        """
        return np.rint((t - self.t0) / self.dt).astype(int)

    def date2ind(self, months=1, days=1, hours=0, mins=0):
        years = (np.asarray(self.year) - 1970).astype("<M8[Y]")
        months = (np.asarray(months) - 1).astype("<m8[M]")
        days = (np.asarray(days) - 1).astype("<m8[D]")
        hours = np.asarray(hours).astype("<m8[h]")
        mins = np.asarray(mins).astype("<m8[m]")
        return self.time2ind(years + months + days + hours + mins)

    def time(self):
        return self.t0 + np.arange(len(self)) * self.dt

    def isleap(self):
        if self.year % 4 == 0 and (self.year % 100 != 0 or self.year % 400 == 0):
            return True
        return False

    def __len__(self):
        """
        Length of time series.
        """
        if self:
            if self.freq:
                days = 365
                if self.isleap():
                    days = 366
                return (days * 24) // self.freq
            else:
                return 1
        return 0

    def month_len(self, months):
        """
        Length of given month(s).
        """
        if self:
            if self.freq:
                month_days = np.array(
                    [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
                )
                if self.isleap():
                    month_days[2] = 29
                return (month_days[months] * 24) // self.freq
            else:
                return 1
        return 0

    def months_full(self):
        """
        Return boolean array listing whether each month has been filled
        """
        i = self.month_len(np.arange(13)).cumsum()
        return [self.isfull(np.s_[i[m] : i[m + 1]]) for m in range(12)]

    def isfull(self, t=np.s_[:]):
        """
        Check whether all of the hours in indices t are full.
        """
        return np.all(self.hdf["full"][t])

    def ispartial(self, t=np.s_[:]):
        """
        Check whether any of the hours in indices t are full.
        """
        return np.any(self.hdf["full"][t])

    def isempty(self, t=np.s_[:]):
        """
        Check whether none of the hours in indices t are full.
        """
        return not self.ispartial(t)

    def count(self, t=np.s_[:]):
        """
        Count full
        """
        return self.hdf["full"][t].sum()

    def min(self, t=np.s_[:]):
        """
        Return minimum time series
        """
        if "min" in self.hdf:
            return self.to_float(self.hdf["min"][t])
        return None

    def max(self, t=np.s_[:]):
        """
        Return maximum time series
        """
        if "max" in self.hdf:
            return self.to_float(self.hdf["max"][t])
        return None


def test():
    pass


if __name__ == "__main__":
    test()
