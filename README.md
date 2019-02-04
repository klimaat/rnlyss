# rnlyss

`rnlyss` is a Python library to download, stack, and manipulated atmospheric reanalyses. It was developed as part of ASHRAE 1745-RP [1], a research project to evaluate the use of reanalyses for building design.

## Installation

Developed in Python3.5 under Ubuntu 16.04 but more recent versions or distributions may work. Untested in Windows.

### Required packages

* `requests`: <https://github.com/requests/requests>
* `numpy`: <https://github.com/numpy/numpy>
* `scipy`: <https://github.com/scipy/scipy>
* `pandas`: <https://github.com/pandas-dev/pandas>
* `h5py`: <https://github.com/h5py/h5py>
* `netCDF4`: <https://github.com/Unidata/netcdf4-python>

For Debian-based distributions:

```bash
sudo apt install python3-requests python3-numpy python3-scipy python3-pandas python3-h5py python3-netcdf4
```

For CFSR/CFSv2, install the package `pygrib` via instructions available at <https://github.com/jswhit/pygrib>.

### Download and install

Clone the repository:

```bash
git clone https://github.com/klimaat/rnlyss.git
```

and run `setup.py` from the newly created `rnlyss` directory:

```bash
cd rnlyss
python3 setup.py develop --user
```

Using the `develop` option allows you to modify the files in `rnlyss` without having to re-install.

Two helper scripts will be placed in your local bin, typically `$HOME/.local/bin`. Make sure this directory is in your `$PATH`.

### Uninstall

```bash
python3 setup.py develop --user --uninstall
```

## Setup

### Authorization

Currently there are two reanalyses products implemented, CFSR/CFSv2 from NOAA and MERRA-2 from NASA. Both products require user accounts in order to download data.  Register online for one or both:

* CFSR/CFSv2 via NCAR Research Data Archive: <https://rda.ucar.edu/>
* MERRA-2 via NASA Earthdata: <https://urs.earthdata.nasa.gov/>

Place the resulting passwords in `$HOME/.netrc` in the format:

```bash
machine urs.earthdata.nasa.gov login ernie password a1b2c3d4
machine rda.ucar.edu login bert@sesame.org password e5g6h7i8
```

### Dataset Paths

`rnlyss` needs to know where to put stuff. Each dataset can have its own location (e.g. if you have multiple hard drives). You have three options:

1. Set individual environment variables e.g. in `$HOME/.bashrc`:

```bash
export MERRA2=/bert
export CFSR=/ernie
export CFSV2=/kermit
```

2. Set paths in `$HOME/.config/rnlyss.conf`:

```INI
[Data]
MERRA2=/bert
CFSR=/ernie
CFSV2=/kermit
```

3. Do nothing and all datasets will default to `$HOME`.

## Usage

### Downloading

`rnlyss` is useless without some data.  Use the helper to download all available dataset variables for Jan (`-m 1`) data for 2018 (`-y 2018`). If the data has not been previously downloaded, it will do so. Go get a coffee. Or two.

```bash
rnlyss_download.py merra2 -y 2018 -m 1
```

### Stacking

Now that you have some data, you need to "stack" it into the HDF5 format to allow rapid access to the time series. Coffee time again.

```bash
rnlyss_stack.py merra2 -y 2018 -m1
```

### Analysis

Once you have sufficient data stacked, you can start to extract time series very rapidly, no coffee required.  Within Python:

```Python
from rnlyss.dataset import load_dataset

# Create MERRA-2 instance
M = load_dataset("MERRA2")

# Extract air temperature at 2m for Atlanta (ASHRAE HQ) into a Pandas Series
# (return the nearest location)
x = M('tas', 33.640, -84.430)
print(x.head())

# The same call but applying bi-linear interpolation of the surrounding
# 4 grid locations and restricting data to the year 2018.
y = M('tas', 33.640, -84.430, hgt=313, order=1, years=[2018])
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
```

### Export to EPW

Once an entire year (or bunch of years) has been stacked, EPW files may be written:

```Python
from rnlyss.epw import write_epw

# Create an Atlanta EPW from the CFSR and CFSV2 datasets for months chosen
# from selected years

years = [2018]*12

meta = {'city': 'Atlanta', 'state': 'GA', 'country': 'USA'}

loc = {'lat': 33.640, 'lon': -84.430, 'hgt': 313}

write_epw('Atlanta.epw', dsets=['MERRA2'], years=years, **meta, **loc)
```

## Disclaimer

Please see `LICENSE`. This code is a research-level product, fragile, and at the mercy of the upstream data providers. Every attempt will be made to keep the code up-to-date and add new features (e.g. ERA5) but either be patient or provide a pull request.

## References

[1] Roth, M. 2019. "Evaluation of Climate Reanalysis Data for Use in ASHRAE Applications". Final Report for ASHRAE 1745-RP.
