# -*- coding: utf-8 -*-
#
# Copyright 2019 Klimaat

from __future__ import division

import datetime
import numpy as np
import matplotlib.pyplot as plt


def join_date(y=1970, m=1, d=1, hh=0, mm=0, ss=0):
    """
    Join date/time components into datetime64 object
    """
    y = (np.asarray(y) - 1970).astype("<M8[Y]")
    m = (np.asarray(m)-1).astype("<m8[M]")
    d = (np.asarray(d)-1).astype("<m8[D]")
    hh = np.asarray(hh).astype("<m8[h]")
    mm = np.asarray(mm).astype("<m8[m]")
    ss = np.asarray(ss).astype("<m8[s]")
    return y+m+d+hh+mm+ss


def split_date(dates):
    """
    Split datetime64 dates into year, month, day components.
    """
    y = dates.astype('<M8[Y]').astype(int) + 1970
    m = dates.astype('<M8[M]').astype(int) % 12 + 1
    d = (dates - dates.astype('<M8[M]')).astype("<m8[D]").astype(int) + 1
    return y, m, d


def split_time(dates):
    """
    Split datetime64 dates into hour, minute, second components.
    """
    hh = (dates - dates.astype('<M8[D]')).astype("<m8[h]").astype(int)
    mm = (dates - dates.astype('<M8[h]')).astype("<m8[m]").astype(int)
    ss = (dates - dates.astype('<M8[m]')).astype("<m8[s]").astype(int)
    return hh, mm, ss


def day_of_year(dates, snap=True):
    """
    Calculate the day of the year (0-365/366)
    """
    dt = np.asarray(dates)-dates.astype('<M8[Y]')
    if snap:
        # Provide value at noon (integer)
        # Jan 1st anytime = 1
        return dt.astype("<m8[D]").astype(int) + 1
    else:
        # Provide value including fractional part (float)
        # Jan 1st at 00:00 = 0, Jan 1st at noon = 0.5
        return dt.astype("<m8[s]").astype(int)/86400


def julian_day(dates):
    """
    Julian day calculator
    """

    # Get Julian Day number
    y, m, d = split_date(dates)
    a = (14-m)//12
    y += 4800 - a
    m += 12*a - 3
    jd = d + ((153*m + 2)//5) + 365*y + y//4 - y//100 + y//400 - 32045

    # Get fractional day (noon=0)
    hh, mm, ss = split_time(dates)
    fd = (hh-12)/24 + mm/1440 + ss/86400

    return jd, fd


def orbit_ashrae(utc):
    """
    Calculate solar parameters based on ASHRAE methodology.

    Ref. ASHRAE HOF 2017, Chap 14
    """

    # Day of year
    n = day_of_year(utc, snap=True)

    # Declination (eqn. 10, radians)
    decl = np.radians(23.45*np.sin(2*np.pi*(n+284)/365))

    # Equation of time (eqns 5 & 6, min)
    gamma = 2*np.pi*(n-1)/365
    eqnOfTime = 2.2918*(0.0075 + 0.1868*np.cos(gamma) - 3.2077*np.sin(gamma) -
                        1.4615*np.cos(2*gamma) - 4.089*np.sin(2*gamma))

    # Convert from minutes to radians
    eqnOfTime *= np.pi/(60*12)

    # Solar constant correction
    solFactor = 1 + 0.033*np.cos(np.radians(360*(n-3)/365))

    return np.sin(decl), np.cos(decl), eqnOfTime, solFactor


def orbit_energyplus(utc):
    """
    Calculate solar coefficients based on EnergyPlus

    Ref. WeatherManager.cc, function CalculateDailySolarCoeffs
    """

    # Day of year
    n = day_of_year(utc, snap=True)

    # Day Angle
    D = 2*np.pi*n/366.0

    sinD = np.sin(D)
    cosD = np.cos(D)

    # Calculate declination sines & cosines

    sinDec = 0.00561800 \
        + 0.0657911*sinD \
        - 0.392779*cosD \
        + 0.00064440*(sinD*cosD*2.0) \
        - 0.00618495*(cosD**2 - sinD**2) \
        - 0.00010101*(sinD*(cosD**2-sinD**2) + cosD*(sinD*cosD*2.0)) \
        - 0.00007951*(cosD*(cosD**2-sinD**2) - sinD*(sinD*cosD*2.0)) \
        - 0.00011691*(2.0*(sinD*cosD*2.0)*(cosD**2 - sinD**2)) \
        + 0.00002096*((cosD**2-sinD**2)**2 - (sinD*cosD*2.0)**2)

    cosDec = np.sqrt(1-sinDec**2)

    # Equation of time (hours)

    eqnOfTime = 0.00021971 \
        - 0.122649*sinD \
        + 0.00762856 * cosD \
        - 0.156308 * (sinD*cosD*2.0) \
        - 0.0530028 * (cosD**2 - sinD**2) \
        - 0.00388702 * (sinD*(cosD**2 - sinD**2) + cosD*(sinD*cosD*2.0)) \
        - 0.00123978 * (cosD * (cosD**2 - sinD**2) - sinD*(sinD*cosD*2.0)) \
        - 0.00270502 * (2.0*(sinD*cosD*2.0)*(cosD**2 - sinD**2)) \
        - 0.00167992 * ((cosD**2 - sinD**2)**2 - (sinD*cosD*2.0)**2)

    # Convert to radians
    eqnOfTime = np.pi*eqnOfTime/12

    # Solar constant correction factor
    solFactor = 1.000047 + 0.000352615*sinD + 0.0334454*cosD

    return sinDec, cosDec, eqnOfTime, solFactor


def orbit_cfsr(utc):
    """
    Calculate solar coefficients based on CFSR methodology

    Ref. radiation_astronomy.f, subroutine solar
    """

    # Get julian day and fractional part of day
    jd, fjd = julian_day(utc)

    # Julian day of epoch which is January 0, 1990 at 12 hours UTC
    jdor = 2415020

    # Days of years
    cyear = 365.25

    # Days between epoch and perihelioon passage of 1990
    tpp = 1.55

    # Days between perihelion passage and March equinox of 1990
    svt6 = 78.035

    # Julian centuries after epoch
    t1 = (jd - jdor)/36525.0

    # Length of anomalistic and tropical years (minus 365 days)
    ayear = 0.25964134e0 + 0.304e-5 * t1
    tyear = 0.24219879E0 - 0.614e-5 * t1

    # Orbit eccentricity and earth's inclination (deg)
    ec = 0.01675104e0 - (0.418e-4 + 0.126e-6 * t1) * t1
    angin = 23.452294e0 - (0.0130125e0 + 0.164e-5 * t1) * t1

    ador = jdor
    jdoe = np.asarray(ador + (svt6 * cyear) / (ayear - tyear), dtype=int)

    # deleqn is updated svt6 for current date
    deleqn = (jdoe - jd) * (ayear - tyear) / cyear

    ayear = ayear + 365

    sni = np.sin(np.radians(angin))
    tini = 1 / np.tan(np.radians(angin))
    er = np.sqrt((1 + ec) / (1 - ec))

    # mean anomaly
    qq = deleqn * 2*np.pi / ayear

    def solve_kepler(e, M, E=1, eps=1.3e-6):
        """
        Solve Kepler equation for eccentric anomaly E by Newton's method
        based on eccentricity e and mean anomaly M
        """
        for i in range(10):
            dE = -(E-e*np.sin(E)-M)/(1-e*np.cos(E))
            E += dE
            dEmax = np.max(np.abs(dE))
            if dEmax < eps:
                break
        else:
            print("Warning: Exceeding 10 iterations in Kepler solver:", dEmax)
        return E

    # Eccentric anomaly at equinox
    e1 = solve_kepler(ec, qq)

    # True anomaly at equinox
    eq = 2.0 * np.arctan(er*np.tan(0.5*e1))

    # Date is days since last perihelion passage
    dat = jd - jdor - tpp + fjd
    date = dat % ayear

    # Mean anomaly
    em = 2*np.pi * date / ayear

    # Eccentric anomaly
    e1 = solve_kepler(ec, em)

    # True anomaly
    w1 = 2.0 * np.arctan(er*np.tan(0.5*e1))

    # Earth-Sun radius relative to mean radius
    r1 = 1.0 - ec*np.cos(e1)

    # Sine of declination angle
    # NB. ecliptic longitude = w1 - eq
    sdec = sni * np.sin(w1 - eq)

    # Cosine of declination angle
    cdec = np.sqrt(1.0 - sdec*sdec)

    # Sun declination (radians)
    dlt = np.arcsin(sdec)

    # Sun right ascension (radians)
    alp = np.arcsin(np.tan(dlt)*tini)
    alp = np.where(np.cos(w1-eq) < 0, np.pi-alp, alp)
    alp = np.where(alp < 0, alp + 2*np.pi, alp)

    # Equation of time (radians)
    sun = 2*np.pi*(date - deleqn)/ayear
    sun = np.where(sun < 0.0, sun+2*np.pi, sun)
    slag = sun - alp - 0.03255

    # Solar constant correction factor (inversely with radius squared)
    solFactor = 1/(r1**2)

    return sdec, cdec, slag, solFactor


def orbit_noaa(utc):
    """
    Orbit as per NOAA Solar Calculation spreadsheet
    https://www.esrl.noaa.gov/gmd/grad/solcalc/calcdetails.html

    Similar to CFSR but faster
    """

    # Julian day (including fractional part)
    jd, fjd = julian_day(utc)
    jd = jd + fjd

    # Julian century
    jc = (jd - 2451545) / 36525

    # Geometric mean longitude (deg)
    gml = (280.46646 + jc * (36000.76983 + jc * 0.0003032)) % 360

    # Geometric mean anomaly Sun (deg)
    gma = 357.52911 + jc * (35999.05029 - 0.0001537 * jc)

    # Eccentricity of Earth's orbit
    ecc = 0.016708634 - jc * (0.000042037 + 0.0000001267 * jc)

    # Sun equation of centre (deg)
    ctr = (
        np.sin(np.radians(gma))
        * (1.914602 - jc * (0.004817 + 0.000014 * jc))
        + np.sin(np.radians(2 * gma)) * (0.019993 - 0.000101 * jc)
        + np.sin(np.radians(3 * gma)) * 0.000289
    )

    # Sun true longitude (deg)
    stl = gml + ctr

    # Sun true anomaly (deg)
    sta = gma + ctr

    # Sun radius vector (AUs)
    rad = (1.000001018 * (1 - ecc * ecc)) / (
        1 + ecc * np.cos(np.radians(sta))
    )

    # Sun apparent longitude (deg)
    sal = stl - 0.00569 - 0.00478 * np.sin(np.radians(125.04 - 1934.136 * jc))

    # Mean obliquity ecliptic (deg)
    moe = (23 + (26 + ((21.448 - jc * (46.815 + jc * (0.00059 - jc * 0.001813)))) / 60) / 60)

    # Obliquity correction (deg)
    obl = moe + 0.00256 * np.cos(np.radians(125.04 - 1934.136 * jc))

    # Sun right ascension (deg)
    sra = np.degrees(
        np.arctan2(
            np.cos(np.radians(obl)) * np.sin(np.radians(sal)),
            np.cos(np.radians(sal)),
        )
    )

    # Sun declination
    sinDec = np.sin(np.radians(obl)) * np.sin(np.radians(sal))
    cosDec = np.sqrt(1.0 - sinDec*sinDec)

    # Var y
    vary = np.tan(np.radians(obl / 2)) * np.tan(np.radians(obl / 2))

    # Equation of time (minutes)
    eqnOfTime = 4 * np.degrees(
        vary * np.sin(2 * np.radians(gml))
        - 2 * ecc * np.sin(np.radians(gma))
        + 4
        * ecc
        * vary
        * np.sin(np.radians(gma))
        * np.cos(2 * np.radians(gml))
        - 0.5 * vary * vary * np.sin(4 * np.radians(gml))
        - 1.25 * ecc * ecc * np.sin(2 * np.radians(gma))
    )

    # Convert from minutes to radians
    eqnOfTime *= np.pi/(60*12)

    # Solar constant correction factor (inversely with radius squared)
    solFactor = 1/(rad**2)

    return sinDec, cosDec, eqnOfTime, solFactor


def orbit_merra2(utc):
    """
    Orbit as per MERRA2 code
    """

    # MERRA-2 solar repeats on a four-year leap-year cycle
    yearlen = 365.25
    days_per_cycle = 1461

    if orbit_merra2.orbit is None:

        # Constants from MAPL_Generic.F90
        ecc = 0.0167
        obliquity = np.radians(23.45)
        perihelion = np.radians(102.0)
        equinox = 80

        omg = (2.0*np.pi/yearlen)/np.sqrt(1-ecc**2)**3
        sob = np.sin(obliquity)

        # TH: Orbit anomaly
        # ZS: Sine of declination
        # ZC: Cosine of declination
        # PP: Inverse of square of earth-sun distance

        # Integration starting at vernal equinox
        def calc_omega(th):
            return omg*(1.0-ecc*np.cos(th-perihelion))**2

        orbit = np.recarray(
            (days_per_cycle, ),
            dtype=[('th', float), ('zs', float), ('zc', float), ('pp', float)]
        )

        def update_orbit(th):
            zs = np.sin(th)*sob
            zc = np.sqrt(1.0-zs**2)
            pp = ((1.0-ecc*np.cos(th-perihelion))/(1.0-ecc**2))**2
            orbit[kp] = th, zs, zc, pp

        # Starting point
        th = 0
        kp = equinox
        update_orbit(th)

        # Runge-Kutta
        for k in range(days_per_cycle-1):
            t1 = calc_omega(th)
            t2 = calc_omega(th+0.5*t1)
            t3 = calc_omega(th+0.5*t2)
            t4 = calc_omega(th+t3)
            kp = (kp + 1) % days_per_cycle
            th += (t1 + 2*(t2+t3) + t4)/6.0
            update_orbit(th)

        # Cache it
        orbit_merra2.orbit = orbit

    else:

        orbit = orbit_merra2.orbit

    # Map into orbit
    year, month, day = split_date(utc)
    doy = day_of_year(utc, snap=True)
    iyear = (year - 1) % 4
    iday = iyear*int(yearlen) + doy - 1

    # Declination
    sinDec = orbit['zs'][iday]
    cosDec = orbit['zc'][iday]

    # MERRA uses *solar* instead of *clock* time; no equation of time
    eqnOfTime = np.zeros_like(sinDec)

    # Inverse square of earth-sun distance ratio to mean distance
    solFactor = orbit['pp'][iday]

    return sinDec, cosDec, eqnOfTime, solFactor


# For caching MERRA-2 orbit
orbit_merra2.orbit = None


def orbit(utc, method=None):

    if method is None:
        method = "ASHRAE"

    if callable(method):
        func = method
        method = "Custom"
    else:
        method = method.upper()
        if method.startswith("A"):
            func = orbit_ashrae
        elif method.startswith("C"):
            func = orbit_cfsr
        elif method.startswith("E"):
            func = orbit_energyplus
        elif method.startswith("M"):
            func = orbit_merra2
        elif method.startswith("N"):
            func = orbit_noaa
        else:
            raise NotImplementedError(method)

    return func(utc)


def total_solar_irradiance_ashrae(utc):
    """
    Return ASHRAE constant solar irradiance value (W/m²)
    """

    return 1367.0*(np.ones_like(utc).astype(float))


def total_solar_irradiance_cfsr(utc):
    """
    Calculate CFSR total solar irradiance (W/m²) based on year and month
    NB.  Interpolates from yearly data
    """

    #
    year, month, _ = split_date(utc)

    # TSI datum
    TSI_datum = 1360.0

    # van den Dool data (1979-2006); assumed valid in July of that year
    dTSI = np.array([
        6.70, 6.70, 6.80, 6.60, 6.20, 6.00, 5.70, 5.70, 5.80, 6.20, 6.50,
        6.50, 6.50, 6.40, 6.00, 5.80, 5.70, 5.70, 5.90, 6.40, 6.70, 6.70,
        6.80, 6.70, 6.30, 6.10, 5.90, 5.70
    ])
    n = len(dTSI)

    # Index into dTSI (float)
    i = np.asarray(year).astype(int) - 1979 + (np.asarray(month) - 7)/12

    # Extend backward and/or forward assuming 11-year sunspot cycle
    while np.any(i < 0):
        i[i < 0] += 11
    while np.any(i > n-1):
        i[i > n-1] -= 11

    # Add base
    return TSI_datum + np.interp(i, np.arange(n), dTSI)


def total_solar_irradiance_merra2(utc):
    """
    Calculate MERRA-2 total solar irradiance (W/m²) based on year and month
    """

    year, month, _ = split_date(utc)

    # CMIP5 data (1980-2008), monthly
    # http://solarisheppa.geomar.de/solarisheppa/sites/default/files/data/CMIP5/TSI_WLS_mon_1882_2008.txt
    TSI = np.array([
        [1366.8707, 1366.6385, 1367.0020, 1366.3137, 1366.4717, 1366.7686,
         1366.7025, 1366.6991, 1366.6078, 1366.5760, 1366.1366, 1366.9659],
        [1367.0270, 1366.5762, 1366.7291, 1367.0487, 1366.7421, 1366.5843,
         1365.8833, 1367.0589, 1366.7669, 1366.4607, 1366.7618, 1366.6833],
        [1367.0527, 1365.9164, 1365.9046, 1366.4697, 1366.4086, 1365.5996,
         1366.1626, 1366.2002, 1366.5021, 1366.6118, 1366.4150, 1366.2152],
        [1366.4198, 1366.2211, 1366.2509, 1366.2035, 1366.1029, 1366.1212,
         1366.2853, 1366.4204, 1366.2336, 1366.0589, 1366.1071, 1366.0605],
        [1365.4259, 1365.6620, 1366.1702, 1365.5668, 1365.7794, 1366.0970,
         1366.1162, 1365.9801, 1365.8692, 1365.7895, 1365.6831, 1365.7649],
        [1365.6116, 1365.7119, 1365.6604, 1365.5154, 1365.6400, 1365.6998,
         1365.6543, 1365.7532, 1365.6687, 1365.5303, 1365.6323, 1365.6828],
        [1365.6780, 1365.5509, 1365.6831, 1365.6565, 1365.7309, 1365.6649,
         1365.6543, 1365.6022, 1365.6068, 1365.6499, 1365.7130, 1365.6751],
        [1365.6707, 1365.6624, 1365.6726, 1365.6419, 1365.7595, 1365.8341,
         1365.8257, 1365.7894, 1365.8603, 1365.8542, 1365.9870, 1366.0384],
        [1366.0580, 1366.1113, 1365.9553, 1366.0675, 1366.3042, 1366.0166,
         1365.8303, 1366.1485, 1366.4650, 1366.1152, 1366.2991, 1366.2632],
        [1366.5443, 1366.6023, 1366.3792, 1366.5935, 1366.7821, 1366.3332,
         1367.0719, 1366.5117, 1366.2650, 1366.9587, 1366.8282, 1366.8817],
        [1366.8792, 1366.6387, 1366.6480, 1366.8708, 1366.5344, 1366.7742,
         1366.4636, 1366.1724, 1366.8062, 1366.6181, 1365.8552, 1366.3904],
        [1366.0560, 1366.3106, 1366.5274, 1367.0611, 1366.4294, 1366.4347,
         1366.6702, 1366.4596, 1366.8890, 1366.1511, 1366.6261, 1365.9471],
        [1366.5259, 1366.4305, 1366.7496, 1366.5985, 1366.4207, 1366.3006,
         1366.0603, 1366.0338, 1366.1649, 1365.9236, 1366.1362, 1366.2879],
        [1366.3059, 1365.9018, 1366.2124, 1366.1830, 1366.1459, 1366.1432,
         1366.0951, 1366.0493, 1365.8926, 1365.7306, 1365.7609, 1365.9120],
        [1365.7409, 1365.9919, 1366.0338, 1365.8676, 1365.7668, 1365.7674,
         1365.7641, 1365.7805, 1365.6507, 1365.7192, 1365.8328, 1365.7086],
        [1365.8283, 1365.8175, 1365.7226, 1365.6256, 1365.6620, 1365.7283,
         1365.6993, 1365.7184, 1365.6976, 1365.6064, 1365.6769, 1365.6436],
        [1365.6443, 1365.6287, 1365.5849, 1365.6109, 1365.6276, 1365.6290,
         1365.6002, 1365.6662, 1365.6821, 1365.6348, 1365.4741, 1365.7028],
        [1365.6989, 1365.6747, 1365.7008, 1365.7047, 1365.7390, 1365.7301,
         1365.7250, 1365.7857, 1365.6768, 1365.9331, 1365.8454, 1365.8881],
        [1365.9627, 1365.9199, 1365.8269, 1366.0931, 1365.9647, 1366.0578,
         1366.2478, 1366.0894, 1366.0800, 1366.3429, 1366.2589, 1366.3730],
        [1366.4806, 1366.2429, 1366.3572, 1366.2549, 1366.3835, 1366.3984,
         1366.4362, 1366.4766, 1366.5841, 1366.2329, 1366.3558, 1366.3730],
        [1366.7211, 1366.6320, 1366.4819, 1366.6498, 1366.3611, 1366.4507,
         1366.5754, 1366.9738, 1366.5276, 1366.9746, 1366.9062, 1366.9492],
        [1366.7811, 1366.8458, 1366.4121, 1366.4659, 1366.5200, 1366.5092,
         1366.7203, 1366.4475, 1366.3010, 1366.8140, 1366.5200, 1366.8910],
        [1367.3162, 1367.1783, 1367.0065, 1366.6454, 1366.6470, 1366.6873,
         1366.1716, 1366.3053, 1366.4584, 1366.5261, 1366.4495, 1366.7773],
        [1366.6034, 1366.5458, 1366.1968, 1366.2227, 1366.1753, 1366.0914,
         1366.2437, 1366.2744, 1366.3611, 1365.5612, 1366.1956, 1366.2899],
        [1366.1038, 1366.0890, 1366.1272, 1366.1742, 1366.0297, 1366.0179,
         1365.7578, 1365.9036, 1366.0957, 1366.1166, 1366.0057, 1366.1552],
        [1365.7864, 1365.9349, 1365.8956, 1365.8800, 1365.8463, 1365.8059,
         1365.8595, 1365.9275, 1365.7988, 1365.8860, 1365.7792, 1365.8549],
        [1365.8986, 1365.8728, 1365.7850, 1365.8058, 1365.9230, 1365.8340,
         1365.8212, 1365.7067, 1365.8419, 1365.8270, 1365.7039, 1365.7087],
        [1365.7173, 1365.7145, 1365.7544, 1365.7228, 1365.6932, 1365.7616,
         1365.7506, 1365.7566, 1365.7159, 1365.7388, 1365.6680, 1365.6927],
        [1365.7163, 1365.7366, 1365.6726, 1365.7146, 1365.7175, 1365.6730,
         1365.6720, 1365.6570, 1365.6647, 1365.6759, 1365.7065, 1365.6926]
    ])
    n = TSI.shape[0]

    # Index year
    i = np.asarray(year).astype(int) - 1980

    # Extend backward assuming 11-year sunspot cycle and forward assuming
    # 13-year
    while np.any(i < 0):
        i[i < 0] += 11
    while np.any(i > n-1):
        i[i > n-1] -= 13

    # Index month
    j = np.asarray(month).astype(int) - 1

    # Return index scaled by TIM correction (Total Irradiance Monitor)
    return 0.9965*TSI[i, j]


def total_solar_irradiance(utc, method=None):
    """
    Calculate the total solar irradiance (W/m²) for given year.
    Year can be fractional.
    """

    if method is None:
        method = "ASHRAE"

    if callable(method):
        func = method

    else:
        method = method.upper()
        if method.startswith("A"):
            func = total_solar_irradiance_ashrae
        elif method.startswith("C"):
            func = total_solar_irradiance_cfsr
        elif method.startswith("E"):
            func = total_solar_irradiance_ashrae
        elif method.startswith("M"):
            func = total_solar_irradiance_merra2
        elif method.startswith("N"):
            func = total_solar_irradiance_ashrae
        else:
            raise NotImplementedError(method)

    return func(utc)


def hour_angle(lon, utc, eot):
    """
    Calculate local hour angle (radians) given longitude (degrees),
    date UTC (datetime64), and equation of time (radians)

    Hour angle is displacement of sun east or west
    """

    # Local solar hour angle (radians, noon = 0)
    hh, mm, ss = split_time(utc)
    H = 2*np.pi*((hh-12)/24 + mm/1440 + ss/86400)

    # Correct based on equation of time
    H += eot

    # Move to longitude location
    H += np.radians(lon)

    # Return centered in -pi to pi
    return ((H + np.pi) % (2*np.pi)) - np.pi


def sunset_hour_angle(sinLat, cosLat, sinDec, cosDec):
    """
    Calculate local sunset hour angle (radians) given sines and cosines
    of latitude and declination.
    """
    return np.arccos(np.clip(-sinDec/cosDec*sinLat/cosLat, -1, 1))


def position(lat, lon, utc, method='ASHRAE'):
    """
    Calculate solar position (x, y, z) in sky given (lat, lon) and UTC time
    """

    # Calculate solar coefficients
    sinDec, cosDec, eqnOfTime, solFactor = orbit(utc, method=method)

    # Calculate hour angle
    H = hour_angle(lon, utc, eqnOfTime)
    sinH = np.sin(H)
    cosH = np.cos(H)

    # Sun position
    sinLat = np.sin(np.radians(lat))
    cosLat = np.cos(np.radians(lat))

    return (
        cosDec*sinH,
        sinDec*cosLat - cosDec*sinLat*cosH,
        sinDec*sinLat + cosDec*cosLat*cosH
    )


def clear_sky_irradiance(z, tb, td, E0):
    """
    Calculate the ASHRAE clear sky beam normal and diffuse horizontal
    irradiance at elevation z, given pseudo-optical coefficients tb and td,
    and extra-terrestrial radiation E0
    """

    # Calculate air mass
    m = air_mass(z)

    # Calculate air mass exponents
    B1, B2, B3, B4 = 1.454, -0.406, -0.286, 0.021
    D1, D2, D3, D4 = 0.507, 0.205, -0.080, -0.190
    ab = B1 + B2*tb + B3*td + B4*tb*td
    ad = D1 + D2*tb + D3*td + D4*tb*td

    # Beam and diffuse irradiance
    return E0*np.exp(-tb*m**ab), E0*np.exp(-td*m**ad)


def elevation(lat, lon, utc, method='ASHRAE', interval=None, h=None):
    """
    Calculate the elevation z and extraterrestrial radiation E0 at
    (lat, lon) and UTC time.

    Result is either "instantaneous" (default) or the average over
    an "hourly" or "daily" interval.

    If hour angle h (rad) is supplied (e.g. solar noon = 0), instantaneous
    elevations will be based on h, otherwise h is calculated from UTC.
    """

    # Calculate solar coefficients at UTC
    sinDec, cosDec, eqnOfTime, solFactor = orbit(utc, method=method)

    # Calculate extraterrestrial radiance at UTC
    E0 = solFactor*total_solar_irradiance(utc, method=method)

    # Latitudinal sines
    sinLat = np.sin(np.radians(lat))
    cosLat = np.cos(np.radians(lat))

    def int_elevation(h):
        """
        Instant elevation at hour angle h
        """
        return np.maximum(sinDec*sinLat + cosDec*cosLat*np.cos(h), 0)

    def avg_elevation(h1, h2):
        """
        Integrated elevation between h1 and h2
        """
        return np.maximum(
            sinLat*sinDec*(h2-h1) + cosLat*cosDec*(np.sin(h2)-np.sin(h1)), 0
        )

    # Default interval is instantaneous
    if interval is None:
        interval = "instant"

    interval = interval.lower()[0]

    # Determine elevation
    if interval == "i":
        """
        Instantaneous
        """
        # Instantaneous hour angle
        if h is None:
            h = hour_angle(lon, utc, eqnOfTime)
        # Instantaneous elevation
        z = int_elevation(h)

    elif interval == "m":
        """
        Instantaneous mid-point of previous hour, i.e. approximate average
        """
        # Instantaneous hour angle at 30 minutes prior
        h = hour_angle(lon, utc - np.timedelta64(30, 'm'), eqnOfTime)
        # Instantaneous elevation
        z = int_elevation(h)

    elif interval == "h":
        """
        Hourly
        """
        # Sunset hour angle
        h0 = np.arccos(np.clip(-sinDec/cosDec*sinLat/cosLat, -1, 1))
        # One hour (radians)
        dh = np.pi/12
        # Start and end hour angles
        h = hour_angle(lon, utc, eqnOfTime)
        a = (h - dh + np.pi) % (2*np.pi) - np.pi
        b = a + dh
        # Default elevation is zero
        z = np.zeros_like(h)
        # Conditions
        a1 = (a < -h0)
        a2 = (a >= -h0) & (a < h0)
        #b1 = (b < -h0)
        b2 = (b >= -h0) & (b < h0)
        b3 = (b >= h0)
        # Dawn
        np.copyto(z, avg_elevation(-h0, b), where=a1 & b2)
        # Comes up very briefly between a & b
        np.copyto(z, avg_elevation(-h0, h0), where=a1 & b3)
        # Sun's up
        np.copyto(z, avg_elevation(a, b), where=a2 & b2)
        # Dusk
        np.copyto(z, avg_elevation(a, h0), where=a2 & b3)
        # Scale by interval
        z /= dh

    elif interval == "d":
        """
        Daily
        """
        # Sunset hour angle
        h = np.arccos(np.clip(-sinDec/cosDec*sinLat/cosLat, -1, 1))
        # Average daily elevation
        z = avg_elevation(-h, h)
        # Scale by 24-hour interval
        z /= 2*np.pi

    else:
        raise ValueError("Interval must be one of 'instant', 'midpoint', "
                         "'hourly', or 'daily'")

    return z, E0


def air_mass(z):
    """
    Calculate air mass based on Kasten & Young 1989
    """
    beta = np.degrees(np.arcsin(z))
    return 1/(z + 0.50572*(6.07995+beta)**-1.6364)


def erbs(Kt, **kwargs):
    """
    Calculate diffuse fraction as a function of clearness index kt
    via Erbs relation
    """
    Kt = np.asarray(Kt)
    Kd = 0.9511 - 0.1604*Kt + 4.388*Kt**2 - 16.638*Kt**3 + 12.336*Kt**4
    np.copyto(Kd, 1.0-0.09*Kt, where=Kt <= 0.22)
    np.copyto(Kd, 0.165, where=Kt > 0.80)
    return Kd


def orgill_hollands(Kt, **kwargs):
    """
    Calculate diffuse fraction as a function of clearness index kt
    via Orgill Hollands relation
    """
    Kt = np.asarray(Kt)
    Kd = 1.557 - 1.84*Kt
    np.copyto(Kd, 1.0-0.249*Kt, where=Kt <= 0.35)
    np.copyto(Kd, 0.177, where=Kt > 0.75)
    return Kd


def ruiz_arias(Kt, z, **kwargs):
    """
    Calculate diffuse fraction as a function of clearness index kt
    via Ruiz-Arias
    """
    m = air_mass(z)
    a = (0.944, 1.538, 2.808, -5.759, 2.276, -0.125, 0.013)
    return np.clip(
        a[0] - a[1]*np.exp(-np.exp(
            a[2]+a[3]*Kt+a[4]*Kt**2+a[5]*m+a[6]*m**2
        )),
        0, 1
    )


def engerer(Kt, Ktc, z, h, **kwargs):
    """
    Calculate diffuse fraction as a function of clearness index kt
    via the Engerer2 relation.

    kt is clearness index (E_gh/E0)
    ktc is clear sky clearness index (E_ghc/E0)
    z is cos(zenith), dimensionless
    h is hour angle, radians
    """

    # Apparent solar time in hours
    AST = 12/np.pi*h
    # Zenith angle in degrees
    theta_z = np.degrees(np.arccos(z))
    dKtc = Ktc - Kt
    Kde = np.maximum(0, 1.0-Ktc/Kt)
    C = 4.2336e-2
    beta = (-3.7912, 7.5479, -1.0036e-2, 3.1480e-3, -5.3146, 1.7073)
    return np.clip(
        C + (1.0-C)/(1.0 + np.exp(beta[0] + beta[1]*Kt + beta[2]*AST +
                     beta[3]*theta_z + beta[4]*dKtc)) + beta[5]*Kde,
        0, 1
    )


def to_utc(lst, tz=0):
    """
    Convert datetime64 in local standard time to UTC
    """
    return lst - np.timedelta64(int(np.rint(tz*60)), 'm')


def to_lst(utc, tz=0):
    """
    Convert datetime64 in UTC to local standard time
    """
    return utc + np.timedelta64(int(np.rint(tz*60)), 'm')


def to_altitude(z):
    """
    Convert z component of solar vector into altitude (deg)
    i.e. angle from horizon
    """
    return np.degrees(np.arcsin(z))


def to_zenith(z):
    """
    Convert z component of solar vector into zenith angle (deg)
    i.e. angle from vertical
    """
    return np.degrees(np.arccos(z))


def to_azimuth(x, y):
    """
    Convert x, y of solar vector into azimuth (deg)
    i.e. angle clockwise from North (+y)
    """
    return (-np.degrees(np.arctan2(x, y))) % 360


def nearest_hour(date):
    """
    Convert datetime64 to nearest hour
    """
    # Add 30 minutes
    date += np.timedelta64(30, "m")
    # Truncate on hour
    return date.astype("<M8[h]")


def fit_taus(zi, Kti, iter_max=42, eps_max=1e-6, plot=False, quiet=False):
    """
    Fit the ASHRAE pseudo-spectral coefficients tau_b & tau_d given a
    set of elevation z and clear sky index Kt values.
    """

    # Need at least two points
    if len(Kti) < 2:
        if not quiet:
            print("Warning: Insufficient points to fit taus")
        return np.nan, np.nan

    # First estimate
    tb, td = 0.4, 2.3

    # tau air mass exponent coefficients
    B1, B2, B3, B4 = 1.454, -0.406, -0.268, 0.021
    D1, D2, D3, D4 = 0.507, 0.205, -0.080, -0.190

    # Calculate air mass
    mi = air_mass(zi)
    logm = np.log(mi)

    # Newton iterations
    def calc(tb, td):
        # Current air mass exponents
        ab = B1 + B2*tb + B3*td + B4*tb*td
        ad = D1 + D2*tb + D3*td + D4*tb*td

        mab = mi**ab
        mad = mi**ad

        Kb = np.exp(-tb*mab)
        Kd = np.exp(-td*mad)/zi
        Kt = Kb + Kd

        # Form Jacobian J
        dKb_dtb = -Kb*mab*(1+tb*logm*(B2+B4*td))
        dKd_dtb = -Kd*mad*td*logm*(D2+D4*td)
        dKb_dtd = -Kb*mab*tb*logm*(B3+B4*tb)
        dKd_dtd = -Kd*mad*(1+td*logm*(D3+D4*tb))

        dKt_dtb = dKb_dtb + dKd_dtb
        dKt_dtd = dKb_dtd + dKd_dtd

        return Kt, dKt_dtb, dKt_dtd

    # Levenberg–Marquardt damping factor
    damping = 1

    taubs = [tb]
    tauds = [td]

    for i in range(iter_max):

        # Calculate current Kt and its gradient
        Kt, dKt_dtb, dKt_dtd = calc(tb, td)

        # Residuals
        dKt = Kti - Kt
        R = np.sum(dKt**2)

        # Form A, [J]^T[J]
        Abb = (1+damping)*np.sum(dKt_dtb**2)
        Abd = np.sum(dKt_dtb*dKt_dtd)
        Add = (1+damping)*np.sum(dKt_dtd**2)

        # Form forcing vector [J]^[dKt]
        Bb = np.sum(dKt_dtb*dKt)
        Bd = np.sum(dKt_dtd*dKt)

        # Solve A*t = B by Kramer's rule, Giddy-up
        try:
            detA = Abb*Add - Abd**2
            dtb = (Bb*Add - Bd*Abd)/detA
            dtd = (Abb*Bd - Abd*Bb)/detA
        except OverflowError:
            if not quiet:
                print("Warning: Determinant overflow while fitting taus")
            return np.nan, np.nan
        except ZeroDivisionError:
            if not quiet:
                print("Warning: Division by zero while fitting taus")
            return np.nan, np.nan
        except:
            raise

        # Test
        Ktt, dKtt_dtb, dKtt_dtd = calc(tb + dtb, td + dtd)
        Rt = np.sum((Kti-Ktt)**2)

        if (Rt >= R):
            # Worse (need more steep descent)
            damping *= 10

        else:
            # Better (need more Newton)
            damping /= 10

            # Correct
            tb += dtb
            td += dtd

            R = Rt

        taubs.append(tb)
        tauds.append(td)

        if (abs(dtb) < eps_max) and (abs(dtd) < eps_max):
            break

    else:
        # Exceeded iterMax iterations
        if not quiet:
            print("Warning: Exceeded", iter_max,
                  "iterations while fitting taus:", tb, td, dtb, dtd)

        return np.nan, np.nan

    if plot:
        plt.rc('text', usetex=True)
        plt.rc('text.latex', unicode=True)
        plt.rc('text.latex', preamble=r'\usepackage{cmbright}')
        f, ax = plt.subplots(figsize=(5, 5), dpi=200)
        ax.plot(mi, Kti, '.', color='orange', markersize=2)
        ax.plot(mi, Kt, '.', color='black', markersize=4)
        ax.set_xlabel("Air Mass $m$", fontsize='smaller')
        ax.set_ylabel("Clearness Index $K_t$", fontsize='smaller')
        txt = "\n".join(
            "%d points" % len(zi),
            "$\\tau_b$ = %.3f" % tb,
            "$\\tau_d$ = %.3f" % td
        )
        ax.text(0.9, 0.9, txt, ha="right", va="top", transform=ax.transAxes)
        plt.tight_layout()
        f.savefig("fit_taus_%d.png" % len(Kti), dpi=f.dpi, bbox_inches="tight")

    return tb, td


def fit_monthly_taus(z, Kt, lat=None, lon=None, noon_flux=False, **kwargs):

    # Loop over months
    months = list(range(1, 13))

    clear_sky = {
        'taub': [],
        'taud': [],
    }

    if noon_flux:
        clear_sky['Ebnoon'] = []
        clear_sky['Ednoon'] = []

    for month in months:
        # Restrict to month
        i = z.index.month == month

        # Fit via non-linear least squares
        taub, taud = fit_taus(z[i], Kt[i], **kwargs)

        clear_sky['taub'].append(taub)
        clear_sky['taud'].append(taud)

        if noon_flux:

            if np.isnan(taub) or np.isnan(taud):
                clear_sky['Ebnoon'].append(np.nan)
                clear_sky['Ednoon'].append(np.nan)
                continue

            # Calculate noon elevation and solar ETR on the 21st day
            utc = join_date(2001, m=month, d=21, hh=12)
            z_noon, E0_noon = elevation(
                lat, lon, utc, method='ASHRAE', interval='instant', h=0
            )

            # Calculate corresponding beam and diffuse irradiance
            Eb, Ed = clear_sky_irradiance(z_noon, taub, taud, E0_noon)
            clear_sky['Ebnoon'].append(Eb)
            clear_sky['Ednoon'].append(Ed)

    return clear_sky


def perez(Eb, Ed, E0, E0h, Td):
    """
    Calculate the global, direct, diffuse, and zenith illuminances from
    the beam, diffuse, extraterrestrial normal and direct irradiances and
    dew point temperature via the Perez (1990) relationships
    """

    # Sun up and working
    d = Ed > 0

    # Calculate elevation z=cosZ
    z = E0h[d]/E0[d]

    # Calculate zenith angle (radians)
    Z = np.arccos(z)
    Z3 = Z**3

    # Calculate air mass
    m = air_mass(z)

    # Sky clearness (eqn 1)
    kappa = 1.04
    epsilon = ((Ed[d]+Eb[d])/Ed[d] + kappa*Z3)/(1 + kappa*Z3)

    # Sky brightness (eqn 2)
    Delta = Ed[d]*m/E0[d]

    # Precipitable water (cm, eqn 3)
    W = np.exp(0.07*Td[d] - 0.075)

    # Sky clearness categories (from overcast to clear)
    bin_edges = [1, 1.065, 1.230, 1.500, 1.950, 2.800, 4.500, 6.200]

    # Find clearnness bin
    i = np.searchsorted(bin_edges, epsilon, side='right') - 1

    # Global luminous efficacy (table 4)
    ai = np.array([96.63, 107.54, 98.73, 92.72, 86.73, 88.34, 78.63, 99.65])
    bi = np.array([-0.47, 0.79, 0.70, 0.56, 0.98, 1.39, 1.47, 1.86])
    ci = np.array([11.50, 1.79, 4.40, 8.36, 7.10, 6.06, 4.93, -4.46])
    di = np.array([-9.16, -1.19, -6.95, -8.31, -10.94, -7.60, -11.37, -3.15])

    # Global illuminance (lux, eqn. 6)
    It = Ed.copy()
    It[d] = (Eb[d]*z+Ed[d])*(ai[i] + bi[i]*W + ci[i]*z + di[i]*np.log(Delta))

    # Direct luminous efficiacy (table 4)
    ai = np.array([57.20, 98.99, 109.83, 110.34, 106.36, 107.19, 105.75, 101.18])
    bi = np.array([-4.55, -3.46, -4.90, -5.84, -3.97, -1.25, 0.77, 1.58])
    ci = np.array([-2.98, -1.21, -1.71, -1.99, -1.75, -1.51, -1.25, -1.10])
    di = np.array([117.12, 12.38, -8.81, -4.56, -6.16, -26.73, -34.44, -8.29])

    # Direct illuminance (lux, eqn. 8)
    Ib = Ed.copy()
    Ib[d] = Eb[d]*(ai[i] + bi[i]*W + ci[i]*np.exp(5.73*Z-5) + di[i]*Delta)
    Ib = np.maximum(0, Ib)

    # Diffuse luminous efficiacy (table 4)
    ai = np.array([97.24, 107.22, 104.97, 102.39, 100.71, 106.42, 141.88, 152.23])
    bi = np.array([-0.46, 1.15, 2.96, 5.59, 5.94, 3.83, 1.90, 0.35])
    ci = np.array([12.00, 0.59, -5.53, -13.95, -22.75, -36.15, -53.24, -45.27])
    di = np.array([-8.91, -3.95, -8.77, -13.90, -23.74, -28.83, -14.03, -7.98])

    # Diffuse illuminance (lux, eqn. 7)
    Id = Ed.copy()
    Id[d] = Ed[d]*(ai[i] + bi[i]*W + ci[i]*z + di[i]*np.log(Delta))

    # Zenith luminance prediction (table 4)
    ai = np.array([40.86, 26.58, 19.34, 13.25, 14.47, 19.76, 28.39, 42.91])
    ci = np.array([26.77, 14.73, 2.28, -1.39, -5.09, -3.88, -9.67, -19.62])
    cip = np.array([-29.59, 58.46, 100.00, 124.79, 160.09, 154.61, 151.58, 130.80])
    di = np.array([-45.75, -21.25, 0.25, 15.66, 9.13, -19.21, -69.39, -164.08])

    # Zenith luminance (Cd/m2, eqn. 10)
    Lz = Ed.copy()
    Lz[d] = Ed[d]*(ai[i] + ci[i]*z + cip[i]*np.exp(-3*Z) + di[i]*Delta)

    return It, Ib, Id, Lz


def test_coeffs(year=2018):

    t1 = np.datetime64("%04d-01-01" % year)
    t2 = np.datetime64("%04d-01-01" % (year+1, ))

    utc = np.arange(t1, t2)

    f, ax = plt.subplots(4)

    for method in ['ashrae', 'energyplus', 'cfsr', 'merra2', 'noaa']:
        coeffs = orbit(utc, method=method)
        for i, ylabel in enumerate(['sinDec', 'cosDec',
                                    'eqnOfTime', 'solFactor']):
            ax[i].plot(utc, coeffs[i], label=method)
            ax[i].set_ylabel(ylabel)

    ax[i].legend(loc=0, fontsize='smaller')
    plt.tight_layout()
    plt.show()


def test_location(lat=33.64, lon=-84.43, dates=None):

    if dates is None:
        dates = [np.datetime64(datetime.datetime.utcnow())]

    for utc in dates:

        # 24 hours centered around UTC
        t = nearest_hour(utc) + np.arange(-12*60, 13*60, dtype="<m8[m]")
        print(nearest_hour(utc))

        f, ax = plt.subplots(3)

        for method in ['ashrae', 'energyplus', 'cfsr', 'merra', 'noaa']:
            x, y, z = position(lat, lon, t, method=method)
            ax[0].plot_date(t.astype(datetime.datetime), to_altitude(z),
                            label=method, fmt='-')
            ax[1].plot_date(t.astype(datetime.datetime), to_azimuth(x, y),
                            label=method, fmt='-')
            z, E0 = elevation(lat, lon, t, method=method)
            ax[2].plot_date(t.astype(datetime.datetime), E0*z, fmt='-',
                            label=method)
            x0, y0, z0 = position(lat, lon, utc, method=method)
            print(to_altitude(z0), to_azimuth(x0, y0))
    ax[0].set_ylabel("Alt")
    ax[1].set_ylabel("Azi")
    ax[2].set_ylabel("TOA Horz")
    ax[2].legend(loc='best', fontsize='smaller')
    f.autofmt_xdate()
    plt.tight_layout()
    plt.show()


def test_integration(lat=33.64, lon=-84.43, utc=None):

    if utc is None:
        utc = np.datetime64(datetime.datetime.utcnow())

    print("***")
    print(lat, lon, utc, to_lst(utc, tz=-5))

    for interval in ['instant', 'hourly', 'daily']:
        print(elevation(lat, lon, utc, method='ASHRAE', interval=None))


def test_solar_irradiance():

    years, months = np.mgrid[1979:2019, 1:13]

    utc = join_date(y=years.flatten(), m=months.flatten())

    f, ax = plt.subplots()

    for method in ['ashrae', 'cfsr', 'merra2']:
        tsi = total_solar_irradiance(utc, method=method)
        ax.plot_date(utc.astype(datetime.datetime), tsi, fmt='-',
                     label=method, clip_on=False)

    ax.set_ylim(1360, 1368)
    ax.get_yaxis().get_major_formatter().set_useOffset(False)
    ax.legend(loc='best', fontsize='smaller')
    ax.set_ylabel("TSI")
    plt.tight_layout()
    plt.show()


def test():

    test_solar_irradiance()
    test_coeffs(year=2011)
    test_location()
    test_integration(lat=43.5448, lon=-80.2482)


if __name__ == '__main__':
    test()
