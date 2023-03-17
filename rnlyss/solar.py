# -*- coding: utf-8 -*-
#
# Copyright 2019 Klimaat

import datetime
import numpy as np
from functools import partial


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


def split_date(dates):
    """
    Split datetime64 dates into year, month, day components.
    """
    y = dates.astype("<M8[Y]").astype(int) + 1970
    m = dates.astype("<M8[M]").astype(int) % 12 + 1
    d = (dates - dates.astype("<M8[M]")).astype("<m8[D]").astype(int) + 1
    return y, m, d


def split_time(dates):
    """
    Split datetime64 dates into hour, minute, second components.
    """
    hh = (dates - dates.astype("<M8[D]")).astype("<m8[h]").astype(int)
    mm = (dates - dates.astype("<M8[h]")).astype("<m8[m]").astype(int)
    ss = (dates - dates.astype("<M8[m]")).astype("<m8[s]").astype(int)
    return hh, mm, ss


def day_of_year(dates, snap=True):
    """
    Calculate the day of the year (0-365/366)
    """
    dt = np.asarray(dates) - dates.astype("<M8[Y]")
    if snap:
        # Provide value at noon (integer)
        # Jan 1st anytime = 1
        return dt.astype("<m8[D]").astype(int) + 1
    else:
        # Provide value including fractional part (float)
        # Jan 1st at 00:00 = 0, Jan 1st at noon = 0.5
        return dt.astype("<m8[s]").astype(int) / 86400


def julian_day(dates):
    """
    Julian day calculator
    """

    # Get Julian Day number
    y, m, d = split_date(dates)
    a = (14 - m) // 12
    y += 4800 - a
    m += 12 * a - 3
    jd = d + ((153 * m + 2) // 5) + 365 * y + y // 4 - y // 100 + y // 400 - 32045

    # Get fractional day (noon=0)
    hh, mm, ss = split_time(dates)
    fd = (hh - 12) / 24 + mm / 1440 + ss / 86400

    return jd, fd


def delta_time(utc, method="NASA"):
    """
    Estimate difference between terrestrial time (TT) and universal time (UT)
    Returns NaNs for years outside approximations.
    """

    # Year fraction (approximate)
    y, m, _ = split_date(utc)
    yf = y + (m - 0.5) / 12

    def poly_val(y, yn=0, an=[]):
        dy = y - yn
        s = np.zeros_like(y)
        for k in range(len(an)):
            s += an[k] * dy ** k
        return s

    if method == "SG2":
        """
        Based on SG2 approximations
        Ref: Blanc and Wald 2012
        "The SG2 algorithm for fast and accurate computation of the position of
        the sun for multi-decadal time period"
        Solar Energy v88, p3072--3083.
        """
        return np.piecewise(
            yf,
            [
                (y >= 1961) & (y < 1986),
                (y >= 1986) & (y < 2005),
                (y >= 2005) & (y < 2050),
            ],
            [
                partial(poly_val, yn=1975, an=[45.45, 1.067, -1 / 260, -1 / 718]),
                partial(
                    poly_val,
                    yn=2000,
                    an=[63.86, 0.3345, -0.060374, 0.0017275, 6.51814e-4, 2.373599e-5],
                ),
                partial(poly_val, yn=2000, an=[62.8938127, 0.32100612, 0.005576068]),
                np.nan,
            ],
        )

    """
    Default
    Based on polynomial approximations from
    https://eclipse.gsfc.nasa.gov/SEcat5/deltatpoly.html
    Just for modern era: 1800-2150
    """

    return np.piecewise(
        yf,
        [
            (y >= 1800) & (y < 1860),
            (y >= 1860) & (y < 1900),
            (y >= 1900) & (y < 1920),
            (y >= 1920) & (y < 1941),
            (y >= 1941) & (y < 1961),
            (y >= 1961) & (y < 1986),
            (y >= 1986) & (y < 2005),
            (y >= 2005) & (y < 2050),
            (y >= 2050) & (y < 2150),
        ],
        [
            partial(
                poly_val,
                yn=1800,
                an=[
                    13.72,
                    -0.332447,
                    6.8612e-3,
                    4.1116e-3,
                    -3.7436e-4,
                    1.21272e-5,
                    -1.699e-7,
                    8.75e-10,
                ],
            ),
            partial(
                poly_val,
                yn=1860,
                an=[7.62, 0.5737, -0.251754, 0.01680668, -0.0004473624, 1 / 233174],
            ),
            partial(
                poly_val,
                yn=1900,
                an=[-2.79, 1.494119, -0.0598939, 0.0061966, -0.000197],
            ),
            partial(poly_val, yn=1920, an=[21.20, 0.84493, -0.076100, 0.0020936]),
            partial(poly_val, yn=1950, an=[29.07, 0.407, -1 / 233, 1 / 2547]),
            partial(poly_val, yn=1975, an=[45.45, 1.067, -1 / 260, -1 / 718]),
            partial(
                poly_val,
                yn=2000,
                an=[63.86, 0.3345, -0.060374, 0.0017275, 6.51814e-4, 2.373599e-5],
            ),
            partial(poly_val, yn=2000, an=[62.92, 0.32217, 0.005589]),
            partial(poly_val, yn=1820, an=[-20 - 330 * 0.5624, 0.5628, 0.0032]),
            np.nan,
        ],
    )


def orbit_ashrae(utc):
    """
    Calculate solar parameters based on ASHRAE methodology.

    Ref. ASHRAE HOF 2017, Chap 14
    """

    # Day of year
    n = day_of_year(utc, snap=True)

    # Declination (eqn. 10, radians)
    decl = np.radians(23.45 * np.sin(2 * np.pi * (n + 284) / 365))

    # Equation of time (eqns 5 & 6, min)
    gamma = 2 * np.pi * (n - 1) / 365
    eqnOfTime = 2.2918 * (
        0.0075
        + 0.1868 * np.cos(gamma)
        - 3.2077 * np.sin(gamma)
        - 1.4615 * np.cos(2 * gamma)
        - 4.089 * np.sin(2 * gamma)
    )

    # Convert from minutes to radians
    eqnOfTime *= np.pi / (60 * 12)

    # Solar constant correction
    solFactor = 1 + 0.033 * np.cos(np.radians(360 * (n - 3) / 365))

    return np.sin(decl), np.cos(decl), eqnOfTime, solFactor


def orbit_energyplus(utc):
    """
    Calculate solar coefficients based on EnergyPlus

    Ref. WeatherManager.cc, function CalculateDailySolarCoeffs
    """

    # Day of year
    n = day_of_year(utc, snap=True)

    # Day Angle
    D = 2 * np.pi * n / 366.0

    sinD = np.sin(D)
    cosD = np.cos(D)

    # Calculate declination sines & cosines

    sinDec = (
        0.00561800
        + 0.0657911 * sinD
        - 0.392779 * cosD
        + 0.00064440 * (sinD * cosD * 2.0)
        - 0.00618495 * (cosD ** 2 - sinD ** 2)
        - 0.00010101 * (sinD * (cosD ** 2 - sinD ** 2) + cosD * (sinD * cosD * 2.0))
        - 0.00007951 * (cosD * (cosD ** 2 - sinD ** 2) - sinD * (sinD * cosD * 2.0))
        - 0.00011691 * (2.0 * (sinD * cosD * 2.0) * (cosD ** 2 - sinD ** 2))
        + 0.00002096 * ((cosD ** 2 - sinD ** 2) ** 2 - (sinD * cosD * 2.0) ** 2)
    )

    cosDec = np.sqrt(1 - sinDec ** 2)

    # Equation of time (hours)

    eqnOfTime = (
        0.00021971
        - 0.122649 * sinD
        + 0.00762856 * cosD
        - 0.156308 * (sinD * cosD * 2.0)
        - 0.0530028 * (cosD ** 2 - sinD ** 2)
        - 0.00388702 * (sinD * (cosD ** 2 - sinD ** 2) + cosD * (sinD * cosD * 2.0))
        - 0.00123978 * (cosD * (cosD ** 2 - sinD ** 2) - sinD * (sinD * cosD * 2.0))
        - 0.00270502 * (2.0 * (sinD * cosD * 2.0) * (cosD ** 2 - sinD ** 2))
        - 0.00167992 * ((cosD ** 2 - sinD ** 2) ** 2 - (sinD * cosD * 2.0) ** 2)
    )

    # Convert to radians
    eqnOfTime = np.pi * eqnOfTime / 12

    # Solar constant correction factor
    solFactor = 1.000047 + 0.000352615 * sinD + 0.0334454 * cosD

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
    t1 = (jd - jdor) / 36525.0

    # Length of anomalistic and tropical years (minus 365 days)
    ayear = 0.25964134e0 + 0.304e-5 * t1
    tyear = 0.24219879e0 - 0.614e-5 * t1

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
    qq = deleqn * 2 * np.pi / ayear

    def solve_kepler(e, M, E=1, eps=1.3e-6):
        """
        Solve Kepler equation for eccentric anomaly E by Newton's method
        based on eccentricity e and mean anomaly M
        """
        for i in range(10):
            dE = -(E - e * np.sin(E) - M) / (1 - e * np.cos(E))
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
    eq = 2.0 * np.arctan(er * np.tan(0.5 * e1))

    # Date is days since last perihelion passage
    dat = jd - jdor - tpp + fjd
    date = dat % ayear

    # Mean anomaly
    em = 2 * np.pi * date / ayear

    # Eccentric anomaly
    e1 = solve_kepler(ec, em)

    # True anomaly
    w1 = 2.0 * np.arctan(er * np.tan(0.5 * e1))

    # Earth-Sun radius relative to mean radius
    r1 = 1.0 - ec * np.cos(e1)

    # Sine of declination angle
    # NB. ecliptic longitude = w1 - eq
    sdec = sni * np.sin(w1 - eq)

    # Cosine of declination angle
    cdec = np.sqrt(1.0 - sdec * sdec)

    # Sun declination (radians)
    dlt = np.arcsin(sdec)

    # Sun right ascension (radians)
    alp = np.arcsin(np.tan(dlt) * tini)
    alp = np.where(np.cos(w1 - eq) < 0, np.pi - alp, alp)
    alp = np.where(alp < 0, alp + 2 * np.pi, alp)

    # Equation of time (radians)
    sun = 2 * np.pi * (date - deleqn) / ayear
    sun = np.where(sun < 0.0, sun + 2 * np.pi, sun)
    slag = sun - alp - 0.03255

    # Solar constant correction factor (inversely with radius squared)
    solFactor = 1 / (r1 ** 2)

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
        np.sin(np.radians(gma)) * (1.914602 - jc * (0.004817 + 0.000014 * jc))
        + np.sin(np.radians(2 * gma)) * (0.019993 - 0.000101 * jc)
        + np.sin(np.radians(3 * gma)) * 0.000289
    )

    # Sun true longitude (deg)
    stl = gml + ctr

    # Sun true anomaly (deg)
    sta = gma + ctr

    # Sun radius vector (AUs)
    rad = (1.000001018 * (1 - ecc * ecc)) / (1 + ecc * np.cos(np.radians(sta)))

    # Sun apparent longitude (deg)
    sal = stl - 0.00569 - 0.00478 * np.sin(np.radians(125.04 - 1934.136 * jc))

    # Mean obliquity ecliptic (deg)
    moe = (
        23
        + (26 + ((21.448 - jc * (46.815 + jc * (0.00059 - jc * 0.001813)))) / 60) / 60
    )

    # Obliquity correction (deg)
    obl = moe + 0.00256 * np.cos(np.radians(125.04 - 1934.136 * jc))

    # Sun right ascension (deg)
    sra = np.degrees(
        np.arctan2(
            np.cos(np.radians(obl)) * np.sin(np.radians(sal)), np.cos(np.radians(sal))
        )
    )

    # Sun declination
    sinDec = np.sin(np.radians(obl)) * np.sin(np.radians(sal))
    cosDec = np.sqrt(1.0 - sinDec * sinDec)

    # Var y
    vary = np.tan(np.radians(obl / 2)) * np.tan(np.radians(obl / 2))

    # Equation of time (minutes)
    eqnOfTime = 4 * np.degrees(
        vary * np.sin(2 * np.radians(gml))
        - 2 * ecc * np.sin(np.radians(gma))
        + 4 * ecc * vary * np.sin(np.radians(gma)) * np.cos(2 * np.radians(gml))
        - 0.5 * vary * vary * np.sin(4 * np.radians(gml))
        - 1.25 * ecc * ecc * np.sin(2 * np.radians(gma))
    )

    # Convert from minutes to radians
    eqnOfTime *= np.pi / (60 * 12)

    # Solar constant correction factor (inversely with radius squared)
    solFactor = 1 / (rad ** 2)

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

        omg = (2.0 * np.pi / yearlen) / np.sqrt(1 - ecc ** 2) ** 3
        sob = np.sin(obliquity)

        # TH: Orbit anomaly
        # ZS: Sine of declination
        # ZC: Cosine of declination
        # PP: Inverse of square of earth-sun distance

        # Integration starting at vernal equinox
        def calc_omega(th):
            return omg * (1.0 - ecc * np.cos(th - perihelion)) ** 2

        orbit = np.recarray(
            (days_per_cycle,),
            dtype=[("th", float), ("zs", float), ("zc", float), ("pp", float)],
        )

        def update_orbit(th):
            zs = np.sin(th) * sob
            zc = np.sqrt(1.0 - zs ** 2)
            pp = ((1.0 - ecc * np.cos(th - perihelion)) / (1.0 - ecc ** 2)) ** 2
            orbit[kp] = th, zs, zc, pp

        # Starting point
        th = 0
        kp = equinox
        update_orbit(th)

        # Runge-Kutta
        for k in range(days_per_cycle - 1):
            t1 = calc_omega(th)
            t2 = calc_omega(th + 0.5 * t1)
            t3 = calc_omega(th + 0.5 * t2)
            t4 = calc_omega(th + t3)
            kp = (kp + 1) % days_per_cycle
            th += (t1 + 2 * (t2 + t3) + t4) / 6.0
            update_orbit(th)

        # Cache it
        orbit_merra2.orbit = orbit

    else:

        orbit = orbit_merra2.orbit

    # Map into orbit
    year, month, day = split_date(utc)
    doy = day_of_year(utc, snap=True)
    iyear = (year - 1) % 4
    iday = iyear * int(yearlen) + doy - 1

    # Declination
    sinDec = orbit["zs"][iday]
    cosDec = orbit["zc"][iday]

    # MERRA uses *solar* instead of *clock* time; no equation of time
    eqnOfTime = np.zeros_like(sinDec)

    # Inverse square of earth-sun distance ratio to mean distance
    solFactor = orbit["pp"][iday]

    return sinDec, cosDec, eqnOfTime, solFactor


# For caching MERRA-2 orbit
orbit_merra2.orbit = None


def orbit_era5(utc):
    """
    Orbit as per ERA5 code.

    From snippet of IFS code (ECMWF)
    """

    # 1 astronomical unit (m)
    REA = 149597870000

    # Number of seconds in a day
    RDAY = 86400

    # Seconds from start of year
    PTIME = day_of_year(utc) * RDAY

    # Fraction of year
    PTETA = PTIME / (RDAY * 365.25)

    REL = 1.7535 + 6.283076 * PTETA
    REM = 6.240075 + 6.283020 * PTETA

    # Sun-Earth distance
    RRS = REA * (1.0001 - 0.0163 * np.sin(REL) + 0.0037 * np.cos(REL))
    solFactor = (REA / RRS) ** 2

    # Relative movement Sun/Earth
    RLLS = 4.8951 + 6.283076 * PTETA

    # Declination
    RLLLS = (
        4.8952
        + 6.283320 * PTETA
        - 0.0075 * np.sin(REL)
        - 0.0326 * np.cos(REL)
        - 0.0003 * np.sin(2.0 * REL)
        + 0.0002 * np.cos(2.0 * REL)
    )

    # Max declination 23.44°
    REPSM = 0.409093

    RDS = np.arcsin(np.sin(REPSM) * np.sin(RLLLS))
    sinDec = np.sin(RDS)
    cosDec = np.sqrt(1 - sinDec ** 2)

    # Equation of time
    RET = (
        591.8 * np.sin(2.0 * RLLS)
        - 459.4 * np.sin(REM)
        + 39.5 * np.sin(REM) * np.cos(2.0 * RLLS)
        - 12.7 * np.sin(4.0 * RLLS)
        - 4.8 * np.sin(2.0 * REM)
    )
    eqnOfTime = RET * 2 * np.pi / RDAY

    return sinDec, cosDec, eqnOfTime, solFactor


def orbit_sg2(utc):
    """
    Orbit based on SG2 algorithm

    Ref: Blanc and Wald 2012
    "The SG2 algorithm for fast and accurate computation of the position of
    the sun for multi-decadal time period"
    Solar Energy v88, p3072--3083.

    Converted from available Matlab functions
    """

    # Calculate difference between terrestrial time (TT) and universal time (UT)
    dt = delta_time(utc, method="SG2")

    # Calculate Julian day in terrestrial time
    jd, fjd = julian_day(utc)
    j_tt = jd + fjd + dt / 86400

    # Evaluate sinusoidal
    def sin_val(j, j0=2444239.5, a=0, b=0, N=0, f=[], rho=[], phi=[]):
        jc = j - j0
        s = np.zeros_like(j)

        # Add linear components
        s += a * jc + b

        # Add sinuisoidal components
        for k in range(N):
            s += rho[k] * np.cos(2 * np.pi * f[k] * jc - phi[k])

        return s

    # Shift values into periodic bounds
    # If s=0, then [0, 2pi] (modulus)
    # If s=0.5 then [-pi, pi]
    def per_val(x, s=0, n=2 * np.pi):
        return x - np.floor(x / n + s) * n

    # Heliocentric radius
    R = sin_val(
        j_tt, a=0, b=1.000140, N=1, f=[1 / 365.254902], rho=[0.016704], phi=[-3.091159]
    )
    solFactor = (1 / R) ** 2

    # Heliocentric longitude
    L = sin_val(
        j_tt,
        a=1 / 58.130101,
        b=1.742145,
        N=10,
        f=1.0
        / np.asarray(
            [
                365.261278,
                182.632412,
                29.530634,
                399.529850,
                291.956812,
                583.598201,
                4652.629372,
                1450.236684,
                199.459709,
                365.355291,
            ]
        ),
        rho=[
            3.401508e-2,
            3.486440e-4,
            3.136227e-5,
            3.578979e-5,
            2.676185e-5,
            2.333925e-5,
            1.221214e-5,
            1.217941e-5,
            1.343914e-5,
            8.499475e-4,
        ],
        phi=[
            1.600780,
            1.662976,
            -1.195905,
            -1.042052,
            2.012613,
            -2.867714,
            1.225038,
            -0.828601,
            -3.108253,
            -2.353709,
        ],
    )

    # Nutation of sun geocentric longitude [-π, π]
    dpsi = per_val(
        sin_val(
            j_tt, a=0, b=0, N=1, f=[1 / 6791.164405], rho=[8.329092e-5], phi=[-2.052757]
        ),
        0.5,
    )

    # Stellar aberrration correction (cf. -9.933735e-5)
    dtau = np.radians(-20.4898 / 3600)  #

    # Apparent sun geocentric longitude
    theta_a = L + np.pi + dpsi + dtau

    # True earth obliquity/tilt
    epsilon = sin_val(
        j_tt,
        a=-6.216374e-9,
        b=4.091383e-1,
        N=1,
        f=[1 / 6791.164405],
        rho=[4.456183e-5],
        phi=[4.091383e-1],
    )

    # Declination
    cos_epsilon = np.cos(epsilon)
    sin_theta_a = np.sin(theta_a)
    sinDec = sin_theta_a * np.sin(epsilon)
    cosDec = np.sqrt(1 - sinDec ** 2)

    # Right ascension
    r_alpha_g = np.arctan2(sin_theta_a * cos_epsilon, np.cos(theta_a))

    # Equation of time [-π, π]
    M = sin_val(j_tt, a=1 / 58.130099904, b=-1.399410798, N=0)
    eqnOfTime = per_val(M - 0.0001 - r_alpha_g + dpsi * cos_epsilon, s=0.5)

    return sinDec, cosDec, eqnOfTime, solFactor


def orbit_spa(utc):
    """
    Orbit based on NREL's SPA algorithm

    Ref: I. Reda and A. Andreas
    "Solar Position Algorithms for Solar Radiation Applications"
    NREL/TP-560-34302, Jan 2008.
    """

    # Julian date
    jd, fjd = julian_day(utc)
    jd = jd + fjd

    # Calculate difference between terrestrial time (TT) and universal time (UT)
    dt = delta_time(utc)

    # Julian emphemeris cenutry
    jce = (jd + dt / 86400.0 - 2451545) / 36525

    # Julian emphemeris millenium
    jme = jce / 10

    # Calculate quantities by summing terms
    def sum_terms(jme, ABC):
        sx = 0
        for i, x in enumerate(ABC):
            sy = 0
            for y in x:
                # y = [A, B, C]
                sy += y[0] * np.cos(y[1] + y[2] * jme)
            sx += sy * jme ** i
        return sx / 1e8

    # Earth heliocentric longitude

    L_ABC = [
        [
            [175347046.0, 0, 0],
            [3341656.0, 4.6692568, 6283.07585],
            [34894.0, 4.6261, 12566.1517],
            [3497.0, 2.7441, 5753.3849],
            [3418.0, 2.8289, 3.5231],
            [3136.0, 3.6277, 77713.7715],
            [2676.0, 4.4181, 7860.4194],
            [2343.0, 6.1352, 3930.2097],
            [1324.0, 0.7425, 11506.7698],
            [1273.0, 2.0371, 529.691],
            [1199.0, 1.1096, 1577.3435],
            [990, 5.233, 5884.927],
            [902, 2.045, 26.298],
            [857, 3.508, 398.149],
            [780, 1.179, 5223.694],
            [753, 2.533, 5507.553],
            [505, 4.583, 18849.228],
            [492, 4.205, 775.523],
            [357, 2.92, 0.067],
            [317, 5.849, 11790.629],
            [284, 1.899, 796.298],
            [271, 0.315, 10977.079],
            [243, 0.345, 5486.778],
            [206, 4.806, 2544.314],
            [205, 1.869, 5573.143],
            [202, 2.458, 6069.777],
            [156, 0.833, 213.299],
            [132, 3.411, 2942.463],
            [126, 1.083, 20.775],
            [115, 0.645, 0.98],
            [103, 0.636, 4694.003],
            [102, 0.976, 15720.839],
            [102, 4.267, 7.114],
            [99, 6.21, 2146.17],
            [98, 0.68, 155.42],
            [86, 5.98, 161000.69],
            [85, 1.3, 6275.96],
            [85, 3.67, 71430.7],
            [80, 1.81, 17260.15],
            [79, 3.04, 12036.46],
            [75, 1.76, 5088.63],
            [74, 3.5, 3154.69],
            [74, 4.68, 801.82],
            [70, 0.83, 9437.76],
            [62, 3.98, 8827.39],
            [61, 1.82, 7084.9],
            [57, 2.78, 6286.6],
            [56, 4.39, 14143.5],
            [56, 3.47, 6279.55],
            [52, 0.19, 12139.55],
            [52, 1.33, 1748.02],
            [51, 0.28, 5856.48],
            [49, 0.49, 1194.45],
            [41, 5.37, 8429.24],
            [41, 2.4, 19651.05],
            [39, 6.17, 10447.39],
            [37, 6.04, 10213.29],
            [37, 2.57, 1059.38],
            [36, 1.71, 2352.87],
            [36, 1.78, 6812.77],
            [33, 0.59, 17789.85],
            [30, 0.44, 83996.85],
            [30, 2.74, 1349.87],
            [25, 3.16, 4690.48],
        ],
        [
            [628331966747.0, 0, 0],
            [206059.0, 2.678235, 6283.07585],
            [4303.0, 2.6351, 12566.1517],
            [425.0, 1.59, 3.523],
            [119.0, 5.796, 26.298],
            [109.0, 2.966, 1577.344],
            [93, 2.59, 18849.23],
            [72, 1.14, 529.69],
            [68, 1.87, 398.15],
            [67, 4.41, 5507.55],
            [59, 2.89, 5223.69],
            [56, 2.17, 155.42],
            [45, 0.4, 796.3],
            [36, 0.47, 775.52],
            [29, 2.65, 7.11],
            [21, 5.34, 0.98],
            [19, 1.85, 5486.78],
            [19, 4.97, 213.3],
            [17, 2.99, 6275.96],
            [16, 0.03, 2544.31],
            [16, 1.43, 2146.17],
            [15, 1.21, 10977.08],
            [12, 2.83, 1748.02],
            [12, 3.26, 5088.63],
            [12, 5.27, 1194.45],
            [12, 2.08, 4694],
            [11, 0.77, 553.57],
            [10, 1.3, 6286.6],
            [10, 4.24, 1349.87],
            [9, 2.7, 242.73],
            [9, 5.64, 951.72],
            [8, 5.3, 2352.87],
            [6, 2.65, 9437.76],
            [6, 4.67, 4690.48],
        ],
        [
            [52919.0, 0, 0],
            [8720.0, 1.0721, 6283.0758],
            [309.0, 0.867, 12566.152],
            [27, 0.05, 3.52],
            [16, 5.19, 26.3],
            [16, 3.68, 155.42],
            [10, 0.76, 18849.23],
            [9, 2.06, 77713.77],
            [7, 0.83, 775.52],
            [5, 4.66, 1577.34],
            [4, 1.03, 7.11],
            [4, 3.44, 5573.14],
            [3, 5.14, 796.3],
            [3, 6.05, 5507.55],
            [3, 1.19, 242.73],
            [3, 6.12, 529.69],
            [3, 0.31, 398.15],
            [3, 2.28, 553.57],
            [2, 4.38, 5223.69],
            [2, 3.75, 0.98],
        ],
        [
            [289.0, 5.844, 6283.076],
            [35, 0, 0],
            [17, 5.49, 12566.15],
            [3, 5.2, 155.42],
            [1, 4.72, 3.52],
            [1, 5.3, 18849.23],
            [1, 5.97, 242.73],
        ],
        [[114.0, 3.142, 0], [8, 4.13, 6283.08], [1, 3.84, 12566.15]],
        [[1, 3.14, 0]],
    ]

    L = np.degrees(sum_terms(jme, L_ABC)) % 360

    # Earth heliocentric latitude

    B_ABC = [
        [
            [280.0, 3.199, 84334.662],
            [102.0, 5.422, 5507.553],
            [80, 3.88, 5223.69],
            [44, 3.7, 2352.87],
            [32, 4, 1577.34],
        ],
        [[9, 3.9, 5507.55], [6, 1.73, 5223.69]],
    ]

    B = np.degrees(sum_terms(jme, B_ABC))

    # Earth radius vector (AU)

    R_ABC = [
        [
            [100013989.0, 0, 0],
            [1670700.0, 3.0984635, 6283.07585],
            [13956.0, 3.05525, 12566.1517],
            [3084.0, 5.1985, 77713.7715],
            [1628.0, 1.1739, 5753.3849],
            [1576.0, 2.8469, 7860.4194],
            [925.0, 5.453, 11506.77],
            [542.0, 4.564, 3930.21],
            [472.0, 3.661, 5884.927],
            [346.0, 0.964, 5507.553],
            [329.0, 5.9, 5223.694],
            [307.0, 0.299, 5573.143],
            [243.0, 4.273, 11790.629],
            [212.0, 5.847, 1577.344],
            [186.0, 5.022, 10977.079],
            [175.0, 3.012, 18849.228],
            [110.0, 5.055, 5486.778],
            [98, 0.89, 6069.78],
            [86, 5.69, 15720.84],
            [86, 1.27, 161000.69],
            [65, 0.27, 17260.15],
            [63, 0.92, 529.69],
            [57, 2.01, 83996.85],
            [56, 5.24, 71430.7],
            [49, 3.25, 2544.31],
            [47, 2.58, 775.52],
            [45, 5.54, 9437.76],
            [43, 6.01, 6275.96],
            [39, 5.36, 4694],
            [38, 2.39, 8827.39],
            [37, 0.83, 19651.05],
            [37, 4.9, 12139.55],
            [36, 1.67, 12036.46],
            [35, 1.84, 2942.46],
            [33, 0.24, 7084.9],
            [32, 0.18, 5088.63],
            [32, 1.78, 398.15],
            [28, 1.21, 6286.6],
            [28, 1.9, 6279.55],
            [26, 4.59, 10447.39],
        ],
        [
            [103019.0, 1.10749, 6283.07585],
            [1721.0, 1.0644, 12566.1517],
            [702.0, 3.142, 0],
            [32, 1.02, 18849.23],
            [31, 2.84, 5507.55],
            [25, 1.32, 5223.69],
            [18, 1.42, 1577.34],
            [10, 5.91, 10977.08],
            [9, 1.42, 6275.96],
            [9, 0.27, 5486.78],
        ],
        [
            [4359.0, 5.7846, 6283.0758],
            [124.0, 5.579, 12566.152],
            [12, 3.14, 0],
            [9, 3.63, 77713.77],
            [6, 1.87, 5573.14],
            [3, 5.47, 18849.23],
        ],
        [[145.0, 4.273, 6283.076], [7, 3.92, 12566.15]],
        [[4, 2.56, 6283.08]],
    ]

    R = sum_terms(jme, R_ABC)
    solFactor = (1 / R) ** 2

    # Geocentric longitude, latitude
    theta = (L + 180) % 360
    beta = np.radians(-B)

    # Third-order polynomial evaluation
    def poly3_val(x, a, b, c, d):
        return ((a * x + b) * x + c) * x + d

    # Mean elongation moon sun
    X0 = poly3_val(jce, 1.0 / 189474.0, -0.0019142, 445267.11148, 297.85036)

    # Mean anomaly sun
    X1 = poly3_val(jce, -1.0 / 300000.0, -0.0001603, 35999.05034, 357.52772)

    # Mean anomaly moon
    X2 = poly3_val(jce, 1.0 / 56250.0, 0.0086972, 477198.867398, 134.96298)

    # Argument latitude moon
    X3 = poly3_val(jce, 1.0 / 327270.0, -0.0036825, 483202.017538, 93.27191)

    # Ascending longitude moon
    X4 = poly3_val(jce, 1.0 / 450000.0, 0.0020708, -1934.136261, 125.04452)

    # Y coefficients from Table A4.3
    Y = [
        [0, 0, 0, 0, 1],
        [-2, 0, 0, 2, 2],
        [0, 0, 0, 2, 2],
        [0, 0, 0, 0, 2],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [-2, 1, 0, 2, 2],
        [0, 0, 0, 2, 1],
        [0, 0, 1, 2, 2],
        [-2, -1, 0, 2, 2],
        [-2, 0, 1, 0, 0],
        [-2, 0, 0, 2, 1],
        [0, 0, -1, 2, 2],
        [2, 0, 0, 0, 0],
        [0, 0, 1, 0, 1],
        [2, 0, -1, 2, 2],
        [0, 0, -1, 0, 1],
        [0, 0, 1, 2, 1],
        [-2, 0, 2, 0, 0],
        [0, 0, -2, 2, 1],
        [2, 0, 0, 2, 2],
        [0, 0, 2, 2, 2],
        [0, 0, 2, 0, 0],
        [-2, 0, 1, 2, 2],
        [0, 0, 0, 2, 0],
        [-2, 0, 0, 2, 0],
        [0, 0, -1, 2, 1],
        [0, 2, 0, 0, 0],
        [2, 0, -1, 0, 1],
        [-2, 2, 0, 2, 2],
        [0, 1, 0, 0, 1],
        [-2, 0, 1, 0, 1],
        [0, -1, 0, 0, 1],
        [0, 0, 2, -2, 0],
        [2, 0, -1, 2, 1],
        [2, 0, 1, 2, 2],
        [0, 1, 0, 2, 2],
        [-2, 1, 1, 0, 0],
        [0, -1, 0, 2, 2],
        [2, 0, 0, 2, 1],
        [2, 0, 1, 0, 0],
        [-2, 0, 2, 2, 2],
        [-2, 0, 1, 2, 1],
        [2, 0, -2, 0, 1],
        [2, 0, 0, 0, 1],
        [0, -1, 1, 0, 0],
        [-2, -1, 0, 2, 1],
        [-2, 0, 0, 0, 1],
        [0, 0, 2, 2, 1],
        [-2, 0, 2, 0, 1],
        [-2, 1, 0, 2, 1],
        [0, 0, 1, -2, 0],
        [-1, 0, 1, 0, 0],
        [-2, 1, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 0, 1, 2, 0],
        [0, 0, -2, 2, 2],
        [-1, -1, 1, 0, 0],
        [0, 1, 1, 0, 0],
        [0, -1, 1, 2, 2],
        [2, -1, -1, 2, 2],
        [0, 0, 3, 2, 2],
        [2, -1, 0, 2, 2],
    ]

    # abcd coefficients from Table A4.3
    abcd = [
        [-171996, -174.2, 92025, 8.9],
        [-13187, -1.6, 5736, -3.1],
        [-2274, -0.2, 977, -0.5],
        [2062, 0.2, -895, 0.5],
        [1426, -3.4, 54, -0.1],
        [712, 0.1, -7, 0],
        [-517, 1.2, 224, -0.6],
        [-386, -0.4, 200, 0],
        [-301, 0, 129, -0.1],
        [217, -0.5, -95, 0.3],
        [-158, 0, 0, 0],
        [129, 0.1, -70, 0],
        [123, 0, -53, 0],
        [63, 0, 0, 0],
        [63, 0.1, -33, 0],
        [-59, 0, 26, 0],
        [-58, -0.1, 32, 0],
        [-51, 0, 27, 0],
        [48, 0, 0, 0],
        [46, 0, -24, 0],
        [-38, 0, 16, 0],
        [-31, 0, 13, 0],
        [29, 0, 0, 0],
        [29, 0, -12, 0],
        [26, 0, 0, 0],
        [-22, 0, 0, 0],
        [21, 0, -10, 0],
        [17, -0.1, 0, 0],
        [16, 0, -8, 0],
        [-16, 0.1, 7, 0],
        [-15, 0, 9, 0],
        [-13, 0, 7, 0],
        [-12, 0, 6, 0],
        [11, 0, 0, 0],
        [-10, 0, 5, 0],
        [-8, 0, 3, 0],
        [7, 0, -3, 0],
        [-7, 0, 0, 0],
        [-7, 0, 3, 0],
        [-7, 0, 3, 0],
        [6, 0, 0, 0],
        [6, 0, -3, 0],
        [6, 0, -3, 0],
        [-6, 0, 3, 0],
        [-6, 0, 3, 0],
        [5, 0, 0, 0],
        [-5, 0, 3, 0],
        [-5, 0, 3, 0],
        [-5, 0, 3, 0],
        [4, 0, 0, 0],
        [4, 0, 0, 0],
        [4, 0, 0, 0],
        [-4, 0, 0, 0],
        [-4, 0, 0, 0],
        [-4, 0, 0, 0],
        [3, 0, 0, 0],
        [-3, 0, 0, 0],
        [-3, 0, 0, 0],
        [-3, 0, 0, 0],
        [-3, 0, 0, 0],
        [-3, 0, 0, 0],
        [-3, 0, 0, 0],
        [-3, 0, 0, 0],
    ]

    # Nutation in longitude and obliquity
    dpsi = np.zeros_like(jce)
    depsilon = np.zeros_like(jce)

    for i in range(len(Y)):
        xy = np.radians(
            X0 * Y[i][0] + X1 * Y[i][1] + X2 * Y[i][2] + X3 * Y[i][3] + X4 * Y[i][4]
        )
        dpsi += (abcd[i][0] + abcd[i][1] * jce) * np.sin(xy)
        depsilon += (abcd[i][2] + abcd[i][3] * jce) * np.cos(xy)

    dpsi /= 36000000
    depsilon /= 36000000

    # Mean obliquity of the ecliptic (temp.)
    epsilon0 = 2.45 * np.ones_like(jme)

    for coeff in [
        5.79,
        27.87,
        7.12,
        -39.05,
        -249.67,
        -51.38,
        1999.25,
        -1.55,
        -4680.93,
        84381.448,
    ]:
        epsilon0 *= jme
        epsilon0 /= 10
        epsilon0 += coeff

    # True obliquity of the ecliptic
    epsilon = np.radians(epsilon0 / 3600 + depsilon)

    # Aberration correction
    dtau = -20.4898 / (3600 * R)

    # Apparent sun longitude
    lambda_ = np.radians(theta + dpsi + dtau)

    # Right ascension (degrees)
    alpha = (
        np.degrees(
            np.arctan2(
                np.sin(lambda_) * np.cos(epsilon) - np.tan(beta) * np.sin(epsilon),
                np.cos(lambda_),
            )
        )
        % 360
    )

    # Declination
    sinDec = np.sin(beta) * np.cos(epsilon) + np.cos(beta) * np.sin(epsilon) * np.sin(
        lambda_
    )
    cosDec = np.sqrt(1 - sinDec ** 2)

    # Sun mean longitude
    M = (1 / 2000000) * np.ones_like(jme)
    for coeff in [-1 / 15300.0, 1 / 49931.0, 0.03032028, 360007.6982779, 280.4664567]:
        M *= jme
        M += coeff
    M %= 360

    # Equation of time (radians)
    eqnOfTime = M - 0.0057183 - alpha + dpsi * np.cos(epsilon)
    eqnOfTime = ((eqnOfTime + 180) % 360) - 180
    eqnOfTime *= np.pi / 180

    return sinDec, cosDec, eqnOfTime, solFactor


def orbit_aa(utc):
    """
    Orbit based on the "low precision" formulas in the Astronomical Almanac

    Ref: The Astronomical Almanac for the year 2019, pg. C5

    NB: Suitable for use between 1950 and 2050
    """

    jd, fjd = julian_day(utc)
    n = jd + fjd - 2451545.0

    # Mean longitude of sun (degrees)
    L = (280.460 + 0.9856474 * n) % 360

    # Mean anomaly (radians)
    g = np.radians((357.528 + 0.9856003 * n) % 360)

    # Ecliptic longitude (radians)
    lambda_ = np.radians((L + 1.915 * np.sin(g) + 0.020 * np.sin(2 * g)) % 360)

    # Obliquity of the ecliptic (radians)
    epsilon = np.radians(23.439 - 0.0000004 * n)

    # Right ascension (radians)
    alpha = np.arctan2(np.cos(epsilon) * np.sin(lambda_), np.cos(lambda_)) % (2 * np.pi)

    # Declination (radians)
    sinDec = np.sin(epsilon) * np.sin(lambda_)
    cosDec = np.sqrt(1 - sinDec ** 2)

    # Earth-sun radius (au)
    R = 1.00014 - 0.01671 * np.cos(g) - 0.00014 * np.cos(2 * g)
    solFactor = (1 / R) ** 2

    # Equation of time (radians)
    eqnOfTime = L - np.degrees(alpha)
    eqnOfTime = ((eqnOfTime + 180) % 360) - 180
    eqnOfTime *= np.pi / 180

    return sinDec, cosDec, eqnOfTime, solFactor


def orbit(utc, method=None):

    if method is None:
        method = "ASHRAE"

    if callable(method):
        func = method
        method = "Custom"
    else:
        method = method.upper()
        if method == "ASHRAE":
            func = orbit_ashrae
        elif method in ["CFSR", "CFSV2"]:
            func = orbit_cfsr
        elif method == "MERRA2":
            func = orbit_merra2
        elif method in ["ERA5", "ERA5LAND"]:
            func = orbit_era5
        elif method in ["EPW", "ENERGYPLUS"]:
            func = orbit_energyplus
        elif method in ["NOAA"]:
            func = orbit_noaa
        elif method in ["SG2"]:
            func = orbit_sg2
        elif method in ["SPA"]:
            func = orbit_spa
        elif method in ["AA"]:
            func = orbit_aa
        else:
            raise NotImplementedError(method)

    return func(utc)


def total_solar_irradiance_ashrae(utc):
    """
    Return ASHRAE constant solar irradiance value (W/m²)
    """

    return 1367.0 * (np.ones_like(utc).astype(float))


def total_solar_irradiance_era5(utc):
    """
    Return ERA5 constant solar irradiance value (W/m²)

    Ref: Hersbach, 2020. "The ERA5 global reanalysis". QJRMS v.146(730) p.2022

    Note: This is an average value. The value of TSI actually changes and
          must be pulled out of the TOA value if necessary.
    """

    return 4 * 340.4 * (np.ones_like(utc).astype(float))


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
    # fmt: off
    dTSI = np.array([
        6.70, 6.70, 6.80, 6.60, 6.20, 6.00, 5.70, 5.70, 5.80, 6.20, 6.50,
        6.50, 6.50, 6.40, 6.00, 5.80, 5.70, 5.70, 5.90, 6.40, 6.70, 6.70,
        6.80, 6.70, 6.30, 6.10, 5.90, 5.70
    ])
    # fmt: on
    n = len(dTSI)

    # Index into dTSI (float)
    i = np.asarray(year).astype(int) - 1979 + (np.asarray(month) - 7) / 12

    # Extend backward and/or forward assuming 11-year sunspot cycle
    while np.any(i < 0):
        i[i < 0] += 11
    while np.any(i > n - 1):
        i[i > n - 1] -= 11

    # Add base
    return TSI_datum + np.interp(i, np.arange(n), dTSI)


def total_solar_irradiance_merra2(utc):
    """
    Calculate MERRA-2 total solar irradiance (W/m²) based on year and month
    """

    year, month, _ = split_date(utc)

    # CMIP5 data (1980-2008), monthly
    # http://solarisheppa.geomar.de/solarisheppa/sites/default/files/data/CMIP5/TSI_WLS_mon_1882_2008.txt
    # fmt: off
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
    # fmt: on
    n = TSI.shape[0]

    # Index year
    i = np.asarray(year).astype(int) - 1980

    # Extend backward assuming 11-year sunspot cycle and forward assuming
    # 13-year
    while np.any(i < 0):
        i[i < 0] += 11
    while np.any(i > n - 1):
        i[i > n - 1] -= 13

    # Index month
    j = np.asarray(month).astype(int) - 1

    # Return index scaled by TIM correction (Total Irradiance Monitor)
    return 0.9965 * TSI[i, j]


def total_solar_irradiance_ceres(utc):
    """
    Calculate MERRA-2 total solar irradiance (W/m²) based on year and month
    """

    year, month, _ = split_date(utc)

    # CERES data (2000-), monthly-averaged from daily data
    # https://ceres.larc.nasa.gov/documents/TSIdata/CERES_EBAF_Ed2.8_DailyTSI.txt
    # fmt: off
    TSI = np.array([
        [1361.7523, 1361.6588, 1361.2588, 1361.5421, 1361.2730, 1361.3547,
         1361.4249, 1361.8603, 1361.3916, 1362.0602, 1361.8426, 1361.9997],
        [1361.9507, 1362.0393, 1361.4560, 1361.3103, 1361.4601, 1361.2170,
         1361.5496, 1361.2858, 1361.1207, 1361.7364, 1361.5359, 1361.8668],
        [1362.2816, 1362.2365, 1362.0543, 1361.5156, 1361.5108, 1361.5701,
         1360.9777, 1361.0790, 1361.3704, 1361.3915, 1361.3571, 1361.7648],
        [1361.6470, 1361.6172, 1361.0791, 1361.1044, 1361.1351, 1360.9831,
         1361.0729, 1361.1855, 1361.2915, 1360.2710, 1361.0254, 1361.2098],
        [1360.9773, 1360.9998, 1360.9793, 1361.0765, 1360.9440, 1360.8673,
         1360.5674, 1360.6825, 1360.9527, 1360.9588, 1360.9009, 1361.0936],
        [1360.6672, 1360.8620, 1360.8105, 1360.7953, 1360.7050, 1360.6961,
         1360.7407, 1360.8234, 1360.6588, 1360.8185, 1360.6546, 1360.7211],
        [1360.7894, 1360.7615, 1360.6741, 1360.6472, 1360.8229, 1360.7264,
         1360.6669, 1360.5537, 1360.6818, 1360.6736, 1360.4766, 1360.5116],
        [1360.5543, 1360.5729, 1360.6214, 1360.5625, 1360.5688, 1360.5950,
         1360.5655, 1360.5839, 1360.5460, 1360.5444, 1360.5241, 1360.5330],
        [1360.6021, 1360.5580, 1360.5020, 1360.5839, 1360.5476, 1360.5168,
         1360.4947, 1360.5073, 1360.5076, 1360.5320, 1360.5206, 1360.5212],
        [1360.5372, 1360.5282, 1360.5189, 1360.5056, 1360.5778, 1360.5473,
         1360.5364, 1360.5262, 1360.5499, 1360.5682, 1360.6239, 1360.5945],
        [1360.6734, 1360.7997, 1360.7470, 1360.8194, 1360.7880, 1360.7623,
         1360.8402, 1360.8097, 1360.8478, 1360.7767, 1360.8630, 1360.8290],
        [1360.8152, 1360.8242, 1360.8258, 1361.1585, 1361.1855, 1361.1634,
         1361.0289, 1360.9692, 1360.9470, 1361.1160, 1361.3190, 1361.4280],
        [1361.3137, 1361.3170, 1361.1408, 1361.2121, 1361.0881, 1361.1641,
         1360.9261, 1361.3606, 1361.4553, 1361.2822, 1361.2450, 1361.3706],
        [1361.1591, 1361.2902, 1361.3131, 1361.1774, 1361.3727, 1361.4137,
         1361.3053, 1361.3203, 1361.3012, 1360.9826, 1360.9674, 1361.1100],
        [1360.9067, 1360.5850, 1361.3852, 1361.2392, 1361.1794, 1361.0998,
         1361.1083, 1361.3099, 1361.3081, 1360.7533, 1361.4756, 1361.4171],
        [1361.5310, 1361.8810, 1361.6700, 1361.6641, 1361.4683, 1361.3103,
         1361.4573, 1361.1615, 1361.1014, 1361.3090, 1361.3597, 1361.2478],
        [1361.2843, 1361.3002, 1361.1816, 1360.8649, 1360.9579, 1361.0261,
         1360.9439, 1360.9476, 1360.9391, 1360.8935, 1360.8895, 1360.8105],
        [1360.3090, 1360.8225, 1360.7377, 1360.7796, 1360.7895, 1360.8099,
         1360.6574, 1360.6783, 1360.5297, 1360.8332, 1360.7905, 1360.7246],
        [1360.6759, 1360.6293, 1360.6488, 1360.6817, 1360.7066, 1360.7269,
         1360.7280, 1360.7113, 1360.7056, 1360.6902, 1360.6697, 1360.6469],
        [1360.6618, 1360.6728, 1360.6655, 1360.6054, 1360.6596, 1360.6880,
         1360.6985, 1360.6561, 1360.6628, 1360.6578, 1360.6468, 1360.6548],
        [1360.6664, 1360.6565, 1360.6744, 1360.6801, 1360.7391, 1360.7473,
         1360.7353, 1360.8115, 1360.7637, 1360.7909, 1360.7609, 1360.9386],
        [1360.9440, 1360.8790, 1360.9140, 1360.8687, 1360.8783, 1360.9566,
         1361.0599, 1361.0228, 1360.9877, 1361.0305, np.nan, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
         np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    ])
    # fmt: on
    n = TSI.shape[0]

    # Index year
    i = np.asarray(year).astype(int) - 2000

    # Return none outside available data
    i[i < 0] = -1
    i[i > n - 1] = -1

    # Index month
    j = np.asarray(month).astype(int) - 1

    # Lookup
    return TSI[i, j]


def total_solar_irradiance(utc, constant=None, method=None):
    """
    Calculate the total solar irradiance (W/m²) for given year.
    Year can be fractional.
    """

    if constant is not None:
        return float(constant) * np.ones_like(utc, dtype=float)

    if method is None:
        method = "ASHRAE"

    if callable(method):
        func = method

    else:
        method = method.upper()
        if method == "ASHRAE":
            func = total_solar_irradiance_ashrae
        elif method in ["CFSR", "CFSV2"]:
            func = total_solar_irradiance_cfsr
        elif method == "MERRA2":
            func = total_solar_irradiance_merra2
        elif method in ["ERA5", "ERA5LAND"]:
            func = total_solar_irradiance_era5
        elif method in ["EPW", "ENERGYPLUS"]:
            func = total_solar_irradiance_ashrae
        elif method in ["NOAA"]:
            func = total_solar_irradiance_ashrae
        elif method in ["CERES"]:
            func = total_solar_irradiance_ceres
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
    H = 2 * np.pi * ((hh - 12) / 24 + mm / 1440 + ss / 86400)

    # Correct based on equation of time
    H += eot

    # Move to longitude location
    H += np.radians(lon)

    # Return centered in -π to π
    return ((H + np.pi) % (2 * np.pi)) - np.pi


def sunset_hour_angle(sinLat, cosLat, sinDec, cosDec):
    """
    Calculate local sunset hour angle (radians) given sines and cosines
    of latitude and declination.
    """
    return np.arccos(np.clip(-sinDec / cosDec * sinLat / cosLat, -1, 1))


def position(lat, lon, utc, method="ASHRAE"):
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
        -cosDec * sinH,
        sinDec * cosLat - cosDec * sinLat * cosH,
        sinDec * sinLat + cosDec * cosLat * cosH,
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
    ab = B1 + B2 * tb + B3 * td + B4 * tb * td
    ad = D1 + D2 * tb + D3 * td + D4 * tb * td

    # Beam and diffuse irradiance
    return E0 * np.exp(-tb * m ** ab), E0 * np.exp(-td * m ** ad)


def elevation(lat, lon, utc, method="ASHRAE", constant=None, interval=None, h=None):
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
    E0 = solFactor * total_solar_irradiance(utc, constant=constant, method=method)

    # Latitudinal sines
    sinLat = np.sin(np.radians(lat))
    cosLat = np.cos(np.radians(lat))

    def int_elevation(h):
        """
        Instant elevation at hour angle h
        """
        return np.maximum(sinDec * sinLat + cosDec * cosLat * np.cos(h), 0)

    def avg_elevation(h1, h2):
        """
        Integrated elevation between h1 and h2
        """
        return np.maximum(
            sinLat * sinDec * (h2 - h1) + cosLat * cosDec * (np.sin(h2) - np.sin(h1)), 0
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
        h = hour_angle(lon, utc - np.timedelta64(30, "m"), eqnOfTime)
        # Instantaneous elevation
        z = int_elevation(h)

    elif interval == "h":
        """
        Hourly
        """
        # Sunset hour angle
        h0 = np.arccos(np.clip(-sinDec / cosDec * sinLat / cosLat, -1, 1))
        # One hour (radians)
        dh = np.pi / 12
        # Start and end hour angles
        h = hour_angle(lon, utc, eqnOfTime)
        a = (h - dh + np.pi) % (2 * np.pi) - np.pi
        b = a + dh
        # Default elevation is zero
        z = np.zeros_like(h)
        # Conditions
        a1 = a < -h0
        a2 = (a >= -h0) & (a < h0)
        # b1 = (b < -h0)
        b2 = (b >= -h0) & (b < h0)
        b3 = b >= h0
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
        h = np.arccos(np.clip(-sinDec / cosDec * sinLat / cosLat, -1, 1))
        # Average daily elevation
        z = avg_elevation(-h, h)
        # Scale by 24-hour interval
        z /= 2 * np.pi

    else:
        raise ValueError(
            "Interval must be one of 'instant', 'midpoint', " "'hourly', or 'daily'"
        )

    return z, E0


def air_mass(z):
    """
    Calculate air mass based on Kasten & Young 1989
    """
    beta = np.degrees(np.arcsin(z))
    return 1 / (z + 0.50572 * (6.07995 + beta) ** -1.6364)


def erbs(Kt, **kwargs):
    """
    Calculate diffuse fraction as a function of clearness index kt
    via Erbs relation
    """
    Kt = np.asarray(Kt)
    Kd = 0.9511 - 0.1604 * Kt + 4.388 * Kt ** 2 - 16.638 * Kt ** 3 + 12.336 * Kt ** 4
    np.copyto(Kd, 1.0 - 0.09 * Kt, where=Kt <= 0.22)
    np.copyto(Kd, 0.165, where=Kt > 0.80)
    return Kd


def orgill_hollands(Kt, **kwargs):
    """
    Calculate diffuse fraction as a function of clearness index kt
    via Orgill Hollands relation
    """
    Kt = np.asarray(Kt)
    Kd = 1.557 - 1.84 * Kt
    np.copyto(Kd, 1.0 - 0.249 * Kt, where=Kt <= 0.35)
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
        a[0]
        - a[1]
        * np.exp(-np.exp(a[2] + a[3] * Kt + a[4] * Kt ** 2 + a[5] * m + a[6] * m ** 2)),
        0,
        1,
    )


def engerer(Kt, Ktc, z, h, **kwargs):
    """
    Calculate diffuse fraction as a function of clearness index kt
    via the Engerer2 relation.

    kt is clearness index (E_gh/E0_h)
    ktc is clear sky clearness index (E_ghc/E0_h)
    z is cos(zenith), dimensionless
    h is hour angle, radians
    """

    # Apparent solar time in hours
    AST = 12 / np.pi * h
    # Zenith angle in degrees
    theta_z = np.degrees(np.arccos(z))
    dKtc = Ktc - Kt
    Kde = np.maximum(0, 1.0 - Ktc / Kt)
    C = 4.2336e-2
    beta = (-3.7912, 7.5479, -1.0036e-2, 3.1480e-3, -5.3146, 1.7073)
    return np.clip(
        C
        + (1.0 - C)
        / (
            1.0
            + np.exp(
                beta[0]
                + beta[1] * Kt
                + beta[2] * AST
                + beta[3] * theta_z
                + beta[4] * dKtc
            )
        )
        + beta[5] * Kde,
        0,
        1,
    )


def to_utc(lst, tz=0):
    """
    Convert datetime64 in local standard time to UTC
    """
    return lst - np.timedelta64(int(np.rint(tz * 60)), "m")


def to_lst(utc, tz=0):
    """
    Convert datetime64 in UTC to local standard time
    """
    return utc + np.timedelta64(int(np.rint(tz * 60)), "m")


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
    return np.degrees(np.arctan2(x, y)) % 360


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
        ab = B1 + B2 * tb + B3 * td + B4 * tb * td
        ad = D1 + D2 * tb + D3 * td + D4 * tb * td

        mab = mi ** ab
        mad = mi ** ad

        Kb = np.exp(-tb * mab)
        Kd = np.exp(-td * mad) / zi
        Kt = Kb + Kd

        # Form Jacobian J
        dKb_dtb = -Kb * mab * (1 + tb * logm * (B2 + B4 * td))
        dKd_dtb = -Kd * mad * td * logm * (D2 + D4 * td)
        dKb_dtd = -Kb * mab * tb * logm * (B3 + B4 * tb)
        dKd_dtd = -Kd * mad * (1 + td * logm * (D3 + D4 * tb))

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
        R = np.sum(dKt ** 2)

        # Form A, [J]^T[J]
        Abb = (1 + damping) * np.sum(dKt_dtb ** 2)
        Abd = np.sum(dKt_dtb * dKt_dtd)
        Add = (1 + damping) * np.sum(dKt_dtd ** 2)

        # Form forcing vector [J]^[dKt]
        Bb = np.sum(dKt_dtb * dKt)
        Bd = np.sum(dKt_dtd * dKt)

        # Solve A*t = B by Kramer's rule, Giddy-up
        try:
            detA = Abb * Add - Abd ** 2
            dtb = (Bb * Add - Bd * Abd) / detA
            dtd = (Abb * Bd - Abd * Bb) / detA
        except OverflowError:
            if not quiet:
                print("Warning: Determinant overflow while fitting taus")
            return np.nan, np.nan
        except ZeroDivisionError:
            if not quiet:
                print("Warning: Division by zero while fitting taus")
            return np.nan, np.nan
        except Exception:
            raise

        # Test
        Ktt, dKtt_dtb, dKtt_dtd = calc(tb + dtb, td + dtd)
        Rt = np.sum((Kti - Ktt) ** 2)

        if Rt >= R:
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
            print(
                "Warning: Exceeded",
                iter_max,
                "iterations while fitting taus:",
                tb,
                td,
                dtb,
                dtd,
            )

        return np.nan, np.nan

    if plot:
        import matplotlib.pyplot as plt

        plt.rc("text", usetex=True)
        plt.rc("text.latex", unicode=True)
        plt.rc("text.latex", preamble=r"\usepackage{cmbright}")
        f, ax = plt.subplots(figsize=(5, 5), dpi=200)
        ax.plot(mi, Kti, ".", color="orange", markersize=2)
        ax.plot(mi, Kt, ".", color="black", markersize=4)
        ax.set_xlabel("Air Mass $m$", fontsize="smaller")
        ax.set_ylabel("Clearness Index $K_t$", fontsize="smaller")
        txt = "\n".join(
            "%d points" % len(zi), "$\\tau_b$ = %.3f" % tb, "$\\tau_d$ = %.3f" % td
        )
        ax.text(0.9, 0.9, txt, ha="right", va="top", transform=ax.transAxes)
        plt.tight_layout()
        f.savefig("fit_taus_%d.png" % len(Kti), dpi=f.dpi, bbox_inches="tight")

    return tb, td


def fit_monthly_taus(z, Kt, lat=None, lon=None, noon_flux=False, **kwargs):

    # Loop over months
    months = list(range(1, 13))

    clear_sky = {"taub": [], "taud": []}

    if noon_flux:
        clear_sky["Ebnoon"] = []
        clear_sky["Ednoon"] = []

    for month in months:
        # Restrict to month
        i = z.index.month == month

        # Fit via non-linear least squares
        taub, taud = fit_taus(z[i], Kt[i], **kwargs)

        clear_sky["taub"].append(taub)
        clear_sky["taud"].append(taud)

        if noon_flux:

            if np.isnan(taub) or np.isnan(taud):
                clear_sky["Ebnoon"].append(np.nan)
                clear_sky["Ednoon"].append(np.nan)
                continue

            # Calculate noon elevation and solar ETR on the 21st day
            utc = join_date(2001, m=month, d=21, hh=12)
            z_noon, E0_noon = elevation(
                lat, lon, utc, method="ASHRAE", interval="instant", h=0
            )

            # Calculate corresponding beam and diffuse irradiance
            Eb, Ed = clear_sky_irradiance(z_noon, taub, taud, E0_noon)
            clear_sky["Ebnoon"].append(Eb)
            clear_sky["Ednoon"].append(Ed)

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
    z = E0h[d] / E0[d]

    # Calculate zenith angle (radians)
    Z = np.arccos(z)
    Z3 = Z ** 3

    # Calculate air mass
    m = air_mass(z)

    # Sky clearness (eqn 1)
    kappa = 1.04
    epsilon = ((Ed[d] + Eb[d]) / Ed[d] + kappa * Z3) / (1 + kappa * Z3)

    # Sky brightness (eqn 2)
    Delta = Ed[d] * m / E0[d]

    # Precipitable water (cm, eqn 3)
    W = np.exp(0.07 * Td[d] - 0.075)

    # Sky clearness categories (from overcast to clear)
    bin_edges = [1, 1.065, 1.230, 1.500, 1.950, 2.800, 4.500, 6.200]

    # Find clearnness bin
    i = np.searchsorted(bin_edges, epsilon, side="right") - 1

    # Global luminous efficacy (table 4)
    ai = np.array([96.63, 107.54, 98.73, 92.72, 86.73, 88.34, 78.63, 99.65])
    bi = np.array([-0.47, 0.79, 0.70, 0.56, 0.98, 1.39, 1.47, 1.86])
    ci = np.array([11.50, 1.79, 4.40, 8.36, 7.10, 6.06, 4.93, -4.46])
    di = np.array([-9.16, -1.19, -6.95, -8.31, -10.94, -7.60, -11.37, -3.15])

    # Global illuminance (lux, eqn. 6)
    It = Ed.copy()
    It[d] = (Eb[d] * z + Ed[d]) * (
        ai[i] + bi[i] * W + ci[i] * z + di[i] * np.log(Delta)
    )

    # Direct luminous efficiacy (table 4)
    ai = np.array([57.20, 98.99, 109.83, 110.34, 106.36, 107.19, 105.75, 101.18])
    bi = np.array([-4.55, -3.46, -4.90, -5.84, -3.97, -1.25, 0.77, 1.58])
    ci = np.array([-2.98, -1.21, -1.71, -1.99, -1.75, -1.51, -1.25, -1.10])
    di = np.array([117.12, 12.38, -8.81, -4.56, -6.16, -26.73, -34.44, -8.29])

    # Direct illuminance (lux, eqn. 8)
    Ib = Ed.copy()
    Ib[d] = Eb[d] * (ai[i] + bi[i] * W + ci[i] * np.exp(5.73 * Z - 5) + di[i] * Delta)
    Ib = np.maximum(0, Ib)

    # Diffuse luminous efficiacy (table 4)
    ai = np.array([97.24, 107.22, 104.97, 102.39, 100.71, 106.42, 141.88, 152.23])
    bi = np.array([-0.46, 1.15, 2.96, 5.59, 5.94, 3.83, 1.90, 0.35])
    ci = np.array([12.00, 0.59, -5.53, -13.95, -22.75, -36.15, -53.24, -45.27])
    di = np.array([-8.91, -3.95, -8.77, -13.90, -23.74, -28.83, -14.03, -7.98])

    # Diffuse illuminance (lux, eqn. 7)
    Id = Ed.copy()
    Id[d] = Ed[d] * (ai[i] + bi[i] * W + ci[i] * z + di[i] * np.log(Delta))

    # Zenith luminance prediction (table 4)
    ai = np.array([40.86, 26.58, 19.34, 13.25, 14.47, 19.76, 28.39, 42.91])
    ci = np.array([26.77, 14.73, 2.28, -1.39, -5.09, -3.88, -9.67, -19.62])
    cip = np.array([-29.59, 58.46, 100.00, 124.79, 160.09, 154.61, 151.58, 130.80])
    di = np.array([-45.75, -21.25, 0.25, 15.66, 9.13, -19.21, -69.39, -164.08])

    # Zenith luminance (Cd/m2, eqn. 10)
    Lz = Ed.copy()
    Lz[d] = Ed[d] * (ai[i] + ci[i] * z + cip[i] * np.exp(-3 * Z) + di[i] * Delta)

    return It, Ib, Id, Lz


def test_coeffs(year=2018):

    import matplotlib.pyplot as plt
    from itertools import cycle

    t1 = np.datetime64("%04d-01-01" % year)
    t2 = np.datetime64("%04d-01-01" % (year + 1,))

    utc = np.arange(t1, t2)

    f, ax = plt.subplots(4, sharex=True, figsize=(12, 9))

    methods = [
        "ashrae",
        "energyplus",
        "cfsr",
        "merra2",
        "era5",
        "noaa",
        "sg2",
        "spa",
        "aa",
    ]
    line_cycler = cycle(["-", "--", "-.", ":"])
    for method in methods:
        coeffs = orbit(utc, method=method)
        ls = next(line_cycler)
        for i, ylabel in enumerate(["sinDec", "cosDec", "eqnOfTime", "solFactor"]):
            ax[i].plot(utc, coeffs[i], ls=ls, lw=2, label=method)
            ax[i].set_ylabel(ylabel)

    ax[0].legend(
        loc="lower right",
        ncol=len(methods),
        bbox_to_anchor=(1, 1),
        frameon=False,
        borderaxespad=0,
        fontsize="smaller",
    )
    plt.tight_layout()
    plt.show()


def test_location(lat=33.64, lon=-84.43, dates=None):

    import matplotlib.pyplot as plt

    if dates is None:
        dates = [np.datetime64(datetime.datetime.utcnow())]

    for utc in dates:

        # 24 hours centered around UTC
        t = nearest_hour(utc) + np.arange(-12 * 60, 13 * 60, dtype="<m8[m]")
        print(nearest_hour(utc))

        f, ax = plt.subplots(3, figsize=(9, 6))

        methods = ["ashrae", "energyplus", "cfsr", "merra2", "era5", "noaa"]
        for method in methods:
            x, y, z = position(lat, lon, t, method=method)
            ax[0].plot_date(
                t.astype(datetime.datetime), to_altitude(z), label=method, fmt="-"
            )
            ax[1].plot_date(
                t.astype(datetime.datetime), to_azimuth(x, y), label=method, fmt="-"
            )
            z, E0 = elevation(lat, lon, t, method=method)
            ax[2].plot_date(t.astype(datetime.datetime), E0 * z, fmt="-", label=method)
            x0, y0, z0 = position(lat, lon, utc, method=method)
            print(to_altitude(z0), to_azimuth(x0, y0))
    ax[0].set_ylabel("Alt")
    ax[1].set_ylabel("Azi")
    ax[2].set_ylabel("TOA Horz")
    ax[0].legend(
        loc="lower right",
        ncol=len(methods),
        bbox_to_anchor=(1, 1),
        frameon=False,
        borderaxespad=0,
        fontsize="smaller",
    )
    f.autofmt_xdate()
    plt.tight_layout()
    plt.show()


def test_integration(lat=33.64, lon=-84.43, utc=None):

    if utc is None:
        utc = np.datetime64(datetime.datetime.utcnow())

    print("***")
    print(lat, lon, utc, to_lst(utc, tz=-5))

    for interval in ["instant", "hourly", "daily"]:
        print(elevation(lat, lon, utc, method="ASHRAE", interval=None))


def test_solar_irradiance():

    import matplotlib.pyplot as plt

    years, months = np.mgrid[1979:2024, 1:13]

    utc = join_date(y=years.flatten(), m=months.flatten())

    f, ax = plt.subplots(figsize=(9, 6))

    methods = ["ashrae", "cfsr", "merra2", "era5", "ceres"]
    for method in methods:
        tsi = total_solar_irradiance(utc, method=method)
        ax.plot_date(
            utc.astype(datetime.datetime), tsi, fmt="-", label=method, clip_on=False
        )

    ax.set_ylim(1360, 1368)
    ax.get_yaxis().get_major_formatter().set_useOffset(False)
    ax.legend(
        loc="lower right",
        ncol=len(methods),
        bbox_to_anchor=(1, 1),
        frameon=False,
        borderaxespad=0,
        fontsize="smaller",
    )
    ax.set_ylabel("TSI")
    plt.tight_layout()
    plt.show()


def test():

    # test_solar_irradiance()
    # test_coeffs(year=2018)
    test_location()
    # test_integration(lat=43.5448, lon=-80.2482)


if __name__ == "__main__":
    test()
