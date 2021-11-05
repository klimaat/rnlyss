# -*- coding: utf-8 -*-
#
# Copyright 2019 Klimaat

"""
Psychrometric relationships from ASHRAE Handbook of Fundamentals, Chapter 1
"""

import numpy as np

# Standard atmospheric pressure at at sea level (Pa)
P0_ = 101325.0

# Water molecular weight
Mw = 18.015268

# Dry air molecular weight
Mda = 28.966

# Universal gas constants (J/kmol/K)
R = 8314.472

# Dry air gas constant (J/kg/K)
Rda = R / Mda

# Water vapor gas constant (J/kg/K)
Rw = R / Mw

# Ratio of molecular weights
epsilon = Mw / Mda


def calc_saturation_vapor_pressure(t, jacobian=False):
    """
    Calculate saturation vapor pressure pws (Pa) given a temperature t (°C).
    From ASHRAE HOF 2013, pg 1.2.
    The two branches are identical at the triple point 0.01°C.
    """
    T = t + 273.15
    with np.errstate(invalid="ignore"):
        pws = np.where(
            # triple point
            t >= 0.01,
            # Over water: equation (6)
            np.exp(
                -5.8002206e3 / T
                + 1.3914993
                - 4.8640239e-2 * T
                + 4.1764768e-5 * T ** 2
                - 1.4452093e-8 * T ** 3
                + 6.5459673 * np.log(T)
            ),
            # Over ice:  equation (5)
            np.exp(
                -5.6745359e3 / T
                + 6.3925247
                - 9.6778430e-3 * T
                + 6.2215701e-7 * T ** 2
                + 2.0747825e-9 * T ** 3
                - 9.4840240e-13 * T ** 4
                + 4.1635019 * np.log(T)
            ),
        )

    if jacobian:
        with np.errstate(invalid="ignore"):
            dpws = np.where(
                # triple point
                t >= 0.01,
                # Over water: derivative of equation (6)
                5.8002206e3 / T ** 2
                - 4.8640239e-2
                + 2 * 4.1764768e-5 * T
                - 3 * 1.4452093e-8 * T ** 2
                + 6.5459673 / T,
                # Over ice: derivation of equation (5)
                5.6745359e3 / T ** 2
                - 9.6778430e-3
                + 2 * 6.2215701e-7 * T
                + 3 * 2.0747825e-9 * T ** 2
                - 4 * 9.4840240e-13 * T ** 3
                + 4.1635019 / T,
            )
        dpws *= pws

        return pws, dpws
    else:
        return pws


def calc_dew_point_temperature(pw, exact=False):
    """
    Calculate dewpoint temperature td (°C) given vapor pressure pw (Pa)
    using either an approximate equation or an exact inversion of the
    saturation vapor pressure equation.
    """

    # Approximation (eqns 37 and 38)
    # N.B. The two branches are identical at pw = 622.08212941 Pa

    alpha = np.log(pw / 1000.0)

    td = np.where(
        pw >= 622.08212941,
        # Over water: equation (39)
        6.54
        + 14.526 * alpha
        + 0.7389 * alpha ** 2
        + 0.09486 * alpha ** 3
        + 0.4569 * (pw / 1000.0) ** 0.1984,
        # Over ice: equation (40)
        6.09 + 12.608 * alpha + 0.4959 * alpha ** 2,
    )

    if exact:

        # Refine above using Newton's method.
        # N.B. Usually only needs 3 iterations to get to well below-8.

        for i in range(3):
            pw_, dwp = calc_saturation_vapor_pressure(td, jacobian=True)
            dtd = (pw - pw_) / dwp
            td += dtd

    return td


def calc_vapor_pressure(Y, p=P0_):
    """
    Calculate vapor pressure pw (Pa) from specific humidity Y (-) and pressure
    p (Pa).
    """

    # Humidity ratio
    W = Y * (1 - Y)

    # Water mole fraction
    xw = W / (epsilon + W)

    return p * xw


def calc_relative_humidity(t, td):
    """
    Calculate the relative humidity (-) given dry-bulb temperature t (°C)
    and dew-point temperature (°C).
    """
    return np.minimum(
        calc_saturation_vapor_pressure(td) / calc_saturation_vapor_pressure(t), 1.0
    )


def calc_humidity_ratio(td, p=P0_):
    """
    Calculate humidity ratio (kg water per kg dry air) given
    dew point temperature td (°C) and pressure (Pa).
    """
    pw = calc_saturation_vapor_pressure(td)
    return epsilon * pw / (p - pw)


def calc_specific_humidity(td, p=P0_):
    """
    Calculate specific humidity (kg water per kg moist air) given
    dew point temperature td (°C) and pressure (Pa).
    """
    W = calc_humidity_ratio(td, p=p)
    return W / (1 + W)


def calc_wetbulb_temperature(t, td, p=P0_, eps=1e-8, n=10):
    """
    Calculate wet-bulb temperature twb (°C) from dry-bulb temperature t (°C) and
    specific humidity Y (-) and pressure p (Pa).
    """

    """
    Options:
    * A Start with the 1/3-rule for wet-bulb.
    * B Start with wet-bulb = dry-bulb
    B has better overall convergence
    """

    # Humidity ratio
    W = calc_humidity_ratio(td, p=p)

    # First guess for wet-bulb (option A)
    twb = 2 / 3 * t + 1 / 3 * td

    dtwb = np.inf
    i = 0
    while np.abs(dtwb).max() > eps:

        # Indices with above zero wet bulb temperatures
        above = twb >= 0

        # Calculate vapor pressure (and derivative) at current wet-bulb
        pws_wb, dpws_wb = calc_saturation_vapor_pressure(twb, jacobian=True)

        # Calculate saturation humidity ratio (and derivative) at wet-bulb
        # temperature
        Ws_wb = epsilon * pws_wb / (p - pws_wb)
        dWs_wb = epsilon * p / (p - pws_wb) ** 2 * dpws_wb

        # Calculate a humidty ratio (and derivative) at current wet-bulb
        # using HOF 2013, Chap 1, eqn 36, split into numerator A and
        # denominator B
        A = np.where(
            above,
            (2501 - 2.326 * twb) * Ws_wb - 1.006 * (t - twb),
            (2830 - 0.24 * twb) * Ws_wb - 1.006 * (t - twb),
        )

        dA = np.where(
            above,
            -2.326 * Ws_wb + (2501 - 2.326 * twb) * dWs_wb + 1.006,
            -0.24 * Ws_wb + (2830 - 0.24 * twb) * dWs_wb + 1.006,
        )

        B = np.where(above, 2501 + 1.86 * t - 4.186 * twb, 2830 + 1.86 * t - 2.1 * twb)

        dB = np.where(above, -4.186, -2.1)

        W_ = A / B
        dW = (dA * B - A * dB) / B ** 2

        # Newton iteration
        dtwb = (W - W_) / dW
        twb += dtwb

        # Limit iterations
        i += 1
        if i > n:
            break

    return twb


def test():
    pass


if __name__ == "__main__":

    test()
