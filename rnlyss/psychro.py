# -*- coding: utf-8 -*-
#
# Copyright 2019 Klimaat

"""
Psychrometric relationships from ASHRAE Handbook of Fundamentals, Chapter 1
"""

import numpy as np

# Standard atmospheric pressure at at sea level (Pa)
P0_ = 101325.0

# Molecular weights
Mw = 18.015268  # Water
Mda = 28.966    # Dry air

# Gas Constants
R = 8314.472    # Universal (J/kmol/K)
Rda = R/Mda     # Dry air (J/kg/K)
Rw = R/Mw       # Water vapor (J/kg/K)

# Ratio of molecular weights
epsilon = Mw/Mda


def calc_saturation_vapor_pressure(t, jacobian=False):
    """
    Calculate saturation vapor pressure pws (Pa) given a temperature t (°C).
    From ASHRAE HOF 2013, pg 1.2.
    The two branches are identical at the triple point 0.01°C.
    """
    T = t + 273.15
    pws = np.where(
        # triple point
        t >= 0.01,
        # Over water: equation (6)
        np.exp(
            -5.8002206e3/T + 1.3914993 - 4.8640239e-2*T + 4.1764768e-5*T**2 -
            1.4452093e-8*T**3 + 6.5459673*np.log(T)
        ),
        # Over ice:  equation (5)
        np.exp(
            -5.6745359e3/T + 6.3925247 - 9.6778430e-3*T + 6.2215701e-7*T**2 +
            2.0747825e-9*T**3 - 9.4840240e-13*T**4 + 4.1635019*np.log(T)
        )
    )

    if jacobian:
        dpws = np.where(
            # triple point
            t >= 0.01,
            # Over water: derivative of equation (6)
            5.8002206e3/T**2 - 4.8640239e-2 + 2*4.1764768e-5*T -
            3*1.4452093e-8*T**2 + 6.5459673/T,
            # Over ice: derivation of equation (5)
            5.6745359e3/T**2 - 9.6778430e-3 + 2*6.2215701e-7*T +
            3*2.0747825e-9*T**2 - 4*9.4840240e-13*T**3 + 4.1635019/T
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

    alpha = np.log(pw/1000.0)

    td = np.where(
        pw >= 622.08212941,
        # Over water: equation (39)
        6.54 + 14.526*alpha + 0.7389*alpha**2 + 0.09486*alpha**3 +
        0.4569*(pw/1000.0)**0.1984,
        # Over ice: equation (40)
        6.09 + 12.608*alpha + 0.4959*alpha**2
    )

    if exact:

        # Refine above using Newton's method.
        # N.B. Usually only needs 3 iterations to get to well below-8.

        for i in range(3):
            pw_, dwp = calc_saturation_vapor_pressure(td, jacobian=True)
            dtd = (pw-pw_)/dwp
            td += dtd

    return td


def calc_vapor_pressure(Y, p=P0_):
    """
    Calculate vapor pressure pw (Pa) from specific humidity Y (-) and pressure
    p (Pa).
    """

    # Humidity ratio
    W = Y*(1-Y)

    # Water mole fraction
    xw = W/(epsilon+W)

    return p*xw


def calc_relative_humidity(t, td):
    """
    Calculate the relative humidity (-) given dry-bulb temperature t (°C)
    and dew-point temperature (°C).
    """
    return np.minimum(
        calc_saturation_vapor_pressure(td)/calc_saturation_vapor_pressure(t),
        1.0
    )


def calc_specific_humidity(td, p=P0_):
    """
    Calculate specific humidity (kg water per kg moist air) given
    dew point temperature td (°C) and pressure (Pa).
    """

    pw = calc_saturation_vapor_pressure(td)

    W = epsilon*pw/(p-pw)

    return W/(1+W)


def test():
    pass


if __name__ == "__main__":

    test()
