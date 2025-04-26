#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

"""
Formulae for the computation of measures of humidity
Ref: WMO No.8 v1, Annex 4.B, 2018, pg.188
"Guide to Instruments and Methods of Observations"
"""


def calc_p_factor(p=None):
    """
    Calculate enhancement factor for pressure (Pa)
    Eq. (4.B.5)
    """
    if p is None:
        return 1.0
    else:
        return np.where(np.isnan(p), 1.0, 1.0016 + 3.15e-8 * p - 7.4 / p)


def calc_sat_vp_water(db, p=None):
    """
    Calculate saturation vapor pressure e_w (Pa) given a dry-bulb db (°C) over water
    Eq. (4.B.1) and (4.B.2)
    """
    return calc_p_factor(p) * 611.2 * np.exp(17.62 * db / (243.12 + db))


def calc_sat_vp_ice(db, p=None):
    """
    Calculate saturation vapor pressure e_i (Pa) given a dry-bulb db (°C) over ice
    Eq. (4.B.3) and (4.B.4)
    """
    return calc_p_factor(p) * 611.2 * np.exp(22.46 * db / (272.62 + db))


def calc_sat_vap(db, p=None):
    """
    Calculate saturation vapor pressure e_w/i (Pa) given a dry-bulb db (°C)
    Branches over ice or water based on db
    """
    return np.where(db < 0, calc_sat_vp_ice(db, p=p), calc_sat_vp_water(db, p=p))


def calc_dp_from_rh(rh, db):
    """
    Calculate dew-point (°C) from RH (%) and dry-bulb (°C)
    Derived from Eq. (4.B.6) and (4.B.11)
    """

    # Vapor pressure (pressure is moot)
    vp = rh / 100 * calc_sat_vp_water(db, p=None)
    # Solve for dew-point
    c = np.log(vp / 611.2)
    return 243.12 * c / (17.62 - c)


def calc_dp_from_q_and_p(q, p):
    """
    Calculate dew-point (°C) from specific humidity (-) and pressure (Pa)
    """

    # Mixing ratio from Eq. (4.A.1) and (4.A.2)
    r = q / (1.0 - q)
    # In-situ vapour pressure = mole fraction × pressure from Eq. (4.A.6)
    vp = r / (0.62198 + r) * p
    # Eq. (4.B.6)
    c = np.log(vp / (611.2 * calc_p_factor(p)))
    return 243.12 * c / (17.62 - c)


def calc_fp_from_rh(rh, db):
    """
    Calculate frost-point (°C) from RH (%) and dry-bulb (°C)
    Derived from Eq. (4.B.6) and (4.B.11)
    """

    # Vapor pressure (pressure is moot)
    vp = rh / 100 * calc_sat_vp_water(db, p=None)
    # Solve for frost-point
    c = np.log(vp / 611.2)
    return 272.62 * c / (22.46 - c)


def calc_fp_from_q_and_p(q, p):
    """
    Calculate frost-point (°C) from specific humidity (-) and pressure (Pa)
    """

    # Mixing ratio from Eq. (4.A.1) and (4.A.2)
    r = q / (1.0 - q)
    # In-situ vapour pressure = mole fraction × pressure from Eq. (4.A.6)
    vp = r / (0.62198 + r) * p
    # Eq. (4.B.7)
    c = np.log(vp / (611.2 * calc_p_factor(p)))
    return 272.62 * c / (22.46 - c)


def calc_dp_or_fp_from_rh(rh, db):
    """
    Calculate dew-point *or* frost-point (°C) from RH (%) and dry-bulb (°C)
    Branches over ice or water based on db
    """
    return np.where(db < 0, calc_fp_from_rh(rh, db), calc_dp_from_rh(rh, db))


def calc_rh_from_db_and_dp(db, dp):
    """
    Calculate the relative humidity (%) given dry-bulb temperature t (°C)
    and dew-point temperature (°C).

    Note: According to WMO Guide, RH should be calculated relative to water,
          even for temperatures less than 0°C.
    """
    return 100.0 * np.minimum(calc_sat_vp_water(dp) / calc_sat_vp_water(db), 1.0)


def main():
    db = 20.0

    vp = calc_sat_vp_water(db)
    print(f"{db=} {vp=}")

    rh = 50
    dp = calc_dp_from_rh(rh, db)  # should be ~9.2°C
    print(f"{db=} {rh=} {dp=}")

    rh = calc_rh_from_db_and_dp(db, dp)
    print(f"{db=} {rh=} {dp=}")

    r = 9.921 / 1000.0
    q = r / (1 + r)
    p = 101325.0
    dp = calc_dp_from_q_and_p(q, p)
    print(f"{q=} {p=} {dp=}")


if __name__ == "__main__":
    main()
