# -*- coding: utf-8 -*-
#
# Copyright 2019 Klimaat
#

import numpy as np


def calc_ground_temperatures(
    Tavg,
    Tmin,
    Tmax,
    coldest_month,
    depth=[0.5, 2, 4],
    leap_year=False,
    diffusivity=2.3225760e-3,
):
    """
    Calculate ground temperatures given the average, minimum, and maximum
    monthly temperatures and the coldest month.

    Replicates code in CalcGroundTemps.f90 provided by Linda Lawrie
    """

    if leap_year:
        year_hours = 8784
        month_day_15 = np.array(
            [np.nan, 15, 47, 75, 96, 136, 167, 197, 228, 259, 289, 320, 350]
        )
    else:
        year_hours = 8760
        month_day_15 = np.array(
            [np.nan, 15, 46, 74, 95, 135, 166, 196, 227, 258, 288, 319, 349]
        )

    beta = np.asarray(depth) * np.sqrt(np.pi / (diffusivity * year_hours))

    ebeta = np.exp(-beta)
    cbeta = np.cos(beta)
    sbeta = np.sin(beta)

    amp = (
        0.5
        * (Tmax - Tmin)
        * np.sqrt((ebeta ** 2 - 2 * ebeta * cbeta + 1) / (2 * beta ** 2))
    )

    phi0 = 0.017214 * month_day_15[coldest_month] + 0.341787

    phi0 = phi0 + np.arctan(
        (1 - ebeta * (cbeta + sbeta)) / (1 - ebeta * (cbeta - sbeta))
    )

    phi = 2 * np.pi / year_hours * 24 * month_day_15[range(1, 13)]

    return np.squeeze(Tavg - amp[:, None] * np.cos(phi - phi0[:, None]))


def test(Tavg, Tmin, Tgnd_org):

    # Annual mean, min and max monthly-mean temperature and coldest month
    days_in_month = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    Tmean_mean = np.round(np.sum(Tavg * days_in_month) / np.sum(days_in_month), 1)
    Tmean_min = np.min(Tavg)
    Tmean_max = np.max(Tavg)

    # Coldest month
    coldest_month = np.argmin(Tmin) + 1

    # Calc
    Tgnd_calc = calc_ground_temperatures(
        Tmean_mean, Tmean_min, Tmean_max, coldest_month
    )

    # Error
    print("Error", Tgnd_calc - Tgnd_org)
    print("Bias", np.mean(Tgnd_calc - Tgnd_org))

    import matplotlib.pyplot as plt

    plt.plot(range(1, 13), Tgnd_org.T, "o-", label="Expected")
    plt.plot(range(1, 13), Tgnd_calc.T, "x-", label="Calculated")

    plt.legend(loc=0)

    plt.show()


def main():
    # fmt: off
    # ABU DHABI

    # Monthly mean temperatures
    Tavg = np.array([18.0, 19.9, 22.2, 26.5, 30.8, 32.8,
                     34.4, 34.8, 32.6, 28.6, 24.4, 20.1])

    # Monthly min temperatures
    Tmin = np.array([9.7, 5.0, 10.5, 17.5, 20.0, 23.0,
                    25.9, 25.1, 24.0, 18.7, 15.8, 9.1])

    # Monthly ground temperatures
    Tgnd = np.array([
        [22.80, 20.20, 19.49, 20.11, 23.61, 27.54,
         31.23, 33.91, 34.69, 33.43, 30.41, 26.59],
        [25.13, 22.70, 21.54, 21.50, 23.35, 26.04,
         28.93, 31.40, 32.68, 32.44, 30.74, 28.11],
        [26.66, 24.77, 23.61, 23.25, 23.93, 25.50,
         27.44, 29.35, 30.63, 30.94, 30.20, 28.67]
    ])

    test(Tavg, Tmin, Tgnd)

    # ATLANTA

    Tavg = np.array([4.0, 7.9, 13.8, 17.2, 20.8, 24.8,
                    26.1, 26.5, 22.5, 16.0, 11.9, 7.7])
    Tmin = np.array([-6.7, -12.8, 1.1, 0.0, 7.8, 11.1,
                     17.2, 18.3, 11.7, 2.2, -3.3, -5.6])
    Tgnd = np.array([
        [10.83, 7.34, 6.39, 7.21, 11.92, 17.20,
         22.17, 25.77, 26.82, 25.13, 21.06, 15.93],
        [13.97, 10.70, 9.14, 9.08, 11.57, 15.18,
         19.07, 22.40, 24.12, 23.79, 21.50, 17.97],
        [16.02, 13.48, 11.92, 11.44, 12.36, 14.46,
         17.07, 19.63, 21.36, 21.77, 20.78, 18.72]
    ])

    test(Tavg, Tmin, Tgnd)

    # fmt: on


if __name__ == "__main__":

    main()
